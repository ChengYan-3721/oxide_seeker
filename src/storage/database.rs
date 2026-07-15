use crate::error::{AppError, Result};
use sqlx::{sqlite::SqlitePoolOptions, FromRow, Row, SqlitePool};
use std::path::Path;

/// Shared database connection pool
pub type DbPool = SqlitePool;

/// File record from the `files` table
#[allow(dead_code)]
#[derive(Debug, Clone, FromRow)]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub filename: String,
    pub file_type: String,
    pub file_size: Option<i64>,
    pub modified_at: Option<i64>,
    pub page_count: i64,
    pub is_excluded: i64,
    pub indexed_at: Option<i64>,
    pub created_at: i64,
    /// Number of consecutive indexing attempts that did not reach a Rust-level
    /// success or error (i.e. the worker process was terminated mid-attempt,
    /// almost always by an FFI structured exception).  Used by
    /// `start_full_index` to auto-blacklist poison-pill files.
    pub crash_attempts: i64,
    /// Free-form note describing why the file is excluded.
    pub exclusion_reason: Option<String>,
    /// SHA-1 hex digest of the file's full byte content as of the last successful
    /// (re-)index.  Used by both the periodic rescan and the watcher to
    /// short-circuit "mtime moved but bytes are identical" re-indexing.
    pub content_sha1: Option<String>,
}

/// Page record from the `pages` table — one row per rendered page.
#[allow(dead_code)]
#[derive(Debug, Clone, FromRow)]
pub struct PageRecord {
    pub id: i64,
    pub file_id: i64,
    pub page_num: i64,
    pub width_px: Option<i64>,
    pub height_px: Option<i64>,
    pub thumb_path: Option<String>,
}

/// A region row joined with its owning page — everything the search path
/// needs to map an ANN / pHash hit back to `(file, page)`.
#[derive(Debug, Clone, FromRow)]
pub struct RegionHit {
    pub region_id: i64,
    pub page_id: i64,
    pub file_id: i64,
    pub page_num: i64,
}

/// Summary counts for the index status API
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub total_files: i64,
    pub indexed_files: i64,
    pub excluded_files: i64,
    pub failed_files: i64,
    pub total_pages: i64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, FromRow)]
pub struct LicenseRow {
    pub id: i64,
    pub install_fingerprint: String,
    pub install_started_at: i64,
    pub license_key: Option<String>,
    pub last_status: String,
    pub last_message: Option<String>,
    pub expires_at: Option<i64>,
    pub customer: Option<String>,
    pub validated_at: Option<i64>,
    pub updated_at: i64,
}

// ── Vector blob codec ────────────────────────────────────────────────────────

/// Encode an embedding as little-endian f32 bytes for the `regions.vector`
/// BLOB column.
pub fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Decode a `regions.vector` BLOB back into an embedding.  The dimension is
/// whatever the byte length implies — model-agnostic by design.
pub fn blob_to_vector(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Initialise the connection pool and run embedded migrations.
pub async fn init_pool(db_path: &Path) -> Result<DbPool> {
    if let Some(parent) = db_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let url = format!("sqlite://{}?mode=rwc", db_path.display());
    let pool = SqlitePoolOptions::new()
        .max_connections(16)
        .connect(&url)
        .await
        .map_err(AppError::Database)?;

    // Run SQL migrations from the `migrations/` directory.  A checksum
    // mismatch here almost always means the database predates the v2 schema
    // rewrite — there is no upgrade path (the embedding model changed), the
    // data directory must be deleted and the corpus re-indexed.
    if let Err(e) = sqlx::migrate!("./migrations").run(&pool).await {
        return Err(AppError::Config(format!(
            "Database migration failed: {}. If this database was created by an \
             older version, delete the data directory and re-index (the v2 \
             schema and embedding model are incompatible with v1 data).",
            e
        )));
    }

    // Post-migration bootstrap for schema additions that must work for both
    // old and new databases.
    ensure_license_schema(&pool).await?;

    sqlx::query("PRAGMA journal_mode=WAL").execute(&pool).await?;
    sqlx::query("PRAGMA synchronous=NORMAL").execute(&pool).await?;
    sqlx::query("PRAGMA foreign_keys=ON").execute(&pool).await?;

    tracing::info!("Database initialised at {}", db_path.display());
    Ok(pool)
}

// ── File operations ───────────────────────────────────────────────────────────

/// Insert or update a file record. Returns the file's `id`.
///
/// **Crash counter / exclusion auto-reset on content refresh.**
/// When the row already exists *and* the new `modified_at` is strictly greater
/// than the stored value (i.e. the user actually edited the file), we reset
/// `crash_attempts` back to 0, clear `is_excluded`, and clear
/// `exclusion_reason`.  This unblocks files that were previously
/// auto-blacklisted (e.g. designer saved an empty placeholder, then later
/// filled it in — without this reset that file would stay in the blacklist
/// forever).
///
/// `indexed_at` is **not** touched here — it's still the responsibility of
/// `mark_file_indexed` after a successful render.  This means a freshly
/// modified file will re-enter the to-index queue on the next scan even
/// though its row pre-existed.
pub async fn upsert_file(
    pool: &DbPool,
    path: &str,
    filename: &str,
    file_type: &str,
    file_size: Option<i64>,
    modified_at: Option<i64>,
) -> Result<i64> {
    let row = sqlx::query(
        r#"
        INSERT INTO files (path, filename, file_type, file_size, modified_at)
        VALUES (?1, ?2, ?3, ?4, ?5)
        ON CONFLICT(path) DO UPDATE SET
            filename    = excluded.filename,
            file_type   = excluded.file_type,
            file_size   = excluded.file_size,
            modified_at = excluded.modified_at,
            -- Reset poison-pill / blacklist state when the file actually changed.
            crash_attempts = CASE
                WHEN excluded.modified_at IS NOT NULL
                 AND files.modified_at IS NOT NULL
                 AND excluded.modified_at > files.modified_at
                THEN 0
                ELSE files.crash_attempts
            END,
            is_excluded = CASE
                WHEN excluded.modified_at IS NOT NULL
                 AND files.modified_at IS NOT NULL
                 AND excluded.modified_at > files.modified_at
                THEN 0
                ELSE files.is_excluded
            END,
            exclusion_reason = CASE
                WHEN excluded.modified_at IS NOT NULL
                 AND files.modified_at IS NOT NULL
                 AND excluded.modified_at > files.modified_at
                THEN NULL
                ELSE files.exclusion_reason
            END
        RETURNING id
        "#,
    )
    .bind(path)
    .bind(filename)
    .bind(file_type)
    .bind(file_size)
    .bind(modified_at)
    .fetch_one(pool)
    .await?;

    Ok(row.get::<i64, _>("id"))
}

/// Persist `sha1_hex` as the file's last-known content fingerprint.  Called
/// once a file has been (re-)indexed successfully so a subsequent rescan can
/// skip the heavy work when the bytes haven't changed.
pub async fn update_file_hash(
    pool: &DbPool,
    file_id: i64,
    sha1_hex: &str,
) -> Result<()> {
    sqlx::query("UPDATE files SET content_sha1 = ?2 WHERE id = ?1")
        .bind(file_id)
        .bind(sha1_hex)
        .execute(pool)
        .await?;
    Ok(())
}

/// Region ids currently attached to `file_id`.  Region ids double as HNSW
/// vector ids, so the caller uses this to tombstone stale ANN entries and to
/// evict stale in-memory pHash entries before re-indexing a modified file.
pub async fn get_region_ids_for_file(pool: &DbPool, file_id: i64) -> Result<Vec<i64>> {
    let ids: Vec<i64> = sqlx::query_scalar(
        "SELECT r.id FROM regions r \
         JOIN pages p ON p.id = r.page_id \
         WHERE p.file_id = ?1",
    )
    .bind(file_id)
    .fetch_all(pool)
    .await?;
    Ok(ids)
}

/// Delete every `pages` row (and, via CASCADE, every `regions` row)
/// belonging to `file_id`.  Called on the re-index path so a file that
/// shrank (e.g. 10 pages → 5) doesn't leave orphan rows behind.
pub async fn delete_pages_for_file(pool: &DbPool, file_id: i64) -> Result<()> {
    sqlx::query("DELETE FROM pages WHERE file_id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Mark a file as excluded (imposition/layout file).
pub async fn mark_file_excluded(pool: &DbPool, file_id: i64) -> Result<()> {
    sqlx::query("UPDATE files SET is_excluded = 1 WHERE id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Mark a file as excluded with a human-readable reason recorded in the
/// `exclusion_reason` column.  Used by the auto-blacklist for crashed files
/// so operators can audit why something was skipped.
pub async fn mark_file_excluded_with_reason(
    pool: &DbPool,
    file_id: i64,
    reason: &str,
) -> Result<()> {
    sqlx::query("UPDATE files SET is_excluded = 1, exclusion_reason = ?2 WHERE id = ?1")
        .bind(file_id)
        .bind(reason)
        .execute(pool)
        .await?;
    Ok(())
}

/// Increment `crash_attempts` for a file row and return the new value.
/// Must be committed BEFORE any FFI work that could terminate the process,
/// so that the increment survives an unrecoverable native crash.
pub async fn bump_crash_attempts(pool: &DbPool, file_id: i64) -> Result<i64> {
    let n: i64 = sqlx::query_scalar(
        "UPDATE files SET crash_attempts = crash_attempts + 1 \
         WHERE id = ?1 RETURNING crash_attempts",
    )
    .bind(file_id)
    .fetch_one(pool)
    .await?;
    Ok(n)
}

/// Reset `crash_attempts` to 0.  Called once a file has been processed to
/// completion at the Rust level (success or a non-crash error/panic) so that
/// only true FFI crashes accumulate the counter.
pub async fn reset_crash_attempts(pool: &DbPool, file_id: i64) -> Result<()> {
    sqlx::query("UPDATE files SET crash_attempts = 0 WHERE id = ?1")
        .bind(file_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Mark a file's indexing as complete and update its page count.
pub async fn mark_file_indexed(pool: &DbPool, file_id: i64, page_count: i64) -> Result<()> {
    sqlx::query(
        "UPDATE files SET indexed_at = unixepoch(), page_count = ?2 WHERE id = ?1",
    )
    .bind(file_id)
    .bind(page_count)
    .execute(pool)
    .await?;
    Ok(())
}

/// Fetch a file record by its path.
pub async fn get_file_by_path(pool: &DbPool, path: &str) -> Result<Option<FileRecord>> {
    let rec = sqlx::query_as::<_, FileRecord>("SELECT * FROM files WHERE path = ?1")
        .bind(path)
        .fetch_optional(pool)
        .await?;
    Ok(rec)
}

/// Delete a file record by its path, cascading to pages and regions.
/// Returns the region ids of the deleted regions so the caller can clean
/// up the in-memory vector index and pHash store.
pub async fn delete_file_by_path(pool: &DbPool, path: &str) -> Result<Vec<i64>> {
    // Collect region ids before deletion (CASCADE will remove pages+regions)
    let region_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT r.id FROM regions r \
         JOIN pages p ON p.id = r.page_id \
         WHERE p.file_id = (SELECT id FROM files WHERE path = ?1)",
    )
    .bind(path)
    .fetch_all(pool)
    .await?;

    sqlx::query("DELETE FROM files WHERE path = ?1")
        .bind(path)
        .execute(pool)
        .await?;

    Ok(region_ids)
}

/// Return all file paths stored in the DB.
pub async fn get_all_file_paths(pool: &DbPool) -> Result<Vec<String>> {
    let paths: Vec<String> = sqlx::query_scalar("SELECT path FROM files")
        .fetch_all(pool)
        .await?;
    Ok(paths)
}

/// Return every row in `files`, used by the indexer's filter stage to do a
/// single bulk DB load instead of N per-path round-trips.  At ~300 k files
/// the result is ≈ 50 MB in process memory — far cheaper than 300 k async
/// SQLite queries each costing a connection-pool round-trip.
pub async fn get_all_file_records(pool: &DbPool) -> Result<Vec<FileRecord>> {
    let recs = sqlx::query_as::<_, FileRecord>("SELECT * FROM files")
        .fetch_all(pool)
        .await?;
    Ok(recs)
}

// ── Index write path (pages + regions) ───────────────────────────────────────

/// One region to insert alongside its page.  `phash` is the raw u64;
/// `vector` is the L2-normalised embedding (stored as an f32-LE BLOB).
#[derive(Debug, Clone)]
pub struct NewRegion {
    /// `"full"` or `"tile"`.
    pub kind: &'static str,
    /// Tile position (row-major); 0 for the full region.
    pub idx: i64,
    /// Normalised bbox `(x, y, w, h)`; the full region is `(0, 0, 1, 1)`.
    pub bbox: (f32, f32, f32, f32),
    pub phash: u64,
    pub vector: Vec<f32>,
}

/// One page to insert, with its regions.
#[derive(Debug, Clone)]
pub struct NewPage {
    pub page_num: i64,
    pub width_px: i64,
    pub height_px: i64,
    /// Path of the page thumbnail relative to the thumbnails dir.
    pub thumb_path: String,
    pub regions: Vec<NewRegion>,
}

/// Insert all pages and regions for a freshly-indexed file in one
/// transaction and return the allocated region ids in the same order the
/// regions were provided (outer order: pages as given; inner order: regions
/// as given).  Region ids double as HNSW vector ids — the caller inserts
/// vectors into the ANN graph and pHashes into the in-memory store *after*
/// this commit succeeds, so a crash between the two leaves only a
/// rebuildable gap, never a dangling reference.
///
/// The caller must have removed any previous rows for this file
/// ([`delete_pages_for_file`]) — this function does plain INSERTs.
pub async fn insert_file_pages(
    pool: &DbPool,
    file_id: i64,
    pages: &[NewPage],
) -> Result<Vec<i64>> {
    if pages.is_empty() {
        return Ok(vec![]);
    }
    let total_regions: usize = pages.iter().map(|p| p.regions.len()).sum();
    let mut region_ids = Vec::with_capacity(total_regions);

    let mut tx = pool.begin().await?;
    for page in pages {
        let page_id: i64 = sqlx::query_scalar(
            "INSERT INTO pages (file_id, page_num, width_px, height_px, thumb_path) \
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
        )
        .bind(file_id)
        .bind(page.page_num)
        .bind(page.width_px)
        .bind(page.height_px)
        .bind(&page.thumb_path)
        .fetch_one(&mut *tx)
        .await?;

        for region in &page.regions {
            let region_id: i64 = sqlx::query_scalar(
                "INSERT INTO regions (page_id, kind, idx, bbox_x, bbox_y, bbox_w, bbox_h, phash, vector) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) RETURNING id",
            )
            .bind(page_id)
            .bind(region.kind)
            .bind(region.idx)
            .bind(region.bbox.0)
            .bind(region.bbox.1)
            .bind(region.bbox.2)
            .bind(region.bbox.3)
            .bind(region.phash as i64)
            .bind(vector_to_blob(&region.vector))
            .fetch_one(&mut *tx)
            .await?;
            region_ids.push(region_id);
        }
    }
    tx.commit().await?;
    Ok(region_ids)
}

// ── Search read path ─────────────────────────────────────────────────────────

/// Maximum number of bind parameters per SQL statement.  SQLite's compile-time
/// `SQLITE_MAX_VARIABLE_NUMBER` defaults to 32766 since 3.32; we stay well
/// below that to leave headroom for sqlx internals and any future composite
/// queries.
pub const SQL_BIND_CHUNK: usize = 16_000;

/// Resolve region ids (== HNSW vector ids / pHash store ids) to their owning
/// page and file.  Chunked to respect the bind-parameter limit.
pub async fn get_region_hits(pool: &DbPool, region_ids: &[i64]) -> Result<Vec<RegionHit>> {
    if region_ids.is_empty() {
        return Ok(vec![]);
    }
    let mut out = Vec::with_capacity(region_ids.len());
    for chunk in region_ids.chunks(SQL_BIND_CHUNK) {
        let placeholders: String = (1..=chunk.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT r.id AS region_id, r.page_id, p.file_id, p.page_num \
             FROM regions r JOIN pages p ON p.id = r.page_id \
             WHERE r.id IN ({})",
            placeholders
        );
        let mut query = sqlx::query_as::<_, RegionHit>(&sql);
        for id in chunk {
            query = query.bind(*id);
        }
        out.extend(query.fetch_all(pool).await?);
    }
    Ok(out)
}

/// Fetch page rows by primary key.  Used by the ranker to enrich surviving
/// candidates with thumbnails and dimensions.
pub async fn get_pages_by_ids(pool: &DbPool, page_ids: &[i64]) -> Result<Vec<PageRecord>> {
    if page_ids.is_empty() {
        return Ok(vec![]);
    }
    let mut out = Vec::with_capacity(page_ids.len());
    for chunk in page_ids.chunks(SQL_BIND_CHUNK) {
        let placeholders: String = (1..=chunk.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("SELECT * FROM pages WHERE id IN ({})", placeholders);
        let mut query = sqlx::query_as::<_, PageRecord>(&sql);
        for id in chunk {
            query = query.bind(*id);
        }
        out.extend(query.fetch_all(pool).await?);
    }
    Ok(out)
}

/// Fetch file rows by primary key.
pub async fn get_files_by_ids(pool: &DbPool, file_ids: &[i64]) -> Result<Vec<FileRecord>> {
    if file_ids.is_empty() {
        return Ok(vec![]);
    }
    let mut out = Vec::with_capacity(file_ids.len());
    for chunk in file_ids.chunks(SQL_BIND_CHUNK) {
        let placeholders: String = (1..=chunk.len())
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("SELECT * FROM files WHERE id IN ({})", placeholders);
        let mut query = sqlx::query_as::<_, FileRecord>(&sql);
        for id in chunk {
            query = query.bind(*id);
        }
        out.extend(query.fetch_all(pool).await?);
    }
    Ok(out)
}

// ── In-memory store bootstrap ────────────────────────────────────────────────

/// Load every `(region_id, phash)` pair.  Called once at startup to populate
/// the in-memory pHash store; incremental updates flow through the indexer
/// afterwards.  ~48 MB for 6 M regions — trivially cheap next to the HNSW
/// graph.
pub async fn load_phash_entries(pool: &DbPool) -> Result<Vec<(i64, u64)>> {
    let rows: Vec<(i64, i64)> =
        sqlx::query_as("SELECT id, phash FROM regions WHERE phash IS NOT NULL")
            .fetch_all(pool)
            .await?;
    Ok(rows.into_iter().map(|(id, h)| (id, h as u64)).collect())
}

/// Number of regions that carry a stored vector.  Used at startup to decide
/// whether the HNSW graph needs to be rebuilt from the database.
pub async fn count_regions_with_vectors(pool: &DbPool) -> Result<i64> {
    let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM regions WHERE vector IS NOT NULL")
        .fetch_one(pool)
        .await?;
    Ok(n)
}

/// Stream one batch of stored vectors ordered by region id, starting strictly
/// after `after_id`.  Used to rebuild the HNSW graph from the database
/// without holding every embedding in memory twice.
pub async fn load_vector_batch(
    pool: &DbPool,
    after_id: i64,
    limit: i64,
) -> Result<Vec<(i64, Vec<f32>)>> {
    let rows: Vec<(i64, Vec<u8>)> = sqlx::query_as(
        "SELECT id, vector FROM regions \
         WHERE vector IS NOT NULL AND id > ?1 \
         ORDER BY id LIMIT ?2",
    )
    .bind(after_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(id, blob)| (id, blob_to_vector(&blob)))
        .collect())
}

// ── OCR text channel ─────────────────────────────────────────────────────────

/// Pages that still lack an OCR row, ordered by id, starting strictly after
/// `after_id`.  Only successfully-indexed files are eligible — their
/// rendering has already been proven safe, which is why the backfill can run
/// in-process instead of behind the crash-isolated worker subprocess.
///
/// The cursor matters: without it the LEFT JOIN re-probes every already-done
/// page on each call, degrading linearly as coverage grows.  The caller
/// resets the cursor to 0 when a sweep comes back empty (catches new pages
/// and rows deleted by re-indexing).
pub async fn claim_ocr_batch(
    pool: &DbPool,
    after_id: i64,
    limit: i64,
) -> Result<Vec<(i64, String, i64)>> {
    let rows: Vec<(i64, String, i64)> = sqlx::query_as(
        "SELECT p.id, f.path, p.page_num FROM pages p \
         JOIN files f ON f.id = p.file_id \
         LEFT JOIN page_ocr o ON o.page_id = p.id \
         WHERE p.id > ?1 AND o.page_id IS NULL \
           AND f.is_excluded = 0 AND f.indexed_at IS NOT NULL \
         ORDER BY p.id LIMIT ?2",
    )
    .bind(after_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Store (or replace) the recognised text for a page.  An empty string is a
/// valid result — it marks "OCR ran, nothing readable" so the page is not
/// re-claimed forever.
pub async fn upsert_page_ocr(pool: &DbPool, page_id: i64, text: &str) -> Result<()> {
    sqlx::query(
        "INSERT INTO page_ocr (page_id, text) VALUES (?1, ?2) \
         ON CONFLICT(page_id) DO UPDATE SET text = excluded.text, ocr_at = unixepoch()",
    )
    .bind(page_id)
    .bind(text)
    .execute(pool)
    .await?;
    Ok(())
}

/// Full-text candidates for an OCR'd query, best-first.  `match_expr` must be
/// a pre-built FTS5 MATCH expression (see `search::build_fts_query`).
/// Returns `(page_id, bm25)` — SQLite's bm25() is lower-is-better.
pub async fn search_ocr(
    pool: &DbPool,
    match_expr: &str,
    limit: i64,
) -> Result<Vec<(i64, f64)>> {
    let rows: Vec<(i64, f64)> = sqlx::query_as(
        "SELECT rowid, bm25(page_ocr_fts) FROM page_ocr_fts \
         WHERE page_ocr_fts MATCH ?1 ORDER BY bm25(page_ocr_fts) LIMIT ?2",
    )
    .bind(match_expr)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// (done, pending) OCR coverage counters for the status API.
///
/// Both sides are cheap single-table COUNTs — the first implementation
/// LEFT-JOINed three tables per call, piled up under WAL write pressure,
/// exhausted the connection pool, and 500'd unrelated search requests.
/// The subtraction is exact because [`mark_orphan_pages_ocr_done`] backfills
/// placeholder rows for pages the claim query can never reach.
pub async fn ocr_progress(pool: &DbPool) -> Result<(i64, i64)> {
    let done: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM page_ocr")
        .fetch_one(pool)
        .await?;
    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM pages")
        .fetch_one(pool)
        .await?;
    Ok((done, (total - done).max(0)))
}

/// Backfill empty `page_ocr` rows for pages [`claim_ocr_batch`] can never
/// claim: files that indexed successfully (writing `pages` rows) but were
/// *later* excluded (crash auto-blacklist) or lost `indexed_at`.  Without
/// this, those rows sit in the pending count forever — production stalled
/// at "23 pending" on exactly this mismatch.  Empty text produces no
/// trigrams, so the placeholders are invisible to the FTS channel.
/// Returns the number of orphans marked; called when a backfill sweep drains.
pub async fn mark_orphan_pages_ocr_done(pool: &DbPool) -> Result<u64> {
    let result = sqlx::query(
        "INSERT INTO page_ocr (page_id, text) \
         SELECT p.id, '' FROM pages p \
         JOIN files f ON f.id = p.file_id \
         LEFT JOIN page_ocr o ON o.page_id = p.id \
         WHERE o.page_id IS NULL \
           AND (f.is_excluded = 1 OR f.indexed_at IS NULL)",
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

// ── License ──────────────────────────────────────────────────────────────────

async fn ensure_license_schema(pool: &DbPool) -> Result<()> {
   sqlx::query(
       r#"
       CREATE TABLE IF NOT EXISTS app_license (
           id                  INTEGER PRIMARY KEY CHECK (id = 1),
           install_fingerprint TEXT    NOT NULL,
           install_started_at  INTEGER NOT NULL,
           license_key         TEXT,
           last_status         TEXT    NOT NULL DEFAULT 'trial',
           last_message        TEXT,
           expires_at          INTEGER,
           customer            TEXT,
           validated_at        INTEGER,
           updated_at          INTEGER NOT NULL DEFAULT (unixepoch())
       )
       "#,
   )
   .execute(pool)
   .await?;

   sqlx::query(
       r#"
       INSERT OR IGNORE INTO app_license (id, install_fingerprint, install_started_at, last_status, updated_at)
       VALUES (1, '', 0, 'trial', unixepoch())
       "#,
   )
   .execute(pool)
   .await?;

   Ok(())
}

pub async fn get_or_init_license_row(
   pool: &DbPool,
   install_fingerprint: &str,
) -> Result<LicenseRow> {
   let now = chrono::Utc::now().timestamp();
   sqlx::query(
       r#"
       INSERT INTO app_license (id, install_fingerprint, install_started_at, last_status, updated_at)
       VALUES (1, ?1, ?2, 'trial', ?2)
       ON CONFLICT(id) DO NOTHING
       "#,
   )
   .bind(install_fingerprint)
   .bind(now)
   .execute(pool)
   .await?;

   sqlx::query(
       r#"
       UPDATE app_license
       SET install_fingerprint = ?1
       WHERE id = 1 AND (install_fingerprint = '' OR install_fingerprint IS NULL)
       "#,
   )
   .bind(install_fingerprint)
   .execute(pool)
   .await?;

   // Backfill install_started_at for rows bootstrapped with 0 in ensure_license_schema().
   // This guarantees trial start time is initialized correctly on first runtime.
   sqlx::query(
       r#"
       UPDATE app_license
       SET install_started_at = ?1,
           updated_at = ?1
       WHERE id = 1 AND install_started_at <= 0
       "#,
   )
   .bind(now)
   .execute(pool)
   .await?;

   let row = sqlx::query_as::<_, LicenseRow>("SELECT * FROM app_license WHERE id = 1")
       .fetch_one(pool)
       .await?;

   Ok(row)
}

pub async fn update_license_state(
   pool: &DbPool,
   status: &str,
   message: Option<&str>,
   expires_at: Option<i64>,
   customer: Option<&str>,
   validated_at: Option<i64>,
) -> Result<()> {
   sqlx::query(
       r#"
       UPDATE app_license
       SET last_status = ?1,
           last_message = ?2,
           expires_at = ?3,
           customer = ?4,
           validated_at = ?5,
           updated_at = unixepoch()
       WHERE id = 1
       "#,
   )
   .bind(status)
   .bind(message)
   .bind(expires_at)
   .bind(customer)
   .bind(validated_at)
   .execute(pool)
   .await?;
   Ok(())
}

pub async fn update_license_key(pool: &DbPool, license_key: Option<&str>) -> Result<()> {
   sqlx::query(
       r#"
       UPDATE app_license
       SET license_key = ?1,
           updated_at = unixepoch()
       WHERE id = 1
       "#,
   )
   .bind(license_key)
   .execute(pool)
   .await?;
   Ok(())
}

// ── Statistics ────────────────────────────────────────────────────────────────

pub async fn get_index_stats(pool: &DbPool) -> Result<IndexStats> {
    let row = sqlx::query(
        r#"
        SELECT
            COUNT(*) AS total_files,
            COUNT(CASE WHEN indexed_at IS NOT NULL AND is_excluded = 0 THEN 1 END) AS indexed_files,
            COUNT(CASE WHEN is_excluded = 1 THEN 1 END) AS excluded_files,
            -- "failed" = file record exists (upserted at start of processing)
            -- but indexing never completed: no indexed_at and not excluded.
            -- This matches the worker_pool behaviour where failed files increment
            -- progress.failed but do NOT call mark_file_indexed.
            COUNT(CASE WHEN indexed_at IS NULL AND is_excluded = 0 THEN 1 END) AS failed_files,
            (SELECT COUNT(*) FROM pages) AS total_pages
        FROM files
        "#,
    )
    .fetch_one(pool)
    .await?;

    Ok(IndexStats {
        total_files: row.get("total_files"),
        indexed_files: row.get("indexed_files"),
        excluded_files: row.get("excluded_files"),
        failed_files: row.get("failed_files"),
        total_pages: row.get("total_pages"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_blob_roundtrip() {
        let v = vec![0.25f32, -1.5, 3.0e-7, 0.0, f32::MAX];
        let blob = vector_to_blob(&v);
        assert_eq!(blob.len(), v.len() * 4);
        assert_eq!(blob_to_vector(&blob), v);
    }

    #[test]
    fn blob_to_vector_ignores_trailing_bytes() {
        let mut blob = vector_to_blob(&[1.0f32, 2.0]);
        blob.push(0xFF); // truncated write — decode the whole f32s only
        assert_eq!(blob_to_vector(&blob), vec![1.0, 2.0]);
    }

    /// End-to-end smoke test of the v2 write/read path against a real
    /// SQLite database (temp file, real migrations).  SQL strings are only
    /// validated at run time, so this is the earliest point schema drift
    /// can be caught.
    #[tokio::test]
    async fn schema_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_pool(&dir.path().join("test.db")).await.unwrap();

        let file_id = upsert_file(&pool, "X:/a/logo.pdf", "logo.pdf", "pdf", Some(123), Some(1000))
            .await
            .unwrap();

        // High-bit pHash exercises the u64 ↔ i64 bit-cast.
        let phash: u64 = 0xDEAD_BEEF_8000_0001;
        let pages = vec![NewPage {
            page_num: 1,
            width_px: 640,
            height_px: 480,
            thumb_path: "1_1.webp".to_string(),
            regions: vec![
                NewRegion {
                    kind: "full",
                    idx: 0,
                    bbox: (0.0, 0.0, 1.0, 1.0),
                    phash,
                    vector: vec![0.5f32; 8],
                },
                NewRegion {
                    kind: "tile",
                    idx: 3,
                    bbox: (0.25, 0.0, 0.5, 0.5),
                    phash: 42,
                    vector: vec![-0.5f32; 8],
                },
            ],
        }];

        let region_ids = insert_file_pages(&pool, file_id, &pages).await.unwrap();
        assert_eq!(region_ids.len(), 2);

        // Search read path: region id → (page, file).
        let hits = get_region_hits(&pool, &region_ids).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.file_id == file_id && h.page_num == 1));

        // pHash bulk load preserves the high bit.
        let entries = load_phash_entries(&pool).await.unwrap();
        assert!(entries.iter().any(|&(id, h)| id == region_ids[0] && h == phash));

        // Vector streaming returns decodable blobs in id order.
        assert_eq!(count_regions_with_vectors(&pool).await.unwrap(), 2);
        let batch = load_vector_batch(&pool, 0, 10).await.unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].1, vec![0.5f32; 8]);

        // Re-index cleanup: region ids drain, CASCADE clears regions.
        let stale = get_region_ids_for_file(&pool, file_id).await.unwrap();
        assert_eq!(stale.len(), 2);
        delete_pages_for_file(&pool, file_id).await.unwrap();
        assert_eq!(count_regions_with_vectors(&pool).await.unwrap(), 0);

        // Delete-by-path returns region ids for in-memory store eviction.
        let ids2 = insert_file_pages(&pool, file_id, &pages).await.unwrap();
        assert_eq!(ids2.len(), 2);
        let deleted = delete_file_by_path(&pool, "X:/a/logo.pdf").await.unwrap();
        assert_eq!(deleted.len(), 2);
    }

    /// FTS5 + trigram availability and the OCR channel's write/search path.
    /// Fails fast if the bundled SQLite was built without FTS5 — a deploy
    /// blocker better caught in CI than on the server.
    #[tokio::test]
    async fn ocr_fts_write_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let pool = init_pool(&dir.path().join("t.db")).await.unwrap();

        let file_id = upsert_file(&pool, "X:/b/tea.pdf", "tea.pdf", "pdf", None, Some(1))
            .await
            .unwrap();
        let pages = vec![NewPage {
            page_num: 1,
            width_px: 100,
            height_px: 100,
            thumb_path: "t.webp".into(),
            regions: vec![NewRegion {
                kind: "full",
                idx: 0,
                bbox: (0.0, 0.0, 1.0, 1.0),
                phash: 1,
                vector: vec![0.1; 4],
            }],
        }];
        let _ = insert_file_pages(&pool, file_id, &pages).await.unwrap();
        mark_file_indexed(&pool, file_id, 1).await.unwrap();

        // Page shows up as pending, then not after upsert.
        let batch = claim_ocr_batch(&pool, 0, 10).await.unwrap();
        assert_eq!(batch.len(), 1);
        let page_id = batch[0].0;
        upsert_page_ocr(&pool, page_id, "金田茶叶精品包装 VUK1457-115T")
            .await
            .unwrap();
        assert!(claim_ocr_batch(&pool, 0, 10).await.unwrap().is_empty());
        assert_eq!(ocr_progress(&pool).await.unwrap(), (1, 0));

        // Trigram substring matching: partial product code, CJK substring.
        for q in ["\"UK1457\"", "\"茶叶精品\""] {
            let hits = search_ocr(&pool, q, 10).await.unwrap();
            assert_eq!(hits.len(), 1, "query {} should hit", q);
            assert_eq!(hits[0].0, page_id);
        }
        // Non-matching query returns nothing.
        assert!(search_ocr(&pool, "\"不存在的词组\"", 10).await.unwrap().is_empty());

        // Cascade: deleting the file clears page_ocr and the FTS index.
        delete_file_by_path(&pool, "X:/b/tea.pdf").await.unwrap();
        assert_eq!(ocr_progress(&pool).await.unwrap().0, 0);
        assert!(search_ocr(&pool, "\"茶叶精品\"", 10).await.unwrap().is_empty());
    }
}
