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
    /// Free-form note describing why the file is excluded.  NULL on rows that
    /// pre-date this column.
    pub exclusion_reason: Option<String>,
}

/// Page record from the `pages` table
#[allow(dead_code)]
#[derive(Debug, Clone, FromRow)]
pub struct PageRecord {
    pub id: i64,
    pub file_id: i64,
    pub page_num: i64,
    pub phash: Option<String>,
    pub vector_id: Option<i64>,
    pub thumb_path: Option<String>,
    pub width_px: Option<i64>,
    pub height_px: Option<i64>,
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

    // Run SQL migrations from the `migrations/` directory
   sqlx::migrate!("./migrations")
       .run(&pool)
       .await
       .map_err(AppError::Migration)?;

   // Post-migration bootstrap for schema additions that were introduced after
   // initial deployment but must work for both old and new databases.
   ensure_license_schema(&pool).await?;

   sqlx::query("PRAGMA journal_mode=WAL").execute(&pool).await?;
    sqlx::query("PRAGMA synchronous=NORMAL").execute(&pool).await?;
    sqlx::query("PRAGMA foreign_keys=ON").execute(&pool).await?;

    tracing::info!("Database initialised at {}", db_path.display());
    Ok(pool)
}

// ── File operations ───────────────────────────────────────────────────────────

/// Insert or update a file record. Returns the file's `id`.
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
            modified_at = excluded.modified_at
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

/// Delete a file record by its path, cascading to pages and index_tasks.
/// Returns the vector_ids of the deleted pages so the caller can clean
/// up the in-memory vector index.
pub async fn delete_file_by_path(pool: &DbPool, path: &str) -> Result<Vec<i64>> {
    // Collect vector_ids before deletion (CASCADE will remove pages)
    let vector_ids: Vec<i64> = sqlx::query_scalar(
        "SELECT vector_id FROM pages WHERE file_id = (SELECT id FROM files WHERE path = ?1) AND vector_id IS NOT NULL",
    )
    .bind(path)
    .fetch_all(pool)
    .await?;

    sqlx::query("DELETE FROM files WHERE path = ?1")
        .bind(path)
        .execute(pool)
        .await?;

    Ok(vector_ids)
}

/// Return all file paths stored in the DB.
pub async fn get_all_file_paths(pool: &DbPool) -> Result<Vec<String>> {
    let paths: Vec<String> = sqlx::query_scalar("SELECT path FROM files")
        .fetch_all(pool)
        .await?;
    Ok(paths)
}

/// A single row to upsert into the `pages` table.  Used by
/// [`upsert_pages_batch`] so the worker pool can persist every page of a
/// freshly-indexed file in a single transaction (one fsync for N pages).
#[derive(Debug, Clone)]
pub struct PageUpsert<'a> {
    pub page_num: i64,
    pub phash: Option<&'a str>,
    pub vector_id: Option<i64>,
    pub thumb_path: Option<&'a str>,
    pub width_px: Option<i64>,
    pub height_px: Option<i64>,
}

/// Upsert every row in `rows` for the given `file_id` inside a single
/// transaction.  This is the hot path for the indexer: a 30-page PDF
/// previously issued 30 independent round-trips to SQLite; now it's one.
pub async fn upsert_pages_batch(
    pool: &DbPool,
    file_id: i64,
    rows: &[PageUpsert<'_>],
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut tx = pool.begin().await?;
    for row in rows {
        sqlx::query(
            r#"
            INSERT INTO pages (file_id, page_num, phash, vector_id, thumb_path, width_px, height_px)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(file_id, page_num) DO UPDATE SET
                phash      = excluded.phash,
                vector_id  = excluded.vector_id,
                thumb_path = excluded.thumb_path,
                width_px   = excluded.width_px,
                height_px  = excluded.height_px
            "#,
        )
        .bind(file_id)
        .bind(row.page_num)
        .bind(row.phash)
        .bind(row.vector_id)
        .bind(row.thumb_path)
        .bind(row.width_px)
        .bind(row.height_px)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Fetch pages by a list of vector IDs.
pub async fn get_pages_by_vector_ids(
    pool: &DbPool,
    vector_ids: &[i64],
) -> Result<Vec<PageRecord>> {
    if vector_ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: String = (1..=vector_ids.len())
        .map(|i| format!("?{}", i))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT * FROM pages WHERE vector_id IN ({})",
        placeholders
    );
    let mut query = sqlx::query_as::<_, PageRecord>(&sql);
    for id in vector_ids {
        query = query.bind(*id);
    }
    Ok(query.fetch_all(pool).await?)
}

/// Return all (page_id, phash_hex) pairs for pHash candidate filtering.
pub async fn find_pages_by_phash_candidates(pool: &DbPool) -> Result<Vec<(i64, String)>> {
    let rows = sqlx::query("SELECT id, phash FROM pages WHERE phash IS NOT NULL")
        .fetch_all(pool)
        .await?;

    Ok(rows
        .into_iter()
        .filter_map(|r| {
            let id: i64 = r.get("id");
            let phash: Option<String> = r.get("phash");
            phash.map(|h| (id, h))
        })
        .collect())
}

// ── Statistics ────────────────────────────────────────────────────────────────

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