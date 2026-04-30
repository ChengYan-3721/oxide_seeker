use crate::error::{AppError, Result};
use sqlx::{sqlite::SqlitePoolOptions, FromRow, Row, SqlitePool};
use std::path::Path;

/// Shared database connection pool
pub type DbPool = SqlitePool;

/// File record from the `files` table
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
}

/// Page record from the `pages` table
#[derive(Debug, Clone, FromRow)]
pub struct PageRecord {
    pub id: i64,
    pub file_id: i64,
    pub page_num: i64,
    pub vector_id: Option<i64>,
    pub thumb_path: Option<String>,
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