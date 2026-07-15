-- OxideSeeker schema — v2 (clean rewrite, no migration path from v1).
--
-- v1 databases are NOT compatible: the vector model changed (CLIP → DINOv2),
-- so every embedding must be recomputed anyway.  Delete the old data
-- directory and re-index.
--
-- Layout
-- ------
--   files    one row per PDF/AI file on disk (scan bookkeeping, crash
--            blacklist, content-hash idempotency)
--   pages    one row per rendered page (thumbnail + pixel dimensions)
--   regions  one row per embedded region: the full page plus 9 overlapping
--            tiles (size = ½ page, stride = ¼ page → any target smaller than
--            ¼ of the page is fully contained in at least one tile).
--
-- regions.id doubles as the HNSW vector id, and regions.vector stores the
-- raw f32-LE embedding so the ANN graph is a *rebuildable cache*: dropping
-- the vectors.* dump files never requires re-running model inference.

CREATE TABLE files (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    path             TEXT    NOT NULL UNIQUE,   -- absolute file path
    filename         TEXT    NOT NULL,
    file_type        TEXT    NOT NULL,          -- 'pdf' | 'ai'
    file_size        INTEGER,                   -- bytes
    modified_at      INTEGER,                   -- Unix timestamp (mtime)
    page_count       INTEGER DEFAULT 1,
    is_excluded      INTEGER DEFAULT 0,         -- 1 = imposition/layout file, skip
    indexed_at       INTEGER,                   -- Unix timestamp of last successful index
    created_at       INTEGER DEFAULT (unixepoch()),
    -- Consecutive attempts that died mid-index (FFI structured exception in
    -- pdfium/onnxruntime).  Bumped before the attempt, reset after any
    -- Rust-level completion, so only true native crashes accumulate.
    crash_attempts   INTEGER NOT NULL DEFAULT 0,
    -- Why the file is excluded (imposition rule, crash blacklist, ...).
    exclusion_reason TEXT,
    -- SHA-1 of the file bytes as of the last successful index.  Lets rescans
    -- skip files whose mtime moved but whose content didn't.
    content_sha1     TEXT
);

CREATE TABLE pages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page_num   INTEGER NOT NULL,                -- 1-based
    width_px   INTEGER,                         -- render size at index time
    height_px  INTEGER,
    thumb_path TEXT,                            -- relative path under thumbnails/
    UNIQUE (file_id, page_num)
);

CREATE TABLE regions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,  -- == HNSW vector id
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    kind    TEXT    NOT NULL DEFAULT 'full',    -- 'full' | 'tile'
    idx     INTEGER NOT NULL DEFAULT 0,         -- tile position 0-8; 0 for 'full'
    -- Normalised bbox in page coordinates ([0,1]); the full region is
    -- (0,0,1,1).  Kept for future "highlight the matched area" UI.
    bbox_x  REAL,
    bbox_y  REAL,
    bbox_w  REAL,
    bbox_h  REAL,
    -- 64-bit perceptual hash stored as a signed i64 (bit-cast of the u64).
    phash   INTEGER,
    -- Raw embedding, f32 little-endian, dim inferred from byte length.
    vector  BLOB,
    UNIQUE (page_id, kind, idx)
);

-- Index job queue / status tracking
CREATE TABLE index_tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER REFERENCES files(id) ON DELETE CASCADE,
    status     TEXT    DEFAULT 'pending',       -- pending|processing|done|failed
    error_msg  TEXT,
    attempts   INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (unixepoch()),
    updated_at INTEGER DEFAULT (unixepoch())
);

CREATE INDEX idx_files_modified     ON files(modified_at);
CREATE INDEX idx_files_content_sha1 ON files(content_sha1);
CREATE INDEX idx_pages_file_id      ON pages(file_id);
CREATE INDEX idx_regions_page_id    ON regions(page_id);
CREATE INDEX idx_tasks_status       ON index_tasks(status);
