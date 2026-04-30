-- Files table: one row per PDF/AI file on disk
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT    NOT NULL UNIQUE,        -- absolute file path
    filename    TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,               -- 'pdf' | 'ai'
    file_size   INTEGER,                        -- bytes
    modified_at INTEGER,                        -- Unix timestamp (mtime)
    page_count  INTEGER DEFAULT 1,
    is_excluded INTEGER DEFAULT 0,              -- 1 = imposition/layout file, skip
    indexed_at  INTEGER,                        -- Unix timestamp of last successful index
    created_at  INTEGER DEFAULT (unixepoch())
);

-- Pages table: one row per rendered page (file_id → many pages)
CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page_num    INTEGER NOT NULL,               -- 1-based
    phash       TEXT,                           -- 16-char hex string (64-bit pHash)
    vector_id   INTEGER,                        -- key in the usearch index
    thumb_path  TEXT,                           -- relative path under thumbnails/
    width_px    INTEGER,
    height_px   INTEGER,
    UNIQUE(file_id, page_num)
);

-- Index job queue / status tracking
CREATE TABLE IF NOT EXISTS index_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER REFERENCES files(id) ON DELETE CASCADE,
    status      TEXT    DEFAULT 'pending',      -- pending|processing|done|failed
    error_msg   TEXT,
    attempts    INTEGER DEFAULT 0,
    created_at  INTEGER DEFAULT (unixepoch()),
    updated_at  INTEGER DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_files_path        ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_modified    ON files(modified_at);
CREATE INDEX IF NOT EXISTS idx_pages_file_id     ON pages(file_id);
CREATE INDEX IF NOT EXISTS idx_pages_phash       ON pages(phash);
CREATE INDEX IF NOT EXISTS idx_pages_vector_id   ON pages(vector_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status      ON index_tasks(status);