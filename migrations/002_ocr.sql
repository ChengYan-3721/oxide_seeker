-- OCR text channel (P4).
--
-- One row per page in `page_ocr`; full-text search via an FTS5 external
-- content table with the trigram tokenizer:
--   * no CJK segmenter needed — trigrams work directly on Chinese text;
--   * substring matching: a screenshot containing half a product code still
--     matches ("UK1457" finds "VUK1457-115T");
--   * tolerant of isolated OCR misreads (the surrounding trigrams still hit).
--
-- Rows are created exclusively by the backfill loop (see
-- indexer::ocr_backfill); a freshly indexed page simply has no row here yet,
-- which is what marks it as pending.  Deleting a page cascades here, and the
-- triggers keep the FTS index in sync.

CREATE TABLE page_ocr (
    page_id INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
    text    TEXT NOT NULL DEFAULT '',
    ocr_at  INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE VIRTUAL TABLE page_ocr_fts USING fts5(
    text,
    content='page_ocr',
    content_rowid='page_id',
    tokenize='trigram'
);

CREATE TRIGGER page_ocr_ai AFTER INSERT ON page_ocr BEGIN
    INSERT INTO page_ocr_fts(rowid, text) VALUES (new.page_id, new.text);
END;
CREATE TRIGGER page_ocr_ad AFTER DELETE ON page_ocr BEGIN
    INSERT INTO page_ocr_fts(page_ocr_fts, rowid, text)
    VALUES ('delete', old.page_id, old.text);
END;
CREATE TRIGGER page_ocr_au AFTER UPDATE ON page_ocr BEGIN
    INSERT INTO page_ocr_fts(page_ocr_fts, rowid, text)
    VALUES ('delete', old.page_id, old.text);
    INSERT INTO page_ocr_fts(rowid, text) VALUES (new.page_id, new.text);
END;
