-- Persistent poison-pill detection for FFI crashes (pdfium / onnxruntime
-- raising a structured exception that Rust catch_unwind cannot intercept).
--
-- crash_attempts is bumped before each indexing attempt and reset on either
-- success or a Rust-level error.  An FFI crash terminates the process before
-- the reset runs, so the counter survives.  When start_full_index sees a file
-- with a non-zero counter it can decide to skip / blacklist the file.
ALTER TABLE files ADD COLUMN crash_attempts INTEGER NOT NULL DEFAULT 0;

-- Free-form note on why a file is excluded (imposition rule, repeated crashes,
-- manual override, ...).  Optional; NULL means the historical reason was just
-- "imposition" because that's the only thing the original schema tracked.
ALTER TABLE files ADD COLUMN exclusion_reason TEXT;
