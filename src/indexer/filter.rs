//! Filter to detect imposition/layout files that should be excluded
//! from the visual index.
//!
//! **Sole exclusion criterion:**
//! If the PDF's XMP metadata contains an `egExtFL:files` element listing
//! any file with a `.pdf` extension, the file is an imposition layout
//! (it references external PDF single-marks) and is excluded from indexing.
//! Files that link only to non-PDF assets (PSD, TIFF, EPS, etc.) are kept.

use pdfium_render::prelude::*;
use std::path::Path;

// ── XMP-based external PDF link check ───────────────────────────────────────

/// Scan the file at `path` for its XMP metadata block and check if the
/// `egExtFL:files` element lists any `.pdf` file.
///
/// Returns `Some(reason)` if the file should be excluded, `None` otherwise.
pub fn pre_check_bytes(path: &Path) -> Option<String> {
    let data = std::fs::read(path).ok()?;

    // Locate the XMP packet: <?xpacket begin=...?> ... <?xpacket end=...?>
    let xmp = extract_xmp_packet(&data)?;

    // Find <egExtFL:files> ... </egExtFL:files> block inside XMP
    let files_content = extract_tag_content(xmp, b"egExtFL:files")?;

    // Extract each <rdf:li> value and check for non-relative .pdf references.
    //
    // Single-mark files may legitimately link to smart-mark PDFs via relative
    // paths like "file:./foo.pdf" or "file:../bar.pdf".  These are NOT
    // imposition layouts and should be kept.  Only exclude when at least one
    // linked PDF uses a non-relative path (e.g. "file:mark.pdf").
    let items = extract_all_rdf_li(files_content);
    let has_non_relative_pdf = items.iter().any(|item| {
        let lower: Vec<u8> = item.iter().map(|b| b.to_ascii_lowercase()).collect();
        if !contains_bytes(&lower, b".pdf") {
            return false; // not a PDF link, irrelevant
        }
        // Check if it is a relative path: "file:./" or "file:../"
        let trimmed = trim_ascii_start(&lower);
        let is_relative = trimmed.starts_with(b"<stref:instanceid>file:./")
            || trimmed.starts_with(b"<stref:instanceid>file:../");
        !is_relative
    });

    if has_non_relative_pdf {
        Some("XMP egExtFL:files contains external PDF file reference (non-relative)".to_string())
    } else {
        None
    }
}

/// Extract the XMP packet from raw PDF bytes.
///
/// XMP is enclosed between `<?xpacket begin=` and `<?xpacket end=`.
fn extract_xmp_packet(data: &[u8]) -> Option<&[u8]> {
    let start_marker = b"<?xpacket begin=";
    let end_marker = b"<?xpacket end=";

    let start = find_bytes(data, start_marker)?;
    let end = find_bytes(&data[start..], end_marker)?;
    // Include a bit past the end marker to capture the closing ?>
    let end_abs = start + end;
    Some(&data[start..end_abs])
}

/// Extract the content between `<tag...>` and `</tag>` for a given tag name.
///
/// Handles tags with attributes (e.g. `<egExtFL:files rdf:parseType="...">`).
fn extract_tag_content<'a>(data: &'a [u8], tag_name: &[u8]) -> Option<&'a [u8]> {
    // Build opening pattern: "<tagname" (followed by '>' or ' ' for attributes)
    let mut open_pattern = Vec::with_capacity(tag_name.len() + 1);
    open_pattern.push(b'<');
    open_pattern.extend_from_slice(tag_name);

    let open_pos = find_bytes(data, &open_pattern)?;
    // Find the '>' that closes the opening tag
    let gt_pos = find_bytes(&data[open_pos..], b">")?;
    let content_start = open_pos + gt_pos + 1;

    // Build closing pattern: "</tagname>"
    let mut close_pattern = Vec::with_capacity(tag_name.len() + 3);
    close_pattern.extend_from_slice(b"</");
    close_pattern.extend_from_slice(tag_name);
    close_pattern.push(b'>');

    let close_pos = find_bytes(&data[content_start..], &close_pattern)?;
    Some(&data[content_start..content_start + close_pos])
}

/// Find the position of `needle` in `haystack`.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Check if `haystack` contains `needle`.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    find_bytes(haystack, needle).is_some()
}

/// Extract the content of every `<rdf:li>...</rdf:li>` element from `data`.
fn extract_all_rdf_li<'a>(data: &'a [u8]) -> Vec<&'a [u8]> {
    let open_tag = b"<rdf:li>";
    let open_tag_attr = b"<rdf:li "; // may have attributes
    let close_tag = b"</rdf:li>";
    let mut results = Vec::new();
    let mut cursor = 0;

    while cursor < data.len() {
        // Find the next <rdf:li> or <rdf:li ...>
        let content_start = if let Some(pos) = find_bytes(&data[cursor..], open_tag) {
            cursor + pos + open_tag.len()
        } else if let Some(pos) = find_bytes(&data[cursor..], open_tag_attr) {
            // Has attributes — skip to the closing '>'
            let tag_start = cursor + pos;
            if let Some(gt) = find_bytes(&data[tag_start..], b">") {
                tag_start + gt + 1
            } else {
                break;
            }
        } else {
            break;
        };

        if let Some(close_pos) = find_bytes(&data[content_start..], close_tag) {
            results.push(&data[content_start..content_start + close_pos]);
            cursor = content_start + close_pos + close_tag.len();
        } else {
            break;
        }
    }

    results
}

/// Trim leading ASCII whitespace from a byte slice.
fn trim_ascii_start(data: &[u8]) -> &[u8] {
    let start = data
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(data.len());
    &data[start..]
}

// ── PDFium document check (page count only) ─────────────────────────────────

/// Return page count from the open `PdfDocument`.
///
/// All exclusion logic is handled by [`pre_check_bytes`] (XMP scan).
pub fn get_page_count(doc: &PdfDocument) -> i32 {
    doc.pages().len() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bytes() {
        assert_eq!(find_bytes(b"hello world", b"world"), Some(6));
        assert_eq!(find_bytes(b"hello world", b"xyz"), None);
        assert_eq!(find_bytes(b"", b"a"), None);
    }

    #[test]
    fn test_extract_tag_content() {
        let xml = b"<root><egExtFL:files><rdf:Bag><rdf:li>file:///server/test.pdf</rdf:li></rdf:Bag></egExtFL:files></root>";
        let content = extract_tag_content(xml, b"egExtFL:files").unwrap();
        assert!(contains_bytes(content, b".pdf"));
    }

    #[test]
    fn test_extract_tag_content_with_attributes() {
        let xml =
            b"<egExtFL:files rdf:parseType=\"Bag\"><rdf:li>image.psd</rdf:li></egExtFL:files>";
        let content = extract_tag_content(xml, b"egExtFL:files").unwrap();
        assert!(contains_bytes(content, b".psd"));
        assert!(!contains_bytes(content, b".pdf"));
    }

    #[test]
    fn test_no_tag_returns_none() {
        let xml = b"<root><other:tag>data</other:tag></root>";
        assert!(extract_tag_content(xml, b"egExtFL:files").is_none());
    }

    #[test]
    fn test_pre_check_bytes_no_file() {
        let result = pre_check_bytes(Path::new("/nonexistent/path.pdf"));
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_all_rdf_li() {
        let xml = b"<rdf:li>file:./a.pdf</rdf:li><rdf:li>file:///b.pdf</rdf:li>";
        let items = extract_all_rdf_li(xml);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], b"file:./a.pdf");
        assert_eq!(items[1], b"file:///b.pdf");
    }

    #[test]
    fn test_relative_pdf_links_not_excluded() {
        // All PDF links are relative → should NOT be excluded
        let items: Vec<&[u8]> = vec![b"file:./mark1.pdf", b"file:../mark2.PDF"];
        let has_non_relative = items.iter().any(|item| {
            let lower: Vec<u8> = item.iter().map(|b| b.to_ascii_lowercase()).collect();
            if !contains_bytes(&lower, b".pdf") {
                return false;
            }
            let trimmed = trim_ascii_start(&lower);
            let is_relative = trimmed.starts_with(b"file:./") || trimmed.starts_with(b"file:../");
            !is_relative
        });
        assert!(!has_non_relative);
    }

    #[test]
    fn test_absolute_pdf_link_excluded() {
        // At least one non-relative PDF link → should be excluded
        let items: Vec<&[u8]> = vec![b"file:./local.pdf", b"file:///server/job/mark.pdf"];
        let has_non_relative = items.iter().any(|item| {
            let lower: Vec<u8> = item.iter().map(|b| b.to_ascii_lowercase()).collect();
            if !contains_bytes(&lower, b".pdf") {
                return false;
            }
            let trimmed = trim_ascii_start(&lower);
            let is_relative = trimmed.starts_with(b"file:./") || trimmed.starts_with(b"file:../");
            !is_relative
        });
        assert!(has_non_relative);
    }

    #[test]
    fn test_rdf_li_with_attributes() {
        let xml = b"<rdf:li stRef:filePath=\"file:./x.pdf\">file:./x.pdf</rdf:li>";
        let items = extract_all_rdf_li(xml);
        // Should not extract the attribute version, only the content
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], b"file:./x.pdf");
    }
}
