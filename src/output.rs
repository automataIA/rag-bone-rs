use crate::search::SearchResult;
use crate::store::{CatalogEntry, Record};
use anyhow::{Context, Result};
use serde::Serialize;
use std::io::Write;

/// Characters of the first non-empty snippet line kept in a compact preview.
const PREVIEW_CHARS: usize = 100;

/// The compact view of a result: metadata plus a one-line preview instead of the
/// full snippet, so an agent can scan many hits cheaply and then `get` the chosen
/// `chunk_id` for full context. Fields mirror `SearchResult`; `preview` replaces
/// `snippet`.
#[derive(Serialize)]
struct CompactResult<'a> {
    chunk_id: &'a str,
    file: &'a std::path::Path,
    lang: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol: &'a Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    headings: &'a Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    corpus_source: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fetched: &'a Option<String>,
    lines: [usize; 2],
    score: f32,
    preview: String,
}

impl<'a> CompactResult<'a> {
    fn from(r: &'a SearchResult) -> Self {
        Self {
            chunk_id: &r.chunk_id,
            file: &r.file,
            lang: &r.lang,
            symbol: &r.symbol,
            headings: &r.headings,
            source: &r.source,
            corpus_source: &r.corpus_source,
            fetched: &r.fetched,
            lines: r.lines,
            score: r.score,
            preview: preview(&r.snippet),
        }
    }
}

/// Print results to stdout. `json` selects a machine array over human text;
/// `compact` swaps full snippets for `chunk_id` + one-line preview. stdout carries
/// only results — logs/progress go to stderr via `tracing`.
pub fn print_results(results: &[SearchResult], json: bool, compact: bool) -> Result<()> {
    match (json, compact) {
        (true, true) => {
            let compact: Vec<CompactResult> = results.iter().map(CompactResult::from).collect();
            println!("{}", serde_json::to_string_pretty(&compact)?);
        }
        (true, false) => println!("{}", serde_json::to_string_pretty(results)?),
        (false, true) => print_compact_text(results),
        (false, false) => print_full_text(results),
    }
    Ok(())
}

/// One line per hit: `chunk_id  file:lines  (score)  symbol  preview`.
fn print_compact_text(results: &[SearchResult]) {
    if results.is_empty() {
        println!("No results.");
        return;
    }
    for r in results {
        let symbol = r
            .symbol
            .as_deref()
            .map(|s| format!("  {s}"))
            .unwrap_or_default();
        println!(
            "{}  {}:{}-{}  (score {:.3}){symbol}  {}",
            r.chunk_id,
            r.file.display(),
            r.lines[0],
            r.lines[1],
            r.score,
            preview(&r.snippet),
        );
    }
}

fn print_full_text(results: &[SearchResult]) {
    if results.is_empty() {
        println!("No results.");
        return;
    }
    for r in results {
        println!(
            "{}:{}-{}  (score {:.3}, {})",
            r.file.display(),
            r.lines[0],
            r.lines[1],
            r.score,
            r.lang
        );
        for line in r.snippet.lines() {
            println!("    {line}");
        }
        println!();
    }
}

/// First non-empty line of the snippet, trimmed and truncated to `PREVIEW_CHARS`
/// characters (an ellipsis marks truncation).
fn preview(snippet: &str) -> String {
    let line = snippet
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    match line.char_indices().nth(PREVIEW_CHARS) {
        Some((byte_idx, _)) => format!("{}…", &line[..byte_idx]),
        None => line.to_string(),
    }
}

/// Metadata-only view of a record — the structural shape `outline` and `find`
/// emit. Excludes the vector and (by default) the full text: an agent scans these
/// cheaply, then `get`s the chunk_id it wants.
#[derive(Serialize)]
struct RecordMeta<'a> {
    chunk_id: &'a str,
    file: &'a std::path::Path,
    lang: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: &'a Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    headings: &'a Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    corpus_source: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fetched: &'a Option<String>,
    lines: [usize; 2],
}

impl<'a> RecordMeta<'a> {
    fn from(r: &'a Record) -> Self {
        Self {
            chunk_id: &r.chunk_id,
            file: &r.file,
            lang: &r.lang,
            symbol: &r.symbol,
            kind: &r.kind,
            parent: &r.parent,
            signature: &r.signature,
            headings: &r.headings,
            source: &r.source,
            corpus_source: &r.corpus_source,
            fetched: &r.fetched,
            lines: [r.start_line, r.end_line],
        }
    }
}

/// Print a set of records as structural metadata (for `outline`/`find`): a JSON
/// array, or one human line per record `chunk_id  file:lines  kind  symbol/section`.
pub fn print_record_meta(records: &[&Record], json: bool) -> Result<()> {
    if json {
        let metas: Vec<RecordMeta> = records.iter().map(|r| RecordMeta::from(r)).collect();
        println!("{}", serde_json::to_string_pretty(&metas)?);
        return Ok(());
    }
    for r in records {
        // Prefer the code symbol; fall back to the Markdown heading breadcrumb.
        let label = r
            .symbol
            .as_deref()
            .map(str::to_string)
            .or_else(|| (!r.headings.is_empty()).then(|| r.headings.join(" > ")))
            .unwrap_or_default();
        let kind = r
            .kind
            .as_deref()
            .map(|k| format!("{k}  "))
            .unwrap_or_default();
        let sig = r
            .signature
            .as_deref()
            .map(|s| format!("  {s}"))
            .unwrap_or_default();
        println!(
            "{}  {}:{}-{}  {kind}{label}{sig}",
            r.chunk_id,
            r.file.display(),
            r.start_line,
            r.end_line,
        );
    }
    Ok(())
}

/// Print the web-sourced document catalog: a JSON array, or one human line per
/// source `URL  (N files, M chunks, fetched DATE)`. The agent's routing menu.
pub fn print_catalog(entries: &[CatalogEntry], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(entries)?);
        return Ok(());
    }
    if entries.is_empty() {
        println!("No web-sourced documents indexed.");
        return Ok(());
    }
    for e in entries {
        let corpus = e
            .corpus_source
            .as_deref()
            .filter(|corpus| *corpus != e.source)
            .map(|corpus| format!(", corpus {corpus}"))
            .unwrap_or_default();
        println!(
            "{}  ({} files, {} chunks, fetched {}{corpus})",
            e.source,
            e.files,
            e.chunks,
            e.fetched.as_deref().unwrap_or("—"),
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct JsonlRecord<'a> {
    query: &'a str,
    results: &'a [SearchResult],
}

#[derive(Serialize)]
struct CompactJsonlRecord<'a> {
    query: &'a str,
    results: Vec<CompactResult<'a>>,
}

/// Write and flush one streaming search response. Each input query produces one
/// complete JSON object, so consumers can process results without buffering EOF.
/// `compact` selects the metadata+preview shape over full snippets.
pub fn write_jsonl_record(
    writer: &mut impl Write,
    query: &str,
    results: &[SearchResult],
    compact: bool,
) -> Result<()> {
    if compact {
        let record = CompactJsonlRecord {
            query,
            results: results.iter().map(CompactResult::from).collect(),
        };
        serde_json::to_writer(&mut *writer, &record).context("serializing batch result")?;
    } else {
        let record = JsonlRecord { query, results };
        serde_json::to_writer(&mut *writer, &record).context("serializing batch result")?;
    }
    writeln!(writer).context("writing batch result")?;
    writer.flush().context("flushing batch result")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn result() -> SearchResult {
        SearchResult {
            chunk_id: "1-abc".into(),
            file: PathBuf::from("src/main.rs"),
            lang: "rust".into(),
            symbol: Some("main".into()),
            kind: Some("function".into()),
            headings: Vec::new(),
            source: None,
            corpus_source: None,
            fetched: None,
            lines: [1, 2],
            score: 0.5,
            snippet: "fn main() {}\n    body".into(),
        }
    }

    #[test]
    fn batch_record_is_one_json_line() {
        let mut output = Vec::new();
        write_jsonl_record(&mut output, "entry point", &[result()], false).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert_eq!(text.lines().count(), 1);
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["query"], "entry point");
        assert_eq!(value["results"][0]["file"], "src/main.rs");
        assert_eq!(value["results"][0]["chunk_id"], "1-abc");
    }

    #[test]
    fn compact_batch_record_has_preview_not_snippet() {
        let mut output = Vec::new();
        write_jsonl_record(&mut output, "q", &[result()], true).unwrap();
        let value: serde_json::Value =
            serde_json::from_str(String::from_utf8(output).unwrap().trim()).unwrap();
        let r = &value["results"][0];
        assert_eq!(r["chunk_id"], "1-abc");
        assert_eq!(r["preview"], "fn main() {}");
        assert!(r.get("snippet").is_none(), "compact drops the full snippet");
    }

    #[test]
    fn preview_uses_first_non_empty_line_and_truncates() {
        assert_eq!(preview("\n\n  hello world  \nmore"), "hello world");
        let long = "x".repeat(250);
        let p = preview(&long);
        assert!(p.ends_with('…'));
        assert_eq!(p.chars().count(), PREVIEW_CHARS + 1);
    }
}
