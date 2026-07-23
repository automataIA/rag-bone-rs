//! Deterministic structural metadata for a chunk: the enclosing symbol, its
//! kind/parent/signature (code, via a second tree-sitter parse) or the Markdown
//! heading breadcrumb (prose). Metadata is optional and degrades to absent
//! fields — a parse error or an unmapped node never fails the file.
//!
//! The metadata feeds three things: the `ranking_text` that is embedded and can
//! be fed to the reranker, the public `chunk_id`, and the internal
//! `embedding_key` used by the (future) embedding cache.

use crate::chunk::Lang;
use tree_sitter::{Node, Parser};

/// Structural facts about one chunk. All code fields are `None` for prose;
/// `headings` is empty for code.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkMeta {
    /// Name of the smallest named definition enclosing the chunk (e.g. `save`).
    pub symbol: Option<String>,
    /// Friendly kind of that definition (`function`, `method`, `struct`, ...).
    pub kind: Option<String>,
    /// Nearest enclosing container definition (`impl Index`, `class Foo`, ...).
    pub parent: Option<String>,
    /// One-line signature of the enclosing definition.
    pub signature: Option<String>,
    /// Markdown heading breadcrumb (ancestors of the chunk), outermost first.
    pub headings: Vec<String>,
}

/// Chunk-id format version. Bumped only if the id derivation changes.
const CHUNK_ID_VERSION: u8 = 1;
/// Cap on a stored signature, in bytes, to keep the ranking header bounded.
const MAX_SIGNATURE_LEN: usize = 200;

/// A file parsed once, ready to answer metadata queries for each of its chunks.
/// For code the tree-sitter tree is built a single time and reused per chunk.
pub struct FileAnalysis<'a> {
    lang: Lang,
    content: &'a str,
    tree: Option<tree_sitter::Tree>,
}

impl<'a> FileAnalysis<'a> {
    /// Parse `content` once for code languages; prose parses lazily per chunk.
    pub fn new(content: &'a str, lang: Lang) -> Self {
        let tree = lang.grammar().and_then(|grammar| {
            let mut parser = Parser::new();
            parser.set_language(&grammar).ok()?;
            parser.parse(content, None)
        });
        Self {
            lang,
            content,
            tree,
        }
    }

    /// Metadata for the chunk spanning `[start_byte, end_byte)`.
    pub fn meta_for(&self, start_byte: usize, end_byte: usize) -> ChunkMeta {
        if self.lang == Lang::Markdown {
            return ChunkMeta {
                headings: markdown_headings(self.content, start_byte),
                ..ChunkMeta::default()
            };
        }
        match &self.tree {
            Some(tree) => self.code_meta(tree.root_node(), start_byte, end_byte),
            None => ChunkMeta::default(),
        }
    }

    /// Walk up from the smallest node spanning the chunk to the nearest named
    /// definition; that is the smallest definition that fully contains the chunk.
    /// A chunk straddling two sibling definitions resolves to their common
    /// ancestor (often none), yielding an absent symbol rather than a wrong one.
    fn code_meta(&self, root: Node<'_>, start_byte: usize, end_byte: usize) -> ChunkMeta {
        let end = end_byte.max(start_byte + 1).min(self.content.len());
        let start = start_byte.min(end.saturating_sub(1));
        let Some(node) = root.descendant_for_byte_range(start, end) else {
            return ChunkMeta::default();
        };
        let Some(def) = self.nearest_definition(node) else {
            return ChunkMeta::default();
        };
        let parent = def
            .parent()
            .and_then(|p| self.nearest_definition(p))
            .filter(|p| is_container(def_kind(p.kind()).unwrap_or("")))
            .map(|p| self.describe_parent(p));
        ChunkMeta {
            symbol: self.name_of(def),
            kind: def_kind(def.kind()).map(str::to_string),
            parent,
            signature: self.signature_of(def),
            headings: Vec::new(),
        }
    }

    /// Nearest ancestor (inclusive) that is a mapped definition kind.
    fn nearest_definition<'b>(&self, node: Node<'b>) -> Option<Node<'b>> {
        let mut cur = Some(node);
        while let Some(n) = cur {
            if def_kind(n.kind()).is_some() {
                return Some(n);
            }
            cur = n.parent();
        }
        None
    }

    /// `kind name` for a container parent (e.g. `impl Index`), or just the kind
    /// when the container is anonymous.
    fn describe_parent(&self, node: Node) -> String {
        let kind = def_kind(node.kind()).unwrap_or(node.kind());
        match self.name_of(node) {
            Some(name) => format!("{kind} {name}"),
            None => kind.to_string(),
        }
    }

    /// The definition's name: the `name` field, then a few grammar-specific
    /// fallbacks, then the first identifier-like named child.
    fn name_of(&self, node: Node) -> Option<String> {
        for field in ["name", "type", "declarator"] {
            if let Some(child) = node.child_by_field_name(field)
                && let Some(id) = self.identifier_text(child)
            {
                return Some(id);
            }
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if let Some(id) = self.identifier_text(child) {
                return Some(id);
            }
        }
        None
    }

    /// Text of `node` if it is an identifier, or of the first identifier nested
    /// inside it (handles C/C++ where the name is buried in a declarator).
    fn identifier_text(&self, node: Node) -> Option<String> {
        if is_identifier(node.kind()) {
            return node
                .utf8_text(self.content.as_bytes())
                .ok()
                .map(str::to_string);
        }
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if let Some(id) = self.identifier_text(child) {
                return Some(id);
            }
        }
        None
    }

    /// One-line signature: the definition text up to its body, whitespace
    /// collapsed and length-capped.
    fn signature_of(&self, node: Node) -> Option<String> {
        let text = node.utf8_text(self.content.as_bytes()).ok()?;
        let end = text
            .find('{')
            .or_else(|| text.find('\n'))
            .unwrap_or(text.len());
        let sig: String = text[..end].split_whitespace().collect::<Vec<_>>().join(" ");
        if sig.is_empty() {
            return None;
        }
        Some(truncate_chars(&sig, MAX_SIGNATURE_LEN))
    }
}

/// Ancestor Markdown headings for a chunk starting at `chunk_start_byte`.
/// A deterministic single scan: maintain a stack of open headings by level and
/// return the ones strictly before the chunk (its own leading heading, if any,
/// stays in the chunk text and is not repeated here).
fn markdown_headings(content: &str, chunk_start_byte: usize) -> Vec<String> {
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut byte = 0usize;
    for line in content.split_inclusive('\n') {
        if byte >= chunk_start_byte {
            break;
        }
        if let Some((level, title)) = parse_atx_heading(line) {
            stack.retain(|(l, _)| *l < level);
            stack.push((level, title));
        }
        byte += line.len();
    }
    stack.into_iter().map(|(_, title)| title).collect()
}

/// Parse an ATX heading line (`## Title`), returning `(level, title)`. Ignores
/// setext headings and fenced content; good enough for a deterministic breadcrumb.
fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.bytes().take_while(|&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.starts_with([' ', '\t']) {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    (!title.is_empty()).then(|| (level, title.to_string()))
}

/// The deterministic ranking header + raw text that is embedded (and optionally
/// reranked). Absent fields are omitted rather than filled with placeholders.
/// Reconstructible from metadata, so it need not be persisted separately.
pub fn ranking_text(path: &str, lang: Lang, meta: &ChunkMeta, raw: &str) -> String {
    let mut header = String::new();
    let mut line = |k: &str, v: &str| {
        if !v.is_empty() {
            header.push_str(k);
            header.push_str(": ");
            header.push_str(v);
            header.push('\n');
        }
    };
    line("path", path);
    line("language", lang.name());
    if !meta.headings.is_empty() {
        line("section", &meta.headings.join(" > "));
    }
    if let Some(s) = &meta.symbol {
        line("symbol", s);
    }
    if let Some(k) = &meta.kind {
        line("kind", k);
    }
    if let Some(p) = &meta.parent {
        line("parent", p);
    }
    if let Some(sig) = &meta.signature {
        line("signature", sig);
    }
    format!("{header}---\n{raw}")
}

/// Public, versioned, deterministic chunk identifier. 128-bit hash of the fields
/// that identify a chunk's location and content; the corpus is local and trusted,
/// so a fast non-cryptographic hash is acceptable (with a build-time collision
/// check on the caller side). The span is the **byte** range, which is unique
/// within a file (unlike the line span, which two chunks can share when the
/// splitter cuts mid-line).
pub fn chunk_id(path: &str, lang: Lang, start_byte: usize, end_byte: usize, raw: &str) -> String {
    let descriptor = format!("{path}\0{}\0{start_byte}\0{end_byte}\0{raw}", lang.name());
    let hash = xxhash_rust::xxh3::xxh3_128(descriptor.as_bytes());
    format!("{CHUNK_ID_VERSION}-{hash:032x}")
}

/// Internal embedding-cache key: the exact bytes of the ranking text bound to the
/// ingestion fingerprint. It deliberately excludes the line span, so inserting
/// lines above an otherwise-unchanged chunk can reuse the vector.
pub fn embedding_key(ingest_fingerprint: u64, ranking_text: &str) -> u64 {
    let mut bytes = Vec::with_capacity(8 + ranking_text.len());
    bytes.extend_from_slice(&ingest_fingerprint.to_le_bytes());
    bytes.extend_from_slice(ranking_text.as_bytes());
    xxhash_rust::xxh3::xxh3_64(&bytes)
}

/// File-level provenance parsed from a leading YAML frontmatter block
/// (`--- ... ---`), as emitted by search2md for web-fetched documents. Only
/// The requested source URL, actual corpus URL, and RFC 3339 fetch timestamp are
/// read; any other key is ignored. A file without a leading frontmatter block
/// yields the default. Hand-parsed to avoid a YAML dependency for three scalars.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Provenance {
    pub source: Option<String>,
    pub corpus_source: Option<String>,
    pub fetched: Option<String>,
}

pub fn frontmatter_provenance(content: &str) -> Provenance {
    let mut prov = Provenance::default();
    let Some(body) = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))
    else {
        return prov;
    };
    for line in body.lines() {
        let line = line.trim_end();
        if line == "---" {
            break;
        }
        if let Some(value) = line.strip_prefix("source:") {
            prov.source = Some(unquote(value));
        } else if let Some(value) = line.strip_prefix("corpus_source:") {
            prov.corpus_source = Some(unquote(value));
        } else if let Some(value) = line.strip_prefix("fetched:") {
            prov.fetched = Some(unquote(value));
        }
    }
    prov
}

/// Trim surrounding whitespace and one layer of matching single/double quotes.
fn unquote(value: &str) -> String {
    let value = value.trim();
    value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
        .unwrap_or(value)
        .to_string()
}

#[cfg(test)]
mod provenance_tests {
    use super::{Provenance, frontmatter_provenance};

    #[test]
    fn reads_source_and_fetched_from_frontmatter() {
        let md = "---\nsource: https://docs.rs/serde\ncorpus_source: https://docs.rs/llms-full.txt\ndepth: llms\nfetched: 2026-07-19T10:00:00Z\ngenerated_by: search2md-rs\n---\n# Serde\nbody";
        let p = frontmatter_provenance(md);
        assert_eq!(p.source.as_deref(), Some("https://docs.rs/serde"));
        assert_eq!(
            p.corpus_source.as_deref(),
            Some("https://docs.rs/llms-full.txt")
        );
        assert_eq!(p.fetched.as_deref(), Some("2026-07-19T10:00:00Z"));
    }

    #[test]
    fn absent_without_block_and_unquotes_values() {
        assert_eq!(
            frontmatter_provenance("# Heading\nno frontmatter"),
            Provenance::default()
        );
        let q = frontmatter_provenance("---\nsource: \"https://x.y/z\"\n---\n");
        assert_eq!(q.source.as_deref(), Some("https://x.y/z"));
        assert_eq!(q.corpus_source, None);
        assert_eq!(q.fetched, None);
    }
}

/// Map a tree-sitter node kind to a friendly definition label, or `None` if the
/// kind is not a definition we surface. Covers the eight supported grammars;
/// unmapped kinds degrade to absent metadata.
fn def_kind(raw: &str) -> Option<&'static str> {
    Some(match raw {
        // functions / methods
        "function_item" | "function_definition" | "function_declaration" => "function",
        "method_definition" | "method_declaration" => "method",
        "constructor_declaration" => "constructor",
        // types / containers
        "struct_item" | "struct_specifier" => "struct",
        "enum_item" | "enum_declaration" | "enum_specifier" => "enum",
        "union_item" | "union_specifier" => "union",
        "trait_item" => "trait",
        "impl_item" => "impl",
        "class_declaration" | "class_definition" | "class_specifier" => "class",
        "interface_declaration" => "interface",
        "mod_item" => "module",
        "type_item" | "type_alias_declaration" | "type_declaration" => "type",
        _ => return None,
    })
}

/// Whether a friendly kind names a container that can be a `parent`.
fn is_container(kind: &str) -> bool {
    matches!(
        kind,
        "impl" | "class" | "struct" | "trait" | "interface" | "enum" | "module" | "union"
    )
}

/// Node kinds that carry a definition's name across the supported grammars.
fn is_identifier(kind: &str) -> bool {
    matches!(
        kind,
        "identifier" | "type_identifier" | "field_identifier" | "property_identifier"
    )
}

/// Truncate to at most `max` characters (not bytes), preserving UTF-8.
fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((byte_idx, _)) => s[..byte_idx].to_string(),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metadata for the chunk whose text is `needle` (found by byte offset).
    fn meta_at(src: &str, lang: Lang, needle: &str) -> ChunkMeta {
        let start = src.find(needle).expect("needle present");
        FileAnalysis::new(src, lang).meta_for(start, start + needle.len())
    }

    #[test]
    fn rust_function_symbol_and_signature() {
        let src = "pub fn save(&self, path: &Path) -> Result<()> {\n    ok()\n}\n";
        let m = meta_at(src, Lang::Rust, "ok()");
        assert_eq!(m.symbol.as_deref(), Some("save"));
        assert_eq!(m.kind.as_deref(), Some("function"));
        assert_eq!(
            m.signature.as_deref(),
            Some("pub fn save(&self, path: &Path) -> Result<()>")
        );
    }

    #[test]
    fn rust_method_has_impl_parent() {
        let src = "impl Index {\n    fn save(&self) -> u8 {\n        1\n    }\n}\n";
        let m = meta_at(src, Lang::Rust, "1");
        assert_eq!(m.symbol.as_deref(), Some("save"));
        assert_eq!(m.parent.as_deref(), Some("impl Index"));
    }

    #[test]
    fn rust_multiline_signature_collapses() {
        let src = "fn wide(\n    a: i32,\n    b: i32,\n) -> i32 {\n    a\n}\n";
        let m = meta_at(src, Lang::Rust, "a\n}");
        assert_eq!(
            m.signature.as_deref(),
            Some("fn wide( a: i32, b: i32, ) -> i32")
        );
    }

    #[test]
    fn python_class_and_method() {
        let src = "class Foo:\n    def bar(self):\n        return 1\n";
        let m = meta_at(src, Lang::Python, "return 1");
        assert_eq!(m.symbol.as_deref(), Some("bar"));
        assert_eq!(m.kind.as_deref(), Some("function"));
        assert_eq!(m.parent.as_deref(), Some("class Foo"));
    }

    #[test]
    fn typescript_interface() {
        let src = "interface Shape {\n  area(): number;\n}\n";
        let m = meta_at(src, Lang::TypeScript, "area");
        assert_eq!(m.symbol.as_deref(), Some("Shape"));
        assert_eq!(m.kind.as_deref(), Some("interface"));
    }

    #[test]
    fn go_function() {
        let src = "func Add(a int, b int) int {\n\treturn a + b\n}\n";
        let m = meta_at(src, Lang::Go, "return a");
        assert_eq!(m.symbol.as_deref(), Some("Add"));
        assert_eq!(m.kind.as_deref(), Some("function"));
    }

    #[test]
    fn java_method_in_class() {
        let src = "class C {\n  int add(int a) {\n    return a;\n  }\n}\n";
        let m = meta_at(src, Lang::Java, "return a");
        assert_eq!(m.symbol.as_deref(), Some("add"));
        assert_eq!(m.kind.as_deref(), Some("method"));
        assert_eq!(m.parent.as_deref(), Some("class C"));
    }

    #[test]
    fn c_function_name_in_declarator() {
        let src = "int add(int a, int b) {\n    return a + b;\n}\n";
        let m = meta_at(src, Lang::C, "return a");
        assert_eq!(m.symbol.as_deref(), Some("add"));
        assert_eq!(m.kind.as_deref(), Some("function"));
    }

    #[test]
    fn cpp_struct() {
        let src = "struct Point {\n    int x;\n    int y;\n};\n";
        let m = meta_at(src, Lang::Cpp, "int x");
        assert_eq!(m.symbol.as_deref(), Some("Point"));
        assert_eq!(m.kind.as_deref(), Some("struct"));
    }

    #[test]
    fn chunk_spanning_two_functions_has_no_symbol() {
        let src = "fn a() {}\nfn b() {}\n";
        // Span both top-level functions: common ancestor is the source file.
        let m = FileAnalysis::new(src, Lang::Rust).meta_for(0, src.len());
        assert_eq!(m.symbol, None);
        assert_eq!(m.kind, None);
    }

    #[test]
    fn parse_error_degrades_to_empty() {
        let src = "fn broken( { { { unterminated";
        let m = FileAnalysis::new(src, Lang::Rust).meta_for(0, src.len());
        // No panic, no wrong symbol — worst case an absent field.
        assert!(m.headings.is_empty());
    }

    #[test]
    fn markdown_breadcrumb_is_ancestor_headings() {
        let src = "# Title\n\nintro\n\n## Section\n\nbody text here\n";
        let m = meta_at(src, Lang::Markdown, "body text here");
        assert_eq!(m.headings, vec!["Title".to_string(), "Section".to_string()]);
    }

    #[test]
    fn markdown_sibling_heading_replaces_same_level() {
        let src = "# A\n\n## One\n\nx\n\n## Two\n\nlast body\n";
        let m = meta_at(src, Lang::Markdown, "last body");
        assert_eq!(m.headings, vec!["A".to_string(), "Two".to_string()]);
    }

    #[test]
    fn ranking_text_omits_absent_fields() {
        let meta = ChunkMeta {
            symbol: Some("save".into()),
            kind: Some("method".into()),
            parent: Some("impl Index".into()),
            signature: Some("fn save(&self)".into()),
            headings: Vec::new(),
        };
        let rt = ranking_text("src/store.rs", Lang::Rust, &meta, "body");
        assert_eq!(
            rt,
            "path: src/store.rs\nlanguage: rust\nsymbol: save\nkind: method\nparent: impl Index\nsignature: fn save(&self)\n---\nbody"
        );
    }

    #[test]
    fn ranking_text_prose_has_only_path_lang_section() {
        let meta = ChunkMeta {
            headings: vec!["Guide".into(), "Setup".into()],
            ..ChunkMeta::default()
        };
        let rt = ranking_text("docs/g.md", Lang::Markdown, &meta, "text");
        assert_eq!(
            rt,
            "path: docs/g.md\nlanguage: md\nsection: Guide > Setup\n---\ntext"
        );
    }

    #[test]
    fn chunk_id_is_deterministic_and_versioned() {
        let a = chunk_id("src/a.rs", Lang::Rust, 0, 20, "code");
        let b = chunk_id("src/a.rs", Lang::Rust, 0, 20, "code");
        assert_eq!(a, b);
        assert!(a.starts_with("1-"));
        // Any identifying field change flips the id.
        assert_ne!(a, chunk_id("src/a.rs", Lang::Rust, 0, 21, "code"));
        assert_ne!(a, chunk_id("src/a.rs", Lang::Rust, 5, 20, "code"));
        assert_ne!(a, chunk_id("src/b.rs", Lang::Rust, 0, 20, "code"));
        assert_ne!(a, chunk_id("src/a.rs", Lang::Rust, 0, 20, "other"));
    }

    #[test]
    fn embedding_key_ignores_line_span_but_tracks_text_and_fingerprint() {
        let rt = ranking_text("src/a.rs", Lang::Rust, &ChunkMeta::default(), "body");
        assert_eq!(embedding_key(7, &rt), embedding_key(7, &rt));
        assert_ne!(embedding_key(7, &rt), embedding_key(8, &rt));
        let rt2 = ranking_text("src/a.rs", Lang::Rust, &ChunkMeta::default(), "other");
        assert_ne!(embedding_key(7, &rt), embedding_key(7, &rt2));
    }
}
