use crate::config::ChunkConfig;
use anyhow::{Context, Result};
use std::path::Path;
use text_splitter::{ChunkConfig as SplitConfig, CodeSplitter, MarkdownSplitter, TextSplitter};

/// A retrievable unit of a source file, with its 1-based line span and 0-based
/// byte offsets into the source (used to map the chunk onto the AST for metadata).
#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

/// Source language, selected from the file extension. Drives the chunker and is
/// reported back in search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Markdown,
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Cpp,
    C,
    Text,
}

impl Lang {
    /// Map an extension to a language, or `None` if unsupported (file skipped).
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "md" | "markdown" | "mdx" => Lang::Markdown,
            "rs" => Lang::Rust,
            "py" | "pyi" => Lang::Python,
            "js" | "mjs" | "cjs" | "jsx" => Lang::JavaScript,
            "ts" | "tsx" | "mts" | "cts" => Lang::TypeScript,
            "go" => Lang::Go,
            "java" => Lang::Java,
            "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Lang::Cpp,
            "c" | "h" => Lang::C,
            "txt" | "text" | "rst" | "adoc" => Lang::Text,
            _ => return None,
        })
    }

    /// Stable short name used in output and filters.
    pub fn name(self) -> &'static str {
        match self {
            Lang::Markdown => "md",
            Lang::Rust => "rust",
            Lang::Python => "python",
            Lang::JavaScript => "javascript",
            Lang::TypeScript => "typescript",
            Lang::Go => "go",
            Lang::Java => "java",
            Lang::Cpp => "cpp",
            Lang::C => "c",
            Lang::Text => "text",
        }
    }

    /// The tree-sitter grammar for code languages; `None` for prose (Markdown/Text).
    pub(crate) fn grammar(self) -> Option<tree_sitter::Language> {
        Some(match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::Java => tree_sitter_java::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Lang::C => tree_sitter_c::LANGUAGE.into(),
            Lang::Markdown | Lang::Text => return None,
        })
    }
}

fn split_config(cfg: &ChunkConfig) -> Result<SplitConfig<text_splitter::Characters>> {
    let base = SplitConfig::new(cfg.max_chars);
    if cfg.overlap == 0 {
        Ok(base)
    } else {
        base.with_overlap(cfg.overlap)
            .context("chunk overlap must be smaller than max_chars")
    }
}

/// Split `content` into chunks using the language-appropriate splitter:
/// tree-sitter for code (cuts on syntactic boundaries), pulldown-cmark for
/// Markdown, plain text otherwise.
pub fn chunk_file(content: &str, lang: Lang, cfg: &ChunkConfig) -> Result<Vec<Chunk>> {
    let indices: Vec<(usize, &str)> = match lang.grammar() {
        Some(grammar) => CodeSplitter::new(grammar, split_config(cfg)?)
            .context("building code splitter")?
            .chunk_indices(content)
            .collect(),
        None if lang == Lang::Markdown => MarkdownSplitter::new(split_config(cfg)?)
            .chunk_indices(content)
            .collect(),
        None => TextSplitter::new(split_config(cfg)?)
            .chunk_indices(content)
            .collect(),
    };

    Ok(indices
        .into_iter()
        .map(|(offset, text)| {
            let start_line = 1 + content[..offset].bytes().filter(|&b| b == b'\n').count();
            let newlines_in = text.bytes().filter(|&b| b == b'\n').count();
            Chunk {
                text: text.to_string(),
                start_line,
                end_line: start_line + newlines_in,
                start_byte: offset,
                end_byte: offset + text.len(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ChunkConfig {
        ChunkConfig {
            max_chars: 200,
            overlap: 0,
        }
    }

    #[test]
    fn lang_from_extension() {
        assert_eq!(Lang::from_path(Path::new("a/b.rs")), Some(Lang::Rust));
        assert_eq!(
            Lang::from_path(Path::new("README.md")),
            Some(Lang::Markdown)
        );
        assert_eq!(Lang::from_path(Path::new("x.unknownext")), None);
    }

    #[test]
    fn rust_chunks_have_line_spans() {
        let src = "fn a() {\n    1\n}\n\nfn b() {\n    2\n}\n";
        let chunks = chunk_file(src, Lang::Rust, &cfg()).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_line, 1);
        assert!(chunks.iter().all(|c| c.end_line >= c.start_line));
    }

    #[test]
    fn markdown_splits_within_budget() {
        let md = "# Title\n\nPara one.\n\n## Section\n\nPara two is here.\n";
        let chunks = chunk_file(md, Lang::Markdown, &cfg()).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn oversized_budget_yields_single_chunk() {
        let src = "fn tiny() {}\n";
        let big = ChunkConfig {
            max_chars: 10_000,
            overlap: 0,
        };
        let chunks = chunk_file(src, Lang::Rust, &big).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
    }
}
