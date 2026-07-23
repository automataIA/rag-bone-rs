use crate::chunk::Lang;
use ignore::WalkBuilder;
use std::path::PathBuf;
use tracing::warn;

/// Walk every source folder (gitignore-aware, hidden files skipped) and return
/// the supported files, deduplicated and sorted for stable ordering.
pub fn discover(sources: &[PathBuf]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = sources
        .iter()
        .filter(|s| s.exists())
        .flat_map(collect_dir)
        .collect();
    files.sort();
    files.dedup();
    files
}

fn collect_dir(dir: &PathBuf) -> Vec<PathBuf> {
    WalkBuilder::new(dir)
        .require_git(false) // honor .gitignore even outside a git repo
        .build()
        .filter_map(|entry| match entry {
            Ok(entry) => Some(entry),
            Err(error) => {
                warn!(directory = %dir.display(), %error, "failed to traverse source entry");
                None
            }
        })
        .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
        .map(|e| e.into_path())
        .filter(|p| Lang::from_path(p).is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn discovers_supported_files_only() {
        let dir = tempdir();
        fs::write(dir.join("a.rs"), "fn main() {}").unwrap();
        fs::write(dir.join("b.md"), "# hi").unwrap();
        fs::write(dir.join("c.bin"), "\0\0").unwrap();
        let found = discover(std::slice::from_ref(&dir));
        let names: Vec<_> = found
            .iter()
            .filter_map(|p| p.file_name()?.to_str())
            .collect();
        assert!(names.contains(&"a.rs"));
        assert!(names.contains(&"b.md"));
        assert!(!names.contains(&"c.bin"));
    }

    #[test]
    fn respects_gitignore() {
        let dir = tempdir();
        fs::write(dir.join(".gitignore"), "ignored.rs\n").unwrap();
        fs::write(dir.join("kept.rs"), "fn a() {}").unwrap();
        fs::write(dir.join("ignored.rs"), "fn b() {}").unwrap();
        let found = discover(std::slice::from_ref(&dir));
        let names: Vec<_> = found
            .iter()
            .filter_map(|p| p.file_name()?.to_str())
            .collect();
        assert!(names.contains(&"kept.rs"));
        assert!(!names.contains(&"ignored.rs"));
    }

    fn tempdir() -> PathBuf {
        let base = std::env::temp_dir().join(format!("ragbone-walk-{}", std::process::id()));
        let unique = base.join(format!("{:?}", std::time::Instant::now()));
        std::fs::create_dir_all(&unique).unwrap();
        unique
    }
}
