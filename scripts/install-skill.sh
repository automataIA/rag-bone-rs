#!/usr/bin/env bash
# Install the rag-bone skill for Sir Bone or Claude Code.
#
# Two variants share an identical body and differ only in frontmatter:
#   skills/rag-bone/SKILL.md              open Agent Skills standard (Sir Bone source)
#   skills/claude-code/rag-bone/SKILL.md  Claude Code extensions (when_to_use, allowed-tools)
# Sir Bone additionally uses `paths:` for *auto-recall*: `main.rs` injects the
# skill body into the system prompt at startup when the working tree matches one
# of the globs, so the model never has to decide to call `load_skill`. `paths`
# stays out of the open-standard file and is added here at install time.
#
# Usage: scripts/install-skill.sh [--check] [--claude|--sirbone] [skills_root]
#   default target/root: Sir Bone, ~/.sirbone/skills
#   --claude default root: ~/.claude/skills
#   --check verifies the installed copy without changing it.
set -euo pipefail

PATHS_LINE='paths: [".rag-bone.toml", ".rag-bone/index.bin"]'

CHECK=0
TARGET=sirbone
while [[ "${1-}" == --* ]]; do
    case "$1" in
        --check) CHECK=1 ;;
        --claude) TARGET=claude ;;
        --sirbone) TARGET=sirbone ;;
        --help)
            echo "usage: scripts/install-skill.sh [--check] [--claude|--sirbone] [skills_root]"
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$TARGET" = claude ]; then
    DEFAULT_SKILLS_ROOT="${CLAUDE_HOME:-$HOME/.claude}/skills"
else
    DEFAULT_SKILLS_ROOT="$HOME/.sirbone/skills"
fi
SKILLS_ROOT="${1-$DEFAULT_SKILLS_ROOT}"
[ "$#" -le 1 ] || { echo "expected at most one skills_root" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$TARGET" = claude ]; then
    SRC="$REPO_ROOT/skills/claude-code/rag-bone/SKILL.md"
else
    SRC="$REPO_ROOT/skills/rag-bone/SKILL.md"
fi
DEST_DIR="$SKILLS_ROOT/rag-bone"
DEST="$DEST_DIR/SKILL.md"

[ -f "$SRC" ] || { echo "missing skill source: $SRC" >&2; exit 1; }

# Guard the invariant the open-standard file must hold: `paths` belongs only in
# the installed Sir Bone copy (added below), never in the source file.
if awk '/^---$/{n++; next} n==1' "$SRC" | grep -q '^paths:'; then
    echo "source $SRC must not carry 'paths:' — it belongs only in the Sir Bone copy" >&2
    exit 1
fi

# The two variants must keep an identical body (frontmatter may differ).
body() { awk '/^---$/{n++; next} n>=2' "$1"; }
if ! diff <(body "$REPO_ROOT/skills/rag-bone/SKILL.md") \
          <(body "$REPO_ROOT/skills/claude-code/rag-bone/SKILL.md") >/dev/null; then
    echo "skill bodies diverged between skills/rag-bone and skills/claude-code/rag-bone" >&2
    exit 1
fi

if [ "$CHECK" -eq 1 ]; then
    [ -f "$DEST" ] || { echo "not installed: $DEST" >&2; exit 1; }
    if [ "$TARGET" = claude ] && cmp -s "$SRC" "$DEST"; then
        echo "in sync: $DEST"
    elif [ "$TARGET" = sirbone ] && grep -Fqx "$PATHS_LINE" "$DEST" \
        && diff <(body "$SRC") <(body "$DEST") >/dev/null; then
        echo "in sync: $DEST"
    else
        echo "OUT OF SYNC: $DEST — re-run scripts/install-skill.sh" >&2
        exit 1
    fi
    exit 0
fi

mkdir -p "$DEST_DIR"
if [ "$TARGET" = claude ]; then
    # Claude Code consumes its variant verbatim (when_to_use, allowed-tools).
    install -m 0644 "$SRC" "$DEST"
    echo "installed $DEST (Claude Code variant)"
else
    # Insert Sir Bone's auto-recall globs just before the frontmatter closes.
    awk -v paths="$PATHS_LINE" '
        /^---$/ { n++; if (n == 2) print paths; print; next }
        { print }
    ' "$SRC" > "$DEST"
    echo "installed $DEST"
    echo "  frontmatter: $(awk '/^---$/{n++; next} n==1' "$DEST" | grep -c .) lines (paths: added for auto-recall)"
    echo "  Sir Bone: enable 'rag-bone' for the project in skills.enabled (or the TUI settings)"
fi
