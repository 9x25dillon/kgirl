#!/usr/bin/env bash
# Install (or, with --uninstall, remove) the kgirl pre-push privacy gate.
#
# Hooks live in the shared git dir, so this covers every worktree of the repo at once.
# An existing pre-push hook (Git LFS's) is kept as pre-push.before-kgirl and still runs after the gate.
# Nothing is pushed or fetched. Opt-in: run it yourself.
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
src="$here/hooks/pre-push"
cd "$here"
hooks=$(cd "$(git rev-parse --git-path hooks)" && pwd)
dst="$hooks/pre-push"
saved="$hooks/pre-push.before-kgirl"
copy="$hooks/kgirl-prepush"          # pinned scanner copy the hook runs, whichever branch is pushed
marker="kgirl pre-push privacy gate"

ours() { [ -f "$dst" ] && grep -q "$marker" "$dst"; }

case "${1:-}" in
  --uninstall)
    if ! ours; then echo "No kgirl gate installed at $dst; nothing changed."; exit 0; fi
    rm -f "$dst"
    rm -rf "$copy"
    if [ -e "$saved" ]; then mv "$saved" "$dst"; echo "Removed the gate; restored the previous pre-push hook."
    else echo "Removed the gate."; fi
    ;;
  "")
    [ -f "$src" ] || { echo "missing $src" >&2; exit 1; }
    if [ -e "$dst" ] && ! ours; then
      if [ -e "$saved" ]; then
        echo "Both $dst and $saved exist and neither is the gate; refusing to overwrite. Inspect them first." >&2
        exit 1
      fi
      mv "$dst" "$saved"
      echo "Kept the existing pre-push hook as $saved; it still runs after the gate."
    fi
    install -d -m 755 "$copy"
    install -m 644 "$here/smoke/prepush_scan.py" "$here/smoke/_support.py" "$copy/"
    install -m 755 "$src" "$dst"
    echo "Installed the kgirl privacy gate at $dst (all worktrees). Undo: $0 --uninstall"
    ;;
  *) echo "usage: $0 [--uninstall]" >&2; exit 2 ;;
esac
