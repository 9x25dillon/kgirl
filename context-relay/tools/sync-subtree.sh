#!/bin/sh
# Copy context-relay/ from its canonical repository into the repository you run this in,
# as one squashed commit plus a merge. The first run adds the directory; later runs pull
# changes with a three-way merge, so edits made in this repository survive.
#
#   sh context-relay/tools/sync-subtree.sh [source-repo] [ref]
#
#   source-repo   URL or local path   (default https://github.com/9x25dillon/myAssistant)
#   ref           branch, tag or sha  (default main)
#   CTXR_PREFIX   directory to sync   (default context-relay)
#
# Same commit shape and git-subtree-dir trailer as `git subtree add/pull --squash`. It exists
# because `git subtree` fails in repositories that track a file named HEAD at their root
# (git reads the bare argument HEAD as ambiguous). No command below passes HEAD bare.
# It only adds commits to the current branch; it never rewrites history.
set -eu

src=${1:-https://github.com/9x25dillon/myAssistant}
ref=${2:-main}
prefix=${CTXR_PREFIX:-context-relay}
prefix=${prefix%/}

die() { echo "sync-subtree: $*" >&2; exit 1; }

[ -d "$src" ] && src=$(cd "$src" && pwd)
cd "$(git rev-parse --show-toplevel)"
git diff --quiet -- && git diff --cached --quiet -- || die "commit or stash your changes first"

git fetch --quiet "$src" "$ref"
source=$(git rev-parse --verify --quiet 'FETCH_HEAD^{commit}') || die "could not fetch $ref from $src"
tree=$(git rev-parse --verify --quiet "$source:$prefix") || die "$src@$ref has no $prefix/ directory"
short=$(git rev-parse --short "$source")
head=$(git rev-parse --verify --quiet 'HEAD^{commit}') || die "this repository has no commits yet"
last=$(git log -1 --format=%H --grep="^git-subtree-dir: $prefix/*\$" "$head" --)

trailers="git-subtree-dir: $prefix
context-relay-source: $src@$source"

if [ -z "$last" ]; then
  git rev-parse --verify --quiet "$head:$prefix" >/dev/null \
    && die "$prefix/ already exists here but was not added by this script or git subtree"
  squash=$(printf "Squashed '%s/' content from %s@%s\n\n%s\n" "$prefix" "$src" "$short" "$trailers" | git commit-tree "$tree")
  git read-tree --prefix="$prefix/" -u "$squash"
  merged=$(printf "Add '%s/' from %s@%s\n" "$prefix" "$src" "$short" | git commit-tree "$(git write-tree)" -p "$head" -p "$squash")
  git update-ref -m "sync-subtree: add $prefix" HEAD "$merged" "$head"
  echo "added $prefix/ from $src@$short"
  exit 0
fi

[ "$(git rev-parse --verify "$last^{tree}")" = "$tree" ] && { echo "$prefix/ is already at $src@$short"; exit 0; }
squash=$(printf "Squashed '%s/' changes up to %s@%s\n\n%s\n" "$prefix" "$src" "$short" "$trailers" | git commit-tree "$tree" -p "$last")
git merge --quiet --no-ff -Xsubtree="$prefix" -m "Merge '$prefix/' from $src@$short" "$squash"
echo "updated $prefix/ to $src@$short"
