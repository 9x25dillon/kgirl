#!/usr/bin/env python3
"""Preview landing several lane worktrees on one base, with no git writes and no tests run.

It copies a clean tree of the base commit (`git archive`) into a temp folder, then overlays each lane's
changes against that commit: committed and uncommitted edits, new untracked files and deletions. It
reports each lane's footprint, 3-way merges overlapping text files against the base, and then checks the
merged tree. A clean automatic merge is a WARN; a real content conflict is a failure with its hunk:
  compile    every .py under src/, tests/ and scripts/ byte-compiles
  js         changed .js files pass `node --check` (skipped when node is absent)
  discovery  unittest discovery of tests/harness, tests/helix, tests/holo and tests/vessel imports cleanly, and every
             folder holding test_*.py contributes cases (imports only: no test is executed)

Example:
  python3 scripts/smoke/merge_preview.py --base ~/work/kgirl-stack \\
      --lane ~/work/kgirl-eng --lane ~/work/kgirl-helix-ui --lane ~/work/kgirl-builder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from _support import Stages, isolated_home, require

SUITES = ("tests/harness", "tests/helix", "tests/holo", "tests/vessel")

DISCOVER = r"""
import json, sys, unittest
from pathlib import Path
root = Path.cwd()
def cases(s):
    for i in s:
        yield from (cases(i) if isinstance(i, unittest.TestSuite) else [i])
out = {}
for rel in json.loads(sys.argv[1]):
    if not (root / rel).is_dir():
        continue
    loader = unittest.TestLoader()
    ids = {c.id() for c in cases(loader.discover(str(root / rel), top_level_dir=str(root)))}
    folders = {}
    for f in sorted({p.parent for p in (root / rel).rglob("test_*.py")}):
        if "__pycache__" in f.parts:
            continue
        dotted = ".".join(f.relative_to(root).parts) + "."
        folders[str(f.relative_to(root))] = sum(i.startswith(dotted) for i in ids)
    out[rel] = {"cases": len(ids), "errors": [str(e)[:400] for e in loader.errors], "folders": folders}
print(json.dumps(out))
"""


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} in {repo}: {r.stderr.strip()}")
    return r.stdout


def footprint(lane: Path, base: str) -> tuple[set[str], set[str]]:
    """(changed-or-added, deleted) paths of a lane against the base commit, uncommitted work included."""
    require(not git(lane, "rev-list", f"HEAD..{base}").strip(),
            f"{lane} does not contain the base commit; rebase it first")
    changed, deleted = set(), set()
    for line in git(lane, "diff", "--name-status", "--no-renames", base).splitlines():
        status, path = line.split("\t", 1)
        (deleted if status == "D" else changed).add(path)
    changed |= set(git(lane, "ls-files", "--others", "--exclude-standard").splitlines())
    return changed, deleted


def blob(repo: Path, commit: str, path: str) -> bytes:
    """A path at commit, or empty bytes when it is new in every lane."""
    r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True)
    if r.returncode == 0:
        return r.stdout
    missing = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}:{path}"],
                             capture_output=True).returncode != 0
    if missing:
        return b""
    raise RuntimeError(f"cannot read {path} from base {commit[:8]}: {r.stderr.decode(errors='replace').strip()}")


def merge_text(ours: bytes, base: bytes, theirs: bytes, labels: tuple[str, str, str], folder: Path) -> tuple[bytes, str]:
    """`git merge-file -p` without touching any checkout; status is clean, conflict, or error."""
    if b"\0" in ours + base + theirs:
        return b"", "binary files cannot be 3-way merged safely"
    files = []
    for name, data in zip(("ours", "base", "theirs"), (ours, base, theirs)):
        path = folder / name
        path.write_bytes(data)
        files.append(path)
    cmd = ["git", "merge-file", "-p"]
    for label in labels:
        cmd += ["-L", label]
    r = subprocess.run([*cmd, *(str(p) for p in files)], capture_output=True)
    if r.returncode == 0:
        return r.stdout, "clean"
    if r.returncode == 1:
        return r.stdout, "conflict"
    detail = r.stderr.decode("utf-8", errors="replace").strip() or f"git merge-file exited {r.returncode}"
    return b"", detail


def deletion_hunk(ours: bytes | None, theirs: bytes | None, ours_label: str, theirs_label: str) -> str:
    """A readable conflict hunk when one lane deletes and another edits the path."""
    left = "(deleted)\n" if ours is None else ours.decode("utf-8", errors="replace")
    right = "(deleted)\n" if theirs is None else theirs.decode("utf-8", errors="replace")
    return f"<<<<<<< {ours_label}\n{left}=======\n{right}>>>>>>> {theirs_label}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="worktree whose HEAD is the landing target")
    parser.add_argument("--lane", type=Path, action="append", required=True, help="lane worktree (repeatable)")
    parser.add_argument("--python", default=sys.executable, help="interpreter for discovery (needs numpy)")
    parser.add_argument("--keep", action="store_true", help="keep the merged tree and print its path")
    args = parser.parse_args()
    stages, lanes = Stages(), {}
    merged_overlaps: dict[str, bytes | None] = {}
    base = git(args.base, "rev-parse", "HEAD").strip()
    tmp = Path(tempfile.mkdtemp(prefix="kgirl-merge-preview-"))
    tree = tmp / "tree"
    try:
        def footprints():
            for lane in args.lane:
                lanes[lane.name] = (lane.resolve(), *footprint(lane.resolve(), base))
            for name, (_, changed, deleted) in lanes.items():
                print(f"  {name}: {len(changed)} changed/added, {len(deleted)} deleted", flush=True)
            return f"{len(lanes)} lanes against {base[:8]}"

        def overlap():
            owner: dict[str, list[str]] = {}
            for name, (_, changed, deleted) in lanes.items():
                for path in changed | deleted:
                    owner.setdefault(path, []).append(name)
            clash = {p: o for p, o in owner.items() if len(o) > 1}
            conflicts = []
            for number, (path, who) in enumerate(sorted(clash.items())):
                base_data = blob(args.base, base, path)
                first = who[0]
                first_root, _, first_deleted = lanes[first]
                current = None if path in first_deleted else (first_root / path).read_bytes()
                current_label = first
                for other in who[1:]:
                    other_root, _, other_deleted = lanes[other]
                    theirs = None if path in other_deleted else (other_root / path).read_bytes()
                    if current is None or theirs is None:
                        if current is None and theirs is None:
                            continue
                        conflicts.append((path, who, deletion_hunk(current, theirs, current_label, other)))
                        break
                    merge_dir = tmp / "merge-files" / str(number)
                    merge_dir.mkdir(parents=True, exist_ok=True)
                    current, status = merge_text(current, base_data, theirs,
                                                 (current_label, "base", other), merge_dir)
                    if status != "clean":
                        detail = current.decode("utf-8", errors="replace") if status == "conflict" else status
                        conflicts.append((path, who, detail))
                        break
                    current_label = f"merged({current_label},{other})"
                else:
                    merged_overlaps[path] = current
                    print(f"  WARN {path}: clean 3-way merge of {', '.join(who)}", flush=True)
            for path, who, detail in conflicts:
                print(f"  CONFLICT {path}: {', '.join(who)}", flush=True)
                print("\n".join("    " + line for line in detail.splitlines()[:80]), flush=True)
            require(not conflicts, f"{len(conflicts)} overlapping paths have real conflicts")
            return f"{len(clash)} overlapping paths merged cleanly" if clash else "no overlapping paths"

        def overlay():
            tree.mkdir()
            archive = tmp / "base.tar"
            with archive.open("wb") as fh:
                require(subprocess.run(["git", "-C", str(args.base), "archive", base], stdout=fh).returncode == 0,
                        "git archive failed")
            with tarfile.open(archive) as tar:
                tar.extractall(tree, filter="data")
            archive.unlink()
            for name, (root, changed, deleted) in lanes.items():
                for path in deleted:
                    if path in merged_overlaps:
                        continue
                    (tree / path).unlink(missing_ok=True)
                for path in changed:
                    if path in merged_overlaps:
                        continue
                    src, dst = root / path, tree / path
                    if src.is_file():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
            for path, data in merged_overlaps.items():
                dst = tree / path
                if data is None:
                    dst.unlink(missing_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
            return "base archive + every lane's changes"

        def compile_all():
            dirs = [str(tree / d) for d in ("src", "tests", "scripts") if (tree / d).is_dir()]
            r = subprocess.run([args.python, "-m", "compileall", "-q", *dirs], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPYCACHEPREFIX": str(tmp / "pycache")})
            require(r.returncode == 0, "byte-compile failed:\n" + (r.stdout + r.stderr)[-2000:])
            return "src, tests and scripts byte-compile"

        def js():
            node = shutil.which("node")
            files = sorted({p for _, changed, _ in lanes.values() for p in changed if p.endswith(".js")})
            if not node or not files:
                return "skipped: " + ("node not installed" if not node else "no changed .js")
            for path in files:
                r = subprocess.run([node, "--check", str(tree / path)], capture_output=True, text=True)
                require(r.returncode == 0, f"{path}: {r.stderr.strip()[:500]}")
            return f"{len(files)} changed .js files parse"

        def discovery():
            with isolated_home():
                env = {**os.environ, "PYTHONPATH": str(tree / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
                r = subprocess.run([args.python, "-c", DISCOVER, json.dumps(SUITES)], cwd=tree, env=env,
                                   capture_output=True, text=True, timeout=300)
            require(r.returncode == 0, "discovery crashed:\n" + r.stderr[-2000:])
            report, problems = json.loads(r.stdout.strip().splitlines()[-1]), []
            for rel, info in report.items():
                print(f"  {rel}: {info['cases']} cases", flush=True)
                problems += [f"{rel}: {e}" for e in info["errors"]]
                problems += [f"{folder}: test files but no cases (missing __init__.py?)"
                             for folder, n in info["folders"].items() if not n]
            for p in problems:
                print(f"  PROBLEM {p}", flush=True)
            require(report and not problems, f"{len(problems)} discovery problems")
            return f"{sum(i['cases'] for i in report.values())} cases discovered; none executed"

        stages.check("footprints", footprints)
        stages.check("overlap", overlap, needs=("footprints",))
        stages.check("overlay", overlay, needs=("footprints", "overlap"))
        stages.check("compile", compile_all, needs=("overlay",))
        stages.check("js", js, needs=("overlay",))
        stages.check("discovery", discovery, needs=("compile",))
        return stages.status()
    finally:
        if args.keep and tree.exists():
            print(f"merged tree kept at {tree}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
