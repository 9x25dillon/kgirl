#!/usr/bin/env python3
"""Preview landing several lane worktrees on one base, with no git writes and no tests run.

It copies a clean tree of the base commit (`git archive`) into a temp folder, then overlays each lane's
changes against that commit: committed and uncommitted edits, new untracked files and deletions. It
reports each lane's footprint, fails on any path that two lanes both change, and then checks the merged
tree:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="worktree whose HEAD is the landing target")
    parser.add_argument("--lane", type=Path, action="append", required=True, help="lane worktree (repeatable)")
    parser.add_argument("--python", default=sys.executable, help="interpreter for discovery (needs numpy)")
    parser.add_argument("--keep", action="store_true", help="keep the merged tree and print its path")
    args = parser.parse_args()
    stages, lanes = Stages(), {}
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
            for path, who in sorted(clash.items()):
                print(f"  CONFLICT {path}: {', '.join(who)}", flush=True)
            require(not clash, f"{len(clash)} paths changed by more than one lane; merge those by hand")
            return "no path is changed by two lanes"

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
                    (tree / path).unlink(missing_ok=True)
                for path in changed:
                    src, dst = root / path, tree / path
                    if src.is_file():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
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
        stages.check("overlay", overlay, needs=("footprints",))
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
