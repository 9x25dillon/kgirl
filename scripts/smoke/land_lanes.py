#!/usr/bin/env python3
"""Land lane worktrees on the integration branch: a printed plan by default, git writes only with --apply.

Steps, in this order (INTEGRATION.md 1, 2 and 4; the push stays a separate, human step):
  rewrite  Every commit not yet on the upstream whose author or committer email is not a GitHub noreply
           address gets the noreply identity from `git config user.email`. Each new commit object is the
           old one byte for byte apart from those lines and its parents (a signature is dropped), so trees
           are unchanged and every worktree, including uncommitted lane work, stays valid. All local
           branches holding such commits move in one atomic `update-ref` transaction, and the old tips are
           kept under refs/kgirl-landing/original/ so the rewrite can be undone.
  commit   Each lane's uncommitted changes (tracked, untracked, deleted) become one commit on its branch,
           made with the noreply identity. Paths that look private (.env, keys, model weights, private/,
           wallet, Hand_off, Tithe) stop the whole run before anything is written.
  merge    Each lane branch is merged into the integration branch with --no-ff, in the order given. A
           conflict aborts that merge, leaves the integration worktree as it was, and stops.

Emails being replaced are never printed, only counted.

Example (plan only):
  python3 scripts/smoke/land_lanes.py --stack ~/work/kgirl-stack \\
      --lane ~/work/kgirl-eng --lane ~/work/kgirl-helix-ui --lane ~/work/kgirl-builder
Add --apply to write, and --steps rewrite,commit,merge to choose steps.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

NOREPLY = re.compile(r"@users\.noreply\.github\.com$", re.I)
IDENT_LINE = re.compile(rb"^(author|committer) (.*) <([^>]*)> (\d+ [+-]\d{4})$")
BACKUP = "refs/kgirl-landing/original/"
STEPS = ("rewrite", "commit", "merge")
PRIVATE = [re.compile(p, re.I) for p in (
    r"(^|/)\.env(\.|$)", r"(^|/)private/", r"(^|/)\.ssh/", r"\.(pem|key|p12|pfx|kdbx)$",
    r"\.(pt|pth|ckpt|safetensors|gguf)$", r"wallet", r"hand_?off(?!.*\.py$)", r"tithe", r"(^|/)\.git-credentials$",
)]


class LandError(RuntimeError):
    pass


def git(repo: Path, *args: str, input: bytes | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=false", *args], input=input,
                       capture_output=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if check and r.returncode:
        raise LandError(f"git {args[0]} failed in {repo}: {r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout.decode(errors="replace")


def git_bytes(repo: Path, *args: str) -> bytes:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode:
        raise LandError(f"git {args[0]} failed: {r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout


@dataclass
class Lane:
    path: Path
    branch: str
    head: str
    changes: list[str] = field(default_factory=list)

    @property
    def short(self) -> str:
        return self.branch.rsplit("/", 1)[-1].removeprefix("kgirl-")


def branch_of(path: Path) -> str:
    name = git(path, "symbolic-ref", "-q", "--short", "HEAD", check=False).strip()
    if not name:
        raise LandError(f"{path} is not on a branch (detached HEAD)")
    return name


def changed_paths(path: Path) -> list[str]:
    out = git(path, "status", "--porcelain=v1", "-z", "-uall")
    paths, items = [], out.split("\0")
    i = 0
    while i < len(items):
        item = items[i]
        if len(item) > 3:
            paths.append(item[3:])
            if item[0] in "RC":          # rename/copy: the source path follows as its own entry
                i += 1
        i += 1
    return paths


def private_hits(paths: list[str]) -> list[str]:
    return [p for p in paths if any(rx.search(PurePosixPath(p).as_posix()) for rx in PRIVATE)]


# ---------------------------------------------------------------------------------------------- rewrite

def noreply_identity(repo: Path) -> tuple[str, str]:
    email = git(repo, "config", "user.email", check=False).strip()
    name = git(repo, "config", "user.name", check=False).strip()
    if not NOREPLY.search(email) or not name:
        raise LandError("git user.name / user.email must be set, and the email must be a GitHub noreply address")
    return name, email


def outgoing(repo: Path, upstream: str, refs: list[str]) -> list[tuple[str, list[str]]]:
    """(commit, parents) for every commit reachable from refs but not upstream, parents before children."""
    if not refs:
        return []
    out = git(repo, "rev-list", "--topo-order", "--reverse", "--parents", *refs, f"^{upstream}")
    return [(line.split()[0], line.split()[1:]) for line in out.splitlines() if line.strip()]


def needs_rewrite(raw: bytes) -> bool:
    for line in raw.split(b"\n\n", 1)[0].split(b"\n"):
        m = IDENT_LINE.match(line)
        if m and not NOREPLY.search(m.group(3).decode(errors="replace")):
            return True
    return False


def rewrite_object(raw: bytes, parents: dict[str, str], ident: tuple[str, str]) -> tuple[bytes, bool]:
    """Return the commit object with mapped parents and noreply identity lines; True when a signature was dropped."""
    head, sep, body = raw.partition(b"\n\n")
    lines, dropped, in_sig = [], False, False
    for line in head.split(b"\n"):
        if in_sig and line.startswith(b" "):
            continue                                  # continuation of a dropped signature header
        in_sig = False
        if line.startswith((b"gpgsig ", b"gpgsig-sha256 ")):
            in_sig = dropped = True
            continue
        if line.startswith(b"parent "):
            old = line[7:].decode()
            line = b"parent " + parents.get(old, old).encode()
        m = IDENT_LINE.match(line)
        if m and not NOREPLY.search(m.group(3).decode(errors="replace")):
            line = b"%s %s <%s> %s" % (m.group(1), ident[0].encode(), ident[1].encode(), m.group(4))
        lines.append(line)
    return b"\n".join(lines) + sep + body, dropped


@dataclass
class RewritePlan:
    affected: int = 0
    remapped: dict[str, str] = field(default_factory=dict)       # old sha -> new sha (only changed ones)
    refs: dict[str, tuple[str, str]] = field(default_factory=dict)  # ref -> (old, new)
    signatures_dropped: int = 0
    exposed: list[str] = field(default_factory=list)            # remote refs already holding affected commits
    detached: list[str] = field(default_factory=list)           # worktrees whose detached HEAD would be orphaned
    untouched: list[str] = field(default_factory=list)          # other branches still holding the old commits


def plan_rewrite(repo: Path, upstream: str, ident: tuple[str, str], write: bool, scope: list[str]) -> RewritePlan:
    """Rewrite only history reachable from the `scope` branches (the stack and its lanes).

    Any local branch whose tip lies in that history moves too (e.g. the lane a stack was built from);
    branches that merely share old commits elsewhere are listed in `untouched`, never moved.
    """
    plan = RewritePlan()
    heads = git(repo, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads/").splitlines()
    tips = {ref: sha for ref, sha in (line.split() for line in heads)}
    commits = outgoing(repo, upstream, [f"refs/heads/{b}" for b in scope])
    mapping: dict[str, str] = {}
    for sha, parents in commits:
        raw = git_bytes(repo, "cat-file", "commit", sha)
        moved = any(p in mapping for p in parents)
        if not (moved or needs_rewrite(raw)):
            continue
        plan.affected += needs_rewrite(raw)
        new_raw, dropped = rewrite_object(raw, mapping, ident)
        plan.signatures_dropped += dropped
        mapping[sha] = git(repo, "hash-object", "-t", "commit", *(["-w"] if write else []), "--stdin",
                           input=new_raw).strip()
    plan.remapped = mapping
    plan.refs = {ref: (sha, mapping[sha]) for ref, sha in tips.items() if sha in mapping}
    if mapping:
        holders = git(repo, "for-each-ref", "--format=%(refname)", *[f"--contains={s}" for s in mapping], "refs/heads/")
        plan.untouched = sorted(r.removeprefix("refs/heads/") for r in holders.split() if r not in plan.refs)
    if mapping:
        remote = git(repo, "for-each-ref", "--format=%(refname)", *[f"--contains={s}" for s in mapping], "refs/remotes/")
        plan.exposed = [r for r in remote.splitlines() if r.strip()]
        for block in git(repo, "worktree", "list", "--porcelain").split("\n\n"):
            fields = dict(l.split(" ", 1) for l in block.splitlines() if " " in l)
            if "detached" in block and fields.get("HEAD") in mapping:
                plan.detached.append(fields.get("worktree", "?"))
    return plan


def apply_rewrite(repo: Path, plan: RewritePlan) -> None:
    if not plan.refs:
        return
    tx = ["start"]
    for ref, (old, new) in sorted(plan.refs.items()):
        tx.append(f"update {ref} {new} {old}")
        tx.append(f"create {BACKUP}{ref.removeprefix('refs/heads/')} {old}")
    tx += ["prepare", "commit"]
    git(repo, "update-ref", "--stdin", input=("\n".join(tx) + "\n").encode())


# ---------------------------------------------------------------------------------------------- run

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, type=Path, help="integration worktree (must be clean)")
    ap.add_argument("--lane", action="append", default=[], type=Path, help="lane worktree, in merge order")
    ap.add_argument("--upstream", default="origin/main", help="already-public base; commits on it are never rewritten")
    ap.add_argument("--message", action="append", default=[], metavar="LANE=TEXT",
                    help="commit message for a lane (by short name, e.g. eng=..., or by branch)")
    ap.add_argument("--steps", default=",".join(STEPS), help="comma list from: " + ", ".join(STEPS))
    ap.add_argument("--apply", action="store_true", help="perform the git writes (default: print the plan only)")
    args = ap.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if bad := [s for s in steps if s not in STEPS]:
        raise SystemExit(f"unknown step(s): {', '.join(bad)}")
    stack = args.stack.expanduser().resolve()
    messages = dict(m.split("=", 1) for m in args.message if "=" in m)
    mode = "APPLY" if args.apply else "PLAN (no writes; add --apply)"
    print(f"land_lanes: {mode}; steps {', '.join(steps)}")

    try:
        ident = noreply_identity(stack)
        stack_branch = branch_of(stack)
        if changed_paths(stack):
            raise LandError(f"{stack} has uncommitted changes; the integration worktree must be clean")
        git(stack, "rev-parse", "--verify", "-q", args.upstream)
        lanes = []
        for p in args.lane:
            p = p.expanduser().resolve()
            lane = Lane(p, branch_of(p), git(p, "rev-parse", "HEAD").strip(), changed_paths(p))
            if hits := private_hits(lane.changes):
                raise LandError(f"{lane.branch}: {len(hits)} path(s) look private, e.g. {hits[0]}; remove or ignore them")
            lanes.append(lane)
        if len({l.branch for l in lanes}) != len(lanes):
            raise LandError("a lane is listed twice")
        print(f"stack {stack_branch} @ {git(stack, 'rev-parse', '--short', 'HEAD').strip()}, identity {ident[0]} <noreply>")

        if "rewrite" in steps:
            plan = plan_rewrite(stack, args.upstream, ident, write=args.apply,
                                scope=[stack_branch, *(l.branch for l in lanes)])
            print(f"rewrite: {plan.affected} commit(s) carry a non-noreply email; "
                  f"{len(plan.remapped)} object(s) re-made; {len(plan.refs)} branch(es) move")
            for ref, (old, new) in sorted(plan.refs.items()):
                print(f"  {ref.removeprefix('refs/heads/')}: {old[:8]} -> {new[:8]}")
            if plan.untouched:
                print(f"  not moved (outside the stack; they keep the old commits): {', '.join(plan.untouched)}")
            if plan.signatures_dropped:
                print(f"  note: {plan.signatures_dropped} signature(s) will be dropped")
            if plan.exposed:
                raise LandError(f"affected commits are already on {', '.join(plan.exposed)}; rewriting cannot hide them")
            if plan.detached:
                raise LandError(f"detached worktree(s) sit on affected commits: {', '.join(plan.detached)}")
            if args.apply:
                apply_rewrite(stack, plan)
                if plan.refs:
                    print(f"  done; undo: git update-ref refs/heads/<branch> {BACKUP}<branch>")
            moved = {ref.removeprefix("refs/heads/"): new for ref, (_, new) in plan.refs.items()}
            for lane in lanes:                     # in plan mode, compare against where the branches WOULD be
                lane.head = moved.get(lane.branch, lane.head)
            stack_head = moved.get(stack_branch, git(stack, "rev-parse", "HEAD").strip())
        else:
            stack_head = git(stack, "rev-parse", "HEAD").strip()

        if "commit" in steps:
            for lane in lanes:
                if not lane.changes:
                    print(f"commit {lane.branch}: nothing uncommitted; skipped")
                    continue
                if lane.head != stack_head:
                    raise LandError(f"{lane.branch} HEAD is not the stack HEAD; rebase or check it by hand first")
                msg = messages.get(lane.short) or messages.get(lane.branch) or f"feat({lane.short}): land the {lane.short} lane"
                print(f"commit {lane.branch}: {len(lane.changes)} path(s); message: {msg.splitlines()[0]}")
                if args.apply:
                    git(lane.path, "add", "-A")
                    git(lane.path, "-c", f"user.name={ident[0]}", "-c", f"user.email={ident[1]}",
                        "commit", "-q", "-m", msg)

        if "merge" in steps:
            for lane in lanes:
                print(f"merge {lane.branch} -> {stack_branch} (--no-ff)")
                if not args.apply:
                    continue
                if changed_paths(lane.path):
                    raise LandError(f"{lane.branch} still has uncommitted changes; run the commit step first")
                r = subprocess.run(["git", "-C", str(stack), "-c", f"user.name={ident[0]}", "-c", f"user.email={ident[1]}",
                                    "merge", "--no-ff", "--no-edit", lane.branch], capture_output=True, text=True)
                if r.returncode:
                    git(stack, "merge", "--abort", check=False)
                    raise LandError(f"merging {lane.branch} failed and was aborted: {r.stdout.strip()[-300:]}")
            if args.apply:
                print(f"stack {stack_branch} now @ {git(stack, 'rev-parse', '--short', 'HEAD').strip()}")
    except LandError as exc:
        print(f"STOP: {exc}")
        return 1
    print("next: run the checks on the stack, then push by hand (the pre-push privacy gate runs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
