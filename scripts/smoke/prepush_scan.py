#!/usr/bin/env python3
"""Read-only outgoing-history privacy scan; print locations/categories, never matched values."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import subprocess
from pathlib import PurePosixPath

from _support import ROOT

REPO = ROOT  # the repository scanned; --repo overrides (the installed hook copy scans the pushing worktree)

MAX_BLOB = 2 * 1024 * 1024
PATTERNS = {
    "private-key marker": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "provider token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                                 r"|AKIA[0-9A-Z]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b"),
    "credential assignment": re.compile(r'''(?ix)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)
                                         ["']?\s*[:=]\s*["'][^"'\s]{8,}["']'''),
    "URL credentials": re.compile(r"[a-z][a-z0-9+.-]*://[^\s/@]+:[^\s/@]+@", re.I),
    "local private path": re.compile(r"(?:/home/[^/\s]+/|~/)(?:kgirl-teach|\.ssh|\.aws|\.gnupg)(?:/|\b)", re.I),
    "private material reference": re.compile(r"\b(?:Tithe|Main Wallet|Hand_off)(?:\b|_)", re.I),
    "possible personal name": re.compile(r'''\b(?:author|owner|full_name|real_name|user_name)["']?\s*[:=]\s*
                                            ["']?[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,3}''', re.X),
}
EMAIL = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:])(?:[A-Fa-f0-9]{0,4}:){2,}[A-Fa-f0-9]{0,4}(?![\w:])")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class ScanError(RuntimeError):
    pass


def git(*args):
    result = subprocess.run(["git", "--no-pager", "-c", "core.quotepath=false", *args], cwd=REPO,
                            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        # Git stderr can contain credential-bearing URLs or content. Do not relay it.
        raise ScanError(f"Git {args[0]} failed (exit {result.returncode}); check the local reference/path")
    return result.stdout


def protected(path):
    """Classify sensitive filenames before reading any diff or Git blob for them."""
    parts = [part.lower() for part in PurePosixPath(path).parts]
    name = parts[-1] if parts else ""
    if any(part in {"private", ".ssh", ".aws", ".gnupg", "kgirl-teach"} for part in parts):
        return "private directory (contents not opened)"
    if any("tithe" in part for part in parts):
        return "Tithe material (contents not opened)"
    if name.startswith(".env") or re.search(r"main[ _-]*wallet.*\.json$", name):
        return "environment/wallet file (contents not opened)"
    if re.search(r"hand[_-]?off", name) or "handoff" in parts:
        return "handoff material (contents not opened)"
    if (name in {"credentials", "hosts.yml", "id_rsa", "id_ed25519", ".netrc", ".npmrc", ".pypirc",
                 ".git-credentials"}
            or name.endswith((".pem", ".key", ".p12", ".pfx"))
            or re.search(r"(?:oauth|credentials?|tokens?|secrets?|(?:api[_-]?)?keys?|auth).*"
                         r"\.(?:json|ya?ml|toml|txt|ini|conf)$", name)):
        return "credential/key file (contents not opened)"
    return ""


def categories(text, names):
    found = {label for label, pattern in PATTERNS.items() if pattern.search(text)}
    # Exempt: GitHub noreply identities and role addresses that point to no person (noreply@anthropic.com in
    # Co-Authored-By trailers, no-reply@...). Everything else is a personal address until a human says otherwise.
    if any(not (e.lower().endswith("@users.noreply.github.com") or e.lower().split("@")[0] in ("noreply", "no-reply"))
           for e in EMAIL.findall(text)):
        found.add("email address")
    for match in (*IPV4.findall(text), *IPV6.findall(text)):
        try:
            address = ipaddress.ip_address(match)
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            found.add("IP address")
    if any(name.casefold() in text.casefold() for name in names):
        found.add("operator-specified personal name")
    return sorted(found)


def additions(diff):
    line = None
    for text in diff.splitlines():
        match = HUNK.match(text)
        if match:
            line = int(match.group(1))
        elif line is not None and text.startswith("+"):
            yield line, text[1:]
            line += 1
        elif line is not None and text.startswith(" "):
            line += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="@{upstream}", help="already-fetched base ref; default: branch upstream")
    parser.add_argument("--name", action="append", default=[], help="additional private name to flag, never printed")
    parser.add_argument("--repo", help="repository/worktree to scan (default: the one containing this script)")
    args = parser.parse_args()
    global REPO
    if args.repo:
        REPO = os.path.abspath(args.repo)
    if any(not name.strip() for name in args.name):
        parser.error("--name cannot be empty")
    findings = set()

    def record(commit, path, line, category):
        key = (commit, path, line, category)
        if key not in findings:
            findings.add(key)
            # A path can itself contain private values; do not print a matching path verbatim.
            location = (f"<redacted path sha256:{hashlib.sha256(path.encode('utf-8', 'surrogateescape')).hexdigest()[:12]}>"
                        if categories(path, args.name) else json.dumps(path, ensure_ascii=True))
            print(f"REVIEW {commit[:12]} {location}:{line}: {category}")

    try:
        base = git("rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}").decode().strip()
        head = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        commits = git("rev-list", "--reverse", head, f"^{base}").decode().splitlines()
        print(f"Scanning {len(commits)} outgoing commits; base {base[:12]}, tip {head[:12]}.")
        print("Working-tree/staged changes are outside this scan. No fetch or push is performed.")
        if not commits:
            print("NO OUTGOING COMMITS: nothing scanned; this is not a privacy clearance.")
            return 0
        for commit in commits:
            metadata = git("show", "-s", "--format=%an%n%ae%n%cn%n%ce%n%B", commit).decode("utf-8", "replace")
            for number, line in enumerate(metadata.splitlines(), 1):
                labels = categories(line, args.name)
                if number in (1, 3) and re.fullmatch(r"[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,3}", line):
                    labels.append("possible personal name in commit identity")
                for label in labels:
                    record(commit, "<commit metadata>", number, label)
            parents = git("rev-list", "--parents", "-n", "1", commit).decode().split()[1:] or [None]
            for parent in parents:
                common = ["--no-ext-diff", "--no-textconv", "--no-renames", "--diff-filter=ACMRT"]
                compare = ["diff", *common, parent, commit] if parent else [
                    "diff-tree", "--root", "--no-commit-id", "-r", *common, commit]
                paths = git(*compare, "--name-only", "-z").split(b"\0")
                for raw_path in paths:
                    if not raw_path:
                        continue
                    path = raw_path.decode("utf-8", "surrogateescape")
                    for label in categories(path, args.name):
                        record(commit, path, 0, label + " in filename")
                    reason = protected(path)
                    if reason:
                        record(commit, path, 0, reason)
                        continue
                    obj = f"{commit}:{path}"
                    kind = git("cat-file", "-t", obj).strip()
                    if kind != b"blob":
                        record(commit, path, 0, "unscanned non-blob (such as a submodule)")
                        continue
                    size = int(git("cat-file", "-s", obj))
                    if size > MAX_BLOB:
                        record(commit, path, 0, "unscanned file above 2 MiB")
                        continue
                    blob = git("cat-file", "blob", obj)
                    if b"\0" in blob or blob.startswith(b"version https://git-lfs.github.com/spec/v1"):
                        record(commit, path, 0, "unscanned binary/LFS payload")
                        continue
                    try:
                        blob.decode("utf-8")
                    except UnicodeDecodeError:
                        record(commit, path, 0, "unscanned non-UTF-8 payload")
                        continue
                    diff = git(*compare, "--unified=0", "-p", "--", f":(literal){path}").decode("utf-8", "replace")
                    for number, line in additions(diff):
                        for label in categories(line, args.name):
                            record(commit, path, number, label)
        if findings:
            print(f"{len(findings)} review findings. Matched values and source lines were withheld.")
            return 1
        print("No heuristic findings in scanned additions/metadata. Manual review is still required.")
        return 0
    except (ScanError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
        # Do not print exception text from subprocess timeouts: command arguments may contain private paths.
        print(f"SCAN INCOMPLETE ({type(exc).__name__}). Confirm --base names a local commit/ref and retry.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
