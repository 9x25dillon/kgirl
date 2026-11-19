"""Workspace: a project directory the chat agent may look at and, with the user's consent, change.

    ls, read, grep   look around. Confined to the workspace root; `.git` is off limits; a file that looks
                     like it holds a secret is refused, so keys never land in a prompt or a transcript.
    edit, write      change files. Each change is shown as a diff and applied only when the user says yes
                     (or the session was started with edits pre-approved). Every applied change is in a
                     journal, so `undo` restores it exactly. Changes and undos enter an append-only audit.
    run              run a command in the root. The model never picks a program on its own authority: the
                     command must be one the user allowed (KGIRL_VERIFY_ALLOWLIST, or `/allow` in this
                     session), or the user approves it when asked. Shell operators require that exact
                     command through `/allow`; every started run records its bounded outcome in the audit.

The model proposes, the person disposes. Without a way to ask (a one-shot message, a pipe, MCP) and
without pre-approval, edits and unlisted commands are refused with the reason, and nothing is changed.
"""

from __future__ import annotations

import difflib
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from ..atlas.secrets import secret_reason
from ..atlas.store import MAX_BYTES, SKIP_DIRS
from ..jev.code_env import _SECRET_ENV, _loose_replace   # the same edit rule and env scrub the swarm uses
from ..util import clip

Confirm = Callable[[str, str], bool]     # (question, what to show the user first) -> yes or no
Notify = Callable[[str], None]

READ_LINES = 200            # most lines one read returns
WRITE_BYTES = 400 * 1024    # largest file the agent may write
GREP_FILES = 4000           # most files one grep opens
RUN_OUTPUT = 3000           # chars of command output handed back
LOOK_OUTPUT = 12_000        # chars from read/grep handed to one model turn
NO_WRITE_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages"})


class WorkspaceError(ValueError):
    """A request the workspace refuses; the message is written for the model to read and act on."""


@dataclass(frozen=True)
class Change:
    path: str               # relative to the root, POSIX form
    before: str | None      # exact previous text; None when the file did not exist
    after: str


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    action: str
    target: str
    outcome: str


def allowlisted_commands() -> list[list[str]]:
    """Commands the user allows a model to run: KGIRL_VERIFY_ALLOWLIST, commands separated by ';'."""
    return [shlex.split(c) for c in os.environ.get("KGIRL_VERIFY_ALLOWLIST", "").split(";") if c.strip()]


def has_shell_operator(command: str) -> bool:
    """Whether command has an unquoted shell control token (quoted `|`/`>` are ordinary arguments)."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return any(re.fullmatch(r"[|&;<>]+", token) for token in lexer)


class Workspace:
    def __init__(self, root: str | os.PathLike, confirm: Confirm | None = None, auto_edit: bool = False,
                 allowed: list[list[str]] | None = None, run_timeout: float = 120.0, notify: Notify | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"{root} is not a directory")
        self.confirm = confirm
        self.auto_edit = auto_edit
        self.allowed: list[list[str]] = [list(c) for c in allowed or []]    # allowed by the user in this session
        self.allowed_shell: set[str] = set()              # only exact text entered through interactive /allow
        self.run_timeout = run_timeout
        self.notify = notify or (lambda text: None)
        self.journal: list[Change] = []
        self.audit: list[AuditEvent] = []

    def _audit(self, action: str, target: str, outcome: str) -> None:
        self.audit.append(AuditEvent(len(self.audit) + 1, action, target, outcome))

    def audit_since(self, seq: int = 0) -> list[AuditEvent]:
        """A copy of events after seq; callers can persist them without being able to rewrite history."""
        return list(self.audit[max(0, int(seq)):])

    @property
    def name(self) -> str:
        return self.root.name

    # ------------------------------------------------------------------ paths

    def resolve(self, path: str | os.PathLike | None, write: bool = False) -> Path:
        """The absolute path of `path` inside the root, or WorkspaceError when it may not be touched."""
        raw = str(path or ".").strip() or "."
        full = Path(raw) if os.path.isabs(raw) else self.root / raw
        resolved = full.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"{raw} is outside the workspace {self.root}")
        parts = resolved.relative_to(self.root).parts
        if ".git" in parts:
            raise WorkspaceError(".git is off limits")
        if write:
            if not parts:
                raise WorkspaceError("that is the workspace root, not a file")
            blocked = NO_WRITE_DIRS.intersection(parts[:-1])
            if blocked:
                raise WorkspaceError(f"kgirl does not write inside {sorted(blocked)[0]}/")
            cursor = full
            through_link = False
            while cursor != self.root and self.root in cursor.parents:
                if cursor.is_symlink():
                    through_link = True
                    break
                cursor = cursor.parent
            if through_link:
                raise WorkspaceError(f"{raw} is a symlink; kgirl does not write through links")
        return resolved

    def rel(self, full: Path) -> str:
        return full.relative_to(self.root).as_posix() or "."

    def _walk(self, base: Path) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                full = Path(dirpath) / fn
                if not full.is_symlink():
                    yield full

    @staticmethod
    def _text(full: Path) -> str | None:
        """A file's text, or None when it is too large, binary or unreadable."""
        try:
            if full.stat().st_size > MAX_BYTES:
                return None
            raw = full.read_bytes()
        except OSError:
            return None
        if b"\x00" in raw[:4096]:
            return None
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ looking

    def ls(self, path: str = ".", depth: int = 1, limit: int = 200) -> str:
        base = self.resolve(path)
        if base.is_file():
            return f"{self.rel(base)}  ({_size(base)})"
        if not base.is_dir():
            raise WorkspaceError(f"{path} does not exist")
        depth = max(1, min(3, int(depth)))
        lines: list[str] = []
        truncated = False

        def walk(directory: Path, level: int) -> None:
            nonlocal truncated
            try:
                entries = sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return
            for entry in entries:
                if entry.name == ".git":
                    continue
                if len(lines) >= limit:
                    truncated = True
                    return
                if entry.is_dir() and not entry.is_symlink():
                    lines.append(f"{self.rel(entry)}/")
                    if level < depth and entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                        walk(entry, level + 1)
                else:
                    lines.append(f"{self.rel(entry)}  ({_size(entry)})")

        walk(base, 1)
        if not lines:
            return f"{self.rel(base)}/ is empty"
        return "\n".join(lines) + ("\n(more entries not shown; list a subdirectory)" if truncated else "")

    def read(self, path: str, start: int = 1, end: int | None = None) -> str:
        full = self.resolve(path)
        if full.is_dir():
            raise WorkspaceError(f"{path} is a directory; use ls")
        if not full.is_file():
            raise WorkspaceError(f"{path} does not exist")
        rel = self.rel(full)
        text = self._text(full)
        if text is None:
            raise WorkspaceError(f"{rel} is binary or larger than {MAX_BYTES // 1024} KB; kgirl does not read it")
        why = secret_reason(rel, text)
        if why:
            raise WorkspaceError(f"{rel} looks like it holds a secret ({why}); kgirl keeps it out of the conversation")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        lo = max(1, int(start))
        hi = min(len(lines), lo + READ_LINES - 1, int(end) if end else lo + READ_LINES - 1)
        if lo > len(lines):
            return f"{rel} has {len(lines)} lines"
        header = f"{rel}: lines {lo}-{hi} of {len(lines)}"
        shown: list[str] = []
        used = len(header) + 1
        next_line = None
        for n in range(lo, hi + 1):
            line = f"{n:>5} {clip(lines[n - 1], 2000)}"
            if used + len(line) + 1 > LOOK_OUTPUT:
                next_line = n
                break
            shown.append(line)
            used += len(line) + 1
        if next_line is not None:
            shown.append(f"…(result truncated; call read_file again with path={rel!r}, start={next_line})")
        return header + "\n" + "\n".join(shown)

    def grep(self, pattern: str, path: str = ".", limit: int = 40) -> str:
        if not pattern:
            raise WorkspaceError("grep needs a pattern")
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))       # not a regular expression: search for it literally
        base = self.resolve(path)
        if not base.exists():
            raise WorkspaceError(f"{path} does not exist")
        hits: list[str] = []
        more = opened = 0
        for full in ([base] if base.is_file() else self._walk(base)):
            if opened >= GREP_FILES:
                break
            opened += 1
            text = self._text(full)
            if text is None:
                continue
            found = [(n, line) for n, line in enumerate(text.split("\n"), 1) if len(line) < 2000 and rx.search(line)]
            if not found or secret_reason(self.rel(full), text):
                continue
            for n, line in found:
                if len(hits) < limit:
                    hits.append(f"{self.rel(full)}:{n}: {clip(line.strip(), 200)}")
                else:
                    more += 1
        if not hits:
            return f"no match for {pattern!r}"
        out = "\n".join(hits) + (f"\n(+{more} more matches; narrow the pattern or the path)" if more else "")
        if len(out) > LOOK_OUTPUT:
            out = out[:LOOK_OUTPUT] + "\n…(result truncated; narrow the grep pattern or path)"
        return out

    # ------------------------------------------------------------------ changing

    def edit(self, path: str, search: str, replace: str) -> str:
        """Replace `search` with `replace` in one file. `search` must occur exactly once (indentation may differ)."""
        full = self.resolve(path, write=True)
        if not full.is_file():
            raise WorkspaceError(f"{path} does not exist; use write_file to create it")
        if not search or not search.strip():
            raise WorkspaceError("edit needs the exact text to replace in `search`")
        rel = self.rel(full)
        raw = _load(full)
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n") if crlf else raw
        search, replace = search.replace("\r\n", "\n"), replace.replace("\r\n", "\n")
        count = text.count(search)
        new = text.replace(search, replace, 1) if count == 1 else \
            (_loose_replace(text, search, replace) if count == 0 else None)
        if new is None:
            raise WorkspaceError(f"the search text occurs {count} times in {rel}; it must match exactly once. "
                                 "Read the file and copy the lines exactly, with enough context to be unique.")
        if new == text:
            return f"{rel}: the edit changes nothing"
        return self._apply(full, raw, new.replace("\n", "\r\n") if crlf else new)

    def write(self, path: str, content: str) -> str:
        """Create a file, or replace one whole (prefer `edit` for a change to an existing file)."""
        full = self.resolve(path, write=True)
        if full.exists() and not full.is_file():
            raise WorkspaceError(f"{path} is not a regular file")
        if len(content.encode("utf-8")) > WRITE_BYTES:
            raise WorkspaceError(f"the content is larger than {WRITE_BYTES // 1024} KB")
        rel = self.rel(full)
        why = secret_reason(rel, content)
        if why:
            raise WorkspaceError(f"the content for {rel} looks like it holds a secret ({why}); not written")
        before = _load(full) if full.exists() else None
        if before == content:
            return f"{rel}: already has exactly this content"
        return self._apply(full, before, content)

    def _apply(self, full: Path, before: str | None, after: str) -> str:
        rel = self.rel(full)
        diff = _diff(rel, before, after)
        if not self.auto_edit:
            if self.confirm is None:
                raise WorkspaceError(f"changing {rel} needs the user's approval, and this session cannot ask for it. "
                                     "Show the user the change you propose instead.")
            if not self.confirm(f"apply this change to {rel}?", diff):
                raise WorkspaceError(f"the user declined the change to {rel}; nothing was changed")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(after.encode("utf-8"))
        self.journal.append(Change(rel, before, after))
        self._audit("create" if before is None else "edit", rel, "applied")
        self.notify(f"{'created' if before is None else 'edited'} {rel}" + (f"\n{diff}" if self.auto_edit else ""))
        return f"{'created' if before is None else 'edited'} {rel}\n{clip(diff, 1500)}"

    def record(self, path: str, before: str | None, after: str) -> None:
        """Add a change something else applied (an accepted swarm diff) to the journal, so `undo` covers it."""
        self.journal.append(Change(path, before, after))
        self._audit("create" if before is None else "edit", path, "recorded external change")

    def undo(self) -> str:
        """Take back the most recent journaled change, if the file is still as kgirl left it."""
        if not self.journal:
            return "nothing to undo"
        change = self.journal[-1]
        full = self.root / change.path
        current = _load(full) if full.is_file() else None
        if current != change.after:
            self._audit("undo", change.path, "refused: file changed")
            return f"{change.path} has changed since kgirl edited it; not undoing"
        if change.before is None:
            full.unlink()
        else:
            full.write_bytes(change.before.encode("utf-8"))
        self.journal.pop()
        self._audit("undo", change.path, "applied")
        return f"undid the change to {change.path}"

    def diff(self) -> str:
        """Everything kgirl changed in this session, as one unified diff against the original text."""
        first: dict[str, str | None] = {}
        for change in self.journal:
            first.setdefault(change.path, change.before)
        parts = []
        for path, before in first.items():
            full = self.root / path
            current = _load(full) if full.is_file() else None
            if current != before:
                parts.append(_diff(path, before, current))
        return "\n".join(p for p in parts if p)

    # ------------------------------------------------------------------ running

    def allowed_commands(self) -> list[list[str]]:
        return [*self.allowed, *allowlisted_commands()]

    def allow(self, command: str) -> list[str]:
        """The user allows a command for this session (`/allow` in the chat)."""
        exact = command.strip()
        argv = shlex.split(exact)
        if not argv:
            raise WorkspaceError("empty command")
        if has_shell_operator(exact):
            self.allowed_shell.add(exact)
        if argv not in self.allowed:
            self.allowed.append(argv)
        return argv

    def run(self, command: str) -> str:
        exact = (command or "").strip()
        try:
            argv = shlex.split(exact)
        except ValueError as exc:
            raise WorkspaceError(f"cannot parse the command: {exc}") from exc
        if not argv:
            raise WorkspaceError("run needs a command")
        if has_shell_operator(exact):
            if exact not in self.allowed_shell:
                raise WorkspaceError("a command with a shell operator is refused unless the user names that exact "
                                     "command with /allow")
            return self.execute_shell(exact)
        if argv not in self.allowed_commands():
            if self.confirm is None:
                choices = ", ".join(shlex.join(c) for c in self.allowed_commands()) or "none yet"
                raise WorkspaceError(f"`{shlex.join(argv)}` is not a command the user has allowed (allowed: {choices}). "
                                     "Ask the user to run it, or to allow it with /allow.")
            if not self.confirm(f"run this in {self.root}?", "$ " + shlex.join(argv)):
                raise WorkspaceError("the user declined to run that command")
        return self.execute(argv)

    def execute_shell(self, command: str) -> str:
        """Run one exact operator-bearing command explicitly entered through /allow."""
        return self._execute(command, command, shell=True)

    def execute(self, argv: list[str]) -> str:
        """Run an approved command in the root: no shell, no terminal input, secrets stripped from the environment."""
        shown = shlex.join(argv)
        return self._execute(argv, shown, shell=False)

    def _execute(self, command: list[str] | str, shown: str, shell: bool) -> str:
        """Run a command with the shared timeout, environment scrub and bounded result."""
        env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
        self.notify(f"$ {shown}")
        try:
            proc = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=self.run_timeout,
                                  env=env, stdin=subprocess.DEVNULL, shell=shell)
        except subprocess.TimeoutExpired:
            self._audit("run", shown, f"timeout after {self.run_timeout:g} s")
            return f"$ {shown}\ntimed out after {self.run_timeout:g} s"
        except OSError as exc:
            self._audit("run", shown, f"could not start: {type(exc).__name__}")
            return f"$ {shown}\ncould not start: {exc}"
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > RUN_OUTPUT:
            out = f"…(output truncated; showing last {RUN_OUTPUT} characters)\n" + out[-RUN_OUTPUT:]
        self._audit("run", shown, f"exit={proc.returncode}")
        return f"$ {shown}\nexit={proc.returncode}\n{out}".rstrip()

    # ------------------------------------------------------------------ description

    def branch(self) -> str:
        """The checked-out git branch, or "" when the root is not a git checkout (read from .git, never run)."""
        try:
            dot = self.root / ".git"
            if dot.is_file():                                        # a worktree: "gitdir: <path>"
                dot = (self.root / dot.read_text(encoding="utf-8").split(":", 1)[1].strip()).resolve()
            head = (dot / "HEAD").read_text(encoding="utf-8").strip()
        except (OSError, IndexError, UnicodeDecodeError):
            return ""
        return head.split("refs/heads/", 1)[-1] if head.startswith("ref:") else head[:10]   # detached: short hash

    def describe(self) -> str:
        branch = self.branch()
        edits = "the user approves each change" if not self.auto_edit else "changes apply at once and are journaled"
        return (f"WORKSPACE: {self.root}" + (f" (git branch {branch})" if branch else "") +
                f". Paths in ls, read_file, grep, edit, write_file and run are relative to it; {edits}.")


def _load(full: Path) -> str:
    """A file's exact text (line endings kept), or WorkspaceError when it is not UTF-8 text."""
    try:
        return full.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceError(f"{full.name} is not UTF-8 text") from exc


def _diff(rel: str, before: str | None, after: str | None) -> str:
    old = (before or "").splitlines(True)
    new = (after or "").splitlines(True)
    return "".join(difflib.unified_diff(old, new, f"a/{rel}" if before is not None else "/dev/null",
                                        f"b/{rel}" if after is not None else "/dev/null", n=2))


def _size(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "?"
    return f"{n} B" if n < 1024 else (f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.1f} MB")
