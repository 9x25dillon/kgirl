"""The user's standing rules: always in the prompt, and checked against every answer.

Similarity recall decides which preferences reach a prompt, so a rule like "never push unless I name it"
only appears when the message happens to look like it. A 7b model then writes `git push` into a script
the user asked for. Two layers close that gap:

  standing_rules  every active preference phrased as a rule ("Never ...", "Always ...", "Only ...",
                  "Before ...", "Don't ...") goes into every system prompt, ahead of anything recalled.
  check           a deterministic look at the code the answer contains (fenced blocks only, so prose that
                  merely mentions a command is never flagged) for steps the user's rules forbid, unless
                  the user's own message names that step. The mind revises once on a hit.

The guards are deliberately few and literal: each is a step that is costly or public and hard to undo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..soup import Soup
from ..soup.pool import Fragment
from ..util import estimate_tokens

RULE_START = re.compile(r"^\s*(never|always|only|before|don'?t|do not|no\b)", re.I)
FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)
TOKEN_SPLIT = re.compile(r"[\s,\"'\[\]()]+")
GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def is_rule(text: str) -> bool:
    return bool(RULE_START.match(text or ""))


def standing_rules(soup: Soup, scope: str | None = None, budget_tokens: int = 700) -> list[Fragment]:
    """Active preferences phrased as rules, global or for this scope, oldest first, within the budget."""
    picked, used = [], 0
    for f in soup.all(status="active", kind="preference"):
        if not is_rule(f.text) or f.scope not in ("*", scope or "*"):
            continue
        cost = estimate_tokens(f.text) + 2
        if used + cost > budget_tokens:
            break
        picked.append(f)
        used += cost
    return picked


def render_rules(rules: list[Fragment]) -> str:
    return ("RULES THE USER SET (binding: they override the request's wording, examples included):\n"
            + "\n".join(f"- {f.text}" for f in rules))


# ---------------------------------------------------------------------------------------------- check

@dataclass(frozen=True)
class Violation:
    name: str
    why: str
    snippet: str


def code_blocks(answer: str) -> list[str]:
    return [m.group(1) for m in FENCE.finditer(answer or "")]


def git_calls(code: str) -> list[list[str]]:
    """Each `git ...` invocation in shell or list form as [subcommand, *rest], options before it skipped."""
    calls = []
    for line in code.splitlines():
        tokens = [t for t in TOKEN_SPLIT.split(line) if t]
        for i, tok in enumerate(tokens):
            if tok != "git" and not tok.endswith("/git"):
                continue
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("-"):
                j += 2 if tokens[j] in GIT_OPTS_WITH_ARG else 1
            if j < len(tokens):
                calls.append(tokens[j:j + 6])
    return calls


def _named(user_text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", user_text or "", re.I) for w in words)


def check(answer: str, user_text: str) -> list[Violation]:
    """Steps in the answer's code that the user's rules forbid and the user's message did not ask for."""
    found: dict[str, Violation] = {}

    def hit(name: str, why: str, snippet: str) -> None:
        found.setdefault(name, Violation(name, why, " ".join(snippet.split())[:80]))

    for code in code_blocks(answer):
        for call in git_calls(code):
            sub, rest = call[0], call[1:]
            text = "git " + " ".join(call)
            if sub == "push" and ("--force" in rest or "-f" in rest or "--force-with-lease" in rest):
                hit("force-push", "force-pushes, which rewrites published history", text)
            elif sub == "push" and not _named(user_text, "push", "publish"):
                hit("push", "pushes; the user's repos can be public and pushing is publication", text)
            elif sub == "pull" and not _named(user_text, "pull"):
                hit("pull", "pulls into an existing checkout that may hold unique uncommitted work", text)
            elif sub == "commit" and not _named(user_text, "commit"):
                hit("commit", "commits, which the user does only when they name it", text)
            elif sub == "add" and rest[:1] and rest[0] in (".", "-A", "--all") and not _named(user_text, "git add"):
                hit("add-all", "stages everything (`git add .`/`-A`), which sweeps up secrets and stray files", text)
            elif sub == "reset" and "--hard" in rest and not _named(user_text, "reset --hard"):
                hit("reset-hard", "hard-resets, destroying uncommitted work", text)
            elif sub == "clean" and any(r.startswith("-") and "f" in r for r in rest) and not _named(user_text, "git clean"):
                hit("clean", "deletes untracked files (`git clean -f`)", text)
            elif sub == "branch" and "-D" in rest and not _named(user_text, "delete branch", "branch -D"):
                hit("branch-delete", "force-deletes a branch", text)
        for m in re.finditer(r"\brm\s+-[a-zA-Z]*(?:rf|fr)[a-zA-Z]*\s+\S+", code):
            if not _named(user_text, "rm -rf", "delete", "remove"):
                hit("rm-rf", "recursively force-deletes files", m.group(0))
        for m in re.finditer(r"\b(torch\.load|pickle\.loads?)\s*\(([^)]*)\)?", code):
            if "weights_only=True" not in (m.group(2) or "") and not _named(user_text, "pickle", "torch.load"):
                hit("pickle-load", "loads a pickle-format file, which can run code", m.group(0))
    return list(found.values())


def revision_prompt(violations: list[Violation], rules: list[Fragment]) -> str:
    lines = "\n".join(f"- `{v.snippet}`: {v.why}" for v in violations)
    ruled = ("\nThe rules:\n" + "\n".join(f"- {f.text}" for f in rules)) if rules else ""
    return ("RULES CHECK: your draft breaks rules the user set. Its code does this:\n" + lines + ruled +
            "\nRewrite the WHOLE answer so no code or command takes those steps. If the task truly needs one, "
            "leave it out of the code and tell the user, in one line, which step they should run themselves "
            "and why. Do not mention this check.")


def warning(violations: list[Violation]) -> str:
    steps = ", ".join(f"`{v.snippet}`" for v in violations)
    return (f"\n\n> ⚠ kgirl's rules check: this answer still contains {steps}, which breaks a rule you set. "
            "Don't run that step unless you mean to.")
