"""The user's standing rules: always in the prompt, and checked against every answer.

Similarity recall decides which preferences reach a prompt, so a rule like "never push unless I name it"
only appears when the message happens to look like it. A 7b model then writes `git push` into a script
the user asked for. Two layers close that gap:

  standing_rules  every active preference phrased as a rule ("Never ...", "Always ...", "Only ...",
                  "Before ...", "Don't ...") goes into every system prompt, ahead of anything recalled.
  check           a deterministic look at fenced code and command-shaped proposals outside fences for
                  steps the user's rules forbid, unless the user's own message names that step. The mind
                  revises once on a hit. Explanations, quoted errors and negative instructions are ignored.

The guards are deliberately few and literal: each is a step that is costly or public and hard to undo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..soup import Soup
from ..soup.pool import Fragment
from ..util import estimate_tokens

RULE_START = re.compile(r"^\s*(never|always|only|before|don'?t|do not|no\b)", re.I)
FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.S)
INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
LIST_COMMAND = re.compile(r"^\s*(?:\$|>|-)\s+(.+)$")
NUMBERED_COMMAND = re.compile(r"^\s*\d+[.)]\s+(?:run|execute|type|use)?\s*:?[ ]*(.+)$", re.I)
NEGATIVE_LINE = re.compile(r"^\s*(?:never|do not|don't|avoid|must not|should not)\b", re.I)
ERROR_LINE = re.compile(r"\b(?:error|fatal|failed|failure|permission denied|not found)\b", re.I)
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


@dataclass(frozen=True)
class Guard:
    """One deterministic detector and the standing-rule wording that activates it."""

    name: str
    rule_words: re.Pattern[str]
    why: str


GUARDS = (
    Guard("force-push", re.compile(r"\b(?:force[- ]?push|push|publish)\b", re.I),
          "force-pushes, which rewrites published history"),
    Guard("push", re.compile(r"\b(?:push|publish)\b", re.I),
          "pushes; the user's repos can be public and pushing is publication"),
    Guard("pull", re.compile(r"\bpull\b", re.I),
          "pulls into an existing checkout that may hold unique uncommitted work"),
    Guard("commit", re.compile(r"\bcommit\b", re.I),
          "commits, which the user does only when they name it"),
    Guard("add-all", re.compile(r"\b(?:git add|stage|staging|sweep)\b", re.I),
          "stages everything (`git add .`/`-A`), which sweeps up secrets and stray files"),
    Guard("reset-hard", re.compile(r"\b(?:reset|destructive git)\b", re.I),
          "hard-resets, destroying uncommitted work"),
    Guard("clean", re.compile(r"\b(?:git clean|delete untracked|destructive git)\b", re.I),
          "deletes untracked files (`git clean -f`)"),
    Guard("branch-delete", re.compile(r"\b(?:delete.*branch|branch.*delete|destructive git)\b", re.I),
          "force-deletes a branch"),
    Guard("rm-rf", re.compile(r"\b(?:rm\s+-[rf]+|delete|remove)\b", re.I),
          "recursively force-deletes files"),
    Guard("curl-pipe-shell", re.compile(r"\b(?:curl|installer|pipe.*shell)\b", re.I),
          "downloads code and pipes it directly into a shell"),
    Guard("no-verify", re.compile(r"\b(?:no-verify|bypass.*(?:hook|check)|hook.*bypass)\b", re.I),
          "bypasses repository safety hooks with `--no-verify`"),
    Guard("pickle-load", re.compile(r"\b(?:pickle|model weights?|\.pt|\.pth|\.ckpt)\b", re.I),
          "loads a pickle-format file, which can run code"),
)


def code_blocks(answer: str) -> list[str]:
    return [m.group(1) for m in FENCE.finditer(answer or "")]


def proposed_commands(answer: str) -> list[str]:
    """Command-shaped proposals outside fences, excluding explanations, errors and prohibitions."""
    outside = FENCE.sub("", answer or "")
    found: list[str] = []
    for line in outside.splitlines():
        if NEGATIVE_LINE.match(line) or ERROR_LINE.search(line):
            continue
        prefixed = LIST_COMMAND.match(line)
        numbered = NUMBERED_COMMAND.match(line)
        if prefixed and _looks_command(prefixed.group(1)):
            found.append(prefixed.group(1).strip())
        elif numbered and _looks_command(numbered.group(1)):
            found.append(numbered.group(1).strip())
        for inline in INLINE.finditer(line):
            value = inline.group(1).strip()
            before = line[:inline.start()].lower()
            if _looks_command(value) and not re.search(r"\b(?:explain|means|example|failed with|error)\b", before):
                found.append(value)
    return found


def _looks_command(text: str) -> bool:
    text = re.sub(r"^(?:run|execute|type|use)\s+", "", text.strip(), flags=re.I)
    return bool(re.match(r"(?:[\w./~-]+/)?(?:git|rm|curl|wget|python\d*|pip\d*|bash|sh|sudo)\b", text))


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


def _enabled(rules: Iterable[Fragment | str] | None) -> set[str]:
    """Detector names activated by mapped rules; None keeps the legacy all-guards API behavior."""
    if rules is None:
        return {guard.name for guard in GUARDS}
    texts = [r.text if isinstance(r, Fragment) else str(r) for r in rules]
    return {guard.name for guard in GUARDS if any(guard.rule_words.search(text) for text in texts)}


def check(answer: str, user_text: str, rules: Iterable[Fragment | str] | None = None) -> list[Violation]:
    """Proposed guarded steps that mapped standing rules forbid and the user did not request."""
    found: dict[str, Violation] = {}
    enabled = _enabled(rules)
    reasons = {guard.name: guard.why for guard in GUARDS}

    def hit(name: str, snippet: str) -> None:
        if name in enabled:
            found.setdefault(name, Violation(name, reasons[name], " ".join(snippet.split())[:80]))

    for code in [*code_blocks(answer), *proposed_commands(answer)]:
        for call in git_calls(code):
            sub, rest = call[0], call[1:]
            text = "git " + " ".join(call)
            if sub == "push" and ("--force" in rest or "-f" in rest or "--force-with-lease" in rest):
                hit("force-push", text)
            elif sub == "push" and not _named(user_text, "push", "publish"):
                hit("push", text)
            elif sub == "pull" and not _named(user_text, "pull"):
                hit("pull", text)
            elif sub == "commit" and not _named(user_text, "commit"):
                hit("commit", text)
            elif sub == "add" and rest[:1] and rest[0] in (".", "-A", "--all") and not _named(user_text, "git add"):
                hit("add-all", text)
            elif sub == "reset" and "--hard" in rest and not _named(user_text, "reset --hard"):
                hit("reset-hard", text)
            elif sub == "clean" and any(r.startswith("-") and "f" in r for r in rest) and not _named(user_text, "git clean"):
                hit("clean", text)
            elif sub == "branch" and "-D" in rest and not _named(user_text, "delete branch", "branch -D"):
                hit("branch-delete", text)
            if "--no-verify" in rest and not _named(user_text, "--no-verify", "no verify"):
                hit("no-verify", text)
        for m in re.finditer(r"\brm\s+-[a-zA-Z]*(?:rf|fr)[a-zA-Z]*\s+\S+", code):
            if not _named(user_text, "rm -rf", "delete", "remove"):
                hit("rm-rf", m.group(0))
        for m in re.finditer(r"\bcurl\b[^\n|]*\|\s*(?:ba)?sh\b", code):
            if not _named(user_text, "curl", "installer", "pipe to shell"):
                hit("curl-pipe-shell", m.group(0))
        for m in re.finditer(r"\b(torch\.load|pickle\.loads?)\s*\(([^)]*)\)?", code):
            if "weights_only=True" not in (m.group(2) or "") and not _named(user_text, "pickle", "torch.load"):
                hit("pickle-load", m.group(0))
    return list(found.values())


def revision_prompt(violations: list[Violation], rules: list[Fragment]) -> str:
    lines = "\n".join(f"- `{v.snippet}`: {v.why}" for v in violations)
    ruled = ("\nThe rules:\n" + "\n".join(f"- {f.text}" for f in rules)) if rules else ""
    return ("RULES CHECK: your draft breaks rules the user set. Its code or proposed commands do this:\n" + lines + ruled +
            "\nRewrite the WHOLE answer so no code or command takes those steps. If the task truly needs one, "
            "leave it out of the code and tell the user, in one line, which step they should run themselves "
            "and why. Do not mention this check.")


def warning(violations: list[Violation]) -> str:
    steps = ", ".join(f"`{v.snippet}`" for v in violations)
    return (f"\n\n> ⚠ kgirl's rules check: this answer still contains {steps}, which breaks a rule you set. "
            "Don't run that step unless you mean to.")
