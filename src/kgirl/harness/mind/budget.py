"""The prompt budget: one token budget for everything the model is shown in a turn.

A turn's prompt is built from blocks. Some are fixed and never cut: the persona, the user's rules, the
tool list, the workspace, the date, the message being answered and the room reserved for tool results.
The rest (WHO, WANTS, MEMORY, EARLIER, CODE, SO FAR and the older turns of the conversation) share what
is left. When they do not fit, the lowest priority block loses its last items first:

    EARLIER < MEMORY < CODE < older turns < WHO < WANTS < SO FAR        (cut first ... cut last)

A block's items are ordered best first, so trimming drops from the end: whole items while it can, and
only the last item that remains is clipped (with an ellipsis), and only if that keeps at least MIN_CLIP
tokens of it. The rule is deterministic: the same blocks and budget always give the same prompt. What was
cut is recorded (`Fit.cut`, and from there the turn's meta["budget"]) and `/context` shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..util import estimate_tokens

MIN_CLIP = 40          # a clipped last item keeps at least this many tokens, else it is dropped whole

# Priorities of the blocks that can be cut: higher survives longer.
PRIORITY = {"earlier": 10, "memory": 25, "code": 35, "history": 40, "who": 50, "wants": 60, "so_far": 70}
# The order blocks appear in /context, which is also the order of the prompt.
ORDER = ("persona", "rules", "tools", "workspace", "date", "who", "wants", "memory", "earlier", "code", "so_far",
         "now", "history", "tool_room")
LABELS = {"persona": "PERSONA", "rules": "RULES", "tools": "TOOLS", "workspace": "WORKSPACE", "date": "DATE",
          "who": "WHO", "wants": "WANTS", "memory": "MEMORY", "earlier": "EARLIER", "code": "CODE",
          "so_far": "SO FAR", "now": "THIS MESSAGE", "history": "OLDER TURNS", "tool_room": "TOOL ROOM"}


@dataclass
class Block:
    name: str
    items: list[str]                  # best first; trimming drops from the end
    header: str = ""                  # printed above the items, kept only while one item is
    priority: int = 0
    fixed: bool = False               # never cut
    clip: bool = True                 # the last item may be clipped instead of dropped
    reserve: int = 0                  # tokens held back for something that is not text here (tool results)

    def render(self, items: list[str] | None = None) -> str:
        items = self.items if items is None else items
        if not items:
            return ""
        return "\n".join(([self.header] if self.header else []) + items)

    def tokens(self, items: list[str] | None = None) -> int:
        text = self.render(items)
        return (estimate_tokens(text) if text else 0) + self.reserve


@dataclass
class Fit:
    budget: int
    blocks: dict[str, Block]
    kept: dict[str, list[str]]
    cut: dict[str, dict] = field(default_factory=dict)      # name -> {"before", "after", "dropped", "clipped"}
    overflow: int = 0                                       # tokens the fixed blocks alone exceed the budget by

    def text(self, name: str) -> str:
        return self.blocks[name].render(self.kept[name]) if name in self.blocks else ""

    def tokens(self, name: str) -> int:
        return self.blocks[name].tokens(self.kept[name]) if name in self.blocks else 0

    @property
    def used(self) -> int:
        return sum(self.tokens(n) for n in self.blocks)

    def meta(self) -> dict:
        """The record kept in a turn's meta["budget"] (keys are stable: other surfaces read them)."""
        return {"limit": self.budget, "used": self.used, "overflow": self.overflow,
                "blocks": {n: self.tokens(n) for n in ORDER if n in self.blocks},
                "dropped": {n: {"items": c["dropped"], "tokens": c["before"] - c["after"], "clipped": c["clipped"]}
                            for n, c in self.cut.items()}}


def fit(blocks: list[Block], budget: int) -> Fit:
    """Cut the cuttable blocks, lowest priority first, until everything fits `budget` tokens."""
    kept = {b.name: list(b.items) for b in blocks}
    by_name = {b.name: b for b in blocks}
    cut: dict[str, dict] = {}

    def total() -> int:
        return sum(b.tokens(kept[b.name]) for b in blocks)

    order = sorted((b for b in blocks if not b.fixed), key=lambda b: (b.priority, -blocks.index(b)))
    for b in order:
        if total() <= budget:
            break
        items = kept[b.name]
        before, n0, clipped = b.tokens(items), len(items), False
        while items and total() > budget:
            over = total() - budget
            last = items[-1]
            room = len(last) - over * 4 - 4          # characters of the last item that can stay
            if b.clip and room >= MIN_CLIP * 4:
                items[-1] = last[:room].rstrip() + "…"
                clipped = True
            else:
                items.pop()
        cut[b.name] = {"before": before, "after": b.tokens(items), "dropped": n0 - len(items), "clipped": clipped}
    return Fit(budget, by_name, kept, cut, max(0, total() - budget))


def render_report(meta: dict, context: int = 0, answer: int = 0) -> str:
    """The `/context` view of one turn's budget (the dict that `Fit.meta()` makes)."""
    if not meta:
        return "no prompt has been built in this conversation yet"
    blocks, dropped = meta.get("blocks", {}), meta.get("dropped", {})
    head = f"prompt budget {meta.get('limit', 0)} tokens, used {meta.get('used', 0)}"
    if context:
        head += f" (context {context}, {answer} kept for the answer)"
    lines = [head]
    for name in ORDER:
        if name not in blocks:
            continue
        note = "never cut" if name in ("persona", "rules", "tools", "workspace", "date", "now", "tool_room") else ""
        d = dropped.get(name)
        if d:
            note = f"cut {d['tokens']} tokens" + (f", {d['items']} item(s) dropped" if d["items"] else "") \
                + (", last one clipped" if d["clipped"] else "")
        lines.append(f"  {LABELS[name]:<13}{blocks[name]:>6}  {note}".rstrip())
    if meta.get("overflow"):
        lines.append(f"  the blocks that are never cut exceed the budget by {meta['overflow']} tokens")
    if not dropped and not meta.get("overflow"):
        lines.append("  nothing was cut")
    return "\n".join(lines)
