"""Evolve: the consolidation pass (kgirl's "sleep"). Run it between sessions: `kgirl evolve`.

    1. reflect    finish any conversation with turns that were never reflected on
    2. curate     one curator step: decay old evidence, promote proposals that were observed again and
                  kept helping, retire what stopped helping, bound the active set
    3. profile    rewrite the consolidated profile of the user from active preferences, facts and recent
                  episodes: a short note recalled into every conversation ("WHO THE USER IS")
    4. skills     forge named skills from recurring verified swarm routines (skills.py)
    5. dataset    export approved exchanges as chat JSONL: the corpus for a later local fine-tune, so the
                  model's weights can follow what memory learned first

Steps 1-4 change only memory and every change is in the Soup ledger (`kgirl soup ledger`), so a bad
consolidation is undone with the curator's rollback. Step 5 only writes a file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..soup import Curator, Soup
from .conversations import ConversationStore
from .reflect import Reflector, quick_feedback
from .text import strip_reasoning

LLM = Callable[[str, str, int, float], str]

PROFILE_SYSTEM = """You maintain a short profile of one user for their personal AI assistant. From the
preferences, facts and conversation summaries given, write at most 10 bullet points: who they are, what
they are building, how they like to work and be spoken to, and what to avoid. Only use what is given;
when items conflict, the newer one wins. Plain bullets, no heading."""


@dataclass
class EvolveReport:
    reflected: list[int] = field(default_factory=list)
    promoted: list[int] = field(default_factory=list)
    retired: list[int] = field(default_factory=list)
    expired: list[int] = field(default_factory=list)
    profile_id: int | None = None
    skills: list[str] = field(default_factory=list)
    dataset: tuple[str, int] | None = None

    def render(self, soup: Soup) -> str:
        lines = [f"reflected on conversations: {self.reflected or 'none pending'}"]
        for label, ids in (("promoted to habit", self.promoted), ("retired", self.retired),
                           ("expired proposals", self.expired)):
            if ids:
                lines.append(f"{label}:")
                for i in ids[:12]:
                    f = soup.get(i)
                    if f:
                        lines.append(f"  #{i} {f.kind}: {f.text[:140]}")
        if self.profile_id:
            lines.append(f"profile rewritten as #{self.profile_id}")
        if self.skills:
            lines.append("skills: " + ", ".join(self.skills))
        if self.dataset:
            lines.append(f"wrote {self.dataset[1]} approved exchanges to {self.dataset[0]}")
        return "\n".join(lines)


def evolve(llm: LLM, soup: Soup, curator: Curator, store: ConversationStore, dataset: str | Path | None = None,
           forge_skills: bool = True) -> EvolveReport:
    rep = EvolveReport()
    reflector = Reflector(llm, store, soup, curator)
    for conv in store.list(limit=50):
        if store.turns(conv.id, since_id=conv.reflected_turn, roles=("user",)):
            if reflector.reflect(conv.id, force=True):
                rep.reflected.append(conv.id)
    cur = curator.step()
    rep.promoted, rep.retired, rep.expired = cur.promoted, cur.retired, cur.expired
    rep.profile_id = rewrite_profile(llm, soup)
    if forge_skills:
        try:
            from ..skills import forge
            rep.skills = [s.name for s in forge(soup)]
        except Exception:  # noqa: BLE001 - skills are optional; never fail the consolidation over them
            rep.skills = []
    if dataset:
        rep.dataset = (str(dataset), export_chat_sft(store, dataset))
    return rep


def rewrite_profile(llm: LLM, soup: Soup) -> int | None:
    prefs = soup.all(status="active", kind="preference")
    facts = [f for f in soup.all(status="active", kind="fact") if f.source != "atlas"]
    episodes = sorted(soup.all(status="active", kind="episode"), key=lambda f: f.created or 0)[-8:]
    if not (prefs or facts or episodes):
        return None
    body = "\n".join([*(f"PREFERENCE (#{f.id}): {f.text}" for f in prefs[-30:]),
                      *(f"FACT (#{f.id}): {f.text}" for f in facts[-30:]),
                      *(f"EPISODE: {f.text[:400]}" for f in episodes)])
    try:
        text = strip_reasoning(llm(PROFILE_SYSTEM, body, 700, 0.2)).strip()
    except Exception:  # noqa: BLE001
        return None
    if not text:
        return None
    for old in soup.db.execute("SELECT id FROM fragments WHERE kind='note' AND tags='profile' AND status='active'"):
        soup.set_status(old["id"], "retired", "superseded profile")
    fid = soup.add(text, kind="note", tags="profile", source="evolve", status="active")
    soup.set_status(fid, "active", "profile")
    return fid


def export_chat_sft(store: ConversationStore, out: str | Path) -> int:
    """Approved exchanges as {"messages": [...]} JSONL: an answer counts when the user's next turn approved it."""
    n = 0
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for conv in store.list(limit=100000):
            turns = store.turns(conv.id, roles=("user", "assistant"))
            for i in range(len(turns) - 2):
                u, a, nxt = turns[i], turns[i + 1], turns[i + 2]
                if u.role == "user" and a.role == "assistant" and nxt.role == "user" \
                        and quick_feedback(nxt.content) == "positive":
                    fh.write(json.dumps({"messages": [{"role": "user", "content": u.content},
                                                      {"role": "assistant", "content": a.content}],
                                         "conv": conv.id}, ensure_ascii=False) + "\n")
                    n += 1
    return n
