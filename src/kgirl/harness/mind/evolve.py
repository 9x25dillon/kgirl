"""Evolve: the consolidation pass (kgirl's "sleep"). Run it between sessions: `kgirl evolve`.

    1. reflect    finish any conversation with turns that were never reflected on
    2. curate     one curator step: decay old evidence, promote proposals that were observed again and
                  kept helping, retire what stopped helping, bound the active set
    3. profile    rewrite the consolidated profile of the user from active preferences, facts and recent
                  episodes: a short note recalled into every conversation ("WHO THE USER IS")
    4. skills     forge named skills from recurring verified swarm routines (skills.py)
    5. dataset    export approved exchanges as chat JSONL: the corpus for a later local fine-tune, so the
                  model's weights can follow what memory learned first

The report also lists what kgirl is still only considering: proposals from conversations that are staged,
waiting to be seen again (or for `kgirl soup promote ID`).

Steps 1-4 change only memory and every change is in the Soup ledger (`kgirl soup ledger`), so a bad
consolidation is undone with the curator's rollback. Step 5 only writes a file.

`kgirl evolve --dry-run` (`preview`) reports what a pass would do and writes nothing: no model call, no
ledger entry, no file. It reads the same rules the curator applies, except that it cannot show the decay the
real pass applies first, so a borderline proposal may differ by a hair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..soup import Curator, Soup
from ..util import now
from .conversations import ConversationStore, Exchange
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
    considering: list[int] = field(default_factory=list)
    profile_id: int | None = None
    skills: list[str] = field(default_factory=list)
    dataset: tuple[str, int] | None = None
    dry_run: bool = False
    profile_from: tuple[int, int, int] | None = None      # dry run: (preferences, facts, episodes) it would be written from

    def render(self, soup: Soup) -> str:
        if self.dry_run:
            return self._render_dry_run(soup)
        lines = [f"reflected on conversations: {self.reflected or 'none pending'}"]
        for label, ids in (("promoted to habit", self.promoted), ("retired", self.retired),
                           ("expired proposals", self.expired),
                           ("still considering (seen once; promoted when seen again, or `kgirl soup promote ID`)",
                            self.considering)):
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

    def _render_dry_run(self, soup: Soup) -> str:
        lines = ["dry run: nothing was changed, no model was called",
                 f"would reflect on conversations: {self.reflected or 'none pending'}"]
        for label, ids in (("would be promoted to habit", self.promoted), ("would be retired", self.retired),
                           ("would expire (proposals nobody repeated)", self.expired),
                           ("would still be considering (seen once)", self.considering)):
            if ids:
                lines.append(f"{label}:")
                for i in ids[:12]:
                    f = soup.get(i)
                    if f:
                        lines.append(f"  #{i} {f.kind} (support {f.support}, utility {f.utility:.2f}): {f.text[:120]}")
        if self.profile_from:
            n_pref, n_fact, n_ep = self.profile_from
            lines.append(f"would rewrite the profile from {n_pref} preferences, {n_fact} facts and {n_ep} episodes")
        else:
            lines.append("would leave the profile alone (nothing to write it from)")
        lines.append("skills: not previewed")
        if self.dataset:
            lines.append(f"would write {self.dataset[1]} approved exchanges to {self.dataset[0]}")
        if not (self.reflected or self.promoted or self.retired or self.expired):
            lines.append("nothing to reflect on, promote or retire")
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
    rep.considering = [r["id"] for r in soup.db.execute(
        "SELECT id FROM fragments WHERE status='staged' AND source LIKE 'reflection:%' ORDER BY support DESC, id DESC"
        " LIMIT 12")]
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


def preview(soup: Soup, curator: Curator, store: ConversationStore, dataset: str | Path | None = None) -> EvolveReport:
    """What `evolve` would do, as a report, with no writes: no reflection, no curator step, no profile, no file.

    The curator's rules are applied to the fragments as they stand now, without the decay the real step applies
    first (decay moves evidence toward its prior, so a borderline fragment can come out differently)."""
    rep = EvolveReport(dry_run=True)
    rep.reflected = [c.id for c in store.list(limit=50) if store.turns(c.id, since_id=c.reflected_turn, roles=("user",))]
    p, t = curator.policy, now()
    for f in soup.all(status="staged"):
        auto = f.source not in p.manual_sources
        if auto and f.support >= p.min_support and f.bound(-1.0) >= p.promote_lcb \
                and (curator.validator is None or curator.validator(f)):
            rep.promoted.append(f.id)
        elif t - (f.created or t) > p.staged_ttl_days * 86400 and (not auto or f.support < p.min_support):
            rep.expired.append(f.id)
    active = soup.all(status="active") + [soup.get(i) for i in rep.promoted]       # promotion comes first
    rep.retired = [f.id for f in active if f.evidence >= p.min_evidence_retire and f.bound(1.0) < p.retire_ucb]
    active = [f for f in active if f.id not in rep.retired]
    if len(active) > p.capacity:
        half = soup.half_life or 1.0

        def keep_score(f) -> float:
            age = t - (f.last_used or f.created or t)
            return f.bound(-1.0) * (0.5 ** (age / half)) + (1.0 if f.kind == "preference" else 0.0)
        rep.retired += [f.id for f in sorted(active, key=keep_score)[: len(active) - p.capacity]]
    rep.considering = [r["id"] for r in soup.db.execute(
        "SELECT id FROM fragments WHERE status='staged' AND source LIKE 'reflection:%' ORDER BY support DESC, id DESC"
        " LIMIT 12") if r["id"] not in rep.promoted and r["id"] not in rep.expired]
    prefs = min(30, len(soup.all(status="active", kind="preference")))         # what rewrite_profile reads
    facts = min(30, len([f for f in soup.all(status="active", kind="fact") if f.source != "atlas"]))
    episodes = min(8, len(soup.all(status="active", kind="episode")))
    rep.profile_from = (prefs, facts, episodes) if prefs or facts or episodes else None
    if dataset:
        rep.dataset = (str(dataset), sum(1 for _ in iter_chat_sft(store)))
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
    for old in soup.db.execute("SELECT id FROM fragments WHERE kind='note' AND tags='profile' AND status='active'"
                               ).fetchall():
        soup.set_status(old["id"], "retired", "superseded profile")
    fid = soup.add(text, kind="note", tags="profile", source="evolve", status="active")
    soup.set_status(fid, "active", "profile")
    return fid


def verdict_on(done: list[Exchange], i: int) -> str:
    """The user's verdict on exchange i: the one recorded on its answer, else read from their next message."""
    answer = done[i].answer
    if answer is None:
        return "none"
    if answer.verdict:
        return answer.verdict
    return quick_feedback(done[i + 1].user.content) if i + 1 < len(done) else "none"


def iter_chat_sft(store: ConversationStore, context_exchanges: int = 2):
    """Approved exchanges as the rows `export_chat_sft` writes: {"messages", "conv", "turn"} and, when it was stored,
    "system". One row per answer the user approved."""
    for conv in store.list(limit=100000):
        done = store.exchanges(conv.id)
        for i, ex in enumerate(done):
            if ex.answer is None or verdict_on(done, i) != "positive":
                continue
            messages: list[dict] = []
            for prev in done[max(0, i - context_exchanges):i]:
                if prev.answer is not None:
                    messages += [{"role": "user", "content": prev.user.content},
                                 {"role": "assistant", "content": prev.answer.content}]
            messages.append({"role": "user", "content": ex.user.content})
            for step in ex.steps:
                messages.append({"role": "assistant", "content": step.content} if step.role == "assistant"
                                else {"role": "user", "content": f"TOOL RESULT {step.content}"})
            messages.append({"role": "assistant", "content": ex.answer.content})
            row = {"messages": messages, "conv": conv.id, "turn": ex.answer.id}
            system = store.context(ex.answer.id)
            if system:
                row["system"] = system
            yield row


def export_chat_sft(store: ConversationStore, out: str | Path, context_exchanges: int = 2) -> int:
    """Approved exchanges as JSONL, in the form the chat model is used in.

    One row per answer the user approved: {"system", "messages", "conv", "turn"}. `messages` holds up to
    `context_exchanges` earlier question/answer pairs, then this exchange with its tool calls (assistant
    turns ending in `<tool>...</tool>`) and their results (user turns starting with TOOL RESULT), then the
    answer. `system` is the prompt the answer was generated from, when it was stored. Tool results are
    written whole; at answer time the model saw the older ones shortened.
    """
    n = 0
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in iter_chat_sft(store, context_exchanges):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
