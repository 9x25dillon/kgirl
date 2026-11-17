"""Helix: the runner. Code routes, counts and keeps state; models write blocks.

One run:

    BYPASS?    the predicate over the budget and the scaffold floor; yes -> one direct call, nothing else
    experts    one block each (one expert by default, two with `lattice`), each closed by a TUNE trailer
               that re-steers only if it quotes its own block
    ORACLE     only when two probes in a row name the same weakness: a calibration with method and falsifier
    CRITIC     on a different model family when one is configured; defects must quote the draft
    SYNTHESIST the answer, plus which conflicts it could not resolve

Every call is charged to the ledger from the provider's own token counts, every state change is a new
version in the store with its diff, and the result's `open_conflicts` is written by code: a synthesis that
does not account for a conflict leaves it open.
"""

from __future__ import annotations

import os
import re

from ..hermes.router import CHAT, Intent, classify, mind_ctx
from ..llm import Message, OllamaBackend
from . import ledger as L
from .prompt import (BYPASS_SYSTEM, GENERATORS, STATIC, assemble, compact_turn, critic_turn, expert_turn, header,
                     oracle_turn, synth_turn)
from .protocol import MAX_PASSES, parse_calibration, parse_review, parse_tunes, parse_unresolved, split
from .state import CONFLICT_CHARS, DEFAULT_VECTOR, MAX_CONFLICTS, capsule
from .store import E_CALL, E_RUN, HelixError, StateStore

CRITIC_WEIGHT = 0.25         # the critic's share of the content pool; the expert blocks split the rest
MIN_CEILING = 2048           # a hard stop far above any target: targets are stated, replies are not cut to them
ROUTES = {Intent.TASK: ("ENGINEER", "LOGICIAN"), Intent.ARCH: ("ENGINEER", "ARCHIVIST"),
          Intent.BLAST: ("LOGICIAN", "ENGINEER"), Intent.RECALL: ("ARCHIVIST", "ORACLE"),
          Intent.ASK: ("ARCHIVIST", "LOGICIAN")}
SAME_FAMILY = "the critic ran on the generator's model family, so its agreement is weak evidence"
UNREVIEWED = "the critic returned no quoted defect and no new evidence, so the draft is unreviewed"


def family(model: str) -> str:
    """`qwen2.5-coder:7b` -> qwen, `someone/Llama3.2:3b` -> llama, `claude-haiku-4-5` -> claude."""
    m = re.match(r"[a-z]+", model.rsplit("/", 1)[-1].lower())
    return m.group(0) if m else model.lower()


def route(task: str, k: int = 1) -> tuple[str, ...]:
    """Which experts write. Decided by the router's intent rules, so routing costs no tokens."""
    return ROUTES[classify(task)][:max(1, k)]


def critic_from_env():
    """The backend named by KGIRL_HELIX_CRITIC_MODEL (an Ollama model), or None."""
    model = os.environ.get("KGIRL_HELIX_CRITIC_MODEL", "").strip()
    return OllamaBackend(model, timeout=600.0, num_ctx=mind_ctx()) if model else None


def _totals(led: L.Ledger) -> dict:
    """The ledger as STATE carries it. `spent_ext` stays null unless every call so far was provider-counted."""
    counted = all(e.ext is not None for e in led.entries)
    return {"alloc": led.B, "spent_est": sum(e.est for e in led.entries),
            "spent_ext": sum(e.ext for e in led.entries) if counted and led.entries else None,
            "reserve_floor": led.synth_floor}


def _conflict(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= CONFLICT_CHARS else text[:CONFLICT_CHARS - 3] + "..."


class Helix:
    def __init__(self, router, *, home=None, store: StateStore | None = None, bus=None, session: str = "main",
                 role: str = CHAT, critic=None, log_limit: int | None = None):
        """`critic` is a Backend for the review stage; None reads KGIRL_HELIX_CRITIC_MODEL, and with neither
        the generator reviews itself and the result says so. `log_limit` is the flight-log size, in estimated
        tokens, past which it is compacted (default: half the role's context window)."""
        self.router, self.bus, self.session, self.role = router, bus, session, role
        self.store = store or StateStore(home)
        self.critic = critic if critic is not None else critic_from_env()
        self.log_limit = log_limit or (router.context_tokens(role) or 8192) // 2

    # ------------------------------------------------------------------------------------------- plumbing

    def _pub(self, topic: str, payload: dict) -> None:
        if self.bus is not None:
            self.bus.publish(topic, payload, sender="helix")

    def _complete(self, system: str, user: str, ceiling: int, temperature: float, backend=None):
        msgs = [Message("user", user)]
        if backend is not None:
            return backend.complete(system, msgs, max_tokens=ceiling, temperature=temperature)
        return self.router.complete(self.role, system, msgs, max_tokens=ceiling, temperature=temperature)

    def _charge(self, led: L.Ledger, node: str, kind: str, text: str, out, target: int, trailer_chars: int = 0):
        entry = led.charge(node, kind, text, out, target, trailer_chars)
        payload = {"node": node, "kind": kind, "alloc": entry.alloc, "target": entry.target, "est": entry.est,
                   "ext": entry.ext, "in_ext": entry.in_ext, "steer_ext": entry.steer_ext, "model": entry.model,
                   "backend": entry.backend, "stop": entry.stop}
        self.store.note(self.session, E_CALL, payload)
        self._pub("helix.call", payload)
        return entry

    # ------------------------------------------------------------------------------------------- the run

    def run(self, task: str, B: int = 4000, *, experts: list[str] | None = None, lattice: bool = False,
            expect: int | None = None, seed: str | None = None) -> dict:
        """Answer `task` within a budget of B generated tokens. `expect` is the caller's guess at the size of
        the answer, used only by the BYPASS predicate. `seed` overrides the task capsule."""
        task = task.strip()
        if not task:
            raise HelixError("there is no task")
        B = int(B)
        if B <= 0:
            raise HelixError("the budget B must be a positive number of tokens")
        ceiling = max(B, MIN_CEILING)
        head = self.store.head(self.session)
        temperature = (head.vector if head else DEFAULT_VECTOR)["temp"]
        floor = self.store.floor()

        reason = L.bypass(B, expect, floor["tokens"])
        if reason:
            led = L.Ledger(B)
            out = self._complete(BYPASS_SYSTEM, task, ceiling, temperature)
            self._charge(led, "DIRECT", L.BYPASS, out.text, out, B)
            return self._close(led, task, out.text.strip(), bypass=reason, floor=floor)

        names = tuple(n.upper() for n in experts) if experts else route(task, 2 if lattice else 1)
        unknown = [n for n in names if n not in GENERATORS]
        if unknown or not 1 <= len(names) <= 3 or len(set(names)) != len(names):
            raise HelixError(f"experts must be one to three different names from {', '.join(GENERATORS)}")
        led = L.Ledger(B, {**{n: (1 - CRITIC_WEIGHT) / len(names) for n in names}, "CRITIC": CRITIC_WEIGHT})

        task_seed = capsule(seed or task)
        resumed = head is not None and head.task_seed == task_seed
        state = self.store.commit(
            self.session, "task", task_seed=task_seed, last_delta=head.last_delta if resumed else None,
            ledger=_totals(led), open_conflicts=head.open_conflicts if resumed else ())
        first = state.v
        log: list[str] = []
        drafts: list[str] = []                            # the expert blocks alone: what a defect may quote
        tunes: list[dict] = []
        conflicts = list(state.open_conflicts)
        probes: list[tuple[str, str, str]] = []           # (axis, direction, quote) of each accepted TUNE
        generator = ""

        for name in names:
            target = led.target(name)
            out = self._complete(STATIC, assemble(task, log, state, expert_turn(name, target)), ceiling,
                                 state.vector["temp"])
            generator = generator or out.model
            body, trailers, chars = split(out.text)
            self._charge(led, name, L.EXPERT, out.text, out, target, chars)
            log.append(header(name, state, out) + "\n" + body)
            drafts.append(body)
            if out.stop_reason == "length":
                conflicts.append(_conflict(f"{name}'s block was cut off at the {ceiling}-token ceiling"))
            passes = 0
            for t in parse_tunes(trailers, body):
                record = {"expert": name, "axis": t.axis, "d": t.d, "quote": t.quote, "accepted": False}
                if not t.ok:
                    record["reason"] = t.reason
                elif passes >= MAX_PASSES:
                    record["reason"] = f"more than {MAX_PASSES} tuning passes on one block"
                elif led.steer_left() <= 0:
                    record["reason"] = "the steering pool is spent"
                else:
                    passes += 1
                    old = state.vector[t.axis]
                    new = round(min(1.0, max(0.0, old + t.d)), 3)
                    if new == old:
                        record["reason"] = f"{t.axis} is already at {old}"
                    else:
                        state = self.store.commit(
                            self.session, "tune", vector={**state.vector, t.axis: new}, ledger=_totals(led),
                            last_delta={"axis": t.axis, "d": round(new - old, 3), "evidence": t.quote})
                        record.update(accepted=True, old=old, new=new, v=state.v, because=t.because)
                        log.append(f'W {t.axis} {old:.2f}->{new:.2f} | quote: "{t.quote}" | because: {t.because}')
                    probes.append((t.axis, "+" if t.d > 0 else "-", t.quote))
                    if len(probes) >= 2 and probes[-1][:2] == probes[-2][:2] and led.steer_left() > 0:
                        self._calibrate(led, task, log, state, probes[-2:], conflicts, ceiling)
                if not record["accepted"]:
                    log.append(f"TUNE rejected: {record['reason']}")
                tunes.append(record)
                self._pub("helix.tune", record)

        log = self._compact(led, task, log, state, ceiling)

        # the critic: another model family when one is configured
        target = led.target("CRITIC")
        out = self._complete(STATIC, assemble(task, log, state, critic_turn(target, len(conflicts))), ceiling,
                             state.vector["temp"], backend=self.critic)
        critic_model = out.model
        decorrelated = family(critic_model) != family(generator)
        _, trailers, _ = split(out.text)
        self._charge(led, "CRITIC", L.CRITIC, out.text, out, target)
        review = parse_review(trailers, "\n".join(drafts), "\n".join(log))
        lines = [header("CRITIC", state, out)]             # only what passed its check enters the log
        for d in review.defects:
            if d.ok:
                conflicts.append(_conflict(f'"{d.quote}" -> {d.repair}'))
                lines.append(f'DEFECT: "{d.quote}" -> {d.repair}')
            else:
                lines.append(f"DEFECT void: {d.reason}")
        if not review.valid:
            lines.append(f"CONCUR: {review.concur}" if review.concur else
                         "REVIEW void" + (f": {review.reason}" if review.reason else ""))
        log.append("\n".join(lines))
        conflicts = conflicts[:MAX_CONFLICTS]
        state = self.store.commit(self.session, "critic", open_conflicts=tuple(conflicts), ledger=_totals(led))

        # the synthesis: the only part a person needs to read
        target = led.synth_target()
        out = self._complete(STATIC, assemble(task, log, state, synth_turn(target, state.open_conflicts)), ceiling,
                             state.vector["temp"])
        answer, trailers, chars = split(out.text)
        self._charge(led, "SYNTHESIST", L.SYNTHESIS, out.text, out, target, chars)
        left = parse_unresolved(trailers, len(conflicts))
        stated = left is not None
        open_now = [conflicts[i - 1] for i in left] if stated else list(conflicts)
        if out.stop_reason == "length":
            open_now.append(_conflict(f"the synthesis was cut off at the {ceiling}-token ceiling"))
        if review.void:
            open_now.append(UNREVIEWED)
        if not decorrelated:
            open_now.append(SAME_FAMILY)

        return self._close(led, task, answer, floor=floor, experts=names, first=first, tunes=tunes, log=log,
                           open_conflicts=open_now[:MAX_CONFLICTS], raised=len(conflicts), stated=stated,
                           models={"generator": generator, "critic": critic_model, "decorrelated": decorrelated})

    # ------------------------------------------------------------------------------------------- stages

    def _calibrate(self, led: L.Ledger, task: str, log: list[str], state, probes: list, conflicts: list,
                   ceiling: int) -> None:
        """Two probes in a row named the same weakness: ask for a calibration, and take it only whole."""
        axis, direction, _ = probes[-1]
        turn = oracle_turn(axis, direction, [q for _, _, q in probes])
        out = self._complete(STATIC, assemble(task, log, state, turn), ceiling, state.vector["temp"])
        self._charge(led, "ORACLE", L.STEER, out.text, out, max(1, led.steer_left()))
        cal = parse_calibration(split(out.text)[1])
        if cal.ok:
            low = f" (below {cal.drops_below:.2f} if so)" if cal.drops_below is not None else ""
            log.append(f"CAL p={cal.p:.2f} | method: {cal.method} | falsifier: {cal.falsifier}{low}")
        else:
            log.append(f"CAL void: {cal.reason}")
            conflicts.append(_conflict(f"uncalibrated: {axis} was probed twice and the calibration was void "
                                       f"({cal.reason})"))
        self._pub("helix.calibrate", {"axis": axis, "ok": cal.ok, "p": cal.p if cal.ok else None, "reason": cal.reason})

    def _compact(self, led: L.Ledger, task: str, log: list[str], state, ceiling: int) -> list[str]:
        """Rewrite a flight log that outgrew `log_limit`. The rewrite is spend, charged where it happens."""
        size = L.estimate("\n\n".join(log))
        if size <= self.log_limit:
            return log
        target = max(1, self.log_limit // 2)
        out = self._complete(STATIC, assemble(task, log, state, compact_turn(target)), ceiling, state.vector["temp"])
        self._charge(led, "LOG", L.COMPACTION, out.text, out, target)
        self._pub("helix.compact", {"from_est": size, "to_est": L.estimate(out.text)})
        return [f"--- LOG compacted at v{state.v} ({size} -> {L.estimate(out.text)} tokens EST) ---\n"
                + split(out.text)[0]]

    def _close(self, led: L.Ledger, task: str, answer: str, *, floor: dict, bypass: str | None = None,
               experts: tuple = (), first: int | None = None, tunes: list | None = None, log: list | None = None,
               open_conflicts: list | None = None, raised: int = 0, stated: bool = True,
               models: dict | None = None) -> dict:
        report = led.report()
        open_conflicts = list(dict.fromkeys(open_conflicts or ()))        # a resumed task may repeat one
        state = self.store.head(self.session)
        if not bypass:
            state = self.store.commit(
                self.session, "close", open_conflicts=tuple(open_conflicts or ()), ledger=_totals(led))
        summary = {"bypass": bypass, "B": led.B, "experts": list(experts), "spent_est": report["spent_est"],
                   "spent_ext": report["spent_ext"], "input_ext": report["input_ext"],
                   "scaffold_ext": report["scaffold_ext"], "overrun": report["overrun"],
                   "versions": [first, state.v] if not bypass else None,
                   "tunes_accepted": sum(1 for t in tunes or [] if t["accepted"]),
                   "tunes_rejected": sum(1 for t in tunes or [] if not t["accepted"]),
                   "open_conflicts": len(open_conflicts or ()), "models": models or {}}
        self.store.note(self.session, E_RUN, summary)
        self._pub("helix.run", summary)
        return {"session": self.session, "bypass": bypass, "answer": answer, "experts": list(experts),
                "models": models or {}, "floor": floor, "state": state.to_dict() if state else None,
                "versions": summary["versions"], "tunes": tunes or [], "conflicts_raised": raised,
                "unresolved_stated": stated, "open_conflicts": list(open_conflicts or ()), "ledger": report,
                "log": log or []}

    # ------------------------------------------------------------------------------------------- handoff

    def handoff(self) -> str:
        """What a handoff carries: the STATE and nothing else."""
        state = self.store.head(self.session)
        if state is None:
            raise HelixError(f"session {self.session!r} has no state yet")
        return state.to_json()

    def ingest(self, obj, source: str = ""):
        return self.store.ingest(self.session, obj, source)
