"""StateStore: every version of every session's STATE, hash-chained, in `$KGIRL_HOME/helix.db`.

The store is a `vessel.matrixdb.MatrixDB` used for its ledger. One entity per session (`session:<name>`);
its chain holds, in order:

    helix.state    a new version: the full state, the diff that justified it, and where it came from
    helix.call     one model call as the ledger measured it
    helix.run      the closing report of one run
    helix.reject   a handoff that failed validation, with the violations

A write bumps `v` and needs a non-empty diff: writing what is already the head writes nothing. Payloads
carry no timestamps, so the same run replayed gives the same chain heads.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from ...vessel.matrixdb import MatrixDB
from ..util import harness_home
from .state import AXES, MAX_DELTA, State, Verdict, default_state, diff, invariants_hash, validate

DB_NAME = "helix.db"
E_STATE, E_CALL, E_RUN, E_REJECT = "helix.state", "helix.call", "helix.run", "helix.reject"
FLOOR_EST = 500          # fixed scaffold of one harness run, in generated tokens, before any run is measured
FLOOR_RUNS = 20          # how many recent measured runs the floor is the median of
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class HelixError(RuntimeError):
    """The harness cannot do what was asked; the message says what to do about it."""


class StateStore:
    def __init__(self, home: str | os.PathLike | None = None):
        self.home = Path(home) if home else harness_home()
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = MatrixDB(self.home / DB_NAME)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------------------------------- sessions

    @staticmethod
    def entity(session: str) -> str:
        if not _NAME.match(session):
            raise HelixError(f"session name {session!r} must be 1-64 characters of A-Z a-z 0-9 _ . -")
        return f"session:{session}"

    def _ensure(self, session: str) -> str:
        name = self.entity(session)
        if not self.db.has_entity(name):
            self.db.create_entity(name, {"layer": "helix"})
        return name

    def sessions(self) -> list[str]:
        return [e.split(":", 1)[1] for e in self.db.entities() if e.startswith("session:")]

    # ------------------------------------------------------------------------------------------- states

    def _states(self, session: str) -> list:
        name = self.entity(session)
        return self.db.events(name, E_STATE) if self.db.has_entity(name) else []

    def head(self, session: str) -> State | None:
        """The latest stored version. The chain is this store's own writing, so its invariants hash is not
        held against the current build; a state that no longer fits the schema counts as no head."""
        events = self._states(session)
        if not events:
            return None
        raw = events[-1].payload.get("state", {})
        verdict = validate(raw, expect_hash=raw.get("invariants_hash") if isinstance(raw, dict) else None)
        return verdict.state if verdict.ok else None

    def get(self, session: str, v: int) -> State | None:
        for e in self._states(session):
            raw = e.payload.get("state", {})
            if raw.get("v") == v:
                verdict = validate(raw, expect_hash=raw.get("invariants_hash"))
                return verdict.state if verdict.ok else None
        return None

    def commit(self, session: str, origin: str, **changes) -> State:
        """Write the head with `changes` applied as the next version. Returns the head unchanged, and writes
        nothing, when the changes amount to no diff."""
        name = self._ensure(session)
        head = self.head(session)
        base = head or default_state()
        last_v = max((e.payload.get("state", {}).get("v", 0) for e in self._states(session)), default=0)
        verdict = validate(replace(base, **changes, v=last_v + 1, invariants_hash=invariants_hash()).to_dict())
        if not verdict.ok:
            raise HelixError("refusing to write an invalid state: " + "; ".join(verdict.violations))
        delta = diff(head, verdict.state)
        if head is not None and not delta:
            return head
        self.db.append(name, E_STATE, {"state": verdict.state.to_dict(), "diff": delta, "origin": origin})
        return verdict.state

    def ingest(self, session: str, obj, source: str = "") -> Verdict:
        """Take a state that crossed a trust boundary (a handoff file, a model's echo).

        It is validated against this build's invariants and rejected fail-closed: on any violation nothing is
        adopted, the rejection is recorded, and the verdict carries the stored head (or the defaults). An
        accepted state becomes the next version with its diff against the head. Two fields are never taken
        on trust: `spent_ext`, which only this harness's own measurement may set, and an axis jump larger
        than MAX_DELTA from the stored head, which is clamped like a tuning pass."""
        name = self._ensure(session)
        head = self.head(session)
        verdict = validate(obj)
        violations = list(verdict.violations)
        if verdict.ok and head is not None and verdict.state.v < head.v:
            violations.append(f"stale: v {verdict.state.v} is behind the stored v {head.v}")
        if violations:
            self.db.append(name, E_REJECT, {"violations": violations, "source": source})
            return Verdict(head or default_state(), False, tuple(violations), verdict.clamped)
        s = verdict.state
        clamped = list(verdict.clamped)
        vector = dict(s.vector)
        if head is not None:
            for axis in AXES:
                lo, hi = head.vector[axis] - MAX_DELTA, head.vector[axis] + MAX_DELTA
                held = round(min(hi, max(lo, vector[axis])), 3)
                if held != vector[axis]:
                    clamped.append(f"vector.{axis} {vector[axis]} -> {held} (more than {MAX_DELTA} from the head)")
                    vector[axis] = held
        ledger = dict(s.ledger)
        ledger["spent_ext"] = head.ledger.get("spent_ext") if head else None
        new = self.commit(session, "ingest" + (f":{source}" if source else ""), task_seed=s.task_seed,
                          vector=vector, last_delta=s.last_delta, ledger=ledger, open_conflicts=s.open_conflicts)
        return Verdict(new, True, (), tuple(clamped))

    def history(self, session: str) -> list[dict]:
        return [{"v": e.payload["state"]["v"], "origin": e.payload.get("origin", ""), "diff": e.payload.get("diff", {}),
                 "hash": e.hash} for e in self._states(session)]

    def diff(self, session: str, a: int, b: int) -> dict:
        old, new = self.get(session, a), self.get(session, b)
        if old is None or new is None:
            raise HelixError(f"session {session!r} has no version {a if old is None else b}")
        return diff(old, new)

    # ------------------------------------------------------------------------------------------- the record

    def note(self, session: str, event: str, payload: dict) -> str:
        return self.db.append(self._ensure(session), event, payload).hash

    def runs(self, session: str | None = None, limit: int = 50) -> list[dict]:
        out = []
        for s in ([session] if session else self.sessions()):
            name = self.entity(s)
            if self.db.has_entity(name):
                out.extend({"session": s, **e.payload} for e in self.db.events(name, E_RUN))
        return out[-limit:]

    def floor(self) -> dict:
        """The fixed scaffold of one run, in generated tokens: the median of what recent runs measured, or
        FLOOR_EST (an estimate, and labelled so) until a run has been measured."""
        seen = sorted(r["scaffold_ext"] for r in self.runs(limit=10_000)[-FLOOR_RUNS:]
                      if not r.get("bypass") and isinstance(r.get("scaffold_ext"), int))
        if not seen:
            return {"tokens": FLOOR_EST, "source": "estimate", "runs": 0}
        return {"tokens": seen[len(seen) // 2], "source": "measured", "runs": len(seen)}

    def verify(self, session: str | None = None) -> dict:
        """Recompute every chain, and check that each session's versions run 1, 2, 3, ... with no gap."""
        out = {}
        for s in ([session] if session else self.sessions()):
            name = self.entity(s)
            if not self.db.has_entity(name):
                continue
            chain = self.db.verify(name)
            versions = [e.payload.get("state", {}).get("v") for e in self.db.events(name, E_STATE)]
            chain["versions"] = len(versions)
            chain["sequential"] = versions == list(range(1, len(versions) + 1))
            chain["ok"] = chain["ok"] and chain["sequential"]
            out[s] = chain
        return out
