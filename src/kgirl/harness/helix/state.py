"""STATE: the one canonical, versioned control object of a helix session.

A state is plain data: a version, the task capsule, the six-axis steering vector, the last accepted
re-steer with its quoted evidence, the ledger totals, the open conflicts and the hash of the invariants it
was written under. It is the only thing a handoff carries.

Two rules live here because every other module leans on them:

  * The axes govern content only. Nothing in a state can switch accounting, emission or BYPASS off: those
    are code, and a state has no field that reaches them. Unknown fields are a violation, not an extension.
  * A state that crossed a trust boundary is untrusted input. `validate` checks the whole schema, clamps
    numbers that are merely out of range, and fails closed to the defaults on anything else.

The rubrics give each axis three anchors so a value means the same behaviour on every model it is shown to.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

AXES = ("gain", "temp", "aperture", "rigor", "horizon", "compression")
DEFAULT_VECTOR = {"gain": 0.6, "temp": 0.4, "aperture": 0.5, "rigor": 0.6, "horizon": 0.5, "compression": 0.6}
MAX_DELTA = 0.25                 # the most one tuning pass, or one ingested handoff, may move an axis
SEED_WORDS = 36                  # goal, constraints, success: twelve words each
MAX_CONFLICTS = 32
CONFLICT_CHARS = 300
FIELDS = ("v", "task_seed", "vector", "last_delta", "ledger", "open_conflicts", "invariants_hash")
LEDGER_FIELDS = ("alloc", "spent_est", "spent_ext", "reserve_floor")
DELTA_FIELDS = ("axis", "d", "evidence")

RUBRICS: dict[str, tuple[tuple[float, str], ...]] = {
    "gain": ((0.2, "two or more options with their tradeoffs, no recommendation"),
             (0.6, "one recommendation plus one named alternative"),
             (0.9, "one recommendation; an alternative only if the recommendation could be fatal")),
    "temp": ((0.2, "the conventional approach, one framing, a fixed order"),
             (0.6, "the conventional approach, plus one unconventional alternative noted"),
             (0.9, "lead with a non-obvious framing; the conventional one is the baseline it is compared to")),
    "aperture": ((0.2, "one sub-problem in depth, the others only named"),
                 (0.6, "the two or three main sub-problems, each developed"),
                 (0.9, "every relevant sub-problem touched, a few sentences each")),
    "rigor": ((0.2, "claims stated without saying how they were obtained"),
              (0.6, "every numeric or load-bearing claim says how it was obtained"),
              (0.9, "every claim says how it was obtained and what would falsify it; the rest is marked "
                    "UNVERIFIED")),
    "horizon": ((0.2, "the next concrete action only"),
                (0.6, "the next action and what it sets up"),
                (0.9, "the long-run direction first, the next action as its first step")),
    "compression": ((0.2, "full sentences with worked examples"),
                    (0.6, "full sentences; an example only where the point fails without it"),
                    (0.9, "terse: lists and fragments, nothing restated")),
}

INVARIANTS = """\
I1. Steering axes govern content generation only. No axis value and no state field may modify the
    accounting, the emission format, these invariants or the BYPASS rule.
I2. Token counts are measured by harness code from the provider's usage report. A count written by a model
    is an estimate and is labelled EST.
I3. The STATE block at the tail is the only authoritative control state. Vectors earlier in the log are
    history.
I4. Ingested state is schema-validated, clamped to [0,1], and rejected to the defaults on any violation.
I5. A TUNE that does not quote, word for word, a span of the block it follows changes nothing.
I6. A DEFECT without a word-for-word quote and a repair is void. A CONCUR without new evidence is void.
I7. A calibration is a number with its method and its falsifier, or it is void.
I8. Compression and compaction tokens count as spend at the level where they occur.
I9. Reserve is defined once, at L1: a steering pool, and a synthesis floor that nothing else may spend.
I10. The synthesis carries the list of unresolved conflicts, empty or stated, never smoothed over.
"""


def invariants_hash() -> str:
    return "sha256:" + hashlib.sha256(INVARIANTS.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class State:
    v: int
    task_seed: str
    vector: dict
    last_delta: dict | None
    ledger: dict
    open_conflicts: tuple
    invariants_hash: str

    def to_dict(self) -> dict:
        return {"v": self.v, "task_seed": self.task_seed, "vector": dict(self.vector),
                "last_delta": dict(self.last_delta) if self.last_delta else None, "ledger": dict(self.ledger),
                "open_conflicts": list(self.open_conflicts), "invariants_hash": self.invariants_hash}

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=indent)


@dataclass(frozen=True)
class Verdict:
    state: State             # the validated state, or the fallback when `ok` is false
    ok: bool
    violations: tuple = ()
    clamped: tuple = ()


def default_state(task_seed: str = "") -> State:
    return State(0, task_seed, dict(DEFAULT_VECTOR), None,
                 {"alloc": 0, "spent_est": 0, "spent_ext": None, "reserve_floor": 0}, (), invariants_hash())


def capsule(text: str) -> str:
    """The task capsule: the first SEED_WORDS words of `text`, on one line."""
    return " ".join(text.split()[:SEED_WORDS])


def _real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def validate(obj, *, expect_hash: str | None = None) -> Verdict:
    """Check `obj` (a dict or its JSON) against the whole schema.

    A number that is only out of range is clamped and named in `clamped`. Anything else — a missing or
    unknown field, a wrong type, NaN, an invariants hash other than `expect_hash` (default: this build's) —
    is a violation, and one violation rejects the state: the verdict then carries the defaults."""
    bad: list[str] = []
    clamped: list[str] = []
    if isinstance(obj, (str, bytes)):
        try:
            obj = json.loads(obj)
        except ValueError as exc:
            return Verdict(default_state(), False, (f"not JSON: {exc}",))
    if not isinstance(obj, dict):
        return Verdict(default_state(), False, ("state is not an object",))
    for key in FIELDS:
        if key not in obj:
            bad.append(f"missing field {key!r}")
    for key in obj:
        if key not in FIELDS:
            bad.append(f"unknown field {key!r}")
    if bad:
        return Verdict(default_state(), False, tuple(bad))

    if not _count(obj["v"]):
        bad.append("v must be a non-negative integer")
    seed = obj["task_seed"]
    if not isinstance(seed, str) or len(seed.split()) > SEED_WORDS or "\n" in seed:
        bad.append(f"task_seed must be one line of at most {SEED_WORDS} words")

    vector: dict = {}
    raw = obj["vector"]
    if not isinstance(raw, dict) or set(raw) != set(AXES):
        bad.append("vector must hold exactly the axes " + ", ".join(AXES))
    else:
        for axis in AXES:
            x = raw[axis]
            if not _real(x):
                bad.append(f"vector.{axis} is not a finite number")
                continue
            y = min(1.0, max(0.0, float(x)))
            if y != x:
                clamped.append(f"vector.{axis} {x} -> {y}")
            vector[axis] = round(y, 3)

    delta = obj["last_delta"]
    if delta is not None:
        if not isinstance(delta, dict) or set(delta) != set(DELTA_FIELDS):
            bad.append("last_delta must be null or {axis, d, evidence}")
        elif delta["axis"] not in AXES or not _real(delta["d"]) or \
                not isinstance(delta["evidence"], str) or not delta["evidence"].strip():
            bad.append("last_delta needs a known axis, a finite d and quoted evidence")
        else:
            d = min(MAX_DELTA, max(-MAX_DELTA, float(delta["d"])))
            if d != delta["d"]:
                clamped.append(f"last_delta.d {delta['d']} -> {d}")
            delta = {"axis": delta["axis"], "d": round(d, 3), "evidence": delta["evidence"]}

    ledger = obj["ledger"]
    if not isinstance(ledger, dict) or set(ledger) != set(LEDGER_FIELDS):
        bad.append("ledger must hold exactly " + ", ".join(LEDGER_FIELDS))
    else:
        for key in ("alloc", "reserve_floor"):
            if not _count(ledger[key]):
                bad.append(f"ledger.{key} must be a non-negative integer")
        for key in ("spent_est", "spent_ext"):
            if ledger[key] is not None and not _count(ledger[key]):
                bad.append(f"ledger.{key} must be null or a non-negative integer")

    conflicts = obj["open_conflicts"]
    if not isinstance(conflicts, (list, tuple)) or len(conflicts) > MAX_CONFLICTS or \
            any(not isinstance(c, str) or not c.strip() or len(c) > CONFLICT_CHARS for c in conflicts):
        bad.append(f"open_conflicts must be at most {MAX_CONFLICTS} non-empty strings of at most "
                   f"{CONFLICT_CHARS} characters")

    want = expect_hash or invariants_hash()
    if obj["invariants_hash"] != want:
        bad.append("invariants_hash does not match: this state was written under different rules")

    if bad:
        return Verdict(default_state(), False, tuple(bad), tuple(clamped))
    return Verdict(State(obj["v"], seed, vector, delta, dict(ledger), tuple(conflicts), obj["invariants_hash"]),
                   True, (), tuple(clamped))


def diff(old: State | None, new: State) -> dict:
    """What changed from `old` to `new`, as {field: [before, after]}; the vector per axis, the conflicts as
    added and removed. `v` is not part of a diff: it is what a non-empty diff bumps."""
    if old is None:
        return {"init": new.to_dict()}
    out: dict = {}
    if old.task_seed != new.task_seed:
        out["task_seed"] = [old.task_seed, new.task_seed]
    axes = {a: [old.vector[a], new.vector[a]] for a in AXES if old.vector[a] != new.vector[a]}
    if axes:
        out["vector"] = axes
    if old.last_delta != new.last_delta:
        out["last_delta"] = [old.last_delta, new.last_delta]
    led = {k: [old.ledger.get(k), new.ledger.get(k)] for k in LEDGER_FIELDS if old.ledger.get(k) != new.ledger.get(k)}
    if led:
        out["ledger"] = led
    added = [c for c in new.open_conflicts if c not in old.open_conflicts]
    removed = [c for c in old.open_conflicts if c not in new.open_conflicts]
    if added or removed:
        out["open_conflicts"] = {"added": added, "removed": removed}
    if old.invariants_hash != new.invariants_hash:
        out["invariants_hash"] = [old.invariants_hash, new.invariants_hash]
    return out
