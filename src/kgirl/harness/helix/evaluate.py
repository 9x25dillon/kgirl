"""evaluate: an offline audit of a HELIX-9 transcript that a model wrote on its own.

Runs of `Helix.run` do not need this: there no model writes a ledger number, so the failures below cannot
occur. This is for the other case, a log from a model following the HELIX-9 prompt in a chat window, where
the ledger, the vector and the capsules are all self-reports. Ported from the user's `helix_kernel.py`
(2026-11-16), detectors unchanged:

    budget_theater       the root ledger is more than 15% off the external token count; or a deeper level
                         claims more spend than the level above it; or sub-ledgers exist with no root
    steering_drift       a vector changed with no GRAD before it, or by more than 0.25 with no JUSTIFY
    missing_ledger       weights were set and no ledger was ever emitted
    capsule_bloat        a SEED field longer than twelve words
    fractal_fractal      a SEED deeper than L3
    ceremony_on_trivial  under 300 external tokens, with capsules and no BYPASS

Only the root ledger (L0) can be reconciled against the external count: a sub-ledger is a part of the spend
and will not match the total. Between levels the comparison is exact, since both numbers are the model's own.
Ledgers are judged after the whole log is read, so the verdict does not depend on emission order.
"""

from __future__ import annotations

import re

from .ledger import MIN_CONTENT
from .state import MAX_DELTA, SEED_WORDS

TOLERANCE = 0.15                 # how far the root ledger may sit from the external count
FIELD_WORDS = SEED_WORDS // 3    # goal, constraints and success get twelve words each
MAX_DEPTH = 3
RE = {
    "seed": re.compile(r"⟨SEED:(L\d+) \| goal: (.*?) \| constraints: (.*?) \| success: (.*?) \| budget: (\d+) tok⟩"),
    "w": re.compile(r"⟨W:([^|⟩]+)"),
    "grad": re.compile(r"⟨GRAD: (\w+)([+\-−])([\d.]+) \| because: (.*?)⟩"),
    "delta": re.compile(r'Δ: "(.*?)"'),
    "ledger": re.compile(r"⟨LEDGER:(L\d+) \| alloc: (\d+) \| spent: (\d+)"),
}
SEVERITY = {"budget_theater": "High", "steering_drift": "High", "missing_ledger": "High",
            "capsule_bloat": "Medium", "fractal_fractal": "Medium", "ceremony_on_trivial": "Medium"}
# detector -> the re-steer it argues for when it fires in two runs in a row
RESTEER = {"budget_theater": ("rigor", 0.15), "steering_drift": ("rigor", 0.10), "capsule_bloat": ("compression", 0.10)}


def reconcile(spent_reported: int, spent_external: int, tol: float = TOLERANCE) -> bool:
    """Whether a reported spend is within `tol` of the externally counted one."""
    return abs(spent_reported - spent_external) / max(spent_external, 1) <= tol


def evaluate(log: str, external_tokens: int | None = None) -> dict:
    """Audit one transcript. Returns the findings (detector, severity, the line as evidence), two rates
    (how many GRADs quote their evidence, how many deltas state a checkable behaviour) and a verdict: FAIL
    when any finding is High."""
    findings: list[dict] = []

    def flag(detector: str, line: str) -> None:
        findings.append({"detector": detector, "severity": SEVERITY[detector], "evidence": line.strip()})

    prev: dict | None = None
    grad_seen = justify_seen = False
    n_grad = n_anchored = n_delta = n_predicate = 0
    ledgers: dict[int, list[tuple[int, str]]] = {}         # level -> [(spent, line)], judged after the loop
    for line in log.splitlines():
        if "⟨JUSTIFY:" in line:
            justify_seen = True
        seed = RE["seed"].search(line)
        if seed:
            if int(seed.group(1)[1:]) > MAX_DEPTH:
                flag("fractal_fractal", line)
            if any(len(field.split()) > FIELD_WORDS for field in seed.group(2, 3, 4)):
                flag("capsule_bloat", line)
        if "⟨GRAD:" in line:
            grad = RE["grad"].search(line)
            grad_seen = True
            n_grad += 1
            if grad and '"' in grad.group(4):
                n_anchored += 1
        if "Δ:" in line:
            n_delta += 1
            d = RE["delta"].search(line)
            if d and ("→" in d.group(1) or "->" in d.group(1)) and \
                    re.search(r"\b(no|every|single|only)\b|[≥<>]", d.group(1)):
                n_predicate += 1
        w = RE["w"].search(line)
        if w:
            vec = {axis: float(value) for axis, value in re.findall(r"(\w+)=([\d.]+)", w.group(1))}
            if prev is not None:
                changed = {a for a in vec if a in prev and vec[a] != prev[a]}
                if changed and not grad_seen:
                    flag("steering_drift", line)
                if any(abs(vec[a] - prev[a]) > MAX_DELTA for a in changed) and not justify_seen:
                    flag("steering_drift", line)
            prev = {**(prev or {}), **vec}
            grad_seen = justify_seen = False
        led = RE["ledger"].search(line)
        if led:
            ledgers.setdefault(int(led.group(1)[1:]), []).append((int(led.group(3)), line))

    def spent(level: int) -> int:                          # a level may emit more than one ledger line
        return sum(s for s, _ in ledgers.get(level, []))

    if 0 in ledgers:
        if external_tokens is not None and not reconcile(spent(0), external_tokens):
            flag("budget_theater", ledgers[0][-1][1])      # only the root meets the external count
        levels = sorted(ledgers)
        for parent, child in zip(levels, levels[1:]):      # spend can only shrink going down
            if spent(child) > spent(parent):
                flag("budget_theater", ledgers[child][-1][1])
    elif external_tokens is not None and ledgers:
        flag("budget_theater", ledgers[min(ledgers)][-1][1])       # sub-ledgers, no root: unreconcilable

    if "⟨W:" in log and "⟨LEDGER:" not in log:
        flag("missing_ledger", "<no ledger emitted>")
    if external_tokens is not None and external_tokens < MIN_CONTENT and "⟨BYPASS:" not in log and "⟨SEED:" in log:
        flag("ceremony_on_trivial", f"external_tokens={external_tokens}")
    return {"findings": findings,
            "metrics": {"evidence_anchor_rate": round(n_anchored / n_grad, 2) if n_grad else None,
                        "delta_predicate_rate": round(n_predicate / n_delta, 2) if n_delta else None},
            "verdict": "FAIL" if any(f["severity"] == "High" for f in findings) else "PASS"}


def propose_resteer(reports: list[dict]) -> list[dict]:
    """A detector that fires in two consecutive audits argues for a re-steer. This only proposes: every
    entry needs a person's approval, and nothing here touches a stored STATE."""
    proposals = []
    for detector, (axis, delta) in RESTEER.items():
        streak = 0
        for i, report in enumerate(reports):
            streak = streak + 1 if detector in {f["detector"] for f in report["findings"]} else 0
            if streak >= 2:
                proposals.append({"axis": axis, "delta": delta, "because": f"{detector} in runs {i - 1},{i}",
                                  "requires_approval": True})
                break
    return proposals
