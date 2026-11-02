"""QINCRS as an optional Soup promotion validator.

The curator promotes a staged fragment once it has support, a high enough
lower confidence bound and (if set) `validator(fragment)` is true. This module
supplies that validator, built on QINCRS's CollapseIntegrityAuditor.

What is audited
    The "seam" is promotion. For each of the five QINCRS bands we compare the
    fragment's point utility (original kappa) with its confidence bound at a
    band-specific z (reconstructed kappa; DELTA z=0 ... GAMMA z=-2). A fragment
    whose utility estimate holds up as the bound tightens has evidence
    that is robust. A thinly-evidenced one loses kappa across the bands
    (large shortfall and drift) and is held back in staging. The thresholds
    below are this harness's choice, not QINCRS's: the auditor computes
    the quantities; `audit_pass` is logged but not used as the gate, because
    its signed sum can cancel. Whether this improves what gets promoted is a
    Level 2 claim; measure it with `soup stats`.

Isolation (by design)
    * Opt-in: nothing runs unless a QincrsValidator is constructed, or
      KGIRL_QINCRS_GATE=1 is set (see Assistant).
    * The engine is loaded from an explicit file path under the private module
      name `_kgirl_qincrs_engine`. It never imports, patches or shadows
      `numbskull`, its coherence scorer or its inference injection, and it
      never mutates Soup beyond the boolean verdict.
    * Fail-open by default: if the engine (numpy/scipy/matplotlib) or its file
      is unavailable the validator approves, so the gate can never block
      memory promotion just because QINCRS is missing. `strict=True` inverts
      this. Every verdict and reason is kept in `.log`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .pool import Fragment

ENGINE_MODULE = "_kgirl_qincrs_engine"
ENV_PATH = "KGIRL_QINCRS_PATH"
Z_BY_BAND = ("DELTA", "THETA", "ALPHA", "BETA", "GAMMA")
Z_LEVELS = (0.0, -0.5, -1.0, -1.5, -2.0)


@dataclass(frozen=True)
class Verdict:
    fragment_id: int
    ok: bool
    reason: str
    shortfall: float | None = None
    drift: float | None = None
    audit_pass: bool | None = None


def load_engine(path: str | os.PathLike):
    """Import QiNCrs.py from `path` under a private module name."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"QINCRS engine not found: {p}")
    if ENGINE_MODULE in sys.modules:
        return sys.modules[ENGINE_MODULE]
    os.environ.setdefault("MPLBACKEND", "Agg")  # engine imports pyplot; never open a window
    spec = importlib.util.spec_from_file_location(ENGINE_MODULE, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ENGINE_MODULE] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(ENGINE_MODULE, None)
        raise
    return mod


class QincrsValidator:
    """Callable `Fragment -> bool`, usable as `Curator(soup, validator=...)`."""

    def __init__(self, engine_path: str | os.PathLike | None = None, max_shortfall: float = 0.20,
                 max_drift: float = 0.15, strict: bool = False, engine=None):
        self.engine_path = engine_path or os.environ.get(ENV_PATH)
        self.max_shortfall = max_shortfall
        self.max_drift = max_drift
        self.strict = strict
        self._engine = engine
        self._auditor = None
        self.log: list[Verdict] = []

    def _load(self):
        if self._auditor is None:
            eng = self._engine
            if eng is None:
                if not self.engine_path:
                    raise FileNotFoundError(f"no QINCRS engine path (set {ENV_PATH})")
                eng = load_engine(self.engine_path)
            self._engine = eng
            self._auditor = eng.CollapseIntegrityAuditor()
        return self._engine, self._auditor

    def __call__(self, f: Fragment) -> bool:
        v = self.check(f)
        self.log.append(v)
        return v.ok

    def check(self, f: Fragment) -> Verdict:
        try:
            eng, auditor = self._load()
            bands = list(eng.FrequencyBand)
            if len(bands) != len(Z_LEVELS):
                raise RuntimeError(f"expected {len(Z_LEVELS)} QINCRS bands, engine has {len(bands)}")
            original = {b: float(f.utility) for b in bands}
            recon = {b: float(f.bound(z)) for b, z in zip(bands, Z_LEVELS)}
            # zero time delta: the seam is an evidence re-read, not elapsed time
            res = auditor.audit(original, recon, 0.0, 0.0, None)  # capsule is unused by the audit math
        except Exception as e:  # engine missing/broken: fail open unless strict
            return Verdict(f.id, not self.strict, f"engine unavailable: {type(e).__name__}: {e}"[:200])
        shortfall, drift = abs(float(res.Δκ)), float(res.D_ω)
        ok = shortfall <= self.max_shortfall and drift <= self.max_drift
        why = "robust" if ok else f"thin evidence (shortfall {shortfall:.3f}, drift {drift:.3f})"
        return Verdict(f.id, ok, why, shortfall, drift, bool(res.audit_pass))
