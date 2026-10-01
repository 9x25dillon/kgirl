"""Calibration, quantification, and the scar on zero.

Nihiline mapping (from the transcript, Levels 8–9):

    Scar on Zero          the blank is never empty: blank + 3σ is the residue every
                          measurement must rise above. The patent never prints the blank,
                          but each curve evaluated at its own LOD *reveals* that scar.
    Stubbornness of 1     dilute a sample by 10 forever and the concentration never
                          reaches zero — but its signal sinks into the scar. Where the
                          dilution series crosses the scar is the LOD.

Pure Python (no numpy) so the MCP tools and CLI work everywhere.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .patent import MARKERS, LogLinearCurve


@dataclass(frozen=True)
class Quant:
    marker: str
    mode: str            # "conc" | "cells"
    signal: float
    value: float         # back-calculated concentration (unit below); NaN if non-physical
    unit: str
    flag: str            # ok | below_lod | below_range | above_range
    scar: float          # signal at the reported LOD (blank + 3σ implied by the patent)

    def to_dict(self) -> dict:
        return asdict(self)


def curve(marker: str, mode: str = "conc") -> LogLinearCurve:
    m = MARKERS[marker.upper()]
    if mode not in ("conc", "cells"):
        raise ValueError("mode must be 'conc' or 'cells'")
    return m.conc_curve if mode == "conc" else m.cell_curve


def signal(c: LogLinearCurve, x: float) -> float:
    if x <= 0:
        raise ValueError("concentration must be > 0 (log-linear curve)")
    return c.slope * math.log10(x) + c.intercept


def invert(c: LogLinearCurve, intensity: float) -> float:
    return 10 ** ((intensity - c.intercept) / c.slope)


def scar(c: LogLinearCurve) -> float:
    """Signal level the patent's LOD implies for blank + 3σ."""
    return signal(c, c.lod)


def quantify(marker: str, intensity: float, mode: str = "conc") -> Quant:
    c = curve(marker, mode)
    s = scar(c)
    x = invert(c, intensity)
    if intensity < s:
        flag = "below_lod"
    elif x < c.lo:
        flag = "below_range"
    elif x > c.hi:
        flag = "above_range"
    else:
        flag = "ok"
    return Quant(marker.upper(), mode, intensity, x, c.unit, flag, round(s, 2))


def quantify_chip(cea_signal: float, afp_signal: float, mode: str = "conc") -> dict[str, Quant]:
    """One c-BPE chip read: anode ECL of the CEA lane and the AFP lane."""
    return {"CEA": quantify("CEA", cea_signal, mode), "AFP": quantify("AFP", afp_signal, mode)}


def lod_from_blank(marker: str, blank_mean: float, blank_sd: float, mode: str = "conc", k: float = 3.0) -> float:
    """LOD for *your* chip: the concentration whose signal equals blank + kσ."""
    return invert(curve(marker, mode), blank_mean + k * blank_sd)


def scar_report() -> list[dict]:
    """Consistency check across the four printed curves.

    Three curves put their LOD at ~170–190 a.u. (one shared blank+3σ); the AFP
    concentration curve sits at ~590 a.u. That outlier is a finding about the
    document, not about the chemistry.
    """
    rows = []
    for name, m in MARKERS.items():
        for mode, c in (("conc", m.conc_curve), ("cells", m.cell_curve)):
            rows.append({"marker": name, "mode": mode, "lod": c.lod, "unit": c.unit,
                         "scar_signal": round(scar(c), 1), "lod_below_linear_range": c.lod < c.lo})
    shared = [r["scar_signal"] for r in rows]
    med = sorted(shared)[len(shared) // 2]
    for r in rows:
        r["outlier"] = abs(r["scar_signal"] - med) > 0.5 * med
    return rows


def dilution_series(marker: str, start: float, factor: float = 10.0, steps: int = 8,
                    mode: str = "conc") -> list[dict]:
    """The stubbornness of 1: x/factorᵏ never reaches 0, but its signal crosses the scar."""
    c = curve(marker, mode)
    out, x = [], start
    for k in range(steps):
        i = signal(c, x)
        out.append({"step": k, "true": x, "signal": round(i, 1), "flag": quantify(marker, i, mode).flag})
        x /= factor
    return out


def per_cell_expression(marker: str, cells_per_ml: float, antigen_ng_per_ml: float) -> float:
    """fg of antigen per cell from paired cell-count and concentration readouts."""
    if cells_per_ml <= 0:
        raise ValueError("cells_per_ml must be > 0")
    return antigen_ng_per_ml * 1e6 / cells_per_ml
