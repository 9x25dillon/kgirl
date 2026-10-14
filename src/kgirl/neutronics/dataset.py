"""Locate, load and validate the NuRea_sim PWR lattice-physics dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from kgirl.paths import find_repo

DATA_DIR = "lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)(1)"
N_RODS = 39
N_COLS = 2 + N_RODS


@dataclass(frozen=True)
class LatticeSplit:
    k_inf: np.ndarray            # (n,)
    pppf: np.ndarray             # (n,)
    enrichment: np.ndarray       # (n, 39) wt% U-235

    @property
    def n(self) -> int:
        return len(self.k_inf)

    @property
    def targets(self) -> np.ndarray:
        return np.stack([self.k_inf, self.pppf], axis=1)


def data_dir(root: Optional[str | Path] = None) -> Path:
    if root:
        return Path(root)
    env = os.environ.get("NUREA_LATTICE_DIR")
    if env:
        return Path(env)
    return find_repo("NuRea_sim") / DATA_DIR


def _load(path: Path) -> np.ndarray:
    head = path.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        raise FileNotFoundError(f"{path} is a Git LFS pointer; run `git lfs pull` in the NuRea_sim repo")
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[1] != N_COLS:
        raise ValueError(f"{path}: expected {N_COLS} columns, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{path}: non-finite values")
    return a


def load(split: str = "train", root: Optional[str | Path] = None) -> LatticeSplit:
    """split: 'train' (raw.csv, 24 000 rows) or 'test' (test.csv, 360 held-out rows)."""
    name = {"train": "raw.csv", "test": "test.csv"}[split]
    a = _load(data_dir(root) / name)
    return LatticeSplit(a[:, 0], a[:, 1], a[:, 2:])


def validate(s: LatticeSplit) -> dict:
    """Physics sanity checks of the column interpretation; raises if they fail."""
    mean_e = s.enrichment.mean(axis=1)
    nonuni = s.enrichment.max(axis=1) / mean_e
    r_k = float(np.corrcoef(s.k_inf, mean_e)[0, 1])
    r_p = float(np.corrcoef(s.pppf, nonuni)[0, 1])
    checks = {
        "k_inf_range": [float(s.k_inf.min()), float(s.k_inf.max())],
        "pppf_range": [float(s.pppf.min()), float(s.pppf.max())],
        "enrichment_range": [float(s.enrichment.min()), float(s.enrichment.max())],
        "corr_kinf_mean_enrichment": round(r_k, 4),
        "corr_pppf_nonuniformity": round(r_p, 4),
    }
    ok = (0.9 < s.k_inf.min() and s.k_inf.max() < 2.0 and s.pppf.min() >= 1.0
          and 0.0 < s.enrichment.min() and s.enrichment.max() <= 20.0 and r_k > 0.8 and r_p > 0.5)
    if not ok:
        raise ValueError(f"column interpretation failed its physics checks: {checks}")
    return checks
