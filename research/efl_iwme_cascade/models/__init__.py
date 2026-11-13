"""The model families and their registry."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from .baseline import (Binomial, HiddenMarkov, MarkovChain, fit_binomial, fit_hidden_markov, fit_markov,
                       hidden_markov_from_arrays, markov_from_table)
from .cascade import all_strings, log2_prob, points_from_strings, sample_indices, signs, string_law, symbolize
from .iwme import AttentionCascade, fit_attention, markov_limit, memory_trace

Model = Union[Binomial, MarkovChain, HiddenMarkov, AttentionCascade]

FAMILIES: Tuple[str, ...] = ("B", "MC1", "MC2", "MC3", "HMM2", "AM")
PARAMETERS: Dict[str, int] = {"B": 1, "MC1": 3, "MC2": 7, "MC3": 15, "HMM2": 7, "AM": 3}

__all__ = ["AttentionCascade", "Binomial", "FAMILIES", "HiddenMarkov", "MarkovChain", "Model", "PARAMETERS",
           "all_strings", "describe", "fit_attention", "fit_binomial", "fit_family", "fit_hidden_markov",
           "fit_markov", "hidden_markov_from_arrays", "log2_prob", "markov_from_table", "markov_limit",
           "memory_trace", "points_from_strings", "project", "sample_indices", "signs", "string_law", "symbolize"]


def fit_family(name: str, strings, weights: Optional[np.ndarray] = None, kind: str = "chaos") -> Model:
    """Fit one family to weighted strings. `kind` selects the stream for HMM2's starting points only."""
    fitters: Dict[str, Callable[[], Model]] = {
        "B": lambda: fit_binomial(strings, weights),
        "MC1": lambda: fit_markov(strings, 1, weights),
        "MC2": lambda: fit_markov(strings, 2, weights),
        "MC3": lambda: fit_markov(strings, 3, weights),
        "HMM2": lambda: fit_hidden_markov(strings, weights, kind=kind),
        "AM": lambda: fit_attention(strings, weights),
    }
    if name not in fitters:
        raise KeyError(f"unknown family {name!r}; choose from {FAMILIES}")
    return fitters[name]()


def project(name: str, law, strings, mass: float = 1e12, kind: str = "chaos") -> Model:
    """The member of a family closest to an exact law in KL: the fit with probability weights. `mass` is the
    notional sample size, large enough that pseudo-counts and ridge are negligible."""
    return fit_family(name, strings, np.asarray(law) * mass, kind)


def describe(model: Model) -> str:
    if isinstance(model, Binomial):
        return f"p = {model.p:.4f}"
    if isinstance(model, MarkovChain):
        return "; ".join("[" + ", ".join(f"{v:.3f}" for v in t) + "]" for t in model.tables)
    if isinstance(model, HiddenMarkov):
        c = model.cascade
        return (f"p = ({c.p[0]:.3f}, {c.p[1]:.3f}); P = "
                + ", ".join(f"{c.P[s][ch][0]:.3f}" for s in (0, 1) for ch in (0, 1))
                + f" (to state 0 from s,c = 00, 01, 10, 11); pi0 = {c.pi0[0]:.3f}")
    tail = "" if model.coupling == 1.0 else f", coupling = {model.coupling:g}"
    return f"rho = {model.rho:.4f}, T = {model.T:.4f}, bias = {model.bias:.4f}{tail}"
