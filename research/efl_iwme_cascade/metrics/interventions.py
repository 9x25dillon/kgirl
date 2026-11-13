"""A change of conditions, and how a fitted model is scored against it.

An intervention changes one mechanism of the generator. A family "has a handle" for it when one of its
terms corresponds to that mechanism; that family edits its fitted model by the known change, with no new
data. Every other family stays as fitted. The score is the regret against the changed generator.

A model can generate the observed law and still have no handle for a change of conditions. The zero-shot
score measures that directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence


@dataclass(frozen=True)
class Intervention:
    name: str
    truth: str                                   # which ground truth is changed
    change: str                                  # in words
    after: object                                # the generator once the condition has changed
    handle_family: str                           # the family with a term for that condition
    edit: Callable[[object], object]             # what that family does to its fitted model


def apply(intervention: Intervention, family: str, fitted: object) -> object:
    """The model a family predicts with after the change: edited if it has the handle, else as fitted."""
    return intervention.edit(fitted) if family == intervention.handle_family else fitted


def catch_up(refit_mean: Dict[int, float], target: float, sizes: Sequence[int]) -> Optional[int]:
    """Smallest post-change sample size at which refitting from scratch reaches `target` regret, or None."""
    return next((n for n in sizes if refit_mean[n] <= target), None)
