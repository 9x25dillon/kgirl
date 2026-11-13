"""Description length: regret against a known truth, and two-part code length on data.

regret          expected excess code length on new data, exact (no test sample):
                strings    KL(truth ‖ model) / n            bits per symbol, summed over all 2^n strings
                Gaussian   ½ [tr(Σ̂⁻¹Σ) − d − ln det(Σ̂⁻¹Σ)] / ln 2    bits per sample
asymptotic      a correct family with k free parameters fitted to m observations has expected regret
                k / (2 m ln 2) bits per observation
two-part        −log2 P(data under the fit) + (k/2) log2(observations): data bits plus parameter bits
"""

from __future__ import annotations

import math

import numpy as np

from ..models.cascade import log2_prob

Array = np.ndarray


def regret_bits(truth_law: Array, model, strings: Array) -> float:
    """KL(truth ‖ model) per symbol, in bits, summed exactly over `strings` (all strings of one length)."""
    keep = truth_law > 0.0
    gap = np.log2(truth_law[keep]) - log2_prob(model, strings[keep])
    return float((truth_law[keep] * gap).sum() / strings.shape[1])


def asymptotic_regret(k: int, observations: float) -> float:
    """k / (2 m ln 2): expected regret per observation of a correct k-parameter family on m observations."""
    return k / (2.0 * observations * math.log(2.0))


def two_part_bits(model, strings: Array) -> float:
    """Data bits under the fitted model plus (k/2) log2(number of symbols) parameter bits."""
    symbols = strings.shape[0] * strings.shape[1]
    return float(-log2_prob(model, strings).sum() + 0.5 * model.k * math.log2(symbols))


def gaussian_regret_bits(truth_cov: Array, model_cov: Array) -> float:
    """Excess code length per sample of the zero-mean Gaussian with `model_cov` on data whose second moments
    are `truth_cov`, relative to the Gaussian with `truth_cov` itself."""
    ratio = np.linalg.solve(model_cov, truth_cov)
    sign, logdet = np.linalg.slogdet(ratio)
    if sign <= 0:
        return math.inf
    return float(0.5 * (np.trace(ratio) - truth_cov.shape[0] - logdet) / math.log(2.0))
