"""Coherence between two scale signals, in the three versions the experiment keeps apart.

    per frequency   Welch magnitude-squared coherence abs(S_xy)² / (S_xx S_yy), one value per frequency bin
    broadband       squared correlation at zero lag: all frequencies pooled
    normal-score    the same after each signal is replaced by the normal scores of its ranks. Normal scores
                    undo any monotone distortion; for a Gaussian copula this is the squared maximal
                    correlation, the measure for which the product rule along a Markov chain is guaranteed.

They can disagree, and the disagreement carries information: per-frequency against broadband separates
"every frequency keeps a floor" from "the pooled coherence decays" in a self-similar cascade; raw against
normal-score separates a nonlinear layer from a coupling between non-adjacent scales.

Estimator facts (Level 0, recomputed in checks.py):
  * K independent segments put a floor of 1/K under zero coherence; `debias` removes it and
    `null_threshold` is its critical value.
  * Coherence is unchanged by a gain or an invertible linear filter on either signal: no normalization is
    needed, and two scales made by linearly filtering one signal have coherence 1 and carry no information.
  * An uncorrected time offset between the signals lowers the estimate.
"""

from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np

Array = np.ndarray


def welch_msc(x: Array, y: Array, seg_len: int = 256) -> Array:
    """Per-frequency coherence. Hann window, non-overlapping segments, so the K = len // seg_len segments are
    independent for white inputs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    k = min(x.size, y.size) // seg_len
    if k < 2:
        raise ValueError("need at least two segments: one segment gives coherence 1 identically")
    window = np.hanning(seg_len)
    fx = np.fft.rfft(x[: k * seg_len].reshape(k, seg_len) * window, axis=1)
    fy = np.fft.rfft(y[: k * seg_len].reshape(k, seg_len) * window, axis=1)
    sxy = np.mean(fx * np.conj(fy), axis=0)
    return np.abs(sxy) ** 2 / (np.mean(np.abs(fx) ** 2, axis=0) * np.mean(np.abs(fy) ** 2, axis=0))


def band_mean(msc: Array) -> float:
    """Average of a per-frequency coherence over frequency, leaving out the DC and Nyquist bins."""
    return float(np.mean(np.asarray(msc)[1:-1]))


def broadband(x: Array, y: Array) -> float:
    """Squared correlation at zero lag, second moments about zero (the signals are zero-mean by design)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean(x * y) ** 2 / (np.mean(x * x) * np.mean(y * y)))


def null_threshold(k: int, alpha: float = 0.05) -> float:
    """Critical value of estimated coherence between unrelated signals at level alpha: 1 − alpha^(1/(K−1))."""
    return 1.0 - alpha ** (1.0 / (k - 1))


def debias(psi: float, k: int) -> float:
    """Remove the 1/K floor that K-segment averaging puts under zero coherence."""
    return (k * psi - 1.0) / (k - 1.0)


@lru_cache(maxsize=32)
def _score_table(n: int) -> Array:
    dist = NormalDist()
    return np.array([dist.inv_cdf((i + 0.5) / n) for i in range(n)])


def normal_scores(x: Array) -> Array:
    """Replace each column by the normal scores of its ranks: rank r of n maps to Φ⁻¹((r − ½)/n)."""
    x = np.asarray(x, dtype=float)
    cols = x.reshape(x.shape[0], -1)
    ranks = np.argsort(np.argsort(cols, axis=0, kind="stable"), axis=0, kind="stable")
    return _score_table(cols.shape[0])[ranks].reshape(x.shape)


def cascade_level_weight(level: int, freq_bin: int = 1) -> float:
    """Power, up to a common factor, that level j of a dyadic multiplicative cascade adds to the log
    amplitude at one frequency bin: 2^j sin²(π m / 2^j). It halves per level once cells are shorter than the
    wavelength, so fixed-frequency coherence deficits are summable while broadband ones (1/n) are not."""
    return (2.0 ** level) * math.sin(math.pi * freq_bin / 2.0 ** level) ** 2
