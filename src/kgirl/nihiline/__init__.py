"""nihiline — the c-BPE-ECL dual tumor-marker assay of CN121933729A, built from K1ll's lab scripts.

    nothing glows below
    every form a stained glass held
    against what isn't

Modules (numpy needed only where marked):

    patent         the patent as data: markers, aptamers, curves, claim ranges, Example 1
    calibration    quantify CEA/AFP from ECL; the scar on zero (blank + 3σ); dilution series
    amplification  fg of antigen per cell → detected photons; predicted LOD vs reported (gap)
    protocol       the transcript's nine levels run as the assay; deterministic chip read
    pulse   [np]   nihil leverage: optimal off-window of the 5 V drive; control × nihil; rate locking
    bloom   [np]   Bloom-integral viability sentinel for the cells on the cathode
    array   [np]   the BPE array as a phase lattice: crosstalk, uniformity, defects, Lyapunov
    plots   [mpl]  figures in the original dark lab style

Originals (unchanged) live in research/nihiline/originals/.
"""

from .calibration import quantify, quantify_chip, scar_report
from .protocol import NIHILINE, run_protocol

__all__ = ["quantify", "quantify_chip", "scar_report", "run_protocol", "NIHILINE"]
