"""iwme — Intra-dimensional Working Memory Emission, rebuilt as testable code (Phases 3–4 of IWME-PRIME-01).

The IWME Draft 1.0 asks whether attention-mediated memory changes physical dynamics or only how an
observer represents them. This package holds the parts of that question that can be computed today,
each tied to a claim in research/iwme/IWME_Phase0-3.pdf and checked by `python -m kgirl.iwme verify`.

Modules (numpy everywhere; sympy only where marked):

    attention   softmax-EMA memory: open- vs closed-loop retention (P2, P2′), the contraction bound,
                the pitchfork at T_c and critical slowing (P3), flip and Neimark–Sacker cases (P3′),
                adiabatic hysteresis, gated forgetting
    cascade     binomial / Markov / EMA-memory cascades, D_q by closed form, transfer operator and box
                counting; the Hutchinson fixed point as the finite model of the EFL "E ≃ Φ_λ E"
    irt         the Inverse Resonant Toroid of Source C: induced metric, det, rank and singular circles,
                curvature by three independent routes; Source C's Appendix A transcribed for comparison
                [sympy for the symbolic proof]
    wake        damped-wave solver (centered and Source B schemes), CFL and order, discrete energy
                identity, memory → wake zero-order hold; Source B §4 checks [sympy for Euler–Lagrange]
    sources     Source D Tier 1–2 checks (Walsh–Hadamard, Z9 → Z3) and Source B's coherence formula
    core        the repaired IWME core, P1 (augmented state), exact persistence-after-decoupling R_persist
    verify      the claim registry: 47 checks, table / JSON / LaTeX output

Evidence levels follow the paper: 0 established, 1 extrapolation, 2 hypothesis, 3 speculative.
Everything in this package is Level 0 or 1 mathematics; nothing here is evidence that memory acts on
matter (that question needs the intervention protocols of the paper, §6).
"""

from . import attention, cascade, core, irt, sources, wake  # noqa: F401

__all__ = ["attention", "cascade", "core", "irt", "sources", "wake"]
