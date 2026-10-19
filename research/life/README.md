# life — the originals and what became of them

`originals/` holds the uploads of 2026-10-19, verbatim:
- `The_Thermodynamics_of_Being_Revised_Manuscript.pdf`, the philosophical framework. The uploaded .docx had the
  same text, so it is not duplicated.
- `Minimal_Algorithmic_Life_Equation.tex` / `.pdf`, the computational scaffold and its listing.
- `Algorithmic_Life_Equation_Mathematics.tex`, the mathematical appendix.
- `life_simulator.py`, the reference implementation.

The working version is `src/kgirl/life/` (standard library only), wired into the harness by
`src/kgirl/harness/metabolism.py`. The guide, with all measured numbers, is `docs/life/README.md`.
`python -m kgirl.life triage` runs `originals/life_simulator.py` beside the rebuild on the same input.

The reference runs as-is (numpy). Its noise comes from `np.random`, which kgirl replaces with ChaosStream.

## Triage

| original | decision | why (measured) |
|---|---|---|
| energy balance F' = [F + ηE − W − D]₊ | **kept** + ledger | It now also returns the deficit and the paid fraction, so F' = F + ηE − W − D + deficit holds exactly, and entropy production is ≥ 0 by construction. |
| organization update (Euler) | **kept, integrated exactly** | With the defaults F → 35.4 and g + ℓ = 2.67, the Euler multiplier is −1.67. A clip at O = 1 masks a period-2 chatter. ẋ = α − βx is now stepped exactly, and `integrator="euler"` keeps the literal map. |
| coherence window | **kept**, made two-sided (v_ref) | C could never exceed C_*, so "rigidity" was unreachable. |
| continuous-time confinement −γ(C − C_min)₊ − γ(C_max − C)₊ | **replaced** | It equals −γ(C_max − C_min) everywhere inside the window: a constant bias, not a barrier. A log-barrier gradient replaces it (`continuous.barrier`). |
| world model, gradient + λΩ | **kept**, normalized (NLMS) | The raw step diverges once ‖φ‖² > 1/η_M. NLMS is stable for η_M < 1. Ridge and an L1 proximal step are offered. The nuclear norm needs an SVD and is left out. |
| memory | **rebuilt** | The listing used a ring buffer and never used ρ. The appendix's EMA is implemented. |
| self-model G, L | **rebuilt** | PDF: G = −tanh‖ε‖ ≤ 0 and L = 0.1\|F − 1\|, so R can only fall. The .py: G = 0.1(1 − \|F − 2\|/5), which is negative for F > 7. R = 0 from step 49, and C_eff = 0 on 88 % of steps. The rebuild uses the appendix's own definitions: G = predictive usefulness of internal state for the next error (online R² skill), and L = self-prediction error + ξ at a discontinuity. |
| reachable potential | **rebuilt** | max(1, int(4·min(1, F/2)·O·C)) ∈ {1} for every reachable state, so P ≡ log 2, fed or starving. The rebuild counts affordable action sequences under O κ(C) F. |
| residual variance | **rebuilt** | `np.var` of a list of vectors pooled time and channels. |
| policy π | **rebuilt** | It was never updated. Now a bandit on the appendix's objective J. |
| learning at F = 0 | **removed** | The model kept learning with no free energy: stationary error was identical at every inflow from 0.6 to 0.02. Work now runs at the paid fraction. |
| functional index C_eff | **kept** | It is a diagnostic. Ablations show that R and C influence nothing else in these equations (docs/life, Open questions). |
| staggered replicas T_k(t) = T(t − (k − 1)τ/2), k = 1..3 | **corrected** | Offsets 0, τ/2, τ ≡ 0 put replica 3 in phase with replica 1. Uniform phases (k − 1)τ/N are optimal. Without coordination every schedule drifts to 1 − (r/τ)ᴺ. |
| "Availability gain > coordination + replication cost" | **implemented** | `replication.best_replication`. The same criterion sizes Jev swarms. |
| "99.9 % ceiling" | **replaced** (as §9.3 asks) | Critical slowing down plus the recovery-time trend: `analysis.rigidity`. |
| five experimental axes | **implemented** | `kgirl.life.experiments`. The markers axis waits for real data. |
| Creation / Nothingness, K(t) ∈ K(∅), life unending | not modelled | The manuscript labels these metaphysical. There is nothing to compute, and nothing here claims otherwise. |
