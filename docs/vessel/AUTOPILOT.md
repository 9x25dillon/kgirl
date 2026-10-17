# Autopilot and the RSI slider

`src/kgirl/vessel/autopilot.py` runs a vessel unattended. A *recursive-self-improvement* (RSI) slider sets how
much the vessel may change itself without a person in the loop. Levels are cumulative.

| RSI | name | adds | never |
|---|---|---|---|
| 0 | observe | measure drift every cycle and record it | changes a weight |
| 1 | steer | gated, reversible repair when drift ≥ engage (0.15); circuit breaker | learns new data |
| 2 | learn | anchored learning from the supplied stream each cycle | moves its anchor |
| 3 | consolidate | moves the anchor β = 0.1 toward the present self after 2 coherent cycles | changes its own controller |
| 4 | self-tune | improves its own steering controller from fidelity-gated sandbox trials | — |

```bash
export PYTHONPATH=src
python -m kgirl.vessel autopilot --rsi 3 --cycles 8            # one level
python -m kgirl.vessel autopilot --sweep --cycles 8 --json      # every level on the same world
python -m kgirl.vessel autopilot --rsi 4 --db ~/.kgirl/vessel/rsi.db   # keep the ledger
```

```python
from kgirl.vessel.autopilot import Autopilot, AutopilotConfig, RSI
ap = Autopilot(vessel, AutopilotConfig(level=RSI.CONSOLIDATE), stream=batches, guard=skill_loss)
ap.run(8)
ap.set_level(RSI.STEER, "operator lowered the slider")       # the slider can move at any time; ledgered
```

MCP: `vessel_autopilot` (`rsi` 0–4, `cycles` 1–8) runs the scenario below in memory.

## Cycle

```
world pushes (disturb) → read drift
  RSI ≥ 2  learn `learn_steps` anchored batches from the stream
  RSI ≥ 1  drift ≥ engage → steer (fresh optimizer moments; RSI 4 first saves the drifted weights as experience)
           live repair failed after a tune → watchdog reinstates the previous controller
  RSI ≥ 1  drift after repair > breaker (0.25) → restore last anchor snapshot, demote to RSI 1
  RSI ≥ 3  `stable_cycles` coherent cycles in a row → consolidate(β)
  RSI 4    every `tune_every` cycles → self_tune()
ledger: autopilot.cycle (+ steer, learn, anchor, autopilot.tune, autopilot.breaker, autopilot.level …)
```

## Self-tuning: the recursive step, and the guards it needed

`self_tune` proposes chaos-seeded log-normal variations of the steering controller (λ_rep, λ_voice, lr_max,
block size). It scores each one in sandboxes: the vessel's own substrate is checkpointed, put into a drifted
state, steered, measured, and restored. A candidate replaces the incumbent only if it passes all three tests:

1. **Faster on every held-out trial**, by at least `tune_margin` (15 %), not just on average.
2. **At least as faithful:** restored voice KL ≤ incumbent × 1.05, and coherence no lower.
3. **Tested on experience:** the trials replay the vessel's own most recent drifted states, saved just before
   each live repair. Chaos weight noise is used only until two such states exist.

Each guard exists because the measured version without it failed:

| version | what happened (8 disturbed cycles) |
|---|---|
| speed only, noise trials | Every adopted controller lowered λ_voice (0.86 → 0.55 → 0.66). Held-out persona voice loss rose to 0.425 (RSI 3: 0.385), and the last repair **stalled** at drift 0.146 with 11 rolled-back blocks. Goodhart: the cheapest repair is to care less about the voice. A controller tuned on synthetic noise also did not transfer to the world's real drift. |
| + fidelity gate, + replayed experience, + watchdog | 3 faster candidates **refused for lower fidelity**, 1 adopted (20–27 % faster on replayed drift). All 8 live repairs coherent, 0 rollbacks. Voice 0.384, the same as RSI 3. |

A second measured fix was **fresh optimizer moments for steering**. When steering shared Adam's state with the
world's updates and the learner's, repair cost grew 130 → 730 steps over 6 cycles, because inherited second
moments shrank every step. With fresh moments it cost 50–140 steps.

The breaker was moved down to RSI 1 for the same reason. With too few steering probes (256), RSI 1 repairs
fell short (drift 0.20, then 0.29) and each cycle stacked on the last. Nothing stopped that ratchet until the
breaker was armed from RSI 1.

## Measured: the slider (same world, 8 cycles, 60 unanchored foreign steps per cycle)

ReferenceLM, 512 steering and 128 audit probes. Held-out losses are on unseen persona prose (voice) and
unseen code (the skill the world pushes).

| RSI | max drift | recognizes itself (own vs sibling) | voice loss (birth 0.330) | skill loss (birth 7.96) | repair steps, cycle 1 → 8 | other |
|---|---|---|---|---|---|---|
| 0 | **0.995** | no (0.995 vs 0.996) | 3.774 | 0.47 | — | it became the world's model |
| 1 | 0.100 | yes (0.100 vs 0.433) | 0.364 | 5.66 | 90 → **510** | erases what the world taught |
| 2 | 0.098 | yes (0.097 vs 0.429) | 0.366 | 5.53 | 50 → 240 | 320 anchored learning steps |
| 3 | 0.100 | yes (0.079 vs 0.459) | 0.385 | 4.66 | 50 → 190 | 4 consolidations |
| 4 | 0.099 | yes (0.079 vs 0.460) | 0.384 | **4.62** | 50 → **128** | 1 controller adopted, 3 refused for fidelity |

What the slider buys, honestly stated:

- RSI 0 shows what happens with no defence: identity is lost in one cycle.
- RSI 1 holds identity, but its upkeep grows every cycle, because the world keeps writing where the probes do
  not look.
- Each higher level makes upkeep cheaper and keeps more of what the world taught. The cost is a small loss of
  native voice (0.364 → 0.385).
- RSI 4's live gain over RSI 3 is modest: repairs of 88–128 steps against 120–190 after its tune.

These are Level 1 measurements on a 33k-parameter model. The same trade-offs at scale are Level 2 until
measured there.

## Invariants (all levels)

- Every cycle and every change is a ledger event. Weight snapshots are content-addressed. The hash chain verifies
  after every run in the tests.
- Sandboxes always restore the substrate. The tests check bit-equal weights after `self_tune`.
- Consolidation never runs from an incoherent state.
- The autopilot never acts on the world. `disturb` belongs to the scenario, not to the vessel.
- Only a person raises the slider. The breaker can only lower it.
