# TA-ULS v2 — a language model built from kgirl

`src/kgirl/tauls/` revives the TA-ULS architecture of `src/kgirl/llm/tauls_*.py` (the Two-level Trans-Algorithmic Universal Learning System) as a causal language model, trained on your own repositories. The training pipeline is ready to run unattended on a 64 GB RAM / 8 GB VRAM desktop.

Evidence levels:
- **L0**: measured in this repo by running code.
- **L1**: derived from reading code or an estimate.
- **L2**: a hypothesis that needs the desktop runs.
- **L3**: speculative.

## 1. Summary

| | |
|---|---|
| Pipeline | byte-level BPE tokenizer → corpus shards → trainer → eval → generation → weight-stability report, all `python -m kgirl.tauls …` |
| Architecture | causal TA-ULS blocks (KFP damping, entropy regulation, two-level control with inter-block recurrence), an optional HRM hierarchical recurrence, softmax or stablemax loss |
| Guarantees (L0, tests) | exact causality and batch independence; KV-cache decoding equal to the full forward (≤ 4e-7); chunked loss and gradient checkpointing equal to the plain path; bit-exact resume; no PRNG anywhere |
| Corpus today (L0) | kgirl alone: 484 distinct text files, 1.65M train + 0.11M val tokens at 8k vocab, 3.88 bytes/token |
| First signal (L2) | at 0.65M parameters and 300 steps, a parameter-matched plain transformer beats TA-ULS by 0.27 nats/token. HRM recurrence (2×2 cycles) is worse still at that scale. The desktop runs (§6) decide |

## 2. Audit of v1 (`kgirl/llm/tauls_transformer.py`, `tauls_model.py`)

| # | Defect | Effect on a language model |
|---|---|---|
| 1 | Attention not causal: `attention_mask` defaults to `None` | every token sees the future; loss collapses, generation impossible |
| 2 | `KFPLayer` variance over the whole batch and all positions, kept in a buffer mutated by `forward()`, also in eval | future and cross-sample leakage; outputs depend on call history |
| 3 | `EntropyRegulationModule` reduces to one scalar per batch × sequence | the same leakage |
| 4 | `torch.autograd.grad` on a buffer (`tauls_transformer.py`) | dead code, always hits `except` |
| 5 | `F` used but not imported (`tauls_model.TAULSRunner.score_text`) | `NameError` (**fixed** in this PR: one import) |
| 6 | post-norm, learned absolute positions, untied 50k head, `need_weights=True` attention | unstable at depth, no flash attention, VRAM waste |
| 7 | no tokenizer, data, training loop or checkpoints | nothing to train |

v1 files are otherwise left as they are.

## 3. Architecture

```
x ─┬─ RMSNorm ─ causal attention (RoPE · SDPA/flash · GQA · KV cache) ─────── a ──┐   stress_t = ‖a_t‖/‖x_t‖
   +◀─────────────────────────────────────────────────────────────────────────────┘
   ├─ RMSNorm ─ EntropyRegulator(stress) ─ TwoLevelControl(c_prev) ─ CausalKFP ── c ┐
   +◀─────────────────────────────────────────────────────────────────────────────┘
   └─▶ next block; c becomes its c_prev
```

| Component | Definition | Starts as |
|---|---|---|
| CausalKFP | `y = x − w ⊙ g_t ⊙ U V x`; `g_t` = inverse per-feature variance over positions ≤ t (cumulative sums), normalised to mean 1, stop-gradient as v1's buffer was | identity (`w = 0`) |
| EntropyRegulator | per token: `s_t = 2σ(W σ(ê_t − target + stress_t) + b)`, `ê_t = σ(MLP(x_t))` | identity (`W, b = 0`) |
| TwoLevelControl | `α ⊙ Meta(x + R c_prev) + (1 − α) ⊙ Auto(x)`, SwiGLU paths, Meta width 2D, Auto D | `α = ½`, `R = 0` |
| HRM mode (`h_cycles > 0`) | z_L ← L-stack(z_L + z_H + x) `l_cycles` times, then z_H ← H-stack(z_H + z_L), all repeated `h_cycles` times; weights shared across cycles; one-step gradient; head reads z_H | — |
| Loss | softmax CE, or stablemax `s(x) = x+1 (x ≥ 0), 1/(1−x) (x < 0)`; sampling uses the matching normalisation | — |
| Init | every tensor from a chaos stream keyed by its parameter name (logistic map r = 4, arcsin map to U(0,1), Box–Muller) | — |

Every TA-ULS component can be switched off. `matched_baseline()` widens the plain MLP until block parameters match within 1%. That makes "does TA-ULS help?" an experiment, not a belief.

HRM comes from `9x25dillon/Fractal_cascade_simulation` (`models/hrm/hrm_act_v1.py`, MIT), after Wang et al. 2025 (arXiv:2506.21734). Changes for language modelling:
- causal attention
- pre-norm blocks
- zero initial states
- a per-(application, block) KV/KFP cache, so incremental decoding is exact
- no ACT halting yet (§7)

The coherence guard reuses the κ thresholds of `CoherenceController` from that repo's `unified_coherence_system.py`.

Presets (non-embedding parameters, L0):

| Preset | Shape | Non-embedding | Notes |
|---|---|---|---|
| `tiny`, `tiny-baseline`, `tiny-hrm` | D 64, 2 blocks, ctx 64 | 0.12M | tests, CPU smoke runs |
| `small` / `small-baseline` | D 384, 6 blocks, ctx 512 | 13.0M / 12.8M | repo-scale corpus |
| `small-hrm` / `small-hrm-tauls` | 4 L + 2 H blocks, 2×2 cycles | ≈ small-baseline / small | 20 block applications per token |
| `desk-8gb` / `desk-8gb-baseline` | D 640, 10 blocks, 10 heads (5 KV), ctx 1024, vocab 16k | 56.1M / 55.3M | needs far more data than kgirl alone |

## 4. Run it on the desktop

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 && pip install -r requirements-tauls.txt
export PYTHONPATH=src

# 1. corpus: kgirl plus any of your repos (vendored copies are read once; nothing is modified)
python -m kgirl.harness soup export-sft runs/tauls/sft.jsonl                     # optional: verified Jev runs
python -m kgirl.tauls tokenizer --roots . ../numbskull ../LiMp --vocab 8192 --out runs/tauls/tokenizer.json
python -m kgirl.tauls data --roots . ../numbskull ../LiMp --tokenizer runs/tauls/tokenizer.json \
                           --out runs/tauls/data [--sft runs/tauls/sft.jsonl]

# 2. budget the card
python -m kgirl.tauls plan --preset small --batch 16 --vram 8

# 3. train (Ctrl-C saves; --resume continues exactly; --train-set stop_at=N for nightly sessions)
python -m kgirl.tauls train --data runs/tauls/data --out runs/tauls/small --preset small --auto-batch \
       --train-set steps=20000 eval_every=500 patience=6
python -m kgirl.tauls train --data runs/tauls/data --out runs/tauls/small --resume

# 4. use and inspect
python -m kgirl.tauls generate  --ckpt runs/tauls/small/best.pt --tokenizer runs/tauls/tokenizer.json --prompt "def "
python -m kgirl.tauls eval      --ckpt runs/tauls/small/best.pt --data runs/tauls/data
python -m kgirl.tauls stability --ckpt runs/tauls/small/best.pt          # Julia bridge if up, else numpy
```

`metrics.jsonl` records four event kinds:
- `train`: loss, LR, ‖g‖, κ and coherence state, TA-ULS internals (KFP fluctuation and weight, regulator entropy, stress and scale, α), tokens/s
- `eval`: loss, perplexity and bits per byte; bits/byte is tokenizer-independent, so use it to compare runs
- `stability`: a weight-spectrum summary
- `guard`: skipped steps

**VRAM (L1, measured slopes).** `plan` fits saved-activation bytes at T = 32, 64, 128 to `a + bT + cT²` and uses the slope `b`:

| Preset | Activations per token | Batch 16 total | Notes |
|---|---|---|---|
| `small` | 219.5 KiB | ≈ 3.0 GiB | fits 8 GB |
| `small-baseline` | 145.7 KiB | — | TA-ULS costs 1.5× the baseline's activations |
| `desk-8gb` | 573 KiB | ≈ 11 GiB | micro-batch 8, or `--grad-checkpoint` (≈ 3.1 GiB), or let `--auto-batch` decide |

**Throughput (L1 estimate).** At about 40% of a 15 TFLOP/s bf16 card, `small` costs about 1e8 FLOPs/token, so it should run at about 5e4 tokens/s: about 0.6B tokens in 3–4 hours. kgirl's 1.65M tokens are then many epochs, so watch `eval` for overfitting (`patience`) and add repositories to the corpus.

## 5. What has been measured (CPU, this container)

| Run (kgirl corpus, ctx 128, batch 16, lr 3e-3 → 3e-4) | Val loss | Bits/byte |
|---|---|---|
| `tiny` TA-ULS, 300 steps, 3 seeds | 5.83 / 5.86 / 5.90 | 2.34–2.37 |
| `tiny-baseline` (matched), 300 steps, 3 seeds | 5.55 / 5.55 / 5.66 | 2.23–2.27 |
| `tiny-baseline` + stablemax, 300 steps, 2 seeds | 6.68 / 6.72 | 2.68–2.70 |
| `tiny-hrm` 1×1 cycle, no injection norm, 150 steps | 6.41 (= flat baseline exactly) | 2.57 |
| `tiny-hrm` 2×2 cycles, one-step gradient, 150 steps | 7.02 | — |
| `tiny-hrm` 2×2 cycles, full backprop, 150 steps | 6.77 | — |

Reading (L2: 0.65M parameters, at most 300 steps, which is far from convergence):
- the TA-ULS paths do not pay for their parameters yet
- stablemax converges more slowly and runs 4–10× slower on CPU
- RMS-normalising the injected sums made the embeddings scale-free and hurt (an injection-norm variant, still at 1×1 cycle, scored 6.82), so `inject_norm` defaults to off
- recurrence did not help in 150 steps, and the one-step gradient costs accuracy against full backprop

None of this rules out TA-ULS or HRM at the scales they are meant for; it says what to test first.

## 6. Experiment plan for the desktop

A 2×2 at `small` scale: {TA-ULS blocks, plain blocks} × {flat, HRM 2×2}. That is `small`, `small-baseline`, `small-hrm-tauls`, `small-hrm`, at equal tokens, 3 chaos seeds each (`--set init_seed=… --train-set seed=…`).

- **Metric:** val bits/byte at a fixed token budget, plus tokens/s.
- **Falsification:** TA-ULS is not adding value if its bits/byte is not below the matched baseline by more than the seed spread at 0.3B tokens. The same test applies to HRM. Then ablate one component at a time (`--set kfp=false`, `entropy_reg=false`, `control_recurrence=false`, `two_level=false`) to find what carries any gain.
- **Watch:** `tauls.kfp_w`, `reg_scale_dev` and `alpha` in `metrics.jsonl`. If training leaves the KFP weight at 0 and the regulator scale at 1, the model is telling you it does not use them.
- **Hyperparameter search:** `kgirl.evolve` (chaos MAP-Elites) can search lr, `control_mult`, `kfp_rank` and cycles with short runs as fitness. Not wired yet.

## 7. Julia bridge

`src/julia/tauls_bridge.jl` serves `stability_analysis`, `kfp_profile` and `entropy_profile`. It uses the `{"function", "args"}` protocol of `src/julia/julia_server_script.jl` and of Fractal_cascade_simulation's `JuliaBackend`, and listens on port 9000.

```bash
julia --project=@tauls -e 'using Pkg; Pkg.add(["HTTP", "JSON3"])'
julia --project=@tauls src/julia/tauls_bridge.jl 9000 &
python -m kgirl.tauls stability --ckpt runs/tauls/small/best.pt --julia http://localhost:9000
# or let training use it: --train-set julia_url=http://localhost:9000
```

`kgirl/tauls/julia_bridge.py` holds the numpy reference with the same formulas. It is used whenever the server is down or a matrix exceeds `max_elems`.
- **Python side (L0):** the protocol is tested against an in-process server. A static test checks that the `.jl` file exposes the same functions and result keys.
- **Julia side (L1):** Julia could not be installed in this container (`julialang-s3` is blocked), so it is unexecuted. Check parity on first run by comparing one matrix with `--local`.

The original `julia_server_script.jl` is truncated (it ends mid-expression in `optimize_matrix`, and `start_http_server` is exported but never defined), so it cannot load. It is left as it is.

## 8. KGIRL: memory and recursive self-improvement

KGIRL stands for Kinetic Geometric Inverted Recursive Learning. It is a geometric holographic database built over the model's own hidden states. That database is then distilled back into the model, coarse to fine, and a round is kept only if held-out loss improves.

- **Database:** `src/kgirl/tauls/holo.py`
- **Loop:** `src/kgirl/tauls/improve.py`

```
hidden h_t ─chaos projection─▶ key k_t (centred, unit)        value: next token y_t
fractal cell tree: deterministic k-means, branching b, depth ≤ L
  a cell splits only if its children code their tokens in fewer bits than they cost (MDL/BIC)
  cell c stores n_c, token counts, and a holographic trace T̂_c = Σ F(k_t) ⊙ F(v_{y_t})
query: descend coarse → fine; p_c = (c_c + β p_parent)/(n_c + β) along the path (Dirichlet back-off)
       leaf vote: ⟨F⁻¹(conj F(q) ⊙ T̂_leaf), v_y⟩ ≈ Σ_{t in leaf, y_t = y} ⟨q, k_t⟩   (holographic unbinding)
mix:   w = σ(logit π + (H_LM − H_mem)/T)    the less uncertain source dominates (free-energy gate)
loop:  per round, deeper memory → fit (η, π, T) on val half A → distil on train windows disjoint from the
       database → keep the weights only if plain-model NLL on val half B drops
```

```bash
python -m kgirl.tauls memory  --ckpt runs/tauls/small/best.pt --data runs/tauls/data --out runs/tauls/memory \
                              --set max_depth=4 branching=8 rank=64
python -m kgirl.tauls generate --ckpt runs/tauls/small/best.pt --tokenizer runs/tauls/tokenizer.json \
                              --memory runs/tauls/memory --prompt "def "
python -m kgirl.tauls improve --ckpt runs/tauls/small/best.pt --data runs/tauls/data --out runs/tauls/kgirl \
                              --set rounds=4 distil_steps=1000 memory_tokens=4000000
# runs/tauls/kgirl/telemetry.json: per round, memory size, fractal dimension, gate, bits/token gain,
# accepted or not; improved.pt loads with eval / generate / info like any checkpoint
```

**L0, tests** (`tests/tauls/test_kgirl.py`):

- On six clusters whose tokens are uniform within each cluster:
  - MDL stops at exactly six leaves; without MDL the tree keeps splitting.
  - NLL reaches ln 5, the ideal value, within 0.05.
- When the token depends on a state's position on a circle inside a single cell:
  - Counts alone are uniform (ln 8).
  - The holographic vote lowers NLL by more than 0.2 nats.
- The free-energy gate weights the lower-entropy source.
- The loop never raises held-out loss: rejected rounds restore the weights.
- `improved.pt` reloads.
- Memory-augmented generation is reproducible.

## 9. Next

1. Run §6 on the desktop. Keep whichever architecture wins on bits/byte.
2. Add per-token ACT halting for HRM mode (HRM's Q-head, with chaos-driven exploration instead of `torch.rand`).
3. Serve the trained model as a Hermes backend (a `tauls` role), so Jev or scout can use it locally.
4. Run an evolve search over hyperparameters, with fitness = val bits/byte after N steps.
