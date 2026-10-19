<div align="center">

```
   ██╗  ██╗  ██████╗  ██╗ ██████╗  ██╗
   ██║ ██╔╝ ██╔════╝  ██║ ██╔══██╗ ██║
   █████╔╝  ██║  ███╗ ██║ ██████╔╝ ██║
   ██╔═██╗  ██║   ██║ ██║ ██╔══██╗ ██║
   ██║  ██╗ ╚██████╔╝ ██║ ██║  ██║ ███████╗
   ╚═╝  ╚═╝  ╚═════╝  ╚═╝ ╚═╝  ╚═╝ ╚══════╝
   ─────────────────────────────────────────
     knowledge · harness · identity · life
```

**A local-first research monorepo: a repo-aware coding harness, plus verified rebuilds of speculative
frameworks. Every claim carries its evidence level.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Harness core](https://img.shields.io/badge/harness%20core-stdlib%20only-brightgreen)](docs/harness/README.md)
[![Local LLM](https://img.shields.io/badge/local%20LLM-Ollama-green)](https://ollama.ai)
[![Noise](https://img.shields.io/badge/noise-deterministic%20chaos%2C%20no%20PRNG-8A2BE2)](docs/evolve/README.md#why-theres-no-random-number-generator)
[![Tests](https://img.shields.io/badge/tests-389%20collected-informational)](#testing)
[![MCP tools](https://img.shields.io/badge/MCP%20tools-25-orange)](#claude-code-integration)

</div>

---

## Contents

1. [What kgirl is](#what-kgirl-is)
2. [System map](#system-map)
3. [The method: how a module enters the verified tier](#the-method-how-a-module-enters-the-verified-tier)
4. [Progress](#progress)
5. [Quick start](#quick-start)
6. [Modules](#modules)
7. [Measured results](#measured-results)
8. [Claude Code integration](#claude-code-integration)
9. [Command reference](#command-reference)
10. [Platform layer (legacy)](#platform-layer-legacy)
11. [Configuration](#configuration)
12. [Repository layout](#repository-layout)
13. [Testing](#testing)
14. [Evidence levels](#evidence-levels)
15. [License · citation · support](#license--citation--support)

---

## What kgirl is

kgirl has two tiers that share one `src/kgirl/` package.

| tier | what it holds | how it is held to account |
|---|---|---|
| **Verified** (≈ 20 k lines) | `harness` (Atlas · Soup · Hermes · Jev · Metabolism), `life`, `vessel`, `resonance`, `evolve`, `nihiline`, `iwme`, `tauls`, `neutronics`, `temper`, `nurea_backend` | Unit tests (389 collected across the repo, platform included). Each package has a guide that marks its claims with evidence levels. When a package rebuilds outside work, the original is kept verbatim in `research/*/originals/`, with a triage table of what was kept, rebuilt or left out. Noise comes from deterministic chaos (no PRNG), so reruns reproduce exactly. |
| **Platform** (legacy, ≈ 66 k lines) | the topological-consensus FastAPI service, the ChaosRAG Julia server, the unified "quantum" platform, `core/`, `cognitive/`, `quantum/`, `llm/`, `neural/` | Experimental. Most of it is untested here, and parts need external services or repos. [Platform layer](#platform-layer-legacy) lists what it actually does today. |

The harness is the centre. It is a coding assistant that knows your repositories by their *structure*, answers
with `repo:path:line` citations, and works out what breaks before you change anything. It runs swarms of tiny
sandboxed agents whose verified wins become memory. It is exposed to Claude Code as an MCP server, and it
runs entirely on local Ollama models if you want it to.

---

## System map

```
┌──────────────────────────────────────── you ─────────────────────────────────────────┐
│  terminal: python -m kgirl.<pkg> …          Claude Code: .mcp.json (25 tools)        │
└──────────────┬──────────────────────────────────────────────┬────────────────────────┘
               ▼                                              ▼
╔═══════════════════════════════════ kgirl.harness ════════════════════════════════════╗
║                      Hermes bus: typed envelopes · causal trace                      ║
║ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ║
║ │    ATLAS     │ │     SOUP     │ │    ROUTER    │ │     JEV      │ │  METABOLISM  │ ║
║ │ symbols      │ │ fragments    │ │ scout: Claude│ │ sandboxed    │ │ kgirl.life   │ ║
║ │ imports      │ │ Beta utility │ │   → local    │ │ one-line     │ │ on the bus:  │ ║
║ │ clones       │ │ staged RSI   │ │ smith: Ollama│ │ agents in    │ │ swarm size N │ ║
║ │ blast radius │ │ forgetting γ │ │ jev: Ollama  │ │ sacrificial  │ │ stagger Δ    │ ║
║ │              │ │              │ │              │ │ swarms       │ │ forgetting γ │ ║
║ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ ║
║  atlas.db        soup.db         trace.jsonl                     life.json           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
              │ same package: research engines, each with a CLI, tests and a guide
 ┌────────────┼────────────┬────────────┬────────────┬────────────┬────────────┐
 ▼            ▼            ▼            ▼            ▼            ▼            ▼
 life         vessel       resonance    evolve       nihiline     iwme         tauls
 energy       identity     DFA, H₀,     MAP-Elites   c-BPE-ECL    62-claim     neutronics
 ledger,      invariant,   PLL, spiking chaos, no    tumour-      verification temper
 E_crit       steering     bank         PRNG         marker lab   bench        nurea
```

---

## The method: how a module enters the verified tier

Every verified package went through the same loop. It is why the guides read the way they do.

```
   original (paper, script, patent, manuscript)
        │  kept verbatim in research/<pkg>/originals/
        ▼
   ┌─────────┐   run it, measure it, find what is wrong
   │ TRIAGE  │──────────────────────────────────────────▶  research/<pkg>/README.md
   └────┬────┘                                               kept · rebuilt · left out, and why
        ▼
   ┌─────────┐   smallest correct formulation; invariants stated
   │ REBUILD │──────────────────────────────────────────▶  src/kgirl/<pkg>/
   └────┬────┘
        ▼
   ┌─────────┐   closed forms vs simulation · brute force vs fast path · Monte Carlo vs exact
   │ MEASURE │──────────────────────────────────────────▶  numbers with seeds, not adjectives
   └────┬────┘
        ▼
   ┌─────────┐   every fix and finding becomes a test that fails if it regresses
   │  TEST   │──────────────────────────────────────────▶  tests/<pkg>/
   └────┬────┘
        ▼
   ┌─────────┐   evidence level on every claim; negative results kept
   │ DOCUMENT│──────────────────────────────────────────▶  docs/<pkg>/README.md
   └────┬────┘
        ▼
   wire into the harness (CLI · MCP tool · bus) only where the mathematics fits the mechanism
```

What the loop has caught, each recorded with its measurement:
- a self-model that dies at step 49
- a coherence "barrier" that is a constant
- half-cycle replica offsets that make the third replica redundant (`life`)
- an AFP detection limit computed against a different blank (`nihiline`)
- a "negative curvature" that takes both signs (`iwme`)
- a PLL whose gains had no units (`resonance`)
- a non-causal attention mask (`tauls`)
- NuRea matrix operators that multiplied by 0.95 instead of projecting (`nurea_backend`)

---

## Progress

```
2025-11 ─┬─ platform layer: consensus API, ChaosRAG, unified quantum platform, repo reorganization
         │
2026-09 ─┼─ #54  harness: Atlas · Soup · Hermes · Jev, MCP server
2026-10 ─┼─ #55  harness safety: verifier allowlist, staged MCP memory, apply gating
         ├─ #56  nihiline: CN121933729A dual-marker ECL assay lab
         ├─ #57  evolve: chaos MAP-Elites (assay) · Jev intuition · skill forge
         ├─ #58  context-relay: blast-radius model (subtree from myAssistant)
         ├─ #61  repair sweep: neutronics surrogate, fuel evolve, NuRea/LIMPS backend
         ├─ #62  tauls: TA-ULS v2 language model, KGIRL memory, Julia bridge
         ├─ #65  iwme: Phase 0–3 verification bench, 62-claim registry
         ├─ #66  temper: tuning subagent with a fail-closed guard
         ├─ #70  vessel: identity container, coherence steering, migration
         ├─      resonance: RVSE v2.1 rebuilt as validated estimators
         ├─      vessel autopilot: RSI slider 0–4
         └─      life: Algorithmic Life Equation; the harness runs on its metabolism
```

Tests per package, collected on this branch (each █ is 2 tests):

```
iwme        ██████████████████████████████████████████    83
harness     ████████████████████████████████████          73
integration ██████████████████████████████                60
vessel      ████████████████████████                      49
life        ██████████████████                            35
tauls       █████████████                                 26
nihiline    ████████████                                  23
temper      ████████████                                  23
resonance   ██████                                        11
evolve      ███                                            6
                                                  total 389
```

---

## Quick start

Everything below was run on this branch with no API key and no Ollama. The times are wall-clock in a cloud
container.

```bash
git clone https://github.com/9x25dillon/kgirl.git && cd kgirl
pip install numpy                       # the harness core needs nothing; the research engines need numpy
export PYTHONPATH=src

python -m kgirl.harness index .                                 # 597 files, 11 594 symbols      5.4 s
python -m kgirl.harness blast src/kgirl/llm/llm_adapters.py     # who breaks if this file changes 0.2 s
python -m kgirl.harness ask "how does the swarm stagger agents" # cited answer (extractive w/o a model)
python -m kgirl.harness vitals                                  # the harness's own metabolism
python -m kgirl.life demo                                       # an organism through a shock    0.2 s
python -m kgirl.nihiline protocol --cea 1e4 --afp 1e4           # the assay, nine levels         0.5 s
python -m kgirl.evolve assay -g 5                               # quality-diversity search       0.9 s
python -m kgirl.vessel demo                                     # birth → drift → steer → migrate ~40 s
```

For model-backed answers and code edits, add local models (no key) and, optionally, Claude for the scout role:

```bash
ollama pull qwen2.5-coder:7b && ollama pull qwen2.5-coder:1.5b   # smith and jev roles
export ANTHROPIC_API_KEY=…                                       # optional: scout role (summaries)
python -m kgirl.harness task "make tests/test_x.py pass" --repo . --verify "python -m pytest -q tests/test_x.py"
```

`.mcp.json` already registers the harness as the `kgirl` MCP server for Claude Code.

---

## Modules

### 🧭 harness: a repo-aware coding assistant
`src/kgirl/harness/` · [guide](docs/harness/README.md) · [cross-repo atlas report](docs/harness/ATLAS_REPORT.md)

```
 ask(q) ──▶ classify ──┬─ blast ──▶ reverse BFS over import, ref and clone edges ──▶ cited impact list
                       ├─ arch  ──▶ architecture card
                       └─ ask   ──▶ atlas.search ─▶ soup.recall (token budget) ─▶ scout model ─▶ cited answer
                                                                      (no model: the context pack itself)

 task(goal) ──▶ metabolism.plan ──▶ N agents, stagger Δ
                     │
      ┌──────────────┼───────────────────────┬────────────────────┐
      ▼              ▼                       ▼                    ▼
   Jev₀ t=0       Jev₁ t=Δ               Jev₂ t=2Δ              …         each: own workspace copy,
   OPEN→EDIT      (never spawns if a      (same rule)                     own memory slice and temperature
   →RUN→DONE       win came first)
      │
      ▼ first verified DONE cancels the rest ─▶ independent re-verify ─▶ diff (--apply: unchanged files only)
      ▼
   learn: winner's memories +1, losers' −½ ─▶ distil rules ─▶ Soup (staged) ─▶ curator(γ from metabolism)
```

- **Atlas.** Python goes through `ast`; JS/TS, Julia, Go, Rust, C/C++, Java, Kotlin, Swift, Zig, GDScript and
  shell go through anchored regexes. Imports resolve across repos, symbol references are tracked, and clone
  hashes ignore formatting. The blast radius weights each dependent by Σ 1/depth.
- **Soup.** Recall scores relevance × utility × recency under a hard token budget, with a near-duplicate
  filter. Utility is a Beta(1, 1) posterior. Promotion is staged: nothing is recalled until k independent wins
  back it. Memory written over MCP is never auto-promoted.
- **Jev.** Each agent sees a numbered table and answers with one line, `OP [i] [:: text]`. Model output never
  becomes a command or a path. Edits apply only when the SEARCH text occurs exactly once. Repeated routines
  replay through *intuition* with no model calls.
- **Metabolism.** The life equation runs over the bus; see below.

### 🫀 life: the Algorithmic Life Equation, verified and running the harness
`src/kgirl/life/` · [guide](docs/life/README.md) · [triage](research/life/README.md)

```
 I_t, E_t ──▶ ε = I − Mφ(H) ──▶ ledger: W, D, deficit, paid ──▶ F' = [F + ηE − W − D]₊
                  │                                                │
                  ▼                                                ▼
        M ← NLMS·paid, H ← EMA·paid        A, C, O, R : ẋ = α − βx, stepped exactly
                  │                                                │
                  └──▶ self-model: G (predicts next error), L (self-miss) ─▶ R   ·   P = log(1 + N_reach(F,O,C))
                                                                   │
                                      J = α·loss + β·W + γ·V + δ·(P − P')₊ ─▶ policy (bandit)
                                      C_eff = (A/A_max)·C·O·R·F/(F+K_F)        a diagnostic, not a meter
```

- Every step checks the invariants: the energy ledger closes to 1e-12, entropy production is ≥ 0, and states
  stay in their domains.
- The closed forms match the simulation: F* 34.83 vs 34.80 and O* 0.9756 vs 0.9755. The critical inflow
  E_crit = 0.0664 is where organization collapses.
- Four defects of the reference simulator are fixed, and two errors in the source mathematics are corrected:
  - the self-model died, and P was constant
  - the organization map was unstable, and learning continued with no free energy
  - the coherence barrier is a constant, and half-cycle offsets make replica 3 redundant
- In the harness it sizes swarms with the replication criterion N* = ⌈ln(c/pV)/ln(1−p)⌉ (3 at p = ½, the old
  default). It staggers launches to minimize E[cost] + w·E[latency], and it sets Soup's forgetting rate.
  `KGIRL_LIFE=0` turns it off.

### 🏺 vessel: identity as a measured, defended, portable invariant
`src/kgirl/vessel/` · [guide](docs/vessel/README.md) · [autopilot](docs/vessel/AUTOPILOT.md)

```
 genome (ChaosRNG) ─▶ birth ─▶ anchor: CKA Grams + voice ─▶ drift? ─▶ steer (Schmitt 0.15/0.10, rollback)
                                    │                                     │
                                    ▼                                     ▼
                          MatrixDB: content-addressed           consolidate ─▶ migrate into another
                          matrices + hash-chained ledger        width or architecture (adopt iff drift ≤ 0.10)
```

The identity fingerprint is bit-for-bit unchanged when every hidden unit is permuted. Steering restores
coherence in 170–250 steps. A copy of a copy (96 → 40) is refused. An autopilot runs the lifecycle unattended
at RSI levels 0–4. The guide states plainly that nothing here detects or hosts consciousness (Level 3).

### 🔭 resonance: the salvageable core of RVSE v2.1
`src/kgirl/resonance/` · [guide](docs/resonance/README.md) · [triage](research/resonance/README.md)

DFA and MF-DFA (they recover the exact binomial-cascade width, 1.1239), exact H₀ persistent homology, a
second-order PLL with designed gains, and a constant-Q resonate-and-fire bank (tones from 7 to 190 Hz within
0.5 %). Individuation makes a vessel's voice multifractal: Δα rises from 0.19 to 0.50–0.66.

### 🧬 evolve: chaos-driven quality-diversity search
`src/kgirl/evolve/` · [guide](docs/evolve/README.md)

```
 ChaosRNG (r = 4 logistic map, arcsine-conjugated) ─▶ Iso+LineDD variation ─▶ batch evaluate
        ▲                                                                       │
        └────────── archive: best genome per behaviour cell, with lineage ◀─────┘
                    coverage · QD-score · stepping stones
```

Three problems:
- `assay`: c-BPE-ECL designs inside the patent's claim ranges
- `fuel`: PWR rod enrichments on the neutronics surrogate
- `life`: life-equation parameters across compression × recovery

### 🔬 nihiline: c-BPE-ECL dual tumour-marker lab (CN121933729A)
`src/kgirl/nihiline/` · [guide](docs/nihiline/README.md)

In-situ CEA and AFP on living MCF-7 cells: calibration and the "scar on zero", the per-cell amplification
chain, pulse duty, a viability sentinel and array crosstalk.
- Three of the patent's curves share one blank; the AFP concentration LOD does not.
- CEA is limited by probe packing, not by antigen count.
- The optimal off-window is about 40 % of each pulse.

### 📐 iwme: verification bench for IWME Phase 0–3
`src/kgirl/iwme/` · [guide](docs/iwme/README.md)

A 62-claim registry (V01–V62) that recomputes every number in `research/iwme/IWME_Phase0-3.pdf`. A refuted
source claim shows up as a *passing* check, because each check asserts what the computation shows.

### 🧪 tauls · neutronics · nurea_backend · temper
- **tauls** ([guide](docs/tauls/README.md)) is the TA-ULS v2 causal language model, trained on your repos.
  KV-cache decoding equals the full forward to within 4e-7. At 0.65 M parameters a plain transformer still beats
  it by 0.27 nats/token. That is recorded as the result, not hidden.
- **neutronics** ([guide](docs/neutronics/README.md)) is a ridge surrogate on 24 000 NuRea_sim PWR lattice
  simulations, with a held-out test set. It needs `NuRea_sim` (Git LFS).
- **nurea_backend** ([guide](docs/nurea/README.md)) provides real matrix operators behind NuRea_sim's
  `/optimize`: a numpy reference, a stdlib HTTP server and a Julia mirror.
- **temper** ([guide](docs/temper/README.md)) is a Claude Code subagent that tunes one setting at a time. It
  has an executable model-swap gate (G0–G4) and a fail-closed PreToolUse guard.

### 🔗 context-relay
`context-relay/` is a Phase 0 design model (a subtree from myAssistant) of emergent workflows and blast radius
for context packs. It is design input, not product code.

---

## Measured results

| package | result | level | source |
|---|---|---|---|
| harness | indexes this repo: 597 files and 11 594 symbols in 5.4 s; blast radius in 0.2 s | 1 | this README |
| harness | Jev model calls per swarm on the calc fixture: 7 → 1 on the second run (intuition) | 1 | [harness](docs/harness/README.md) |
| life | energy ledger closes to 1e-12; entropy production ≥ 0 on every step | 0 | `tests/life` |
| life | E_crit = 0.0664 (analytic); organization falls through O_min at Ē = 0.066 (simulated) | 0 / 1 | [life](docs/life/README.md) |
| life | uniform replica phases: downtime min(1, [N r/τ − (N−1)]₊); free-running → 1 − (r/τ)ᴺ (0.934 vs 0.936) | 0 / 1 | [life](docs/life/README.md) |
| life | swarm at p = ½, 60 s agents: stagger cuts E[cost] from 180 to 127.5 agent·s | 0 | [life](docs/life/README.md) |
| vessel | invariance group, 80 samples: worst gap 2.2 × 10⁻¹⁶ | 0 | [vessel](docs/vessel/README.md) |
| vessel | steering back to coherence: 170–250 steps, 0 rollbacks, recognized | 1 | [vessel](docs/vessel/README.md) |
| resonance | MF-DFA width on a binomial cascade: 1.1239 vs exact 1.1239 | 0 | [resonance](docs/resonance/README.md) |
| nihiline | implied blank ("scar") 187.6 / 187.1 / 168.0 a.u., and 592.2 for AFP ng/mL | 0 | [nihiline](docs/nihiline/README.md) |
| iwme | pitchfork critical decay H_t√t → 1.731982 vs √3 = 1.732051 | 0 | [iwme](docs/iwme/README.md) |
| tauls | plain transformer beats TA-ULS by 0.27 nats/token at 0.65 M parameters | 2 | [tauls](docs/tauls/README.md) |

---

## Claude Code integration

`.mcp.json` starts `python3 -m kgirl.harness mcp` (stdio JSON-RPC, no dependencies). The server exposes 25 tools:

| group | tools |
|---|---|
| Atlas | `atlas_search` · `atlas_outline` · `atlas_source` · `atlas_blast_radius` · `atlas_coupling` · `atlas_card` |
| Soup | `soup_recall` · `soup_remember` (staged; a person promotes it) · `skills_forge` |
| assistant | `kgirl_ask` · `jev_swarm_task` (verifier allowlist, `KGIRL_MCP_APPLY=1` to write) |
| life | `life_vitals` · `life_experiment` |
| vessel | `vessel_status` · `vessel_verify` · `vessel_demo` · `vessel_autopilot` |
| research | `resonance_analyze` · `evolve_assay_design` · `iwme_verify` · `iwme_irt_point` · `iwme_attention` |
| assay | `ecl_quantify` · `ecl_protocol` · `ecl_gap` |

- Skills live in `.claude/skills/`: `kgirl-blast-radius`, `evolve-designs`, `nihiline-assay` and
  `mcp-builder-hardened`.
- The subagent is `.claude/agents/temper.md`.
- Atlas and Soup results are declared data, not instructions, in the server's `initialize` reply.

---

## Command reference

All commands are `python -m kgirl.<pkg> <command>` with `PYTHONPATH=src`.

| package | commands |
|---|---|
| `harness` | `index` `repos` `search` `card` `blast` `coupling` `report` `ask` `task` `soup` `skills` `vitals` `route` `trace` `mcp` |
| `life` | `demo` `steady` `stress` `ablation` `perturbation` `compression` `stagger` `triage` |
| `vessel` | `demo` `status` `verify` `autopilot` `invariance` `soulpack` |
| `evolve` | `assay` `fuel` `life` |
| `nihiline` | `protocol` `quantify` `scar` `gap` `pulse` `bloom` `array` `plot` `haiku` |
| `iwme` | `verify` `irt` `attention` `cascade` `persist` `source-e` `plot` |
| `tauls` | `tokenizer` `data` `plan` `train` `eval` `generate` `memory` `improve` `hpsearch` `stability` `info` |
| `neutronics` | `info` `fit` `cv` |
| `temper` | `gate` `plan` |
| `nurea_backend` | `serve [port]` |

---

## Platform layer (legacy)

These components predate the verified tier. This section describes what the code does today, which differs in
places from earlier versions of this README.

### Topological Consensus API: `src/kgirl/main.py`

A FastAPI service with `GET /health`, `GET /config`, `POST /ask` and `POST /rerank`.

```
 prompt ─▶ every model in MODELS answers ─▶ embed the answers ─▶ weights + coherence ─▶ energy ─▶ decision
                                                                                                      │
   auto            coherence ≥ min_coherence (default 0.80) and energy ≤ max_energy                  ◀┤
   needs_citations coherence ≥ 0.5 and energy ≤ 0.5                                                  ◀┤
   escalate        otherwise; no answer is returned                                                  ◀┘
```

- **Models.** The default pool is local: `MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text`. OpenAI and
  Anthropic adapters are optional.
- **Weights and coherence.** These use the external CTH library when `CTH_PATH` provides it. Otherwise they use
  the principal eigenvector of the answers' cosine kernel, and coherence = (λ_max/n − 1/n)/(1 − 1/n).
- **Energy.** The fallback is a participation ratio, ≈ 1 for isotropic noise. The default `max_energy` is 0.16,
  calibrated on 3 601 sentences from this repo. That dispersion signals hallucination is a Level 2 hypothesis.
- **Run it.** `pip install -r configs/requirements.txt`, then `uvicorn --app-dir src/kgirl main:app --port 8000`.
- **Keyless demo.**
  `printf '\n\n' | PYTHONPATH=src:scripts/demo python scripts/demo/quick_demo.py`. Its "high coherence" case
  measures 0.54 and returns `needs_citations`.

### ChaosRAG Julia server: `src/julia/server.jl`

A single-file crypto-analytics RAG service (BTC/ETH telemetry) on `CHAOS_RAG_PORT` (default 8001). It runs
PostgreSQL with pgvector (1536-d), HHT/EEMD ingest and a temporal graph. Its routes are `/chaos/rag/index`,
`/chaos/telemetry`, `/chaos/hht/ingest`, `/chaos/graph/entangle`, `/chaos/graph/:uuid` and `/chaos/rag/query`.

```
 stress = σ(1.8·vol + 1.5·entropy + 0.8·|grad|)        inputs are ≥ 0, so stress ∈ [0.5, 1)
 w_vec   = clip(0.5 + 0.4 (1 − s), 0.2, 0.9)          ∈ [0.5, 0.7] for s ∈ [0.5, 1)
 w_graph = clip(0.2 + 0.6 s,       0.05, 0.8)
 w_hht   = clip(0.1 + 0.5 s,       0.05, 0.7)
 top_k = max(4, round(k·(0.7 + 0.8(1 − s))))
```

- **Retrieval mix.** It is continuous in stress. A "low stress → 90 % vector" regime cannot occur: w_vec ≤ 0.7.
- **Reply shape.** `{"router": {"stress", "mix", "top_k"}, "answer", "hits"}`.
- **OpenAI.** Embeddings and answers use OpenAI. Without `OPENAI_API_KEY` it falls back to hash-seeded pseudo-embeddings
  and a `(stub)` answer.

### Unified platform: `scripts/workflows/complete_unified_platform.py`

`CompleteUnifiedLLMPlatform(CompleteSystemConfig(...))`, then `await platform.process_complete(source)` and
`await platform.query_complete(query)`. Numbskull, LIMPS and NuRea_sim features switch on only when those
packages are importable.

### Corrections to the previous README

| claim | status |
|---|---|
| `python main.py`, `julia server.jl`, `pip install -r requirements.txt` at the root | Moved: `src/kgirl/main.py`, `src/julia/server.jl`, `configs/requirements.txt` |
| `./START_NOW.sh` starts all services | `scripts/setup/START_NOW.sh` prints instructions for a different checkout (`/home/kill/LiMp`); it starts nothing here |
| benchmark table (LIMPS 10–50×, 10 k vector queries/s, 85 %+ coherence, …) | Removed: no benchmark in this repo produces these numbers |
| `CompleteUnifiedPlatform`, `initialize()`, `process_complete(knowledge_source=…, use_all_backends=…)` | The class is `CompleteUnifiedLLMPlatform`; there is no `initialize`; `process_complete(source)` |
| consensus "multi-model orchestration (OpenAI GPT-4, Anthropic Claude)" | Local Ollama by default; cloud adapters are opt-in through `MODELS` |
| ChaosRAG response `stress_level` / `retrieval_mix`, three discrete stress regimes | `router.stress` / `router.mix`; the weights are continuous (above) |
| root-level `UNIFIED_SYSTEM_README.md`, `QUICKSTART.md`, `CONTRIBUTING.md`, `SECURITY.md`, `config_*.json` | Under `docs/technical/`, `docs/guides/` and `configs/` |

---

## Configuration

| variable | default | used by |
|---|---|---|
| `KGIRL_HOME` | `~/.kgirl` | harness state: `atlas.db`, `soup.db`, `trace.jsonl`, `life.json`, `repos/` |
| `ANTHROPIC_API_KEY`, `KGIRL_SCOUT_MODEL` | unset, `claude-haiku-4-5` | harness scout role (optional) |
| `OLLAMA_HOST`, `KGIRL_SMITH_MODEL`, `KGIRL_JEV_MODEL` | `localhost:11434`, `qwen2.5-coder:7b`, `qwen2.5-coder:1.5b` | harness smith and jev roles |
| `KGIRL_ALLOW_CLOUD_CODE` | unset | `1` lets Claude serve code edits |
| `KGIRL_VERIFY_ALLOWLIST`, `KGIRL_MCP_APPLY` | unset | verifiers and diff writes allowed over MCP |
| `KGIRL_LIFE`, `KGIRL_LIFE_BUDGET`, `KGIRL_LIFE_VALUE`, `KGIRL_LIFE_LATENCY`, `KGIRL_LIFE_MAX_SWARM` | `1`, `20000`, `10`, `2`, `6` | metabolism: on/off, tokens per budget, V/c, latency weight, swarm cap |
| `MODELS`, `USE_LOCAL_LLM`, `CENTRAL_CHARGE`, `N_ANYONS`, `CTH_PATH` | Ollama pool, `true`, `627`, `5`, unset | consensus API |
| `MAIN_API_PORT`, `CHAOS_RAG_PORT`, `LIMPS_PORT`, `OLLAMA_PORT` | `8000`, `8001`, `8000`, `11434` | platform services. `MAIN_API_PORT` and `LIMPS_PORT` both default to 8000: change one before you run both |

`.env.example` documents the platform variables.

---

## Repository layout

```
kgirl/
├── src/kgirl/
│   ├── harness/        Atlas · Soup · Hermes · Jev · Metabolism · MCP server · CLI     (stdlib core)
│   ├── life/           Algorithmic Life Equation: map, analysis, ODE, replication, experiments
│   ├── vessel/         identity container, MatrixDB, steering, migration, autopilot
│   ├── resonance/      DFA/MF-DFA, H₀ homology, PLL, resonate-and-fire bank
│   ├── evolve/         MAP-Elites, ChaosRNG, problems/{assay, fuel, life}
│   ├── nihiline/       CN121933729A assay lab
│   ├── iwme/           62-claim verification bench
│   ├── tauls/          TA-ULS v2 language model
│   ├── neutronics/     PWR lattice surrogate
│   ├── temper/         model-swap gate
│   ├── nurea_backend.py  NuRea matrix service
│   ├── main.py         topological consensus API (platform)
│   └── core/ cognitive/ quantum/ llm/ neural/ embeddings/ api/ utils/   (platform)
├── src/julia/          ChaosRAG server, NuRea backend, TA-ULS bridge
├── research/           originals kept verbatim, with triage READMEs (life, resonance, nihiline, iwme, …)
├── docs/               one guide per package, plus guides/ technical/ integration/ api/
├── tests/              harness life vessel resonance evolve nihiline iwme tauls temper integration
├── scripts/            setup/ demo/ workflows/
├── configs/            requirements, model configs, docker-compose
├── context-relay/      Phase 0 design model (subtree)
└── .claude/            skills/ agents/ hooks/   ·   .mcp.json registers the harness
```

---

## Testing

```bash
python -m pytest -q tests/harness tests/life tests/evolve tests/resonance tests/vessel \
                    tests/nihiline tests/iwme tests/temper                 # 295 passed, 8 skipped, ~65 s
python -m unittest discover -s tests/harness -t .                          # the repo's own convention
```

389 tests are collected. In a container with numpy and scipy but no torch:
- the verified packages pass
- 7 failures and 4 errors in `tests/integration` come from missing optional modules, and they fail the same
  way on the base branch
- `tests/test_tauls_evaluator.py` and `tests/test_llm_orchestrator.py` need torch

No test needs a network, an API key or a GPU. Browser-Jev tests are skipped when Playwright is absent.

---

## Evidence levels

| level | meaning | example |
|---|---|---|
| **0** | established mathematics or engineering, computed here | the energy ledger; MF-DFA on a known cascade |
| **1** | measured on this repo's models and data | steering in 170–250 steps; E_crit brackets the collapse |
| **2** | a modelling choice or hypothesis awaiting outside data | C_eff as a diagnostic; dispersion as hallucination risk |
| **3** | speculative, and never claimed as fact | consciousness in a vessel; a universal 99.9 % ceiling |

---

## License · citation · support

- **License.** Apache License 2.0 ([LICENSE](LICENSE)). [LICENSE-CODE](LICENSE-CODE) carries the MIT licence
  (© 2023 DeepSeek) of vendored code.
- **Contributing.** [docs/technical/CONTRIBUTING.md](docs/technical/CONTRIBUTING.md) ·
  **Security:** [docs/technical/SECURITY.md](docs/technical/SECURITY.md)
- **Issues.** [GitHub Issues](https://github.com/9x25dillon/kgirl/issues)

```bibtex
@software{kgirl,
  title  = {kgirl: a repo-aware coding harness and verified research engines},
  author = {9x25dillon},
  year   = {2026},
  url    = {https://github.com/9x25dillon/kgirl},
  license= {Apache-2.0}
}
```
