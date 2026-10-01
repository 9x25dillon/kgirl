# HANDOFF — kgirl harness, nihiline assay lab, evolve

_Session window: 2026-09-28 → 2026-10-01. Branch: `claude/dazzling-sagan-0uf4y7`. Written for whoever (human or agent) picks this up next._

## 1. State at handoff

| item | state |
|---|---|
| PR [#54](https://github.com/9x25dillon/kgirl/pull/54) — harness (Atlas · Soup · Hermes · Jev) | **merged** (`8e9f421`) |
| PR [#56](https://github.com/9x25dillon/kgirl/pull/56) — nihiline (CN121933729A assay lab) | **merged** (`e04e463`) |
| commit `80f8b85` — evolve, Jev intuition, skill forge, `.claude/skills` | **pushed, not merged, no PR yet** |
| this file | on the same branch as `80f8b85` |
| tests | harness 43 · nihiline 23 · evolve 6 = **72**, all passing; `pyflakes` clean |
| MCP server (`.mcp.json`, server name `kgirl`) | 15 tools, connected in Claude Code |
| `~/.kgirl/atlas.db` (the index the MCP tools read) | **empty**: index again before relying on the `atlas_*` tools |
| live model runs (Ollama / Claude) | **never done**: every model call was tested with `ScriptedBackend` |

## 2. What was built

### 2.1 Harness — `src/kgirl/harness/` (PR #54)

- **Atlas.** A structural index across repos. It stores symbols, signatures, imports, refs and clone hashes; it never stores file bodies.
  - It resolves bare-module imports across repos, computes blast radius over file, symbol and clone edges, ranks repo coupling, and builds architecture cards.
  - Ingest is incremental and keyed by `PARSER_VERSION ‖ sha1`.
  - It indexed 33 repos and about 34k symbols in roughly 15 s.
- **Soup.** A memory pool with just-in-time recall under a token budget, Beta-posterior credit assignment, and a staged curator. The curator uses a support gate, LCB/UCB bounds, decay, a capacity cap, and a ledger with rollback.
- **Hermes.** An envelope bus with a causal trace, plus a role router:
  - `scout` goes to Claude Haiku 4.5.
  - `smith` and `jev` go to Ollama.
  - Each role has a fallback chain.
- **Jev.** Meeseeks agents acting on a numbered action table:
  - Decisions are one strictly parsed line.
  - An edit only applies if its SEARCH text occurs exactly once.
  - Agents work in sandbox copies and run as sacrificial swarms.
  - Winners are verified independently, then distilled into Soup.
  - There is a code world and a Playwright browser world.
- **Interfaces.** A CLI (`python -m kgirl.harness …`), a stdio MCP server, and `docs/harness/ATLAS_REPORT.md`, which reports kgirl's coupling with 32 neighbour repos.

### 2.2 nihiline — `src/kgirl/nihiline/` (PR #56)

The user's lab scripts, re-aimed at **CN121933729A**: CEA/AFP detection on living MCF-7 cells with closed bipolar electrode (c-BPE) ECL. The originals are kept byte for byte in `research/nihiline/originals/`.

| module | job |
|---|---|
| `patent` | the patent as data |
| `calibration` | quantification and the scar on zero (blank + 3σ) |
| `amplification` | per-cell antigen → photon chain and the LOD gap |
| `protocol` | the transcript's nine levels run as the assay |
| `pulse` | the optimal off ("nihil") duty of the drive |
| `bloom` | viability sentinel |
| `array` | BPE lattice: crosstalk, uniformity, defects |
| `plots` | figures |

**Findings.**
- One shared blank lies behind three of the patent's four curves. The AFP concentration curve is the outlier at 592 a.u., and the printed AFP cell slope "44703 4473" must be 4473.
- CEA is limited by probe packing; AFP is limited by antigen count.
- The reported LODs need only about 2–3 × 10⁻⁴ of the bound MB to be electroactive.
- The optimal off-fraction is about 0.4.
- Phase variance warns 53–330 s before intensity collapses.
- A real bug in the original `phase_topology_lab.py` was fixed: its winding-number sum always telescoped to 0.

### 2.3 Unmerged (`80f8b85`)

- **`kgirl.evolve`.** MAP-Elites driven by ChaosRNG, which maps the r = 4 logistic map onto U(0,1) through arcsin√x. There is no PRNG anywhere. Lineage is tracked, along with coverage, QD-score and stepping stones.
  - The first problem evolves c-BPE designs inside the claim ranges.
  - The patent default under continuous drive scores 0. The champion scores about 0.35: TPA 70 mM, Ru 2 mM, 5 V, an off-fraction of about 0.47, 0.5 s pulses, about 93 % viable.
- **Jev intuition (System 1).** Replays routines, keyed by line-free targets, from accepted runs without calling a model. A failed intuitive step switches the agent back to the model ("surprise").
  - Model calls dropped from 7 to 1 on a repeated task.
  - Only half the swarm uses intuition, so the rest keeps exploring.
- **Skill forge.** Turns verified routines into Soup skills and Claude Code `SKILL.md` exports. It never overwrites hand-written skills.
- **`.claude/skills/`.** Three hand-written skills: `kgirl-blast-radius`, `nihiline-assay`, `evolve-designs`.

## 3. User preferences learned (binding)

1. **Never delete or "clean" the user's copies, vendored duplicates or originals.** The root clutter is intentional: venv shims, git internals, clones of numbskull. Leave it for future agents.
2. **Noise and randomness come from deterministic chaos (logistic maps), never `random` or `np.random`.**
3. **Model routing.** Claude handles retrieval and summaries; local models handle code edits.
4. **Keep building.** The user wants features, not proofs or experiments, unless they ask. They value emergent and evolutionary mechanisms and their own vocabulary: nihil, scar on zero, Bloom, Meeseeks.
5. **PR flow.** The user explicitly asks for PRs and merges. Do not open or merge on your own initiative.
6. **Communication.** State confidence levels (0–3) on scientific claims. Never present a Level 2 model output as a measured result.

## 4. Critique — three places I could have operated more efficiently

1. **Repo discovery before asking.** "Jev", "Hermes" and "soup" did not appear in kgirl.
   - I then cloned about 30 repos (583 MB in the scratchpad, including two over 150 MB) and grepped them in three rounds before asking the user what the names meant.
   - One clarifying question after the first miss would have saved about ten minutes and most of that disk.
   - The user's later reply ("I just need their architecture and blast radius") showed full clones were never needed. Metadata or shallow sparse checkouts would have done.
2. **Writing test assertions before looking at real output.** Several red-green cycles came from wrong expectations, not wrong code:
   - the dilution flag at 100 cells/mL
   - the vortex sign convention
   - the test decider matching `test_calc.py` instead of `calc.py`
   - the seeded table not containing a bare `calc.py` row
   - a fixture that dedented badly and broke clone hashing
   
   Running the function once and *then* pinning behaviour in a test would have removed four or five extra test runs.
3. **Indexing into a scratch `KGIRL_HOME`.**
   - The Atlas was built twice in scratch homes (`kh`, `kh2`), so the MCP server, which reads `~/.kgirl`, started empty and still is.
   - Each `ingest` also calls `resolve_imports()` over *all* imports, which is O(repos × imports). It should resolve once after a batch.
   - Indexing the default home, with one deferred resolve, would have made the MCP tools useful immediately and the run faster.

## 5. Critique — three things I could have improved

1. **No live model validation.**
   - Jev's prompt and decision grammar, the smith SEARCH/REPLACE format, distillation, and Claude scout answers have only met scripted backends.
   - Small local models (1.5B–7B) are exactly where format drift happens.
   - The "7 → 1 model calls" intuition result comes from a scripted agent, not a real one.
2. **Level 2 assumptions drive the headline numbers.** Several conclusions rest on response curves I chose, not on data:
   - the evolved champion
   - the optimal off-fraction
   - "probe at 25 µg/mL"
   
   Those curves are fouling and healing rates, V² stress, background ∝ probe, binding saturation and ECL yield. I documented the levels but added no **robustness check**: re-scoring the elites under perturbed or alternative assumptions to see which designs stay elite. The fitness function also bakes in a product of yield, reproducibility, viability and sensitivity whose weights are a choice.
3. **Engineering hygiene gaps.**
   - No CI workflow, so the 72 tests never run on PRs. kgirl's only workflows are Docker publishing and a stale-issue bot.
   - Dependencies are undeclared: numpy and matplotlib for nihiline/evolve; anthropic and playwright as optional extras.
   - PRs #54 and #56 were merged without review.
   - The harness MCP server now imports nihiline and evolve lazily. That works, but it couples the generic harness to one domain. A plugin registry for MCP tools would scale better.

## 6. Next session — three strategies

1. **Boot sequence (first five minutes).**
   ```bash
   git fetch origin && git log --oneline -3 origin/main          # was 80f8b85 merged? if not, ask whether to PR it
   PYTHONPATH=src python -m kgirl.harness index . 9x25dillon/numbskull 9x25dillon/Eopiez 9x25dillon/LiMp
   for s in harness nihiline evolve; do python -m unittest discover -s tests/$s -t .; done
   PYTHONPATH=src python -m kgirl.harness route                  # which backends are actually up
   ```
   - Index into the **default** home so the `atlas_*` MCP tools work.
   - Read `.claude/skills/*` and sections 3–5 of this file before touching anything.
2. **Close the live-model loop before adding features.**
   - With Ollama (`qwen2.5-coder:7b` and `:1.5b`) and an Anthropic key, run one real Jev swarm on a small failing kgirl test.
   - Record success, model calls, intuition steps and surprises, then run `skills forge --export .claude/skills`.
   - Fix whatever format drift shows up in `parse_decision` or the smith block parser.
   - Only then tune `intuition_threshold` and the curator policy, ideally by evolving them with `kgirl.evolve`, which closes the self-improvement loop.
3. **Harden what exists, in small reviewed PRs.**
   - (a) A GitHub Actions workflow running the three suites plus pyflakes.
   - (b) Declared dependencies (`requirements-harness.txt` or `pyproject` extras).
   - (c) A robustness ensemble in `evolve/problems/assay.py`: score each elite under N chaos-perturbed assumption sets and report designs that stay in the top decile.
   - (d) Defer `resolve_imports` to once per `index` batch.
   - (e) An MCP tool registry so domain packages register their own tools.
   - Keep each PR focused, and let the user decide merges.

## 7. Map of the repo additions

```
src/kgirl/harness/      atlas/ soup/ hermes/ llm/ jev/ (agent, intuition, swarm, code_env, browser_env) skills.py mcp_server.py cli.py report.py
src/kgirl/nihiline/     patent calibration amplification protocol pulse bloom array plots __main__
src/kgirl/evolve/       chaos space mapelites pareto plots problems/assay __main__
research/nihiline/      originals/ (verbatim uploads) + README (provenance, patent data)
docs/harness/ docs/nihiline/ docs/evolve/   guides, figures, ATLAS_REPORT.md
.claude/skills/         kgirl-blast-radius, nihiline-assay, evolve-designs
tests/harness/ tests/nihiline/ tests/evolve/
.mcp.json               registers the kgirl MCP server for Claude Code
```
