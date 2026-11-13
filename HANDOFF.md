# Hand-off — kgirl

## Resume here
**Newest close:** 13 November 2026, 09:20 PST · Claude Code, Opus 5.5 · session `e0adf913`
**Where things stand:** `main` is at `6802f8c` with 0 open PRs and CI green. Today merged #85–#89.
- **/ask:** works end to end on local Ollama, with retrieval from the Julia ChaosRAG service (Postgres :5433). It answers instead of always escalating: the energy cutoff now calibrates to the serving embedder (#89). Answers are never token-capped (#89).
- **Correction to the 19 Oct hand-off:** `80f8b85` is in main. The Atlas home is now `~/.kgirl-sync`, not `~/.kgirl`.
- **Correction to the 11 Nov `~/Hand_off.md`:** kgirl had 3 more merged PRs (#83, #84 research) before today.

**Decision needed:**
1. Do the ~38 uncommitted April files in `~/Synaps` belong in the repo? They include `src/kgirl/blockchain/`, `metering/`, `coord/` and `quarantine/`.
2. May the superseded local branches be deleted? These are `local/ask-gate-fix` and `feat/qincrs-soup-validator`.

**Start with:**
1. `cd ~/Synaps && git fetch && git branch -f main origin/main`: syncs local main (main is not checked out anywhere).
2. Bring the stack up with the commands under "Running the stack", then ask `/ask` something with `"use_rag":true`.
3. Check in with session `kgirl-f2` (cwd `~/kgirl`), which was doing harness work. See "In flight".

## Progress track (newest first)
| Date | Session | Shipped (hash / PR) | State after |
|---|---|---|---|
| 2026-11-13 | Opus 5.5 `e0adf913` | #89 `6802f8c` /ask energy cutoff calibrated per embedder; no token cap; Ollama num_ctx 8192 | /ask answers on nomic-embed-text (auto ≤ 0.948) |
| 2026-11-13 | Opus 5.5 `e0adf913` | #88 `7318a9f` ignore `/.pgdata/` `/.cthroot/` | local Postgres can't be staged by `git add -A` |
| 2026-11-13 | Opus 5.5 `e0adf913` | #87 ChaosRAG over Ollama behind /ask (/index /query /telemetry, use_rag, concurrent generation); Julia `IPAddr` startup crash fixed | full stack runs locally |
| 2026-11-13 | Opus 5.5 `e0adf913` | #86 `sydv.py` parses (paste moved to `docs/archive/`) | every `src/kgirl` file listed in quarantine parses |
| 2026-11-13 | Opus 5.5 `e0adf913` | #85 efl_iwme_cascade test fixed for Py 3.14; research CI job | CI: `harness` + `research-efl-iwme-cascade` |
| 2026-11-12/13 | Claude (cloud) | #83, #84 research/efl_iwme_cascade preregistered runs | 10/11 hypotheses replicate; A5 doesn't (RESULTS.md §4) |
| 2026-11-11 | Opus 5.5 `7304eed2` | #82 untracks 75 root junk files; 20 merged branches deleted | `main` only |
| 2026-11-03 | Claude (porter) | #80 embedding cache, #81 drill tests | stronger default route not shipped |
| 2026-10-19 | Claude | `kgirl.life` + metabolism; this file's previous version | archived: `docs/handoff/HANDOFF-2026-10-19-claude-opus.md` |

## In flight
- **Harness setup "at full capability"** by peer session `kgirl-f2` (cwd `~/kgirl`, maybe worktree `~/work/kgirl-run`): Atlas refresh, hybrid retrieval, the MCP server and a launcher. Not this session's work; its state is unknown. Ask it, or check `porter_brief`.
- **Parked on purpose:** `feat/stronger-default-route` in `~/work/kgirl-sync`. Round-5 drills showed no gain (`~/kgirl-teach/route/RESULTS.md`).
- **Superseded, not merged:**
  - `local/ask-gate-fix` (`15c1e72`): everything useful is in #61 and #87.
  - `feat/qincrs-soup-validator` in `~/work/kgirl-harness`: already upstream via #77. Its 10 dirty files are older drafts.
  - `feat/jev-blocked-gate`, `sync/harness-router-retrieval`, `test/*`, `feat/incremental-embeddings` (worktree `~/work/kgirl-vectors`): merged; their remote branches are gone.
- **Uncommitted April work in `~/Synaps`** (branch `chore/claude-md-gitignore`):
  - Parts already merged elsewhere: the Julia Ollama port and the `sydv.py` repair.
  - Never committed: `src/kgirl/{blockchain,metering,api/services}/`, `core/marl_portfolio.py`, `coord/`, `quarantine/`, `tests/test_runtime_smoke.py`, and duplicate `qincrs_*` files at the root and in `docs/`. Don't sweep these into a commit; ask the user.

## Decisions (why)
- 2026-11-13 **Energy cutoffs calibrate per embedder:** `auto` ≤ mean + 1 SD and `needs_citations` ≤ mean + 2 SD of the fallback energy over 24 reference prompts (`ENERGY_CALIBRATION_PROMPTS`, `energy_cutoffs()` in `src/kgirl/main.py`). It's the same rule that produced the old 0.16 on the demo adapter. Rejected: a new constant, because the scale moves with the embedder (demo ≈ 0.11, nomic ≈ 0.88).
- 2026-11-13 **No token caps:** the user hates truncation. Ollama gets no `num_predict`. Anthropic needs `max_tokens`, so it's 8192, the default model's output limit (`ANTHROPIC_MAX_TOKENS`). Ollama `num_ctx` = 8192 (`KGIRL_OLLAMA_NUM_CTX`), because Ollama silently drops prompt starts beyond 4096. 16384 evicted a model on the 8 GB GPU.
- 2026-11-13 **Kept #61's coherence and energy math over the August local fix;** only the features #61 lacked were ported (#87).
- 2026-11-13 **The efl test fix touched only the test,** so RESULTS.md §7 ("models/metrics/experiments unchanged since 8180617") stays true.
- 2026-11-13 **Truncated paste moved, not deleted** (`docs/archive/sydv_appended_bundle.txt`; the two files concatenated equal the old one).
- 2026-11-13 **`*.zip` not ignored:** `data/` tracks zips on purpose.

## Dead ends
- Julia 1.13 precompiling LibPQ: `Memento` fails ("Method overwriting is not permitted during Module precompilation"). Workaround: `--compiled-modules=no` (slow first start, about 2 min).
- `/ask` default energy on nomic before #89: always `escalate`, even when both models agreed and were right.
- Harness `ask` with the current Atlas index found only `def ask(...)`, not its body, so the index is stale (13 Nov). Re-indexing writes into `~/.kgirl-sync/repos`; that's for `kgirl-f2` or the user to do.

## Open questions
- On nomic-embed-text, isotropic noise scores about 0.93 and the calibrated cutoff is 0.948, so the energy gate barely separates anything there. Is it worth keeping, or should coherence alone decide? To settle: log energy vs. answer quality over real questions.
- Coherence measures agreement, not truth: two models both saying "I don't know" scored 0.95. Is a correctness check wanted?
- The `sydv.py` demo still crashes (autograd on a no-grad tensor; a 3-output Sequential unpacked into 2). Is TA-ULS v2 (`src/kgirl/tauls/`) the maintained one?

## State (verified 09:05–09:20 PST)
- **kgirl** `main` `6802f8c`; 0 open PRs; CI on #89 pass (harness + research). Public repo: pushes are publications.
- **Local `main`** = origin/main. `~/work/kgirl-tests` is detached at origin/main and clean.
- **Running (started this session):**
  - Postgres :5433 (`~/Synaps/.pgdata`)
  - Julia ChaosRAG :8001
  - kgirl API :8765 with qwen2.5:3b and llama3.2:3b, on nomic-embed-text, num_ctx 8192, about 6.2 GB VRAM
- **Data:** the `chaos` DB got one test doc (an "Omarchry" note) plus a TEST telemetry row.
- **Machine:** Python 3.14.10 (multiprocessing default is forkserver), Julia 1.13.3 with packages in the v1.12 env, Ollama 0.12.11.

## Running the stack
```bash
pg_ctl -D ~/Synaps/.pgdata -l /tmp/pg.log -o "-p 5433" start
cd ~/work/kgirl-tests   # any checkout of main
DATABASE_URL=postgresql://127.0.0.1:5433/chaos julia --compiled-modules=no \
  --project=$HOME/.julia/environments/v1.12 src/julia/server.jl &          # :8001
PYTHONPATH=src MODELS='ollama:chat=qwen2.5:3b,embed=nomic-embed-text|ollama:chat=llama3.2:3b,embed=nomic-embed-text' \
  ~/Synaps/.venv/bin/python -m uvicorn kgirl.main:app --host 127.0.0.1 --port 8765 &
curl -s 127.0.0.1:8765/ask -H 'content-type: application/json' -d '{"prompt":"…","use_rag":true,"return_all":true}'
```
To stop: `pg_ctl -D ~/Synaps/.pgdata stop`, then kill the processes listening on :8001 and :8765. The user's shell is fish, so wrap these lines in `bash -c '…'`.

## Project health
- Tests:
  - `tests/integration/test_consensus.py`: 12 pass
  - `tests/iwme/test_sources.py` and `tests/harness/test_privacy.py`: pass
  - harness: 146 OK (2 skipped)
  - research: 119 tests, 30/30 checks
- CI: both jobs green on #89.
- Fragile:
  - `/ask` calibration runs on the first request per adapter (24 embeddings, about 1–2 s).
  - The Julia server's DB default is `postgresql://127.0.0.1:5433/chaos`.
  - A stray file named `main` in `~/Synaps` makes a bare `main` ambiguous; use `refs/heads/main`.

## Next steps, in order
1. Answer the two decisions above. Reply e.g. `delete the superseded kgirl branches` / `review the April files in Synaps`.
2. Coordinate with `kgirl-f2`, then re-index Atlas so the harness `ask` sees current code.
3. Log `/ask` energy and coherence against judged answers over about 30 real questions, then decide the energy gate's future.
4. Optional: a `docs/` page or script for "Running the stack" (above), so it isn't only here.

## One-shot check
    bash ~/.claude/skills/hanoffed/scripts/scan.sh --root ~/Synaps --fetch
    bash ~/.claude/skills/hanoffed/scripts/checks.sh ~/work/kgirl-tests --run

## Rules that still apply
- PR flow. The user asks for PRs and merges; don't merge on your own initiative.
- Never delete or "clean" the user's copies, originals or vendored duplicates. Root clutter was untracked (#82) but the files stay on disk.
- Randomness comes from ChaosRNG (logistic map), never `random` or `np.random`, in kgirl modules. Tests may use numpy RNG for fixtures.
- State confidence levels on scientific claims. Never present Level 2 output as measured.
- Tithe material never goes into this repo, `~/.kgirl-sync` or the router index.
- Never print credentials. Mask URL userinfo.

## Vocabulary so far
squash merge, dangling commit (11-11) · preregistration, Goodhart's law (11-02/03) · superseded, calibration (11-13). Full ledger: `~/.claude/skills/hanoffed/vocabulary.md`.

## Opening prompt for next session (edit the brackets)
> Read kgirl's `HANDOFF.md` on main (`git -C ~/Synaps show origin/main:HANDOFF.md`, or `~/work/kgirl-tests/HANDOFF.md` after `git -C ~/work/kgirl-tests checkout --detach origin/main`) and run its one-shot check. Then [next step 1 or 3]. Done means [truth condition]. Live work; open PRs for changes and merge when CI is green. When we wrap up: /hanoffed kgirl.
