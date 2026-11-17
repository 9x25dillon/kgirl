# Hand-off — kgirl

## Resume here
**Newest close:** 16 November 2026, 19:20 PST · Claude Code, Fable 5.2 · session `2e0e8c56`
**Where things stand:** `main` is still `afd666e` (PR #95). Five feature lanes sit on top of it as local
branches that are not on GitHub: mind, holo, console, mind-tests (all 15 Nov, other sessions) and **helix**
(today). This file lives on `feat/kgirl-helix`; `main`'s copy is the 13 Nov one.
- **helix** (`src/kgirl/harness/helix/`, 118 tests): the HELIX-9 prompt review built as code. Code counts
  tokens from the provider, keeps a versioned STATE, and only accepts a re-steer or a defect that quotes its
  target. Runs end to end on local Ollama. Whether it gives better answers than a plain call is not measured.
- **Correction to the 13 Nov hand-off:** `main` moved from `6802f8c` to `afd666e` (#90–#95 merged). The
  "kgirl-f2" harness set-up it lists as in flight is done: `~/.kgirl` is the live home.
- **Backups exist now.** A second session built `kgirl-vault` today: every local branch here is mirrored to
  two other disks nightly. See `~/.claude/plans/kgirl-vault-and-two-path-link.md`.

**Decision needed:** publish or hold. `feat/kgirl-helix` carries six commits that are not on GitHub (two
helix, one holo, three mind). The repo is public. Reply `push helix` to publish all six, or leave them local.
**Start with:**
1. `cd ~/work/kgirl-helix && PYTHONPATH=src ~/.kgirl/venv/bin/python -m unittest discover -s tests/helix -t .`:
   118 tests, about a second. Confirms the lane is intact.
2. Read `docs/harness/HELIX.md`: the table mapping each review finding to its code, and "Decisions that are
   yours" (seven items).
3. First task: measure helix against a direct call on about 10 real questions (next step 1).

## Progress track (newest first)
| Date | Session | Shipped (hash / PR) | State after |
|---|---|---|---|
| 2026-11-16 | Fable 5.2 `2e0e8c56` | `31d2a97` fix: answers keep their code fences | helix 118 tests OK; local only |
| 2026-11-16 | Fable 5.2 `2e0e8c56` | `1249f0d` helix lane: state, store, ledger, protocol, prompt, harness, evaluate, CLI | runs on Ollama; 2 live runs in a scratch home |
| 2026-11-16 | Opus `185e9e77` | `kgirl-vault` tool, vault on two disks (not in this repo) | restore drill passed; daily 03:30 timer |
| 2026-11-15 | other sessions | `b6f6973` holo; `31bbc3f` console; `eb8bdb7` mind tests; `c04ad72` `0e560d4` `e05143d` mind | all local branches, none pushed |
| 2026-11-14 | Opus 5.5 `bacd` | #92–#95 merged (index refresh, verify hint, all-repo chunk index, eval flags) | `main` = `afd666e` |
| 2026-11-13 | Opus 5.5 `e0adf913` | #85–#91 (/ask on Ollama + ChaosRAG, energy calibration, hand-off) | /ask answers locally |
| 2026-11-12/13 | Claude (cloud) | #83, #84 research/efl_iwme_cascade | 10/11 hypotheses replicate |
| 2026-11-11 | Opus 5.5 `7304eed2` | #82 untracks 75 root junk files | `main` only |
| 2026-11-03 | Claude (porter) | #80 embedding cache, #81 drill tests | stronger default route not shipped |
| 2026-10-19 | Claude | `kgirl.life` + metabolism | archived: `docs/handoff/HANDOFF-2026-10-19-claude-opus.md` |

## In flight
- **helix** on `feat/kgirl-helix` (local only, worktree `~/work/kgirl-helix`): done: the lane, its doc, 118
  tests, the transcript audit ported from `~/work/helix_kernel`. Left: quality measurement, CI coverage,
  the seven decisions in HELIX.md. Verify: start-with step 1.
- **The branch stack.** `main` ← mind (3 commits) ← holo (1) ← helix (2). Console and mind-tests also branch
  from mind. Helix needs mind: it uses the `chat` role and `mind_ctx`, which are not on `main`. So helix
  cannot merge before mind does.
- **kgirl-vault** (session `185e9e77`, tool at `~/.local/bin/kgirl-vault`, not in this repo): stage 1 passed
  its gate at 19:15 by that session's report. Next there: router index refresh over the cable, then a search
  page and a restricted chat on the home Wi-Fi. Its plan file has a Trajectory section; trust that over this.
- **Still parked from 13 Nov:** `feat/stronger-default-route` (no gain in round-5 drills); about 38 uncommitted
  April files in `~/Synaps` (`blockchain/`, `metering/`, `coord/`, `quarantine/`). Don't sweep them into a
  commit; ask.

## Decisions (why)
- 2026-11-16 **helix is code, not a prompt.** The review's point was that a model cannot count its own tokens
  or audit itself. Rejected: shipping the patched prompt, because every ledger in it stays a self-report.
- 2026-11-16 **Only the review's enhancements were built** (the owner's instruction). The prompt's
  model-written ROUTE/SEED/BUDGET tags and the L2/L3 budget split are not implemented.
- 2026-11-16 **An out-of-range axis value is adjusted into range, not rejected** (owner: "since they're out of
  range we should adjust them to within range of the axis"). A non-number is still rejected. The kernel at
  `~/work/helix_kernel` was changed to match (20/20).
- 2026-11-16 **Budget B counts generated tokens.** Prompt tokens are measured and reported, not charged.
- 2026-11-16 **Replies are not cut to their targets.** The hard stop is max(B, 2048); a reply that reaches it
  becomes an open conflict. Follows the 13 Nov "no token caps" rule as far as Ollama's API allows.
- 2026-11-16 **Lane rule kept:** helix edits only its package, `tests/helix/` and `docs/harness/HELIX.md`,
  like holo. That is why there is no `kgirl helix` verb in `cli.py` yet.
- 2026-11-16 **Committed locally, not pushed:** the branch would publish four commits other sessions made.
- Earlier decisions (energy cutoffs per embedder, no token caps, `*.zip` not ignored) are in
  `docs/handoff/HANDOFF-2026-11-13-claude-opus.md`.

## Dead ends
- Matching a critic's quote against the whole flight log: the first live run accepted a defect whose "quote"
  was the harness's own log line. Quotes are checked against the expert blocks only.
- Copying the critic's prose into the log: llama3.2:3b echoed the entire prompt back (842 tokens). Only
  checked trailer lines enter the log now.
- A broad "clean up after trailers" pattern: it stripped code fences from every answer (`31d2a97`).
- From 13 Nov, still true: Julia 1.13 cannot precompile LibPQ (use `--compiled-modules=no`).

## Open questions
- Does helix beat a direct call? Both live answers were generic; in run 2 the synthesis mostly copied the
  draft. Settle with next step 1.
- Is llama3.2:3b a usable critic? In run 2 it produced no quoted defect, so the draft was reported as
  unreviewed. Try `deepseek-r1-distill-7b` or `qwen2.5:3b` as `--critic-model` on the same questions.
- The token proxy overestimated Ollama's count by 26% and 41% (two samples). Not a calibration.
- From 13 Nov: is the /ask energy gate worth keeping on nomic-embed-text; is a correctness check wanted.

## State (verified 19:10–19:20 PST)
- `origin/main` `afd666e`. `feat/kgirl-helix` `31d2a97`: 6 ahead, 0 behind, clean, no upstream.
- `git push --dry-run origin feat/kgirl-helix` succeeds, so pushing works. `gh` is logged out (`gh auth
  status` fails), so PRs cannot be opened or merged until `gh auth login`.
- Helix has not been run in `~/.kgirl`: no `helix.db` there yet.

## Project health
- Tests at 19:12: `tests/helix` 118 OK; `tests/harness` 159 OK (2 skipped). `tests/holo` (49) has never been run.
- CI runs only `tests/harness` and the research suite (`.github/workflows/tests.yml:25`). Helix, holo and
  mind tests are not in CI.
- `checks.sh --run` reports 2 FAIL and 1 SKIP, all unrelated to the harness: `scripts/setup/Makefile` points
  at a `server/` folder that does not exist, and pytest is not installed for the system Python.
- Fragile: `helix/store.py:71,99` reads every stored state on each write, fine now and slow after thousands of
  versions. `helix/protocol.py:31` treats any line starting `TUNE:` or `CAL:` as a trailer, even in prose.

## Next steps, in order
1. Measure: about 10 real questions, `helix run` against a direct call with the same model, judged blind.
   Say first how long it will take. If helix does not win, the lane stays a tool for audits and handoffs.
2. Decide the order to land the stack: mind first (it has never had a smoke test), then holo, then helix.
3. Add `tests/helix` and `tests/holo` to `.github/workflows/tests.yml` (outside the lane, so a separate PR).
4. Answer the seven decisions in `docs/harness/HELIX.md`.
5. Optional: a `kgirl helix` verb in `cli.py`, once the console lane's edits to the CLI have landed.

## One-shot check
    bash ~/.claude/skills/hanoffed/scripts/scan.sh --root ~/Synaps --fetch
    cd ~/work/kgirl-helix && PYTHONPATH=src ~/.kgirl/venv/bin/python -m unittest discover -s tests/helix -t .

## Rules that still apply
- PR flow. The owner asks for PRs and merges. The repo is public: a push is a publication.
- Never delete or "clean" the owner's copies, originals or vendored duplicates.
- Randomness comes from ChaosRNG in kgirl modules, never `random` or `np.random`.
- State confidence levels on scientific claims. Never present Level 2 output as measured.
- Tithe material never goes into this repo, `~/.kgirl-sync` or the router index.
- Never print credentials. Mask URL userinfo.
- One lane, one package: a lane edits its own package, tests and doc, nothing else.
- The owner wants visible results first and verification alongside, not in front (their usage report, 16 Nov).

## Running the stack
Unchanged since 13 Nov: see `docs/handoff/HANDOFF-2026-11-13-claude-opus.md`, "Running the stack". Helix needs
only Ollama: `PYTHONPATH=src ~/.kgirl/venv/bin/python -m kgirl.harness.helix run "question" -B 4000
--critic-model llama3.2:3b`. Add `--home <dir>` to keep it out of `~/.kgirl`.

## Vocabulary so far
superseded, calibration (11-13) · consolidation, smoke test (11-15) · fail-closed, least privilege (11-16).
Full ledger: `~/.claude/skills/hanoffed/vocabulary.md`.

## Opening prompt for next session (edit the brackets)
> Read `~/work/kgirl-helix/HANDOFF.md` and run its one-shot check. Then [measure helix against a direct call
> on these questions: ... / land the mind branch]. Stipulate: live work, visible results first, [push allowed
> / local commits only]. When we wrap up: /hanoffed.
