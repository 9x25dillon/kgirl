# helix: orchestration where code keeps the books and models write the blocks

`src/kgirl/harness/helix/` is the HELIX-9 prompt design after its review, built as instrumentation instead of
as a prompt. The review's verdict was that the prompt asked a model to do three things a model cannot do
honestly: count its own tokens, hold persistent state by repeating it, and audit itself. This layer moves
each of those into code and leaves the models one job, writing a block when it is their turn.

    python -m kgirl.harness.helix run "your question" -B 4000 --critic-model llama3.2:3b
    python -m kgirl.harness.helix state | history | diff 1 2 | verify | floor
    python -m kgirl.harness.helix ingest handoff.json
    python -m kgirl.harness.helix audit transcript.txt --external 3600

Only the review's enhancements are implemented. The original prompt's ceremony (model-written ROUTE, SEED,
BUDGET and LEDGER tags, the L2 and L3 fractal split) is not.

## The bar (written before the code)

1. Byte-compiles; `PYTHONPATH=src python3 -c "import kgirl.harness.helix"` resolves and pulls in no torch.
2. No number in a ledger is written by a model. `spent_ext` is the provider's count or it is null.
3. An axis value cannot reach accounting, emission or BYPASS: a state has no field that does.
4. A handed-off state is validated before use and rejected whole on any violation.
5. A re-steer or a defect that does not quote its target word for word changes nothing.
6. The static prefix is byte-identical on every call; everything before STATE only grows by appending.
7. Deterministic: the same run replayed in another home gives the same ledger heads.
8. Nothing outside `src/kgirl/harness/helix/**`, `tests/helix/**` and this file is edited.

## What each review finding became

| Finding | What the review asked for | Where it lives |
|---|---|---|
| F1 | make the "holographic database" a real, versioned object you can query and diff | `state.py` `State`; `store.py` `StateStore` over `vessel.matrixdb.MatrixDB` at `$KGIRL_HOME/helix.db`: every version hash-chained with its diff; `get`, `history`, `diff`, `verify` |
| F2 | accounting out of the prose, reconciled against the provider's `usage` | `ledger.py`: `ext` from `Completion.output_tokens`, `est` from the stated proxy max(words x 1.35, chars / 3.5), their ratio reported; a backend that does not count leaves `spent_ext` null |
| F2 | BYPASS as a predicate derived from the scaffold floor | `ledger.bypass`: bypass when floor / B exceeds 20%, or the expected answer is under 300 tokens. `StateStore.floor()` is the median scaffold of recent measured runs, and 500 (the review's estimate, labelled) before any |
| F3 | axes govern content only; ingested state schema-validated, clamped, fail-closed | invariant I1; `state.validate`; `StateStore.ingest`; the `invariants_hash` field |
| F4 | compression counts as spend where it occurs | compaction is a ledger entry with no allocation, so it shows as overrun and shrinks the next target |
| F4 | reserve defined at L1 only, split into a steering pool and a synthesis floor | `Ledger`: 5% steering, 10% synthesis floor, 85% content split by weight; no reserve below L1 |
| F5 | TUNE must quote the offending span | `protocol.parse_tunes` + `quoted`: the quote must be a word-for-word span of the block, three words or more |
| F5 | ORACLE gives number + method + falsifier | `protocol.parse_calibration`; a bare number is void and becomes an open conflict |
| F6 | one canonical STATE at the tail; handoff passes STATE only | `prompt.assemble`; `Helix.handoff()` |
| F6 | anchor rubrics per axis | `state.RUBRICS`: three anchors for each of the six axes, in the static prefix |
| F6 | W calls emit changed axes only | the log line is `W rigor 0.60->0.70 | quote: ... | because: ...`; the full vector exists only in STATE |
| F7 | static prefix, task, append-only log, STATE tail | `prompt.py`; system message is `STATIC`, never restates the vector |
| F7 | compact the log into STATE periodically and count it | `Helix._compact`, past `log_limit` estimated tokens |
| F7 | ASCII tags | every tag and trailer is ASCII |
| F8 | CRITIC on a different model family | `critic=` or `KGIRL_HELIX_CRITIC_MODEL`; `family()` compares; a same-family review is stated as an open conflict |
| F8 | CRITIC quotes a defect span with a repair; no concurrence without new evidence | `protocol.parse_review`; a review with neither is void, and the draft is reported as unreviewed |
| F8 | SYNTHESIST carries the unresolved-conflicts list, never smoothed | `UNRESOLVED: [...]` trailer; missing or malformed means every conflict stays open |

## One run

    BYPASS?     predicate over B and the floor. Yes: one direct call, no state written, the run recorded.
    experts     one block each. One expert by default, chosen by the router's intent rules (no tokens);
                `--lattice` gives two, `--expert NAME` names them. Each block ends with a TUNE trailer.
    ORACLE      only when two accepted probes in a row name the same axis and direction.
    CRITIC      DEFECT trailers that quote the expert blocks, or a CONCUR with new evidence.
    SYNTHESIST  the answer, and which numbered conflicts it could not resolve.

The answer is the only part a person needs to read. `render()` prints it first and the flight recorder under
it: per-call targets against provider counts, accepted and rejected tunes, the open conflicts.

## STATE

    {"v": 4,
     "task_seed": "<the first 36 words of the task>",
     "vector": {"gain": 0.6, "temp": 0.25, "aperture": 0.5, "rigor": 0.6, "horizon": 0.5, "compression": 0.6},
     "last_delta": {"axis": "temp", "d": -0.15, "evidence": "<the quoted span>"},
     "ledger": {"alloc": 4000, "spent_est": 2138, "spent_ext": 1521, "reserve_floor": 400},
     "open_conflicts": ["..."],
     "invariants_hash": "sha256:..."}

- One chain per session (`--session`, default `main`). The vector persists across tasks in a session; open
  conflicts belong to a task and are cleared when the task capsule changes.
- A write bumps `v` and needs a non-empty diff. Writing the head again writes nothing.
- `validate` rejects on: a missing or unknown field, a wrong type, NaN, a foreign `invariants_hash`. A number
  that is only out of range is clamped and reported. A rejected state yields the defaults.
- `ingest` records every rejection on the chain (`helix.reject`). It never adopts `spent_ext` from outside,
  since only this harness's measurement may set it, and it rejects a version behind the stored head.

## Auditing a transcript a model wrote on its own

`evaluate.py` is the evaluator from the user's `helix_kernel.py`, ported on 2026-11-16 with its detectors
unchanged. `Helix.run` does not need it: there no model writes a ledger number. It is for a log from a model
following the HELIX-9 prompt in a chat window, where the ledger, vector and capsules are self-reports.

| Detector | Severity | Fires when |
|---|---|---|
| `budget_theater` | High | the L0 ledger is more than 15% off `--external`; a deeper level claims more spend than the level above; sub-ledgers exist with no L0 |
| `steering_drift` | High | a vector changed with no GRAD before it, or by more than 0.25 with no JUSTIFY |
| `missing_ledger` | High | weights were set and no ledger was emitted |
| `capsule_bloat` | Medium | a SEED field over twelve words |
| `fractal_fractal` | Medium | a SEED deeper than L3 |
| `ceremony_on_trivial` | Medium | under 300 external tokens, with capsules and no BYPASS |

The verdict is FAIL when any finding is High, and `audit` exits 1 on FAIL. Only L0 is reconciled against the
external count; the kernel used to compare every level with it, which failed every honest multi-level log
(its T8). `propose_resteer(reports)` turns a detector that fires in two audits in a row into a proposed
re-steer marked `requires_approval`; it applies nothing.

Not ported from the kernel, because the package already has its own: `DurableStore` (here `StateStore`, on
MatrixDB's hash chain), `validate_state` (here `state.validate`) and `count_tokens` (here the provider's
count, with `ledger.estimate` as the labelled proxy). See decisions 6 and 7.

## Decisions that are yours

1. **B is a budget of generated tokens.** Prompt tokens are measured and reported (`input_ext`) but not
   charged against B. Three calls each re-read the static prefix, roughly 1,100 tokens on qwen2.5-coder.
   Ollama reuses an evaluated prefix; the Anthropic backend sends no cache breakpoints, and adding them is
   outside this lane.
2. **Targets are stated, replies are not cut to them.** Every call's hard stop is max(B, 2048). A reply that
   reaches it becomes an open conflict rather than a silent truncation.
3. **Three things here are not in the review.** The `temp` axis is also passed as the sampling temperature.
   An ingested vector may sit at most 0.25 per axis from the stored head; the excess is clamped. Routing
   uses `hermes.router.classify` instead of a model-written ROUTE line. Say so if any should go.
4. **Three things in the review are not here.** The model does not echo STATE back: code validates it
   instead, which costs no tokens. The L0 capsule is not charged to reserve, because code writes it and it
   costs no generated tokens. The saving from ASCII tags over the old delimiters was not measured.
5. **The scaffold floor counts steering spend plus the critic's output** (generated tokens that are neither
   draft nor answer). On the one measured run below that is 177, which would put the BYPASS line at B < 885
   instead of the review's 2,500. The floor is the median of the last 20 runs in the home it is run from.

6. **Decided 2026-11-16: an out-of-range axis value is adjusted into range, not rejected.** That is what
   `state.validate` already did (clamp to [0,1], report it in `clamped`). The kernel used to reject such a
   state outright; it now adjusts too and records the adjustment. A value that is not a finite number
   cannot be adjusted and is still rejected, here and in the kernel.
7. **The kernel's store is stricter about writes than this one.** It fsyncs each write and keeps a fault
   table for failed writes; `StateStore` relies on SQLite and records only rejected handoffs. Say so if
   the fault table should come over.

## What has been checked, and what has not

Checked on 2026-11-16: `tests/helix/` (117 tests, offline, fake backends with exact token counts) passes;
`import kgirl.harness.helix` loads no torch. The ported evaluator returns the same report as the fixed
kernel (15/15 on its own tests) on ten logs, including every log those tests use. Two live runs on local Ollama in a scratch home, B = 4000,
qwen2.5-coder:7b writing and llama3.2:3b as critic:

| Run | Calls | Generated (provider) | Proxy estimate / provider | Tune | Review |
|---|---|---|---|---|---|
| 1 (before the fixes below) | 3 | 1,681 | 1.26 | gain 0.60 -> 0.50, quote verified | one defect accepted wrongly |
| 2 | 3 | 1,521 | 1.41 | temp 0.40 -> 0.25, quote verified | void: no quoted defect, reported as unreviewed |

Run 1 found three faults, fixed before run 2: a defect quote was matched against the harness's own log line
instead of the expert's block; the critic's prose, which echoed the whole prompt, was copied into the log;
and the ledger shown to later stages was stale.

Not checked: answer quality. Both live answers were generic lists, and the synthesis in run 2 mostly copied
the expert block; whether the harness improves answers over a direct call has not been measured. The proxy
overestimates this tokenizer by 26% to 41% on two samples, which is not a calibration. Calibration (ORACLE),
compaction and the two-expert lattice ran only against fakes. Nothing has been run in `~/.kgirl`.
