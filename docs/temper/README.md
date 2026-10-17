# TEMPER: kgirl's tuning specialist

TEMPER is a Claude Code subagent with one discipline. It makes kgirl measurably better one setting at a time,
and proves every gain before claiming it. It proposes; you decide. It never edits `src/` or `tests/`, never
commits or pushes, and writes only under `runs/temper/`, which is gitignored.

| File | What it is |
|---|---|
| `.claude/agents/temper.md` | The agent: loop, control surface, invariants, report schema |
| `.claude/hooks/temper_guard.py` | The PreToolUse guard that enforces the boundaries while TEMPER runs (stdlib only) |
| `tests/temper/test_guard.py` | Runs the guard as Claude Code does: a subprocess fed hook JSON |
| `src/kgirl/temper/gate.py` | The model-swap gate (G0–G4) and the suite-size planner; stdlib only, deterministic |
| `tests/temper/test_gate.py` | Planning, the exact and paired tests, leakage by containment, every gate's REJECT, CLI exit codes |

## Using it

In an interactive Claude Code session in this repository, ask for it by name:

> Use the temper agent to record a baseline for the current commit.

> Use the temper agent to test the max-type PPPF features against the degree-2 surrogate.

Every task ends with three lines: what changed, what was measured (with the report path), and the next most
informative experiment.

## The loop

1. **Baseline.** The bench results for one commit go in `runs/temper/baseline-<sha>/`.
2. **Hypothesis.** One setting at a time. The primary metric, the guard metrics and the acceptance rule are
   written to `plan.json` before anything runs.
3. **Experiment.** Both arms get the same budget, with at least 5 chaos seeds, compared seed by seed against
   the baseline.
4. **Verdict.** ACCEPT only if the primary metric wins on at least 4 of 5 seeds, it also improves on held-out
   tasks or rows, and no guard metric regresses. Otherwise REJECT, and the rejection is recorded.
5. **Ledger.** One line per attempt in `runs/temper/ledger.jsonl`. The attempt counts make repeated tries
   on one setting visible.

A proposal reaches `main` only when you ask for it as a PR.

## What it may tune, and what it may not

It may tune:
- `kgirl.evolve` variation settings
- the neutronics surrogate's features and regularization
- `SwarmConfig` and `JevConfig`
- `CurationPolicy` (on a copy of the memory pool)

It may never touch these, because each trades safety for score:
- `manual_sources` and `allow_unverified`
- the verify allowlist and the scrubbed verifier environment
- staged MCP trajectories and staged forged skills
- ChaosRNG-only randomness
- the 0.7–5.0 wt% enrichment bounds

The full tables are in the agent file.

## What the guard enforces

The guard sees every tool call TEMPER makes (matcher `"*"`).

**Passes:**
- Read, Grep, Glob and LS.
- Write, only to paths that really resolve inside `runs/temper/`.
- Bash, when every command in the line passes.

**Refused in Bash:**
- Git commands that change state. Read-only `status`, `log`, `show`, `diff`, `rev-parse` and similar pass.
- The network, installers and privilege escalation: `curl`, `ssh`, `gh`, `pip`, `npm`, `sudo`, `ollama pull`.
- Inline code: `python -c`, `bash -c`, heredocs, `$(...)`, backticks, `awk`, `sed`, `eval`, `xargs`.
- Writes outside `runs/temper/`, whether through redirection, `tee`, `cp`, `mv`, `rm`, `mkdir`, `touch` or
  `ln`, or through output flags such as `--out`, `--export`, `--latex` and `--plot`.
- kgirl benches whose default output path lies outside `runs/temper/` (`neutronics fit`, `iwme plot`, the
  `tauls` trainers), unless TEMPER passes an explicit output under it.
- Harness commands that change state (`index`, `task`, `soup add`/`curate`, `skills forge`), unless
  `--home` or `KGIRL_HOME` points under `runs/temper/`. To experiment on memory, copy it there first:
  `cp -r ~/.kgirl runs/temper/<id>/home`.
- Always: `soup promote`, `task --apply`, `ask`, `card --summarize`, `mcp`, and indexing remote repos. A
  `task --verify` command is itself checked as a Bash command.

**Every other tool is refused.** Any error inside the guard also refuses the call. The agent file sets
`onFailure: block`, so a crash, a timeout or a missing `python3` blocks as well. That setting needs Claude
Code v2.1.295 or later; older versions ignore it.

### Limits

- **It is a guard, not a sandbox.** It stops mistakes and drift. A program TEMPER is allowed to run (a bench,
  or an analysis script it wrote under `runs/temper/scripts/`) can still do whatever its own code does.
  Review those scripts like any other evidence.
- **It runs only in a trusted, interactive session.** Claude Code runs a project subagent's frontmatter hooks
  only after you accept the workspace-trust dialog for this folder. A headless `claude -p` session doesn't
  count as accepting it, so **in `-p` mode the guard does not load**. Run TEMPER interactively.
- **It doesn't control permission prompts.** When the main session runs in auto or bypass mode, TEMPER runs
  in that mode too. The tool allowlist and this guard are the boundaries. The guard never approves anything;
  an allowed call still goes through the normal permission flow.

## Tests

```bash
python -m unittest discover -s tests/temper -t .
```

The suite covers:
- the commands TEMPER needs: benches with outputs under `runs/temper/`, read-only git, analysis scripts,
  harness experiments on an isolated home;
- every refusal above: symlink escapes, `(cd runs/temper) ; echo x > a`, newline-separated commands,
  `--verify 'curl x'`, harness scripts run by path instead of `-m`;
- bad hook input, which fails closed;
- the wiring in the agent's frontmatter.

When the guard is mutated to allow everything, 76 subtests fail.

For the gate (`test_gate.py`): sample sizes against the closed form, the binomial tail in log space, an
improvement that is accepted, seeds counted per item, each of G0–G4 rejecting on its own, unpaired runs,
INCOMPLETE without the shadow run or the leakage check, containment catching what Jaccard misses, and the CLI's
exit codes.

## Stage 2: from specialist agent to fine-tuned specialist model

A fine-tuned local model replaces its base only through the gate in `kgirl.temper.gate`, which turns the
acceptance rule into code with a verdict: ACCEPT, REJECT or INCOMPLETE. An INCOMPLETE swap never happens.

```bash
python3 -m kgirl.temper plan --items 200                     # what a suite of 200 can detect
python3 -m kgirl.temper gate --eval runs/temper/<id>/runs.jsonl --suite suite.jsonl \
    --train runs/temper/<id>/jev.jsonl --shadow runs/temper/<id>/shadow.jsonl \
    --artifact <sha256 of the served GGUF> --out runs/temper/<id>/gate.json      # exit 0 / 1 / 2
```

| Gate | Passes when | Why |
|---|---|---|
| G0 preconditions | ≥ 200 paired items; base and tuned ran on identical (item, seed, temperature) sets; every tuned row came from one artifact, the served GGUF by sha; no training example contains a held-out item | Without these, the other gates measure something else |
| G1 validity | one-sided lower bound of the valid-decision change > −0.02 | Small models mostly fail by drifting off format |
| G2 success | one-sided p < 0.05 and a positive change: exact McNemar with one run per item, a paired test on per-item means with several seeds | Seeds are repeated measures of an item, not independent trials |
| G3 families | no task family loses more than 0.05 success | An average gain can hide a broken family |
| G4 shadow | on unscored real traffic, ≤ 25 % of parsed decisions change | The suite cannot score everything the model will see |

The rules behind the gate:

- **Evaluate what ships.** Score the quantized GGUF Ollama will serve, at Jev's own temperatures (0.1, 0.5,
  0.9) with chaos seeds. A model that passes as an fp16 adapter and fails as Q4 fails.
- **Leakage is containment, not equality.** A held-out item leaks when at least 55 % of its 3-word shingles
  appear inside one training example, or its normalized text equals one. A long training prompt that embeds
  a short held-out task verbatim has low Jaccard overlap and containment 1.0.
- **Size the suite before training.** At 20 % discordant items, α 0.05 one-sided and 80 % power: 47 items
  detect only a 0.159 gain, 200 detect 0.078, and a 0.050 gain needs 493 (`plan`). A model trained before the
  suite exists cannot be judged.
- **`kgirl.iwme verify` is a platform guard, not a model evaluation.** Its 62 checks re-derive the IWME paper's
  numbers. They run in every baseline, and any FAIL blocks every verdict, but they say nothing about a coding
  model.

What trains the model, and what doesn't:

- **Training rows** come from `python -m kgirl.harness soup export-sft <out> --role jev` (and `--role smith`)
  for runs on the decision log (#68): exact step prompt → one-line decision, EDIT prompt → applied
  SEARCH/REPLACE. The `diff` role (goal → whole diff) matches neither model's format. Only active,
  human-promoted trajectories export.
- **Preference pairs** come from `runs/temper/ledger.jsonl`: an accepted against a rejected proposal for the same
  setting. Tag every rejection with its reason (format or content), and drop untagged pairs.
- **Never trained on:** the Jev task suite. It doesn't exist yet: building it (≥ 200 items, each with its own
  oracle) is the prerequisite for Stage 2, together with live-model runs on the decision log.

## First assignments

1. **Baseline registry** for the current `main`, including `iwme verify`.
2. **Surrogate features: done.** Cross-validated on 2026-10-17 (L0): PPPF R² 0.704 → 0.754, ΔR² +0.050 in 15/15
   folds, 95% CI [+0.046, +0.053]. Using the better model alone in the fuel search was exploited (optimism gap
   +0.135), so the search now uses a pessimistic ensemble held to the data's support (#67).
3. **Jev task suite (≥ 200 items) and live-model runs on the decision log** (#68), on a machine with Ollama and
   `qwen2.5-coder:7b` and `:1.5b`. This produces the first real Stage 2 data, and the suite the gate needs.
