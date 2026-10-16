# TEMPER: kgirl's tuning specialist

TEMPER is a Claude Code subagent with one discipline. It makes kgirl measurably better one setting at a time,
and proves every gain before claiming it. It proposes; you decide. It never edits `src/` or `tests/`, never
commits or pushes, and writes only under `runs/temper/`, which is gitignored.

| File | What it is |
|---|---|
| `.claude/agents/temper.md` | The agent: loop, control surface, invariants, report schema |
| `.claude/hooks/temper_guard.py` | The PreToolUse guard that enforces the boundaries while TEMPER runs (stdlib only) |
| `tests/temper/test_guard.py` | Runs the guard as Claude Code does: a subprocess fed hook JSON |

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

## Stage 2: from specialist agent to fine-tuned specialist model

TEMPER's ledger and kgirl's own memory become training data for a local specialist:

- **Training examples:** `python -m kgirl.harness soup export-sft runs/temper/<id>/sft.jsonl`. This writes
  only active, human-promoted trajectories; MCP-started runs stay out until promoted (#60).
- **Preference pairs:** an accepted proposal against a rejected one for the same setting, from
  `runs/temper/ledger.jsonl`.
- **Held out, never trained on:** `python -m kgirl.iwme verify` and a fixed Jev task suite.
- **Acceptance:** the fine-tuned model replaces its base only if it raises task success and the rate of valid
  decisions on the held-out suite. This is the same keep-only-improvements rule as `kgirl.tauls improve`.

Prerequisite: live-model Jev runs. HANDOFF.md still records them as never done. That baseline is TEMPER's
third assignment, and it needs Ollama with `qwen2.5-coder:7b` and `:1.5b`.

## First assignments

1. **Baseline registry** for the current `main`.
2. **Surrogate features.** The max-type PPPF features measured on 2026-10-16 lifted held-out R² from 0.737 to
   0.784 (L0, 360 rows). TEMPER confirms or refutes this with cross-validation.
3. **Live-model Jev baseline**, which unblocks tuning `SwarmConfig`, `JevConfig` and intuition.
