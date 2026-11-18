# Engineering checks

These checks are opt-in. The engineering lane writes them during build-forward work;
a person chooses when to run them. They have not yet been executed on this branch.
A green check demonstrates the named behavior, not the correctness of model-generated
solutions or a measured quality improvement.

## Offline mind smoke

From the worktree, with Python 3.14:

```fish
python3 scripts/smoke/mind_smoke.py
```

The script imports code from its own worktree, uses a fake streaming backend, and
creates a temporary `KGIRL_HOME`. Inherited kgirl/model settings are cleared. It
does not use the live launcher state, contact Ollama, download models or load weights.
Parent-process network calls are blocked and reported. Only fixed source literals
in the script run in child interpreters; those children retain the builder's time
and resource limits. `KGIRL_SANDBOX=off` avoids host-dependent namespace probes for
these trusted fixtures. This is not a check of bubblewrap isolation for model code.

| Stage | What a pass establishes |
| --- | --- |
| imports | Mind, stores, router, builder and writer initialize against scratch state. |
| chat | Multiple streamed pieces match the persisted reply and its saved context. |
| reflect | Fenced JSON creates a staged proposal; malformed JSON leaves the summary unchanged. |
| curate | One observation stays staged; a second conversation reinforces it and permits recall. |
| run_suite | The real child reports three passes, one assertion failure, one error, and a skip correctly. |
| builder | Two fixed samples agree, produce agreed tests, and store an unreviewed exemplar as staged. |
| writer | The built code reaches planning, drafting, critique and section revision. |
| offline | No parent network operation was attempted, even if an internal caller swallowed the error. |

The expected failures inside the `run_suite` fixture test reporting; they are not
failures of the smoke command. The current builder reads a nonce-prefixed JSON
record from `_suite_runner.py`, not unittest's human-readable terminal output.
Running with Python 3.14 checks that actual interpreter path; another version only
checks that version. The interpreter version is printed at startup.

Every stage prints `PASS`, `FAIL`, or `SKIP` with a reason. Dependencies of a failed
stage are skipped, independent stages continue, and any failure/skip returns exit
status 1. Normal runs are intended to take seconds; timing has not been measured.
Temporary state is removed when the command exits normally, including failure paths.

## Offline suites and CI

```fish
python3 scripts/smoke/suite_check.py harness
python3 scripts/smoke/suite_check.py helix
python3 scripts/smoke/suite_check.py holo
python3 scripts/smoke/suite_check.py vessel
# Only the mind subpackage, when narrowing a failure:
python3 scripts/smoke/suite_check.py mind
```

CI uses the first four commands as separate Python 3.14 matrix jobs. `vessel` is there because
holo stores its matrices in `vessel.matrixdb`, so a change to that shared module must run its own
tests; the torch-backed vessel cases skip themselves when torch is absent, as it is in CI. Harness
discovery includes `tests/harness/mind`, whose `__init__.py` already exists, and
fails explicitly if no mind cases are discovered. Empty suites and import errors
also fail. So does any folder under the suite that holds `test_*.py` files but
contributes no discovered case (usually a missing `__init__.py`), so a new lane's
tests, such as `tests/harness/console`, cannot silently drop out of CI. Mind is
not run a second time in the default matrix.

The suites use fake backends and require no model service. CI installs NumPy for
the numeric fixtures; package installation needs network access, test execution
needs no external model. The research job is retained separately.

## Console HTTP smoke

```fish
python3 scripts/smoke/console_smoke.py
# After the Helix/Holo UI lane is integrated:
python3 scripts/smoke/console_smoke.py --extended
```

This starts a real HTTP server on an automatically assigned loopback port, drives
the API, then shuts it down. It never attaches to an existing console. Atlas,
Soup, conversations, the workspace and exports are temporary fixtures. Porter
is pointed at a nonexistent scratch database. Connections outside this one
loopback endpoint are blocked in the parent process.

The script covers baseline GET routes and static assets; conversation creation,
a fake streaming chat job, feedback and deletion; memory transitions; workspace
settings; approval denial; job results; local fixture indexing; MCP search; input
errors; Host/Origin/token/content-type guards; path traversal; and an SSE hello
followed by a real event from the bus. Build, writing and evolution computations
are stubbed here: their HTTP/job boundaries are checked, not their model behavior.
The task and matrix routes are checked through validation errors, without starting
a swarm or embedding a query. Reading `app.js` is not a browser rendering test.

The launch token is parsed from the served HTML and never printed. A console
without the UI lane's new CSRF/token protection is expected to fail the bootstrap
and guard stages. Do not weaken the check to make the older console pass.

`--extended` additionally requires the Helix/Holo read routes, critic-host guard,
lane view and explicit errors for missing task/query/vector inputs. A Holo build
job must report its missing scratch chunk index, not claim a successful build.
No actual Helix model run or Holo embedding is attempted.

## One check-day command

```fish
bash scripts/check_all.sh
# Integrated console routes, with a different per-command time limit:
bash scripts/check_all.sh --extended-console --timeout 240
# Use an existing Python 3.14 interpreter if it is not the default:
env KGIRL_CHECK_PYTHON=/path/to/python3.14 bash scripts/check_all.sh
```

The runner executes the engineering tooling's regression checks, harness
(including nested mind tests), Helix, Holo, mind smoke and console smoke in order.
Mind tests run once. Individual failures do not hide
later results. Each command has its own log and a default 180-second deadline;
a timed-out command's own process group receives SIGTERM. An interruption stops
the current check and exits. Logs stay in the printed temporary directory so a
failure can be inspected. The final screen lists status and elapsed time for
each command. Exit 0 means all commands passed; exit 1 means a failure/timeout;
exit 2 signals an operational error; exit 130 means interrupted.

This runner installs nothing, builds no package and contacts no model server.
Use an existing environment with NumPy for the numeric suites. The research
suite remains a separate CI job and is not part of this harness-focused command.

The tooling checks can also be selected with
`python3 scripts/smoke/engineering_tests.py`. They use a fake Git transport and
fake child processes to check secret redaction, refusal to read protected files,
intermediate-history scanning, environment restoration, failure reporting,
timeouts and interruption. They create no Git commits, launch no real children,
and load no models. CI runs them once in the harness matrix job. They too are
written but not yet run.

## Makefile helpers

`scripts/setup/Makefile` is retargeted to the current source-tree harness, replacing
references to the nonexistent `server/` application. Its default target prints
help. It resolves the worktree root from the Makefile location, so both forms work:

```fish
make -f scripts/setup/Makefile help
make -C scripts/setup help
```

`install` explicitly creates `.venv` and installs NumPy, Black and Ruff; it is a
separate operator action and is never invoked by CI smoke or the check-day runner.
`dev` uses the existing `kgirl` launcher with this worktree's `KGIRL_CODE`, a fresh
temporary state directory, and the console on port 8765 (`PORT` overrides it).
The printed scratch directory is retained after exit. Dev mode may use local
Ollama when you request model work; it is not the fake-model smoke environment.
`test` runs the three offline suite families; `smoke` runs mind smoke; `fmt` and
`lint` apply only to `scripts/smoke`, with a 120-column setting, so they cannot
reformat another active lane's source or tests. Override
`PY` or `VENV` to use an existing environment. None of these targets run implicitly.

## Packaging audit

At the engineering branch's starting revision (`c6843f0`), no root
`pyproject.toml`, `setup.cfg` or `MANIFEST.in` exists. The two `setup.py` files
belong to legacy embedding/numbskull layouts. They do not establish a root wheel
contract for the current harness. The source-tree launcher therefore should not
be treated as evidence that a wheel includes the new packages or static assets.

A future packaging change needs explicit package selection for the harness and
its internal dependencies, plus these console resources:

- `kgirl/harness/console/static/index.html`
- `kgirl/harness/console/static/app.js`
- `kgirl/harness/console/static/style.css`

Confirm `mind`, `helix`, `holo` and `console` in an actual built wheel, including
new helper modules, and check imports from an isolated installation outside the
source tree. A package list alone does not prove dependency completeness. The
coordinator directed this lane to document the gap and keep the source-tree
launcher; choosing a distribution layout belongs to the owner. No root packaging
configuration was added, and no package build/install was performed.

## Previewing a lane merge

Before landing lane worktrees on the integration branch, preview the result with no git writes:

```sh
python3 scripts/smoke/merge_preview.py --base ~/work/kgirl-stack \
    --lane ~/work/kgirl-eng --lane ~/work/kgirl-helix-ui --lane ~/work/kgirl-builder \
    --python ~/.kgirl/venv/bin/python
```

It extracts the base commit with `git archive` into a temp folder, then overlays each lane's
changes against that commit, committed or not, including untracked files and deletions. It
fails on any path two lanes both change, then byte-compiles `src/`, `tests/` and `scripts/`,
runs `node --check` on changed `.js` files, and runs unittest discovery (imports only; no test
executes) to confirm every test folder contributes cases. A lane that does not contain the
base commit is refused: rebase it first. `--keep` leaves the merged tree for inspection.
Its own offline tests, `python3 scripts/smoke/merge_preview_tests.py`, run in throwaway
repositories; `check_all` and CI run them too.

## Landing the lanes

After the preview passes, `land_lanes.py` does the git side of landing. It only prints a plan
unless given `--apply`:

```sh
python3 scripts/smoke/land_lanes.py --stack ~/work/kgirl-stack \
    --lane ~/work/kgirl-eng --lane ~/work/kgirl-helix-ui --lane ~/work/kgirl-builder
```

Three steps run in order (choose a subset with `--steps`):

- **rewrite**: commits on the stack or its lanes that are not yet on `origin/main` and carry a
  non-noreply author or committer email get the noreply identity from `git config user.email`.
  The commit objects are otherwise byte-identical (any signature is dropped), so trees don't
  change and lane worktrees keep their uncommitted work. Only the stack, its lanes and branches
  whose tip lies inside that history move, in one atomic transaction. Other branches that
  share the old commits are listed, never moved. The old tips stay under
  `refs/kgirl-landing/original/` for undo. It refuses when an affected commit is already on a
  remote. Emails are counted, never printed.
- **commit**: each lane's uncommitted changes become one commit (`--message eng=...`), made
  with the noreply identity. A path that looks private (`.env`, keys, weights, `private/`,
  wallet, Hand_off, Tithe) stops the run before anything is written.
- **merge**: `--no-ff` merges into the stack, in the order the lanes are given. A conflict
  aborts that merge and stops, leaving any earlier merges in place.

It never pushes. Its offline tests, `python3 scripts/smoke/land_lanes_tests.py`, run in
throwaway repositories; `check_all` and CI run them too.

## Before a public push

```fish
bash scripts/prepush_scan.sh --base origin/main
# Optionally match a particular private name as well:
bash scripts/prepush_scan.sh --base origin/main --name 'Private Person'
```

The default base is the branch's upstream; a branch without one must name an
already-fetched base explicitly. The scanner never fetches or pushes. It examines
every commit reachable from HEAD but not the base, including intermediate commits
and merge-parent diffs, because deleting a secret later does not remove it from
outgoing history. Uncommitted/staged changes are not covered.

It flags private directories, environment files, wallets, keys, handoff material
and Tithe filenames **before opening their contents**. Other changed text is
checked for token patterns, credential assignments, URL credentials, personal
emails, non-loopback IP addresses and likely personal names in labelled fields or
commit identities. GitHub noreply emails and loopback/unspecified IPs are exempt.
`--name` adds a literal match but is visible in your shell history/process arguments;
omit it if that is inappropriate. Binary/LFS payloads and files over 2 MiB are
reported as unscanned, not certified clean. No weights or archives are loaded.

Output contains commit, quoted path, line and category; matched values, source
lines, commit messages and Git error text are withheld. Paths themselves are
shown so the file can be found, unless the path itself matches a private-value
pattern; then only a short SHA-256 identifier is printed. Non-UTF-8 text is also
reported as unscanned. Sensitive filenames and synthetic examples can
produce false positives. Names are heuristics, not reliable identity detection;
manual privacy review is still required even when no pattern matches.

Exit 0 means no heuristic findings (or no outgoing commits, explicitly stated);
exit 1 means review findings or unscanned payloads; exit 2 means the scan could
not complete. A failed or empty scan is not permission to publish.

### Automatic gate on every push (opt-in)

```fish
bash scripts/install_prepush_hook.sh             # install for every worktree (shared git dir)
bash scripts/install_prepush_hook.sh --uninstall # remove; restores the previous hook
```

The installer copies `scripts/hooks/pre-push` into the repo's hooks directory. The existing Git LFS
`pre-push` hook is kept as `pre-push.before-kgirl` and still runs after the gate, with the same input.
On each push the gate:
- scans exactly what is being pushed: new commits against the remote branch, or against
  `<remote>/main` for a new branch. A failed scan, or an exit code of 1 or 2, blocks the push.
- fails closed. It blocks when the pushed commit isn't the checked-out one, when the branch has no
  scanner (branches older than this lane), or when no base can be found (fetch first).
- reads extra private names from `$(git rev-parse --git-common-dir)/kgirl-private-names`, one per line,
  outside every worktree. The names never enter shell history, but while the scan runs they're visible
  in the process list.

`git push --no-verify` skips the gate. Use it only after reading every finding yourself.

The gate has its own offline tests, `python3 scripts/smoke/hook_tests.py`, which `check_all` and CI also run.
They build throwaway repos, never touch the real hooks, and cover:
- install, reinstall and uninstall, with the LFS hook kept and restored exactly
- a clean pass that hands identical input to the LFS hook
- blocks on a personal email, secrets, wallet files and private names, without printing them
- every fail-closed path
- a branch that ships a weakened scanner still being caught by the installed copy
