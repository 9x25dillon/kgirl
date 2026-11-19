# kgirl console — one live surface over the whole harness

    kgirl console                      # http://127.0.0.1:8765, workspace = the current directory
    kgirl console --dir ~/proj --port 9000
    kgirl console --no-workspace       # indexed repos and memory only, no file tools

A single-page app served by the harness itself (stdlib only, nothing leaves the machine). It drives the
same objects the CLI uses — no parallel implementation. Fourteen views:

| View | What it does |
|---|---|
| Console | chat with streaming answers, tool chips, recalled-memory markers, good/bad feedback, `/apply` for a waiting swarm diff, workspace diff/undo/auto/verifier. When the mind-rules lane is present, a stored answer whose rules check fired shows chips: "rules check: revised (push)" for `rules_hits`, a red "rules check: still breaks push" when `rules_left` is non-empty (feature-detected per turn; older turns show nothing) |
| Specialists | dispatch a Jev swarm on one goal; live agent table (steps, llm calls, tokens, verified), the winner's diff colored line by line, apply only after you read it |
| Memory | Soup fragments with utility/support/LCB, search and filters, promote/retire/credit, "remember …", the transition ledger. A read-only Rules panel lists the active preferences the mind-rules lane's `is_rule()` marks as standing rules (id, text, scope); it hides when that lane is absent. Editing rules stays with `kgirl soup` |
| Atlas | indexed repos, hybrid keyword+matrix search with citations, numbered source reading (secret-guarded), re-index |
| Forge | spec→code builder (confidence, agreement, module/tests/report tabs, accept as exemplar), long-form writer, evolve, skill forge + SKILL.md export |
| Builder | a read-only deep-dive into one spec→code build: the candidate matrix (style, temperature, tests passed/failed, repaired/dropped, the winner), the winning module, the agreed tests, the disputed tests with their output, and the suites as generated. When the builder provides `explain()`/`summary_dict()` (feature-detected per build), a collapsible "Why this winner" block and a candidate × suite vote table appear; older builders show neither. Nothing is applied or accepted here |
| Vitals | metabolism dashboard (A/C/O/F/R/P, predictions, affordable swarm), the swarm plan, and a read-only GPU & Ollama panel (nvidia-smi VRAM/util/temp, loaded models with the GPU/CPU split and context window, the active KGIRL_MIND_CTX; warns when a model is not 100% GPU-resident or the GPU hits 95 °C) |
| Matrix | a sampled spatial projection of the KGVM store (ChaosRNG random projection, colored by repo) and live inference: coarse sign-bit Hamming pass → fine int8 dot pass, with scores |
| Helix | run a task within a budget of generated tokens (experts → CRITIC → SYNTHESIST) as a job; the answer and flight recorder, the versioned STATE (six axis bars, ledger, open conflicts), per-version history with diffs, a diff of any two versions, a fail-closed handoff ingest, an offline transcript audit (the ported HELIX-9 detectors), the scaffold floor, and chain verification |
| Holo | the overlapping-matrix store: build it from the chunk index (offline, no model), a status panel (entities, shared digests, storage, chain health, the HoloMemory prior), live inference with the guidance block and per-hit provenance, a shared-digests panel (one content-addressed vector and the regions that own it), a read-only "verify chains" table (per-entity hash-chain health and unproven matrices), and `holo.built/query/regulate` bus events streaming into a panel |
| Lanes | a read-only window onto the shared porter ledger: every active agent session with its liveness and path claims, and the recent events, filterable by project and auto-refreshing — watch all the agents at once. A read-only Landing card shows each landing worktree's uncommitted path count (git status) and the exact `land_lanes.py` plan command; display only, the console never runs git writes |
| Pocket | read-only phone/node status from kgirl-pocket's local state files (last backup, restore drill, model benchmark) and a local TCP check of the adb-forwarded Ollama port. It never invokes adb and never sends a command to the phone |
| Rescue | read-only rescue scan: when the rescue lane's CLI exists (feature-detected), runs only its `scan --json` mode with a timeout and shows the table — platform, session, worktree, uncommitted count, STALE/live, last activity (relative age plus clock time) and last task, sorted STALE first then by uncommitted count. Rows carry the core's newer fields as chips: "+N older sessions" (hover lists them) and "N unmerged commits"; the worktree cell tooltips the full path and branch, and the uncommitted cell's tooltip splits the count into safe-to-commit vs private-skipped paths, with a "M private skipped" chip when any were skipped. An opt-in "auto" checkbox re-scans every 60 s while the view is active (off by default; a busy guard keeps scans from overlapping). A failed scan raises the persistent error banner too (a missing script stays a quiet hint — that is feature detection, not failure). Display only: the console never calls the rescue CLI's record or commit modes |
| Services | every MCP tool kgirl offers other platforms, callable from the browser with a JSON argument form |

A live rail streams every bus envelope (`jev.spawn/step/done`, `build.start/end`, `soup.propose`, …) and
job event. Workspace edits and unlisted commands raise an approval dialog; an unanswered approval times
out to "no".

## How it stays correct

- **One heavy lock.** Chat, task, build, write, evolve and index serialize: the metabolism keeps one
  episode slot and the GPU holds one model. Reads (memory, atlas, vitals, matrix) never wait on it.
- **The bus bridge only enqueues.** Bus handlers run on the publisher's thread (swarm workers), so the
  console's handler puts events on bounded queues; a slow browser drops events, it never stalls a run.
- **Jobs.** Long operations run in worker threads with bounded event logs; the browser watches deltas
  over SSE and fetches results by id. Reconnects get a hello snapshot (pending approvals, running jobs).
- **Failures are loud.** The header carries a job tray with running and failed counts; clicking it opens
  a drawer of the recent jobs (kind, label, elapsed time, status, error, and the last event text so a
  stuck or queued job says what it was last doing — a row opens the job's event card; Escape closes the
  drawer). A running job with no events for 120 s is flagged **stuck**: job summaries carry `last_event`
  (the newest event's timestamp, else the creation time) and a 5 s browser ticker measures the silence.
  A failed POST or a failed job raises a persistent error banner with a dismiss button, not only a
  fleeting toast; a newly stuck job banners once too. Tray, drawer and banner text is model-safe:
  it goes in through textContent only. Banners are `role=alert` (announced by screen readers) and
  drawer rows are keyboard-operable (Tab + Enter/Space).
- **The restart hint.** At startup the server fingerprints `KGIRL_CODE` (default `~/work/kgirl-stack`):
  git HEAD plus a short hash of the uncommitted diff and untracked `src` files — the same shape the
  helix-eval runner records as `code_rev`. `GET /api/code` re-reads it at most once a minute (git reads
  only: `rev-parse`, `diff`, `ls-files`); when it drifts from the startup value, the header shows
  "code changed on disk — restart the console to load it", tooltipped with the root and the
  startup → current fingerprints so you can see what moved. The running server keeps its loaded code;
  the hint only says so.
- **Regression cover.** `tests/harness/console/test_post_jobs.py` drives chat, build, write and task
  through the real `_post` HTTP path (real Assistant, Console and server; only the model is scripted)
  and asserts each job reaches done or error with exactly one matching `job_end` event — never stays
  "running". A dead model must end the chat job in error, and a POST without the launch token is 403.
- **Localhost only.** Binds 127.0.0.1 and rejects foreign Host headers (DNS rebinding). Static paths are
  confined; source reads pass the same secret guard chat uses.
- **No cross-site forgery.** Every `/api` call (GET, POST, DELETE and the SSE stream) must carry the
  per-launch token (`Console.token`, `secrets.token_urlsafe(32)`) as `X-Kgirl-Token`, or `?token=` for the
  EventSource which cannot set headers. The token is injected into `index.html` at serve time, so only a
  same-origin page can read it. A present `Origin` must be the console's own loopback origin, and a POST
  must be `Content-Type: application/json` — a cross-site "simple" request is refused (415) and a JSON
  request forces a CORS preflight this server never answers. A `critic_host` named from the browser must
  be loopback or listed in `KGIRL_CRITIC_HOSTS`, so a forged run cannot ship prompts and code off-box.
- **Approvals bridge.** `Workspace.confirm` becomes an HTTP round-trip with a 10-minute timeout that
  refuses — an unattended console never edits or runs anything silently.

## Layout

    console/core.py     Hub (SSE fan-out), JobManager, Approvals, Console (Assistant + Minds + workspace + token),
                        and the operations both front ends share: chat/task/build/helix jobs, search, rules,
                        the GPU probe, and the JSON shapes (plain, swarm_dict, build_dict, rules_meta)
    console/server.py   ThreadingHTTPServer, JSON API, static serving, the CSRF guards
    console/static/     index.html · style.css · app.js (no framework, no build step)

`kgirl ui` (docs/harness/UI.md) is the terminal front end over the same `Console`: it subscribes to the Hub
like the SSE stream does and calls the shared operations in-process, so a job started from either surface
has the same kind, label and result. A chat job's result now carries `rules_hits`, `rules_left` and
`rules_revised` (empty when the rules check did not fire), read from the stored answer turn.

Turn meta from the other lanes, each part feature-detected (older turns show nothing): an answer's
`budget` becomes a chip (`context 3100/4000 · cut earlier, memory`; click it for the `/context` report from
`GET /api/context?conv_id=`), each `related` entry a chip that opens that conversation (hover: why it was
pulled in), and a tool turn's `audit` records (action, target, outcome) one chip each, red when refused or
failed. The chat job's result carries the same `budget`, `related` and this exchange's `audit`.

Writing: `POST /api/write` results strip the writer's "(saved as draft #N …)" footer into `draft: N`, shown as
a chip above the text. Forge lists saved drafts (`GET /api/drafts`, `{available, drafts: [{id, conv_id,
title, progress}]}`) with a resume button (`POST /api/drafts/resume {id}` → a `write` job). Health: the Vitals
view's panel shows `GET /api/doctor` — `kgirl doctor`'s checks (`{available, checks: [{name, status, detail,
fix}]}`), computed in-process by the guard lane's `doctor.collect`, read-only. Without those lanes each
returns `available: false` and its panel hides or says so.

Builder route: `POST /api/builder/run` (the same builder call as `/api/build`, job kind `builder`, display-only).
Helix routes: `GET /api/helix/state|history|verify|diff`, `POST /api/helix/run|ingest|audit`. Holo routes:
`GET /api/holo/status|owners|overlaps|verify`, `POST /api/holo/build|query`. `verify` recomputes every
entity's hash chain and lists matrices recorded without a ledger entry (an unbuilt store returns `{}`
without creating the file); the Holo view shows it as a per-entity table behind "verify chains". The run/build/query POSTs start heavy jobs and
return `{job}`; the browser watches them over SSE and fetches the result by id, like every other operation.
Lanes route: `GET /api/lanes?project=&limit=`. It opens `porter.sqlite` (env `AURIC_PORTER_DB`, else
`~/.local/share/auric/porter.sqlite`) with a read-only SQLite URI and never writes; liveness mirrors
`auric.porter` (not ended, seen within `AURIC_PORTER_TTL_HOURS`, and a live pid on this host). No auric import
— the console stays stdlib-only. The same response carries `landing`: the stack and lane worktrees
(`KGIRL_LANDING_STACK` / `KGIRL_LANDING_LANES`, else `~/work/kgirl-stack` + eng/helix-ui/builder) with their
branch and uncommitted path count from `git status --porcelain` (a read; the only other git command is
`rev-parse --abbrev-ref HEAD`), plus the exact `scripts/smoke/land_lanes.py` plan command to run yourself. Pocket route: `GET /api/pocket`. It reads `~/.local/state/kgirl-pocket/
{backup,drill,bench}.json` and checks the forwarded port `127.0.0.1:11435` with a connect-and-close TCP probe
(no data sent, no adb, no phone command). Rescue route: `GET /api/rescue` — feature-detects
`~/work/kgirl-rescue/scripts/rescue.py` (env `KGIRL_RESCUE_SCRIPT` overrides) and runs
`python3 <script> scan --json` with a 30 s timeout; the mode is hard-coded to scan, so record and commit
are unreachable from the console. A missing script returns `{available: false}`; a failed, timed-out or
non-JSON scan returns an `error` the panel shows. `tests/harness/console/test_rescue.py` pins this contract:
a stub script logs its argv, and the test asserts only `["scan", "--json"]` ever runs. GPU route: `GET /api/gpu` — `nvidia-smi` (read-only query),
Ollama `/api/ps` plus a best-effort `/api/show` for the native context window, and `KGIRL_MIND_CTX`; it mutates nothing.
Rules route: `GET /api/rules` — `try: from ..mind.rules import is_rule` (the mind-rules lane); on ImportError it
returns `{available: false}` and the Memory panel hides. Otherwise it lists the active preferences `is_rule`
marks as rules, as `{id, text, scope}`; read-only. Code route: `GET /api/code` — the `CodeWatch` startup
fingerprint of `KGIRL_CODE`, throttled to one disk re-read per minute
(`{available, root, startup, current, changed}`).
`tests/harness/console/test_rules_meta.py`, `test_rules_panel.py` and `test_code_watch.py` pin the rules-meta
passthrough, the panel's feature detection and the fingerprint/throttle/read-only-git contracts.

The header carries the kgirl-ui block-letter KGIRL logo with a drifting rainbow gradient (pure CSS
`background-clip: text`, no library); `prefers-reduced-motion` shows a static gradient. When the
mind-rules lane is present, a "rules: on" chip appears beside the route chips (hover: how many
standing rules are active), read once from `/api/rules` at page load.

The `console` verb lives in `cli_mind.py`. The `helix` verb lives in `cli.py` (it forwards to the helix CLI);
the console calls the same helix and holo objects directly, not through that verb.