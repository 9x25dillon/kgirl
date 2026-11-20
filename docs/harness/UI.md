# kgirl ui — the terminal front end

    kgirl ui                           # workspace = the current directory, resumes its latest conversation
    kgirl ui --dir ~/proj              # work in another project directory
    kgirl ui --new                     # start a fresh conversation
    kgirl ui --conv 42                 # reopen conversation 42
    kgirl ui --no-workspace            # indexed repos and memory only, no file or command tools
    kgirl ui --yes                     # apply edits without asking (each one still undoable with /undo)

A split-screen terminal app: the rainbow KGIRL title over **chat** on the left and **status**, the
**job tray** and **code search** on the right. Below 100 columns the panes stack (chat on top, jobs and
search below, the status box hidden) so it stays usable in a narrow split.

It needs the optional `textual` package. Without it, `kgirl ui` says how to install it
(`python3 -m pip install textual`) and exits 2; nothing else in kgirl imports textual.

## How it works

`kgirl ui` builds the same in-process `Console` the web console uses (`console/core.py`): one Assistant,
one workspace, a Mind per conversation, long work as jobs under the one heavy lock, approvals as events.
The app is one more subscriber to the console Hub, so it sees exactly what the browser's SSE stream
sees: chat deltas, job progress, job ends, workspace notices and approval requests. There is no HTTP in
between and no second implementation: chat, task, build and helix jobs, code search, the standing rules
and the GPU probe are `Console` methods shared by both surfaces. The TUI-side adapter is
`ui/backend.py` (`HarnessBackend`), the only thing the app calls; tests hand the app a fake with the
same methods.

Nothing leaves the machine. The model is reached through the existing router; the status pane reads
`nvidia-smi` and Ollama on 127.0.0.1 only.

## Panes

| Pane | What it shows |
|---|---|
| chat | the resumed conversation's last messages, then your messages and kgirl's answers. An answer streams into a live strip under the log; when it ends, the final text (after any rules-check revision) goes into the log rendered as Markdown (`/md off` keeps it raw). Under each answer: rules chips (below), the turn meta lines (below), how long it took, recalled and newly remembered memories. Workspace notices (an edit applied, a command run) appear in yellow |
| mode line | the conversation id, `thinking…` while an answer streams, the project directory and whether edits ask first or apply automatically, whether events are shown, and **code changed · restart kgirl ui** when the restart hint fires |
| status | the chat and build models the router picked, the GPU (temperature, memory; yellow from 80 °C, red at 95 °C), the models Ollama has loaded (with the GPU share when one is not fully resident), whether a chunk index is configured, and any GPU warnings. Refreshes every 30 s and on ctrl+r |
| jobs | the six most recent console jobs: status dot, kind, id prefix, label and age. A running job silent for 2 minutes is flagged *stuck?*; a failed one shows its error |
| code search | Atlas symbols plus retriever chunks for the query (the console's search), query words highlighted |

### Rules chips

When the mind-rules lane's check fires on an answer, the answer turn's meta carries `rules_hits` (the rules
the first draft broke, so kgirl revised it) and `rules_left` (rules the final answer still breaks). The TUI
shows them like the console: a yellow **rules check: revised (…)** chip and a red **rules check: still
breaks …** chip. They show on live answers (the chat job's result carries the fields) and on resumed
history. `/rules` lists the standing rules themselves; editing them stays with `kgirl soup`.

### Turn meta lines

Answers carry more than text when the other lanes are present, and the TUI shows each part when it is
there (older turns show nothing):

- **budget** (voice lane): `context 3100/4000 · cut earlier, memory` — how much of the prompt budget the
  turn used and which blocks were cut. `/context` prints the full block-by-block report.
- **related** (voice lane): `related: conv #3 (same topic: deploy); conv #5 (matching words)` — the
  earlier conversations pulled into the prompt, and why.
- **audit** (guard lane): `⚙ edit x.py → applied`, `⚙ run make test → refused` — the workspace edits and
  runs of the tool calls behind the answer, in red when refused or failed.

### Approvals

With a project directory attached, kgirl asks before it edits a file or runs a command it has not been
allowed. The request opens a modal with the question and the diff or command: **y** approves, **n** or
**esc** declines. Several requests queue and open one after another. An unanswered request times out to
"no" after 10 minutes (the console's rule); when that happens the modal closes by itself.

### The restart hint

At start the TUI fingerprints the code under `KGIRL_CODE` (HEAD plus a hash of the uncommitted diff and
untracked `src` files, git reads only) and re-checks once a minute. When it drifts, the running process
no longer matches the code on disk: the chat pane says so once and the mode line keeps the hint. `/code`
checks on demand.

## Commands

| Command | What it does |
|---|---|
| `/new [title]` | start a new conversation |
| `/list` · `/open ID` | recent conversations · switch to one (shows its last messages) |
| `/find words` | search past messages across conversations |
| `/ask QUESTION` | a one-shot answer from the indexed repos and memory (`kgirl ask`) as a job; shown in the chat pane with its intent, model and citations, but not saved to the conversation |
| `/recap` | what this conversation is about |
| `/good` · `/bad` | your verdict on the last answer (overrides an inferred one) |
| `/rules` | the standing rules the rules check enforces |
| `/diff` | everything kgirl changed in the project this session, as a coloured unified diff |
| `/undo` | take back kgirl's last change (refused if the file changed since) |
| `/auto on\|off` | apply edits without asking, or ask first again |
| `/ws` | the project directory, the edit mode and the commands kgirl may run without asking |
| `/apply` | write the last task's verified diff into the project |
| `/verify CMD` | the command that proves a task's change (for every conversation in this session) |
| `/allow CMD` | let kgirl run CMD in the project without asking, for this session |
| `/context` | what the latest prompt held: each block's size and what the budget cut |
| `/write BRIEF` | long-form writing as a job; the result names the draft it was saved as |
| `/drafts` · `/resume N` | saved drafts with where each stands · continue draft N as a job |
| `/doctor` | the read-only health checks of `kgirl doctor`, run in-process |
| `/md on\|off` | render answers as Markdown (on) or keep the raw text |
| `/task GOAL` | sandboxed specialists work the goal as a job; when it finishes with a verified diff, `/job ID` shows it and `/apply` writes it |
| `/build SPEC` · `/build @file` | spec → tested code as a job; `/job ID` shows the report and the code |
| `/explain` | the latest finished build's explanation: generated suites, the candidate × suite votes, repairs and why the winner won (empty for a builder without `explain()`) |
| `/helix TASK` | a HELIX-9 run as a job (default budget 4000 tokens) |
| `/jobs` · `/job ID` | every job · one job's result (an id prefix is enough) |
| `/s words` | code search (also the search box, ctrl+f) |
| `/events` | show or hide tool and memory events while kgirl answers |
| `/code` | is the running code still the code on disk? |
| `/paste [text]` | open the multi-line composer (also ctrl+o): ctrl+s sends it as a message, ctrl+b starts a build with it as the spec, esc cancels |
| `/clear` · `/help` · `/quit` | clear the chat pane · the command list · leave |

## Keys

| Key | Action |
|---|---|
| ctrl+l | focus the chat prompt |
| ctrl+f | focus code search |
| ctrl+o | the multi-line composer |
| ctrl+n | new conversation |
| ctrl+r | refresh the status pane |
| f1, or `?` outside a text box | the help overlay (esc closes) |
| y / n / esc | answer an approval |
| ctrl+q | quit |

## What differs from the web console

- **Feature-detected extras.** `/context`, `/drafts`, `/resume` and `/doctor` say so when the lane behind
  them is not in this build, instead of failing.
- **A subset of views.** The TUI is the daily chat surface: chat, approvals, workspace, jobs, status and
  search. Memory curation, the Specialists table, Forge, Builder, Matrix, Helix state, Holo, Lanes,
  Pocket, Rescue and Services stay in `kgirl console`.
- **One answer at a time.** A second message waits until the current answer ends (the console would queue
  it on the heavy lock); jobs from `/task`, `/build` and `/helix` do queue there.
- **Long input goes through the composer.** The prompt is one line; ctrl+o (or `/paste`) opens a
  multi-line editor that sends a message or starts a build, and `/build @file` reads a spec file.
- **No forced reflection on quit.** Like closing the console, quitting joins any reflection already
  running and leaves the rest to the next turn; `kgirl chat`'s `/quit` reflects before it exits.
- **No HTTP surface.** There is no token, port or Origin check because nothing listens; the console's
  CSRF hardening is untouched.

## What differs from the old kgirl-ui

The earlier TUI lived outside the repo and ran `kgirl chat` as a subprocess per message. This one runs
in-process, so it can answer approvals (the old one had to refuse every edit), stream from the model,
show rules chips, keep a job tray and drive the workspace. It does not show the vault, second-disk or
router-sync rows, which came from a personal script outside kgirl. `/ask` is kept, now in-process as a
console job (the old one ran `kgirl ask` from a separate install). Code search uses the console's search (Atlas plus the configured chunk
index) instead of opening the chunk database directly. The project directory is attached by default,
as in `kgirl console`; `--no-workspace` turns file and command tools off.

## Layout

    src/kgirl/harness/ui/__init__.py   run(): feature-detect textual, pick the workspace, start the app
    src/kgirl/harness/ui/backend.py    HarnessBackend: the app's whole view of the harness (no textual)
    src/kgirl/harness/ui/app.py        the textual app: panes, event pump, approval and help modals
    tests/harness/ui/                  pilot tests on a fake backend (skipped without textual),
                                       backend tests on a real Console with a scripted model
