# kgirl's mind: conversation, memory that learns, spec → code, writing

Everything runs on local models through the Hermes router (`chat` role for talking and writing, `smith`
for code). Nothing leaves the machine unless `KGIRL_CHAT_CLOUD=1` and local-only mode is off.

## Talk to it

    kgirl chat                 # resume the latest conversation (interactive; /help for commands)
    kgirl chat --new           # start a new one
    kgirl chat "one question"  # one turn, then exit
    kgirl chat --dir ~/proj --verify "python3 -m pytest -q"   # workspace + verifier for the task tool
    kgirl convo list | show ID | search QUERY | delete ID

Each turn kgirl builds its prompt from:

| Block | Source |
|---|---|
| WHO THE USER IS | the profile `kgirl evolve` maintains |
| WHAT THE USER WANTS | active `preference` fragments relevant to the message |
| MEMORY | facts, lessons, skills, antipatterns and episodes (summaries of other conversations) |
| EARLIER CONVERSATIONS | full-text matches over every stored turn |
| THIS CONVERSATION SO FAR | the rolling summary and open threads |
| messages | the most recent turns that fit the history budget |

It can call tools in the middle of an answer: `search_code`, `read_file`, `recall`, `remember`, `repos`,
`build`, `write`, `task`. Text streamed before a mid-answer tool call is stored with the turn, so the
record and the next prompt both keep it. With a workspace attached (`kgirl chat --dir .`, the default in a
project directory) it also gets `ls`, `grep`, `edit`, `write_file` and `run`; edits ask first and stay
undoable (`/diff`, `/undo`, `/auto on` to stop asking, `/allow CMD` to permit a command).

`task` hands a coding goal to a swarm of sandboxed Jev agents and returns a verified diff for review —
never applied on its own. Its verifier must be the session verifier (`/verify CMD`) or one of the user's
`KGIRL_VERIFY_ALLOWLIST` commands; the person reads the diff and applies it with `/apply`, which is also
what promotes the run's staged lessons. `/tools` lists what a session offers.

## How it learns (and why it stays sane)

1. **Feedback**: "thanks, that works" or "no, that's wrong" credits or debits every memory the previous
   answer was built from. Useful memories rise in recall; misleading ones sink and are retired. A reaction
   counts only when it leads the message and addresses the answer — "the test fails when the input is
   empty" describes a problem and debits nothing; a question is never a verdict; "great, but it still
   fails on empty input" is negative. Reflection still reads the whole message. `/good` and `/bad` say it
   outright.
2. **Explicit memory**: "remember that …" is stored at once as your own words.
3. **Reflection** (every 3 user turns, and on exit): the model rewrites the conversation summary and
   *proposes* facts, preferences and lessons. A proposal is staged. The same thing observed again in
   another reflection reinforces it, and the curator promotes it to an active habit only when its support
   is ≥ 2 and its utility lower bound clears the policy. One conversation can suggest a habit; only
   repetition makes it one.
4. **Evolve** (`kgirl evolve`, run between sessions): reflects on anything pending, runs the curator
   (decay, promote, retire, capacity), rewrites the user profile, forges skills, and with `--dataset`
   exports approved exchanges as chat JSONL for a later local fine-tune.

Every transition is in the Soup ledger (`kgirl soup ledger`) and can be rolled back.

## Spec → code

    kgirl build spec.md -o lru.py                                  # writes lru.py and test_lru*.py
    kgirl build spec.md -n 6 -k 3 -r 2 --explain                   # sample counts, repair rounds, full vote breakdown
    kgirl build spec.md --repo kgirl --context src/kgirl/paths.py  # build against an existing indexed repo file
    kgirl build spec.md --context ./helpers.py -o mod.py           # build against a local companion file
    kgirl build spec.md --no-exemplars                             # cold-start build without recalling Soup exemplars
    kgirl build spec.md --accept                                   # confirm and stage the winner as a Soup exemplar

The builder (`mind/builder.py` and `mind/build_spec.py`) writes `K` test suites from the spec alone
(suite 1 direct, later suites focused on boundaries/errors and contracts/invariants with a static
contract checklist pulled from the spec), normalizes each suite so stateless multi-assertion methods
split into one test per assertion, samples `N` implementations, and runs every implementation against
every test in a sandbox. It picks the highest-scoring CodeT dual-agreement group (`members × weighted
tests passed`, where `test_<name>_1..M` siblings split from one multi-assert method carry weight `1/M`
each so splitting a method never inflates its vote weight; tie-broken by raw tests passed, suite
breadth, per-test candidate consensus, fewer static contract mismatches, and code length; zero-passing
clusters score 0), then runs up to `R` targeted repair rounds toward credible tests some other candidate
passes or static contract mismatches against the spec (ranked by candidate support, showing the failing
test method source, a passing donor candidate snippet, and any regressions from an earlier round).

### Knobs
| Flag | Default | Meaning |
|---|---|---|
| `-o, --out PATH` | stdout | Write the winning module, any companion modules (`extra_files`), and retargeted agreed test files (`test_<stem>.py`, `test_<stem>_2.py`, …) beside `PATH` |
| `-n INT` | `6` | Candidate implementations to sample across spread temperatures and alternating `direct` / `edge-first` styles |
| `-k INT` | `3` | Independent unittest suites generated from the spec before any code is written |
| `-r, --repair-rounds INT` | `2` | Targeted repair rounds toward credible failing tests |
| `--explain` | off | Print the four-part explanation (`SUITES`, `CANDIDATE VOTES & AGREEMENT GROUPS`, `REPAIR STEPS`, `WHY THE WINNER WON`) instead of the short summary (also `/explain` in `kgirl chat`) |
| `--repo NAME` | workspace | Indexed Atlas repository to read `--context` paths from |
| `--context PATH` | none | Repeatable path (in `--repo` or on disk) whose source is checked for secrets, added to the prompt context, and emitted alongside the module |
| `--accept` | off | Store the result as an **active** exemplar right away (also `/accept` in chat: the user's word makes it active immediately and credits the exemplars it built from) |
| `--no-exemplars` | off | Skip recalling active Soup exemplars for a clean cold-start build |
| `--allow-io` | off | Permit file I/O in static lint when the spec requires reading or writing files |

### How to read `--explain`
- **`=== SUITES ===`**: Lists each surviving generated suite (`test_gen0`, `test_gen1`, …) and every atomic test ID (`test_gen0:Class.test_x`), with the count and `#idx` list of live candidates that passed it (`0/N` means no candidate satisfied that test, so it is not treated as credible).
- **`=== CANDIDATE VOTES & AGREEMENT GROUPS ===`**: Shows each sampled candidate (`#0..#N-1`), its style (`direct` or `edge-first`), temperature, character length, either `DROPPED — <reason>` or the exact list of tests it passed, followed by the ranked dual-agreement groups (`members × tests -> score` with `1/M` sibling weighting, suites spanned, and shortest member).
- **`=== REPAIR STEPS ===`**: For each repair round (`#100`, `#101`, …), shows `before -> after` test counts, `+gained` test IDs, `-regressed` test IDs, and whether the round was `ACCEPTED (strict superset)` or `REJECTED`.
- **`=== WHY THE WINNER WON ===`**: States in plain steps which initial agreement group won (and how its dual-agreement score beat the runner-up group) and whether a repair candidate superseded the initial pick by passing a strict superset of tests.

### Exemplars & evaluation
`/accept` and `kgirl build --accept` are the user's word, so the exemplar becomes **active** at once
(with the agreed test suite stored in `fragment.data`) and credits the exemplars it was built from.
Unconfirmed automatic write-back (and anything arriving over MCP) is stored as **staged** until a
person promotes it (`kgirl soup promote ID`). Active exemplars are recalled into later builds when
their `SPEC:` header shares domain terms with the new spec, ranked by domain overlap and Soup's
smoothed `(wins + 1) / (wins + losses + 2)` utility. In the preregistered `spec/run.py` experiment,
primary Arm B runs with `soup=None` so the pipeline test is not confounded by prior memory. Over MCP
the tool is `kgirl_build`; it needs `KGIRL_MCP_BUILD=1` because it executes model-written code.

## Writing

    kgirl write "a design note on X for the team" -w 900 -o note.md

The writer plans a JSON outline (using your preferences and relevant facts), drafts one section at a time,
critiques the draft against the brief, and revises only the sections the critique names.

## State

`$KGIRL_HOME/conversations.db` holds conversations and turns. Memory stays in `soup.db`. Both are private
(the home is created 0700).
