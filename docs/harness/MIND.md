# kgirl's mind: conversation, memory that learns, spec → code, writing

Everything runs on local models through the Hermes router (`chat` role for talking and writing, `smith`
for code). Nothing leaves the machine unless `KGIRL_CHAT_CLOUD=1` and local-only mode is off.

## Talk to it

    kgirl chat                 # resume the latest conversation (interactive; /help for commands)
    kgirl chat --new           # start a new one
    kgirl chat "one question"  # one turn, then exit
    kgirl chat --dir ~/proj --verify "python3 -m pytest -q"   # workspace + verifier for the task tool
    kgirl convo list | show ID | search QUERY | export ID | done ID C2 | delete ID

Each turn kgirl builds its prompt from:

| Block | Source | Cut order |
|---|---|---|
| persona, **RULES**, tools, workspace, date | fixed text; RULES are the user's "Never/Always ..." preferences | never |
| WHO THE USER IS | the profile `kgirl evolve` maintains | 5th |
| WHAT THE USER WANTS | active `preference` fragments relevant to the message | 6th |
| MEMORY | facts, lessons, skills, antipatterns and episodes (summaries of other conversations) | 2nd |
| EARLIER CONVERSATIONS | earlier conversations on the same **topic** (shared tags), then full-text matches over stored turns | 1st |
| CODE THIS MESSAGE NAMES | exact Atlas matches for the files and symbols the message names | 3rd |
| THIS CONVERSATION SO FAR | the rolling summary, **open commitments** and open threads | 7th (last) |
| messages | this exchange, whole; older turns fill what the budget leaves | older turns 4th |

### One budget for all of it

The context window (`KGIRL_MIND_CTX`; the router default is 8192 and the launcher sets 24576) is shared, not filled block by block
(`mind/budget.py`). The prompt may use `(context - answer room) * 0.92` tokens. The fixed blocks, the message
being answered and a reserve for tool results always stay. If everything else does not fit, blocks lose their
last items in the order of the table (the "Cut order" column; the lowest goes first), whole items first and the
last item that remains clipped only when 40 tokens of it survive. Older turns are dropped oldest first and never
clipped. The rule is deterministic: the same blocks and budget always give the same prompt. The rules are never
cut; if the fixed blocks alone are over the limit, the overflow is reported, not hidden. Memory behind a cut item
is not counted as "recalled", so it is not credited or blamed for the answer.

What the budget did is kept with each answer (`meta["budget"]`: `limit`, `used`, `overflow`, per-block `blocks`,
and `dropped` with items, tokens and clipped) and `/context` in `kgirl chat` prints it:

    prompt budget 20300 tokens, used 6120 (context 24576, 1600 kept for the answer)
      PERSONA         312  never cut
      RULES            48  never cut
      ...
      EARLIER           0  cut 190 tokens, 3 item(s) dropped

### Commitments, threads and topics

Reflection (every 3 user turns and on exit) reads the open threads and commitments from before and returns the
new state. A **commitment** is something said would be done later, by either side ("I'll check the lockfile",
"next time I will read the changelog first"). Each is stored per conversation with an owner (`kgirl` or `user`)
and a status, shown in THIS CONVERSATION SO FAR while open, and closed when the reflection reads a stated
outcome or the user saying it is done, or by hand: `/done C2` in chat or `kgirl convo done ID C2` (a piece of its
text works too). `kgirl convo show ID` lists every commitment with its status; `convo export` includes the
summary, topics, threads, commitments and drafts. Reflection also writes a few **topic tags** for the conversation.

EARLIER CONVERSATIONS picks earlier conversations that share a tag with this conversation or with words in the
message (rarer tags count for more; a conversation of another project is left out), at most 3, and says why in the
prompt ("same topic: rust, parser") and in `meta["related"]` (`via: "tags"` with the shared tags, or `via: "words"`).
Threads, commitments and tags are conversation state: they never become habits.

It can call tools in the middle of an answer: `search_code`, `read_file`, `recall`, `remember`, `repos`,
`build`, `write`, `task`. Text streamed before a mid-answer tool call is stored with the turn, so the
record and the next prompt both keep it. With a workspace attached (`kgirl chat --dir .`, the default in a
project directory) it also gets `ls`, `grep`, `edit`, `write_file` and `run`; edits ask first and stay
undoable (`/diff`, `/undo`, `/auto on` to stop asking, `/allow CMD` to permit a command).

`task` hands a coding goal to a swarm of sandboxed Jev agents and returns a verified diff for review —
never applied on its own. Its verifier must be the session verifier (`/verify CMD`) or one of the user's
`KGIRL_VERIFY_ALLOWLIST` commands; the person reads the diff and applies it with `/apply`, which is also
what promotes the run's staged lessons. `/tools` lists what a session offers.

The `build` tool with `edit` (a workspace file, plus optional `tests`) uses the builder's edit mode: it comes back
as a diff for review and is never applied by the model. `/apply` writes it through the workspace (so you approve it
and `/undo` takes it back) and refuses if the file changed since the build. `/doctor` runs the health checks
(`kgirl doctor`) in-process and prints one line per check.

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
   *another* conversation reinforces it (also when worded differently), and the curator promotes it to an
   active habit only when its support is ≥ 2 and its utility lower bound clears the policy. A conversation that
   says the same thing again on a later pass adds nothing. One conversation can suggest a habit; only
   repetition makes it one.
4. **Evolve** (`kgirl evolve`, run between sessions): reflects on anything pending, runs the curator
   (decay, promote, retire, capacity), rewrites the user profile, forges skills, and with `--dataset`
   exports approved exchanges as chat JSONL for a later local fine-tune. `kgirl evolve --dry-run` reports what
   it would promote, retire, expire or rewrite, and changes nothing (no model call, no ledger entry, no file;
   it cannot show the decay the real pass applies first).

Every transition is in the Soup ledger (`kgirl soup ledger`) and can be rolled back.

## Spec → code

    kgirl build spec.md -o lru.py                                  # writes lru.py and test_lru*.py
    kgirl build spec.md --edit path/in/workspace.py                # edit an existing file; outputs a unified diff for review
    kgirl build spec.md --edit calc.py --tests tests/test_calc.py  # edit mode with explicit existing test file(s)
    kgirl build spec.md -n 6 -k 3 -r 2 --explain                   # sample counts, repair rounds, full vote breakdown
    kgirl build spec.md --strict --budget-s 120                    # lint spec before sampling; cap wall-clock/GPU seconds
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
passes or static contract mismatches against the spec (ranked by candidate support, fed exact failing
test IDs, trimmed tracebacks with last non-unittest frame plus assertion capped by tokens, the failing
test method source, a passing donor candidate snippet, and any regressions from an earlier round).

- **Edit mode (`--edit FILE`, optional `--tests PATH`)**: samples unified-diff candidates against `FILE`,
  applies each diff in pure Python, and runs the file's existing tests (auto-discovered beside `FILE` or
  named with `--tests`) plus the new spec tests in the same sandbox. A candidate passes only when all
  existing tests and all credible spec tests are green; the winner is returned as a unified diff for
  review and is never applied automatically.
- **Multi-file specs**: a spec with `## module: a.py` / `## module: b.py` sections is built, gated
  (`imports_all_ok`), and tested as one unit, with companion modules emitted in `BuildResult.extra_files`.
- **Chat summary (`BuildResult.summary()`)**: compact ≤ 6-line summary of what won and why for the chat
  `build` tool.

### Knobs
| Flag | Default | Meaning |
|---|---|---|
| `-o, --out PATH` | stdout | Write the winning module, any companion modules (`extra_files`), and retargeted agreed test files (`test_<stem>.py`, `test_<stem>_2.py`, …) beside `PATH` |
| `--edit FILE` | none | Edit an existing Python file from the spec; runs existing + new spec tests and prints a unified diff for review (never applied) |
| `--tests PATH` | auto | Repeatable existing test file to run in `--edit` mode (otherwise auto-discovered beside `FILE`) |
| `-n INT` | `6` | Candidate implementations to sample across spread temperatures and alternating `direct` / `edge-first` styles |
| `-k INT` | `3` | Independent unittest suites generated from the spec before any code is written |
| `-r, --repair-rounds INT` | `2` | Targeted repair rounds toward credible failing tests |
| `--strict` | off | Lint the spec for ambiguities and missing examples before sampling, stopping early on warnings |
| `--budget-s FLOAT` | none | Wall-clock/GPU budget in seconds; stops sampling/repair cleanly when reached and reports what it had |
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
    kgirl write --list                 # saved drafts, newest first
    kgirl write --resume 3             # continue draft 3 where it stopped

The writer (`mind/writer.py`) works in steps, each a small call:

1. **Outline**: brief + the user's STYLE + relevant facts → title, audience, thesis, and sections with points and a
   length target. With `-w`, sections without a target share the total and a plan that misses it by over a quarter is
   scaled to it.
2. **Draft**: one call per section, given the outline, what the earlier sections already covered (so it does not repeat
   them) and the tail of the previous one.
3. **Critique**: one call against the brief. With `-w`, a section more than 1.5x or less than 0.5x its target is also
   an issue (found by counting words, not asked of the model).
4. **Revise**: only the sections an issue names, once each.

STYLE is the user's voice: the profile `kgirl evolve` keeps and the preferences recalled for the brief. It goes into every
call, apart from CONTEXT (facts and past conversations to use, not imitate).

Every step is saved as a conversation artifact (kind `draft`; `kgirl write` keeps it in the latest conversation, or
`--conv ID`; the `write` tool and `/write` in chat use the current one). Saving a draft does not make a conversation the
"latest", so `kgirl chat` still resumes the one you were in. `--resume ID` (or `/write --resume ID`) continues from the
step that stopped, with the brief, context, style and length target the draft started with: no section is drafted twice
and no revision repeated. Ctrl-C leaves the draft saved.

## State

`$KGIRL_HOME/conversations.db` holds conversations and turns. Memory stays in `soup.db`. Both are private
(the home is created 0700).
