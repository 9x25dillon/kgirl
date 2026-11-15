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

    kgirl build spec.md -o lru.py      # writes lru.py and test_lru.py
    kgirl build spec.md --accept       # also keep the result as an active exemplar

The builder writes K test suites from the spec alone, samples N implementations, runs every
implementation against every test in a sandbox, picks the largest group of implementations that agree,
and repairs toward tests some other implementation passes. Accepted results become `exemplar` memories
recalled into later builds as worked examples. Over MCP the tool is `kgirl_build`; it needs
`KGIRL_MCP_BUILD=1` because it executes model-written code (the sandbox is a temp dir with rlimits, not a
security boundary), and its exemplars stay staged until a person promotes them.

## Writing

    kgirl write "a design note on X for the team" -w 900 -o note.md

The writer plans a JSON outline (using your preferences and relevant facts), drafts one section at a time,
critiques the draft against the brief, and revises only the sections the critique names.

## State

`$KGIRL_HOME/conversations.db` holds conversations and turns. Memory stays in `soup.db`. Both are private
(the home is created 0700).
