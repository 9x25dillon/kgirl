# kgirl harness — a repo-aware personal coding assistant

`src/kgirl/harness/` is an agentic assistant that knows your repositories by their
**structure**: signatures, docstrings, imports, symbol references, and clone hashes. It does
not hold their text. It combines four subsystems:

| layer | role | one-line contract |
|---|---|---|
| **Atlas** | structural memory of every repo | `ingest(path)`, `search(q)`, `blast_radius(target)`, `coupling(hub)`, `architecture_card(repo)` |
| **Soup** | shared, task-adaptive memory pool | `recall(q, budget)` (just-in-time), `credit(ids, ok)`, `Curator.propose/step` (staged RSI) |
| **Hermes** | messenger: bus + router | every cross-layer call is an `Envelope` on a topic; roles → model fallback chains |
| **Jev** | Mr. Meeseeks agents | one goal, tiny context, one-line decisions over a numbered table; run as sacrificial swarms |

The core is standard-library only (Python ≥ 3.10, SQLite FTS5). The optional extras are
`anthropic` (Claude), a running Ollama (local models), and `playwright` (browser Jev).

```
                         ┌───────────────────────── Hermes bus ──────────────────────────┐
 user / Claude Code ──▶  │ assistant.ask ─▶ atlas.search ─▶ soup.recall ─▶ llm.complete   │──▶ answer + citations
   (CLI or MCP)          │ assistant.task ─▶ swarm.start ─▶ jev.spawn×N ─▶ jev.step …     │
                         │                 ─▶ jev.done / jev.poof ─▶ soup.propose/curate  │──▶ verified diff
                         └────────────────────────────────────────────────────────────────┘
        Atlas (atlas.db)               Soup (soup.db)                  Router (role → chain)
  files · symbols · imports       fragments · FTS · ledger      scout: Claude → local coder
  refs · clone hashes · FTS       utility = Beta posterior      smith: local coder (→ Claude if allowed)
                                                                jev:   small local → local coder
```

## Quick start

```bash
export PYTHONPATH=src                     # from the repo root
python -m kgirl.harness index . owner/other-repo ../local-repo   # paths, owner/repo, or git URLs
python -m kgirl.harness coupling --hub kgirl                     # which repos kgirl is entangled with
python -m kgirl.harness blast src/kgirl/llm/llm_adapters.py      # who breaks if this changes
python -m kgirl.harness blast "llm_adapters.py::LocalStubAdapter"
python -m kgirl.harness card numbskull --summarize               # architecture card (+ Claude summary)
python -m kgirl.harness ask "how does the dual LLM orchestrator pick a backend?"
python -m kgirl.harness task "make tests/test_x.py pass" --repo . \
       --verify "python -m pytest -q tests/test_x.py" -n 3        # Jev swarm; add --apply to write the diff
python -m kgirl.harness report -o docs/harness/ATLAS_REPORT.md
python -m kgirl.harness route                                    # which backend serves which role
python -m kgirl.harness soup stats | soup recall "…" | soup ledger | soup export-sft out.jsonl
```

State lives in `$KGIRL_HOME` (default `~/.kgirl`): `atlas.db`, `soup.db`, `trace.jsonl`, and
`repos/` (a shallow-clone cache).

### Inside Claude Code (MCP)

`.mcp.json` at the repo root registers the harness as the `kgirl` MCP server. Claude Code
then gets these tools: `atlas_search`, `atlas_outline`, `atlas_source`, `atlas_blast_radius`,
`atlas_coupling`, `atlas_card`, `soup_recall`, `soup_remember`, `kgirl_ask`,
`jev_swarm_task`, `skills_forge`, `evolve_assay_design`, and the assay tools `ecl_quantify`,
`ecl_protocol` and `ecl_gap`
(see `docs/nihiline/README.md`). Run `index` once, before first use.

Three tools change state, so the MCP server holds them to what the user allows, not what the model asks:

| tool | rule over MCP | setting |
|---|---|---|
| `soup_remember` | stored as `staged` with source `mcp`; never auto-promoted, so never recalled until a person runs `python -m kgirl.harness soup promote <id>` | — |
| `jev_swarm_task` `verify` | must equal one of the user's allowed commands; verifiers run without `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*` variables | `KGIRL_VERIFY_ALLOWLIST="python -m pytest -q; npm test"` |
| `jev_swarm_task` `apply` | only into the root of an indexed repo, and only when the user opted in | `KGIRL_MCP_APPLY=1` |

`skills_forge` also writes to Soup, and isn't yet held to a user setting:

- It stores forged skills as `active` fragments with source `forge`.
- The model chooses the support and utility thresholds.
- It writes no files. Exporting to `.claude/skills` is CLI-only (`skills forge --export`).

`initialize` answers with a protocol version the server supports (`2025-06-18`, `2025-03-26`, `2024-11-05`) and
sends instructions that Atlas and Soup text is data, not instructions.

## Model routing (Hermes router)

| role | used for | default chain | env |
|---|---|---|---|
| `scout` | answers from context packs, repo summaries, distilling trajectories | `claude-haiku-4-5` → local coder | `KGIRL_SCOUT_MODEL`, `ANTHROPIC_API_KEY` |
| `smith` | code edits (SEARCH/REPLACE blocks) | Ollama `qwen2.5-coder:7b` (Claude only if `KGIRL_ALLOW_CLOUD_CODE=1`) | `KGIRL_SMITH_MODEL`, `OLLAMA_HOST` |
| `jev` | one-line step decisions | Ollama `qwen2.5-coder:1.5b` → smith model | `KGIRL_JEV_MODEL` |

A backend error moves the call to the next link in its chain. When a chain is exhausted,
`NoBackend` is raised with every link's reason. `ask` then degrades to an extractive answer:
the cited context pack itself.

## Core theory

### Atlas — structure instead of text *(Level 0)*
- **Extraction.** Python goes through `ast`, which gives exact signatures, a normalized body
  hash that ignores docstrings and formatting, calls and names, and `__main__` guards. JS/TS,
  Julia, Go, Rust, C/C++, Java, Kotlin, Swift, Zig, GDScript and shell use line-anchored regexes.
  Markdown headings become `section` symbols. A Python file that fails to parse falls back to the
  regex extractor and is flagged.
- **Cross-repo resolution.** Many of your repos use bare module imports
  (`from numbskull_dual_orchestrator import …`). An import resolves to any indexed file whose
  dotted path ends with the import path. The preference order is: same repo, then longest shared
  directory prefix, then shortest path. Stdlib names are never resolved.
- **Blast radius.** This is a reverse BFS over resolved-import edges up to depth *d*. When the
  target is a symbol, the first hop keeps only importers that actually name or reference that
  symbol (`ref` edges). Clone siblings, meaning identical normalized bodies elsewhere, are added at
  depth 1. Weight = Σ over discovered paths of 1/depth.
- **Incremental.** Files are keyed by `sha1(PARSER_VERSION ‖ content)`. Unchanged files are
  skipped, and bumping the parser version re-parses everything. Indexing all 33 repos in
  `ATLAS_REPORT.md` takes about 15 s in total.

### Soup — just-in-time, task-adaptive memory *(mechanism Level 0; benefit Level 2)*
- A **fragment** has one of the kinds `fact | abstraction | trajectory | preference | antipattern | note`
  and a scope (`*` or a repo name).
- **Recall** takes BM25 candidates and scores each as
  relevance × (0.35 + 0.65·utility) × (0.5 + 0.5·recency) × support. It then packs fragments
  greedily under a hard token budget and drops any fragment whose word-bigram Jaccard with an
  already-picked one is ≥ θ. The `offset` parameter gives each swarm agent a different slice.
- **Utility** = (wins+1)/(wins+losses+2), the Beta(1,1) posterior mean. After a swarm, the
  winner's recalled fragments get +1 and the losers' recalled fragments get −½. This is the
  "inference-time training": memory adapts to outcomes without touching model weights.
  `soup export-sft` writes accepted trajectories as JSONL for when you do want LoRA fine-tuning
  of the local coder.

### Staged, regularized RSI (Soup curator) *(Level 0 mechanics)*
The harness improves itself **only through memory**. It never rewrites its own code or weights.

```
propose ─▶ staged ──(support ≥ k ∧ LCB(utility) ≥ p ∧ validator)──▶ active
              └─(TTL, support < k)─▶ retired ◀─(UCB < r after ≥ n evidence │ capacity overflow)
```
The regularizers are:
- staging: nothing a single run proposes is recalled until k independent successes reinforce it
- manual sources: fragments a model wrote over MCP (source `mcp`) are never auto-promoted; a person promotes them
- decay γ: evidence shrinks toward the prior each step, so old wins have to be re-earned
- bounded active capacity
- confidence bounds on both promotion and retirement
- an append-only ledger with `rollback(ts)`

### Jev — Meeseeks agents *(Level 0 design, measured in tests)*
- **Indexed action space.** Every observation is a numbered element table. The model replies with
  one line, `OP [i] [:: text]`, and `parse_decision` enforces the closed op set, that the index
  exists in the *current* table, and that the argument is inert text. Model output never becomes a
  command, selector or path.
- **Code world** ops: `SEARCH`, `OPEN`, `EDIT`, `BLAST`, `RUN`, `DONE`, `BLOCKED`.
  - Only `EDIT` generates text, through the *smith* role. Each SEARCH/REPLACE block applies only
    if its SEARCH text occurs **exactly once** in the file.
  - `RUN` executes the verifier **you** supplied.
  - Workspaces are per-agent copies. Files over 5 MB are read-only symlinks, and edits that
    resolve outside the workspace are refused.
- **Browser world** (Jev Ultrafast style, needs Playwright): `CLICK`, `TYPE_TEXT`, `SELECT`,
  `SCROLL_UP/DOWN`, `WAIT`, `DONE`, `BLOCKED`.
  - Observations are structured: an interactive-element table plus a 300-char text excerpt. There
    are no screenshots.
  - Targets are re-validated against a fresh snapshot signature before any action; a stale target
    is refused.
- **Context discipline.** There is no chat history. Every step is one fresh prompt: goal +
  memory slice + last 3 results + the table, at about 300–800 tokens.
- **Sacrificial swarm.** N agents run with different temperatures and memory slices. The first
  *verified* DONE cancels the rest. The swarm re-runs the verifier independently. The winner is
  the accepted result with the fewest steps and tokens. The losers vanish; their only trace is
  credit assignment. `--apply` writes the diff only to files unchanged since the copy was taken,
  and reports conflicts otherwise.

## Intuition (System 1) and the skill forge

**Intuition.** Accepted runs are stored with their *routine*: the ordered list of
`(op, target key)` pairs. Target keys drop line numbers, so routines survive code motion.

On a later run, Jev checks intuition before calling the model:
- It looks for routines whose goal resembles this one and whose steps so far match its own.
- It resolves the routine's next target against the current table.
- If the combined confidence clears the threshold, it acts **without a model call**.
- If an intuitive step fails, the agent is *surprised*: intuition switches off for that run and the model takes over.

Routines that carry a verified win gain utility; routines that mislead an agent lose it. Only
`intuition_fraction` of the swarm uses intuition, so the rest keeps exploring.

On the calc fixture, Jev model calls per swarm dropped from 7 to 1 on the second run, and an
agent that failed the first time solved the task with zero model calls.

**Skill forge.**

```bash
python -m kgirl.harness skills forge --min-support 2 --export .claude/skills
```

- It groups verified trajectories by routine shape and keeps those with enough support and utility.
- Each one is stored in Soup as `kind="skill"`.
- Each one is exported as a Claude Code `SKILL.md`, with when-to-use goals, steps, the files it touched and the verifier that proved it.
- Hand-written skills are never overwritten.

Three hand-written skills ship in `.claude/skills/`: `kgirl-blast-radius`, `nihiline-assay` and `evolve-designs`.

## Testing

```bash
python -m unittest discover -s tests/harness -t .     # 43 tests, ~10 s; no network, no API keys
```
The tests cover:
- parsing: Python, TypeScript, Julia, broken files, clone hashes
- cross-repo, relative and TS import resolution; blast radius (file, symbol, clone); coupling;
  incremental ingest; venv-shim skipping
- Soup: recall budget, diversity, credit ranking, scope, staging → promotion, validator gate,
  retirement, capacity, decay, rollback
- Hermes: ordering, glob routing, causal trace, hop bound, handler isolation, request/reply,
  fallback chains
- Jev: decision contract, exactly-once edits, sandbox escape, a full open→edit→run→done fix,
  swarm accept/apply/learn/promote, rejection of unverified runs, apply conflicts
- MCP: handshake, tool calls and errors, a real stdio round trip
- Browser Jev: a real Chromium form fill, skipped when Playwright is absent

## Failure modes and mitigations

| failure | effect | mitigation |
|---|---|---|
| name collision in bare-module resolution (two repos define `utils.py`) | wrong cross-repo edge | same-repo and closest-directory preference; stdlib excluded; edges carry evidence lines |
| regex languages miss exotic syntax | missing symbols | Python is exact; others are best-effort by design; bump `PARSER_VERSION` after improving patterns |
| small local model emits junk | wasted steps | strict parser; 2 consecutive invalid replies end the agent; swarm diversity |
| model "fixes" tests instead of code | false success | the verifier is yours; the independent re-run happens in the swarm, not the agent |
| memory poisoning by one lucky run | bad rules recalled | staging with k ≥ 2, LCB promotion, decay, UCB retirement, ledger rollback |
| prompt-injected agent writes memory over MCP | instruction recalled in every later session | `soup_remember` stages under source `mcp`; the curator never auto-promotes it; `soup promote` is a person's call |
| model-chosen verifier command | arbitrary program runs with the user's keys | `KGIRL_VERIFY_ALLOWLIST`; secret-named variables removed from verifier environments |
| model-chosen apply target | diff written into an unrelated directory | `KGIRL_MCP_APPLY=1` and an indexed repo root required |
| Claude unreachable / no key | no prose answers | falls back to the local model, then to an extractive context pack |

## Evidence vs speculation

- **Level 0 (established):** AST extraction, FTS5/BM25 retrieval, reverse-dependency closure,
  Beta-posterior credit assignment, confidence-bound admission, and a verifier-gated agent loop.
  All are implemented and tested here.
- **Level 1:** a tiny-context, indexed-action agent with a 1.5B–7B local model can close small,
  test-specified tasks. This follows from the constrained action space, but it has not been
  benchmarked on your repos yet.
- **Level 2:** curated Soup memory measurably raises swarm success rate and lowers tokens per
  task over time. Measure it with `soup stats` and the `swarm.end` / `jev.done` events in
  `trace.jsonl` (steps, tokens, verified).
- **Next experiment:** 20 seeded bug-fix tasks from kgirl's history. Run with memory off, then
  on. Compare success@N and median tokens, and use a paired sign test to check for significance.
