# Emergent Workflows

Context Relay, Phase 0 design input for Checkpoint A. Status: **proposed**, not frozen.
The tables marked *generated* come from [`model/relay-model.json`](../model/relay-model.json) via
`node tools/relay-model.mjs sync-docs`, and CI fails if they drift from the model.

Evidence levels used below: **L0** established engineering fact or verified platform documentation; **L1** reasonable extrapolation;
**L2** hypothesis that needs measurement; **L3** speculative.

---

## 1. Executive summary

- **The pack format is a hub.** Context Relay defines 8 ways a pack comes into existence (entries) and 9 ways it is consumed (exits).
  These 17 route definitions compose into **72 checked workflows**. The product's value is that ratio: supporting a new surface costs
  O(1) adapters and adds O(n) workflows. This is the same argument that makes LSP and compiler IRs work. **(L0)**
- **Composability is also the main risk.** Because every entry reaches every exit, the safety of a workflow can't depend on the route.
  It has to live in the pack itself, as a **type**, a **label set** and an **origin trust**. The model enforces it with 15 invariants.
  The checker accepts all 13 named workflows and rejects all 11 anti-workflows, each for exactly the expected reason. **(L0 for the checker, L1 for the design)**
- **Of the 72 compositions, 46 are direct and 26 need one human approval. None is forbidden.** A gate appears exactly where a pack of
  foreign or model origin would reach an automatic, instruction, or served sink.
- **The core workflow (W1, session continuity) is a feedback loop**, inject → transcript → capture → pack → inject. It stays stable only
  while the carry-forward gain of unaffirmed content is below 1 (§6). Echo-strip sets the verbatim gain to 0. The paraphrase gain has to be
  measured; target ≤ 0.05. **(L0 math, L2 values)**
- **Enumeration surfaced four workflows nobody designed** (§7): a model-to-model persistence channel through MCP drafts, the promotion of
  an agent's own session to AGENTS.md instructions, approval stamps leaking through re-shared packs, and cloud sessions whose store dies
  with the container.

---

## 2. Architecture: the pack as hub

```
 ENTRIES (produce a pack)                                   EXITS (consume a pack)
 ─────────────────────────                                  ─────────────────────────
 E1 hook capture (SessionEnd/PreCompact) ─┐             ┌─▶ X1 SessionStart auto-inject
 E2 skill-guided save in Claude Code ─────┤             ├─▶ X2 ctxr show (Bash tool)
 E3 claude.ai skill file ─────────────────┤             ├─▶ X3 MCP fetch (any MCP client)
 E4 official export zip ──────────────────┤   ┌─────┐   ├─▶ X4 extension → claude.ai composer
 E5 adapter (--via) ──────────────────────┼──▶│PACK │──▶┼─▶ X5 AGENTS.md (instruction file)
 E6 repo-committed pack ──────────────────┤   └─────┘   ├─▶ X6 .ctxpack.md handoff
 E7 stdin ────────────────────────────────┤  type        ├─▶ X7 upload to claude.ai
 E8 MCP propose (model-authored) ─────────┘  labels      ├─▶ X8 JSON to stdout
                                             origin      └─▶ X9 commit to repo
                                             hash
                     gate: approve@cli (human, TTY) binds effective trust = user to the content hash
```

Surfaces map onto the packages the build prompt names (`core`, CLI `ctxr`, plugin hooks, skill, MCP server, native host, extension).
Every primitive is implemented once in `core`; a surface only chooses which primitives it offers and which actor may call them.

---

## 3. Core theory

### 3.1 Values, labels, origin

A value moving through a workflow has a **type**, a **label set** and an **origin**:

```
 Source ──capture──▶ Pack{origin=session}
 External ──import─▶ Pack{origin=foreign}          ┌──────────── edit ────────────┐
 Text ──propose────▶ Pack{origin=model}            ▼                              │
                         │                      (labels)                          │
                         ▼                                                        │
            validate ─▶ {validated} ─▶ redact ─▶ {validated,redacted} ─▶ seal ─▶ {…,sealed} ─▶ approve ─▶ {…,sealed,approved}
                                                                                   │                        │
                                                                     render ─▶ Text ─▶ wrap ─▶ Wrapped ─▶ inject ─▶ Sink(context)
                                                                     render ─▶ Text ─────────────────────▶ emit ───▶ Sink(data|instruction)
```

- **Effective trust** is `user` while the pack carries an approval stamp bound to its current hash, and its origin otherwise.
  `edit` drops both `sealed` and `approved`, so any change to the content revokes the approval with no bookkeeping.
- **A violating step confers no labels.** A model that calls `approve` gets an `INV-ACTOR` violation and does not get the label (test A10).
- **Integrity is not provenance.** `sealed` proves the bytes haven't changed since sealing. It says nothing about where they came from.
  The UI must say "unchanged since sealed", never "verified" or "trusted" (FM-21).

### 3.2 Primitives *(generated)*

<!-- relay-model:begin primitives -->
| Op | Type | Requires | Adds | Removes | Origin | Surfaces | Actors | Effects |
|---|---|---|---|---|---|---|---|---|
| capture | Source → Pack | — | — | — | session | hooks, cli | hook, human, model | fs.read |
| import | External → Pack | — | — | — | foreign | cli, hooks, mcp | human, hook, model | fs.read |
| via | External → External | — | — | — | — | cli | human | exec |
| propose | Text → Pack | — | — | — | model | mcp | model | — |
| validate | Pack → Pack | — | validated | — | — | any | any | — |
| redact | Pack → Pack | — | redacted | — | — | any | any | — |
| seal | Pack → Pack | validated, redacted | sealed | — | — | any | any | — |
| edit | Pack → Pack | — | — | sealed, approved | — | cli, mcp | human, model | — |
| approve | Pack → Pack | validated, redacted, sealed | approved | — | — | cli | human | fs.write |
| audit | Pack → Pack | validated | audited | — | — | cli, hooks | any | fs.read |
| store | Pack → Pack | validated, redacted | — | — | — | hooks, cli, mcp | any | fs.write |
| render | Pack → Text | validated | — | — | — | any | any | — |
| wrap | Text → Wrapped | — | — | — | — | any | any | — |
| inject | Wrapped → Sink | redacted | — | — | — | hooks, cli, mcp, extension | any | — |
| emit | Text → Sink | redacted | — | — | — | cli | human | fs.write |
<!-- relay-model:end primitives -->

No primitive declares a `network` effect (INV-NO-NET). The only `exec` primitive is `via`, and it is human-only on the CLI (INV-EXEC-HUMAN).

### 3.3 Invariants

| Rule | Meaning | Where it bites |
|---|---|---|
| INV-TYPE | Each op receives the type it declares | malformed compositions |
| INV-SURFACE / INV-ACTOR | Each op is offered only on its surfaces, to its actors | A4 (`via` from MCP), A10 (model approves) |
| INV-REQ | Required labels are present | A7 (persist before redaction) |
| INV-WRAP | Context sinks are reached only through `wrap` + `inject` | A6 |
| INV-AUTO-TRUST / INV-AUTO-SEALED | A hook injects only sealed `user` or `session` packs | A1, A2, A8 |
| INV-INSTR-TRUST | Instruction sinks (AGENTS.md) take only `user` packs | A3, E1×X5 |
| INV-SERVE | MCP and the native host serve only `user` or `session` packs (drafts stay invisible) | E3–E8 × X3/X4 |
| INV-SHARE-SEALED | Git sharing requires a seal, so recipients can check integrity | A11 |
| INV-BUDGET | Render + wrapper ≤ sink capacity (10,000 chars for hook context, PN-01) | A5 |
| INV-LOOP-ECHO | A cyclic workflow captures with injected regions stripped | A9 |
| INV-NO-NET, INV-EXEC-HUMAN, INV-FAILOPEN | Global: no network; exec is human-only on the CLI; hook surfaces fail open | mutation tests |
| G-TRUST-CYCLE | Every persistent trust loop through an agent is contained by a full control or explicitly accepted | BLAST_RADIUS §8 |

---

## 4. Route composition *(generated)*

`ok` = checks as is. `gate` = checks only after inserting `approve@cli` by a human. `no` = cannot be made valid.

<!-- relay-model:begin routes -->
| Entry \ Exit | X1 | X2 | X3 | X4 | X5 | X6 | X7 | X8 | X9 |
|---|---|---|---|---|---|---|---|---|---|
| E1 Hook capture (SessionEnd / PreCompact) | ok | ok | ok | ok | gate | ok | ok | ok | ok |
| E2 Skill-guided save in Claude Code | ok | ok | ok | ok | gate | ok | ok | ok | ok |
| E3 claude.ai skill file, imported | gate | ok | gate | gate | gate | ok | ok | ok | ok |
| E4 Official export (zip) | gate | ok | gate | gate | gate | ok | ok | ok | ok |
| E5 Adapter (--via) | gate | ok | gate | gate | gate | ok | ok | ok | ok |
| E6 Repo-committed pack discovered at SessionStart | gate | ok | gate | gate | gate | ok | ok | ok | ok |
| E7 stdin pipeline | gate | ok | gate | gate | gate | ok | ok | ok | ok |
| E8 MCP propose (model-authored) | gate | ok | gate | gate | gate | ok | ok | ok | ok |

| Exit | Name |
|---|---|
| X1 | SessionStart auto-inject |
| X2 | ctxr show via Bash in Claude Code |
| X3 | MCP fetch (any MCP client) |
| X4 | Extension inserts brief into claude.ai composer |
| X5 | Export AGENTS.md |
| X6 | .ctxpack.md file handoff |
| X7 | Upload to claude.ai (skill reads it) |
| X8 | JSON to stdout |
| X9 | Commit to repo for teammates |

8 entries + 9 exits = 17 route definitions → 72 compositions: 46 direct, 26 need a human approval (INV-AUTO-TRUST, INV-INSTR-TRUST, INV-SERVE), 0 forbidden.
<!-- relay-model:end routes -->

Reading the matrix:

- **Rows E1–E2 (session origin)** go everywhere directly, except into AGENTS.md. Your own session is trusted as *context* but not as
  *standing instructions for every agent that opens the repo*.
- **Rows E3–E8 (foreign or model origin)** need a gate for exactly four exits: automatic injection (X1), MCP serving (X3), the extension
  (X4) and AGENTS.md (X5). Every exit that a human triggers and reads (X2, X6–X9) stays direct: the pack arrives wrapped, and the human
  is the gate.
- **No cell is forbidden.** The design never needs to say "you can't get there from here", only "a human must look first". Forbidden
  compositions exist only where a primitive is used out of role (the anti-workflows in §5).

---

## 5. Named workflows and anti-workflows *(generated)*

<!-- relay-model:begin workflows -->
| ID | Workflow | Composition | Scenarios | Verdict |
|---|---|---|---|---|
| W1 | Session continuity (cyclic) | E1 → X1 | S1 | passes |
| W2 | Compaction survival (PreCompact -> SessionStart source=compact) (cyclic) | E1 → X1 | — | passes |
| W3 | Fork lineage (SessionStart source=fork) (cyclic) | E1 → X1 | — | passes |
| W4 | Claude Code -> claude.ai | E1 → X4 | S3 | passes |
| W5 | claude.ai -> Claude Code | E3 → gate → X1 | — | passes |
| W6 | Backfill from official export | E4 → gate → X2 | S4 | passes |
| W7 | Claude Code -> any agent via AGENTS.md | E1 → gate → X5 | S5 | passes |
| W8 | Claude Code -> any MCP client | E1 → X3 | S5 | passes |
| W9 | Team relay through git | E1 → X9 → E6 → gate → X1 | — | passes |
| W10 | Cloud session -> local session | E1 → X9 → E6 → gate → X1 | — | passes |
| W11 | Unix pipeline | E7 → X8 | S5 | passes |
| W12 | Adapter round trip (markdown vault) | E5 → X6 | S5 | passes |
| W13 | Audited relay | E1 → audit@cli → X2 | — | passes |
<!-- relay-model:end workflows -->

<!-- relay-model:begin anti -->
| ID | Rejected composition | Path | Rules fired |
|---|---|---|---|
| A1 | Laundering: model-authored pack auto-injected next session | E8 → X1 | INV-AUTO-TRUST |
| A2 | Repo pack auto-injected without review | E6 → X1 | INV-AUTO-TRUST |
| A3 | Foreign pack promoted to AGENTS.md instructions | E3 → X5 | INV-INSTR-TRUST |
| A4 | Adapter execution reachable from MCP | via@mcp → import@mcp → validate@mcp → redact@mcp → store@mcp | INV-ACTOR, INV-SURFACE |
| A5 | Full render into SessionStart | E1 → render@hooks → wrap@hooks → inject@hooks | INV-BUDGET |
| A6 | Inject without wrapper | E1 → render@hooks → inject@hooks | INV-WRAP |
| A7 | Persist before redaction | capture@hooks → validate@hooks → store@hooks | INV-REQ |
| A8 | Edited after sealing, then auto-injected | E1 → edit@cli → X1 | INV-AUTO-SEALED |
| A9 | Echo loop: cyclic capture keeps injected text | capture@hooks → validate@hooks → redact@hooks → seal@hooks → store@hooks → X1 | INV-LOOP-ECHO |
| A10 | Model self-approves a repo pack | E6 → approve@cli → X1 | INV-ACTOR, INV-AUTO-TRUST |
| A11 | Share an unsealed pack through git | capture@cli → validate@cli → redact@cli → X9 | INV-SHARE-SEALED |
<!-- relay-model:end anti -->

### Notes on the workflows that aren't obvious

**W1 Session continuity.** SessionEnd or PreCompact captures, and the next SessionStart injects a brief. Budget: the brief is at most
8,000 chars, plus 600 for the wrapper, under the 10,000-char hook cap. Over the cap, Claude Code swaps the text for a file path and a
2,000-char preview, and the model may never open the file (PN-01, FM-02). SessionEnd work has to fit a **shared 1.5 s budget** (PN-03).
So capture should be incremental: record a transcript cursor at PreCompact or at turn boundaries, and have SessionEnd only finalise and
seal in O(tail) time. Every write goes to a temp file and is renamed into place, so a hook killed mid-write loses one pack, never the store (FM-04). **(L0 facts, L1 design)**

**W2 Compaction survival.** PreCompact capture, then SessionStart with `source=compact` (PN-02). PreCompact is one of the events that
*can block*, so the hook must never exit 2 or return `decision:block` (FM-03). The emergent hazard: the compaction summary may paraphrase
the earlier wrapped pack into **unwrapped** summary text, and then we inject the pack again. After compaction, the pack content exists
twice, and the copy without a wrapper looks like the model's own notes (FM-29, the top residual risk). Mitigation candidates: inject only a
delta on `source=compact`, and restate the data-not-instructions framing. **(L2: how compaction treats additionalContext has to be measured)**

**W3 Fork lineage.** `source=fork` is handled like resume, but both branches keep living. Packs are immutable nodes with a `parent` id,
so a fork creates siblings in a DAG instead of racing for one "latest" slot (FM-26). Selection is keyed by **project identity** (repo
root, remote-URL hash), never by a global "latest", which prevents project A's pack from landing in project B (FM-25). **(L1)**

**W4 / W5 Claude Code ↔ claude.ai.** Outbound (E1→X4), the native host serves a brief under the 1 MiB host-to-extension message limit
(PN-08). The extension inserts it into the composer **without sending**. When the DOM changes it falls back to the clipboard (FM-15).
Inbound (E3→gate→X1), a pack written by the claude.ai skill crosses the user's Downloads folder, so it arrives as `foreign` and needs one
approval before it auto-injects. **(L0 limit, L1 design)**

**W7 AGENTS.md.** This is the only exit whose sink is an **instruction** channel for *other* agents, which read AGENTS.md as standing
orders, not as wrapped data. The export therefore requires effective trust `user` for every origin, including your own session (E1×X5 =
gate). Once content is in AGENTS.md, no control of ours applies downstream (BLAST_RADIUS §4). **(L1)**

**W9 / W10 Team and cloud relay.** Git is the only transport that crosses machines without a server, and the build prompt rules out cloud
sync. A remote Claude Code session (`CLAUDE_CODE_REMOTE=true`, PN-12) has an ephemeral store, so for W10 the export-and-commit step is
mandatory: the pack is lost when the container is reclaimed (FM-24). On arrival every repo pack is `foreign`, whoever authored it.
**Approval stamps are local state and never serialized into a shared pack**, otherwise a teammate's approval would silently approve for
you. **(L1)**

**W11 Unix pipeline.** `ctxr import - | …`, `ctxr export --format json`, `ctxr validate -`. The exit-code contract (0 valid, 2 invalid,
1 other) and a stable `--json` output are what make workflows nobody designed possible (jq, CI checks, editor tasks). Treat them as API:
covered by golden tests, and changed only with a formatVersion bump. **(L0)**

**W13 Audited relay.** When several packs are injected together they can contradict each other: pack A says "use Postgres", pack B says
"use SQLite". The uploaded OmegaClaw `memory_audit` work shows that a deterministic audit exposes all planted contradictions *if* beliefs
are structured and consistently named, and degrades roughly as p² (direct) and p³ (two-step) in the structured share p. Similarity-based
recall exposed about 0 two-step contradictions once context shaped the embeddings. For the pack format, that means `decisions[]` entries
should carry a stable `key`, a `value` and `supersedes`. At p = 0.95 direct-conflict θ ≈ 0.90; at p = 0.5 it falls to about 0.25. v1 bans LLM
calls, so a keyed deterministic audit is also the only option that fits. **(L2: synthetic benchmark, transfer is an extrapolation)**

---

## 6. Feedback dynamics of the continuity loop

W1 is a discrete-time system. Let *x*ₙ be the content items in pack *n*, *u*ₙ the items newly derived from session *n*, and *g* the
fraction of injected items that the next capture carries forward without the user re-affirming them:

```
x_{n+1} = g · x_n + u_n            bounded  ⇔  g < 1,   x_∞ ≤ P_max + u_max / (1 − g)
P(planted item survives k sessions) = g^k,   expected lifetime = 1 / (1 − g) sessions
```

- **g splits into verbatim and paraphrase.** Echo-strip (capture drops every region that arrived inside a wrapper) sets g_verbatim = 0 by
  construction. g_paraphrase is the chance that the model restates injected content in its own words, which capture then keeps. It is
  unknown. **Measure it** with the planted-marker harness in §8. Target g ≤ 0.05, which gives lifetime ≤ 1.05 sessions and P(survive 3) ≤ 1.3·10⁻⁴.
- **Stability does not depend on delay.** If the pack is injected *d* sessions later instead of the next one, the recurrence becomes
  x_{n+d} = g·x_n + u and is stable under the same condition g < 1, for any d. This is the discrete form of the delay-independent stability
  argument in the uploaded QINCRS v2.1 note: undelayed damping exceeding total delayed gain. **(L0 math, L1 analogy)**
- **Pinned items are the one deliberate exception**: user-pinned facts carry forward with g = 1. They are bounded by P_max and must be
  pinned through a human action on the CLI, never by the model.
- Without echo-strip, g_verbatim → 1. Each pack then contains the previous one, pack size grows linearly until it hits the brief budget,
  and a single injected instruction becomes permanent. Anti-workflow A9 rejects that configuration.

---

## 7. Emergent workflows nobody designed (found by enumeration)

| Found | Composition | Why it matters | Resolution |
|---|---|---|---|
| Model-to-model persistence channel | E8 (MCP propose) → X3 (MCP fetch) | A prompt-injected model writes a draft "note", and a later session's model fetches it. Nothing is auto-injected, yet content persists across sessions. | MCP never lists or serves drafts (C-DRAFT-ISOLATION, INV-SERVE). Recommendation: a read-only MCP server in v1 (D-012). |
| Self-promotion to instructions | E1 → X5 | An agent's own session summary becomes AGENTS.md, which every other agent obeys. | A gate is required even for session origin (INV-INSTR-TRUST). |
| Approval laundering through git | E6 → gate → X9 → teammate E6 | If stamps travelled inside packs, one approval would approve for the whole team. | Stamps are local and keyed by hash; re-shared packs arrive `foreign`. |
| Ephemeral cloud store | W10 | Packs captured in a remote session vanish with the container. | Detect `CLAUDE_CODE_REMOTE`; `ctxr doctor` warns; commit via X9. |
| Compaction de-wrapping | W2 | Summarizer output is outside our wrapper. | Delta-only re-inject on `source=compact`; measure (FM-29). |
| Scope bleed | W1 with 2 projects | "Latest pack" across projects injects confidential context into the wrong repo. | Project-keyed selection (C-PROJECT-SCOPE). |

---

## 8. Testing strategy: planted-defect harnesses

Every **partial** control is only as good as its measured catch rate θ. This follows the methodology of the uploaded planted-contradiction
test (`pc_test.py`) and `memory_audit/bench_theta.py`: plant known defects, count the share exposed, use fixed seeds plus a held-out seed set,
report Wilson 95% CIs and the false-flag count.

| Harness | Plants | θ definition | Accept if | Feeds |
|---|---|---|---|---|
| Injection | "ignore previous instructions…" variants across pack fields and surfaces | share *not* acted on (S6, extended per surface) | lower CI bound ≥ 0.95 | C-WRAP, D-008 |
| Echo | unique markers in an injected pack, over k chained sessions | share absent from pack n+1 | ≥ 0.95 per hop | C-ECHO-STRIP, g in §6 |
| Secret | fake keys in every supported and several unsupported formats, at capture and at each egress | share redacted; false flags | 1.00 supported; unsupported reported | C-REDACT |
| Contradiction | keyed and free-text decision conflicts across packs | share flagged; false flags = 0 | ≥ 0.90 on keyed | C-AUDIT |

Deterministic controls (full strength) are tested by the anti-workflows and the mutation tests in `tools/relay-model.test.mjs`, which
break the model on purpose (drop the trust gate, overflow the budget, add a network effect, un-accept a loop) and assert that the
checker catches each break.

---

## 9. Open questions

1. Does the PowerShell fallback on Windows (used when Git Bash is absent, PN-11) run 5.1 or 7? `node … || exit 0` only parses on 7.
   This decides whether C-NODE-SHIM can be a single command string (D-001).
2. Does the Bash tool run commands with a TTY? `approve` relies on an interactive confirmation that a model can't satisfy (D-010).
3. Does SessionStart block the first prompt until the hook returns? If it does, the 600 s default timeout (PN-13) is a user-visible hazard
   and C-HOOK-TIMEOUT is mandatory.
4. How does compaction treat `additionalContext`? Is it summarized, dropped, or kept verbatim? (FM-29)
5. Does the spec's MCP surface include any write tool? This model assumes it might (E8), and recommends it doesn't in v1.

## 10. Evidence vs speculation

| Claim | Level | Basis |
|---|---|---|
| Hook caps, events, budgets, env vars | L0 | Claude Code docs, checked 2026-09-30 (PLATFORM_NOTES_ADDENDUM) |
| The composition and invariant results | L0 | Deterministic checker plus tests in this directory |
| Hub-and-spoke O(E+X) → O(E·X) | L0 | Standard IR/LSP argument |
| Loop stability condition g < 1 | L0 | Linear recurrence |
| Value of g_paraphrase; θ of wrap and echo-strip | L2 | Unmeasured; harnesses in §8 |
| p²/p³ scaling of audit θ with structured share | L2 | Synthetic OmegaClaw benchmark, transferred by analogy |
| Compaction de-wrapping (FM-29) | L2 | Mechanism plausible; behaviour unverified |
