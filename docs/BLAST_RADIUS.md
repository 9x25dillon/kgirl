# Blast Radius

Context Relay, Phase 0 risk register input for Checkpoint A. Status: **proposed**.
The *generated* tables come from [`model/relay-model.json`](../model/relay-model.json) via `node tools/relay-model.mjs sync-docs`, and CI fails on drift.
Evidence levels: **L0** established or verified, **L1** extrapolation, **L2** needs measurement, **L3** speculative.

---

## 1. Executive summary

1. **All five of the top residual risks are trust risks**: compaction de-wrapping, the laundering loop, secret egress, forged approval, and
   instructions inside packs. No runtime or change failure makes the top five. Conventional controls (atomic writes, fail-open hooks, golden
   tests) contain those well. What remains is at the level of prompts and data provenance. (§10)
2. **One accepted loop.** Among the persistent trust loops that the relay itself adds (host capabilities excluded), exactly one class is
   not fully contained: `capture → store → inject → session → capture`, reached through the SessionStart hook or through MCP. It *is* the
   product (automatic continuity). It can only be damped, by the wrapper and echo-strip, both of which are partial. D-008 accepts it on a
   measured condition and otherwise turns auto-inject off by default. (§8)
3. **The trust gate is the highest-value control.** Removing it exposes 41 more (source, node) pairs, more than any other control. It is
   cheap: a label check plus a hash-bound approval stamp. Build it first. (§9)
4. **Redaction is a single point of protection for confidentiality toward git and other agents.** Those egress points are reached only
   through the one partial control C-REDACT. The claude.ai path has a second barrier: the native host serves only packs the user marked.
   Redacting at capture and again at egress with the *same* patterns is not defence in depth, because the misses are correlated. A second,
   *different* mechanism is needed at egress: an allow-list of structured fields for AGENTS.md and git, plus entropy flags. (§5)
5. **The plugin's own data directory is a data-loss trap.** Claude Code deletes `${CLAUDE_PLUGIN_DATA}` when the plugin is uninstalled
   from its last location, and the directory isn't visible to the Bash tool or the native host. Store packs elsewhere. (PN-05, FM-06, D-007) **(L0)**
6. **Hooks turn supply-chain compromise into automatic execution.** A compromised release runs at every SessionStart on every install,
   without the user invoking anything. Ship hooks as one pre-bundled file with no runtime installs, and pin the plugin version. (FM-18)
7. **Measure blast radius as a delta over the host.** Inside Claude Code the model already has Bash and Write (subject to permissions), so
   `--via` adds no new capability there. It would be new on MCP, the native host, or claude.ai, which is why it is CLI-only. Likewise, a
   model with write permission can forge an approval by editing the store (FM-30). The relay cannot remove host capabilities; it must not
   add to them. (D-009)

---

## 2. Method

**Graph.** 42 components in 6 layers (external, contract, core, surface, data, sink) and 3 edge kinds:

| Kind | Edge u → v means | Views |
|---|---|---|
| `change` | changing u's contract can break v | change |
| `runtime` | u failing makes v fail | runtime |
| `trust` | content flows from u to v (data flow) | integrity (attacker content forward from its entry point), confidentiality (secrets forward from where they originate) |

**Exposure lattice.** Propagation from a source marks each node *unmitigated* (3), *damped* (2) or *contained* (1). An edge with an applicable
**full** control lowers the level to contained. A **partial** control caps it at damped. No control passes it through unchanged. A node keeps
the worst level over all paths, computed by BFS with level upgrades in O(V·(V+E)) per view.

**Threat-scoped controls.** A control declares the threats it addresses. Redaction dampens confidentiality paths and does nothing for
prompt injection; the wrapper does the reverse.

**Baseline edges.** Capabilities the host already grants (the model's Bash and Write tools, the user running an adapter) are marked
`baseline`. They are excluded from the default delta analysis and shown separately. This is the "must not add blast radius" rule (D-009).

**Persistence loops.** These are elementary cycles of the trust graph that pass through both a data node (where content persists) and an
agent (where content is acted on). A loop is *contained* if any edge on it has a full integrity control. Otherwise it must be listed in
`acceptedCycles` with a decision, or the check fails (G-TRUST-CYCLE).

Elementary cycles miss loops that visit a node twice. So the check also removes every edge that a full integrity control closes, and
flags any agent that still shares a strongly connected component with a data node (D-016). This second test found the Soup loop in the
kgirl harness (KGIRL_HARNESS.md).

**Control value.** For each control: the number of (source, node) pairs that become unmitigated when that control alone is disabled,
summed over all views.

**Risk.** FMEA-style RPN = S × L × D (each 1–5, with D inverted so that 5 means silent), given both *inherent* (no controls) and
*residual* (with the listed controls) values. The scales are defined in the model file.

**Limits of the method.**
- It is coarse: one component node stands for every instance, and the model doesn't grade severity along a path.
- "Contained" means the dominant failure on that edge is stopped, not that every failure is.
- S/L/D values are expert estimates **(L1)** until the harnesses in WORKFLOWS §8 produce data.

---

## 3. Component map

```
 external   node  cc-hooks  cc-plugins  marketplace  cai-skills  cai-dom  chrome-nm  mcp-sdk  export-fmt  npm  cws  git-remote
               \      |          |                        |          |         |          |          |              |
 contract       \   spec ────────┼──── wrapper-proto      |          |         |          |          |              |
                 \    |          |          |             |          |         |          |          |              |
 core     schema canon redact render wrap store capture importer audit         |          |          |              |
                 \    |     |     |     |     |     |        |                 |          |          |              |
 surface   shim hooks plugin skill skill-zip cli mcp nmh extension mcpb adapters ◀──────────┘          |              |
                  |         |                  |   |   |                                               |              |
 data       store-data  store-drafts  repo-packs  agents-md ◀──────────────────────────────────────────┴──────────────┘
                  |              |           |          |
 sink       cc-session      cai-chat     other-agents     user-host        (agents: cc-session, cai-chat, other-agents)
```

The full edge list, with kinds and controls, is in the model file. `node tools/relay-model.mjs report <section>` prints any table below.

---

## 4. Integrity: where attacker content goes *(generated)*

<!-- relay-model:begin radius-integrity -->
| Source | Unmitigated | Damped | Contained | Unmitigated reach | Agent / egress / host impact |
|---|---|---|---|---|---|
| capture | 7 | 6 | 2 | cli, git-remote, hooks, mcp, repo-packs, store-data, store-drafts | git-remote unmitigated, cai-chat damped, cc-session damped, other-agents damped, user-host contained |
| store-data | 6 | 7 | 2 | cli, git-remote, hooks, mcp, repo-packs, store-drafts | git-remote unmitigated, cai-chat damped, cc-session damped, other-agents damped, user-host contained |
| extension | 5 | 0 | 10 | cai-chat, cli, importer, nmh, store-drafts | cai-chat unmitigated, cc-session contained, git-remote contained, other-agents contained, user-host contained |
| agents-md | 4 | 10 | 1 | cc-session, cli, other-agents, store-drafts | cc-session unmitigated, other-agents unmitigated, cai-chat damped, git-remote damped, user-host contained |
| cai-chat | 3 | 0 | 12 | cli, importer, store-drafts | cc-session contained, git-remote contained, other-agents contained, user-host contained |
| export-fmt | 3 | 0 | 13 | cli, importer, store-drafts | cai-chat contained, cc-session contained, git-remote contained, other-agents contained, user-host contained |
| git-remote | 3 | 0 | 12 | cli, repo-packs, store-drafts | cai-chat contained, cc-session contained, other-agents contained, user-host contained |
| repo-packs | 3 | 0 | 12 | cli, git-remote, store-drafts | git-remote unmitigated, cai-chat contained, cc-session contained, other-agents contained, user-host contained |
| cc-session | 2 | 11 | 2 | cli, store-drafts | cai-chat damped, git-remote damped, other-agents damped, user-host contained |
| importer | 2 | 0 | 13 | cli, store-drafts | cai-chat contained, cc-session contained, git-remote contained, other-agents contained, user-host contained |
| store-drafts | 1 | 0 | 14 | cli | cai-chat contained, cc-session contained, git-remote contained, other-agents contained, user-host contained |
| hooks | 0 | 13 | 2 | — | cai-chat damped, cc-session damped, git-remote damped, other-agents damped, user-host contained |
| mcp | 0 | 13 | 2 | — | cai-chat damped, cc-session damped, git-remote damped, other-agents damped, user-host contained |
| cws | 0 | 6 | 10 | — | cai-chat damped, cc-session contained, git-remote contained, other-agents contained, user-host contained |
| nmh | 0 | 5 | 10 | — | cai-chat damped, cc-session contained, git-remote contained, other-agents contained, user-host contained |
| marketplace | 0 | 1 | 0 | — | user-host damped |
| npm | 0 | 1 | 0 | — | user-host damped |
<!-- relay-model:end radius-integrity -->

Reading it:

- **`agents-md` is the only source that reaches an agent unmitigated** (`cc-session`, `other-agents`). An instruction file has no wrapper
  downstream, which is why the gate sits *before* it (C-INSTR-GATE).
- **`capture` and `store-data` reach `git-remote` unmitigated.** Poisoned session content can be committed and pushed. Teammates are
  protected only because every repo pack re-enters as `foreign` behind the trust gate.
- **Everything behind the store reaches agents only damped.** The wrapper is the last barrier, and it is partial (θ unmeasured).
- **`npm` and `marketplace` reach `user-host` damped.** Supply-chain controls are partial by nature, and hooks make the payload run automatically.

## 5. Confidentiality: where secrets go *(generated)*

<!-- relay-model:begin radius-confidentiality -->
| Source | Unmitigated | Damped | Contained | Unmitigated reach | Egress impact |
|---|---|---|---|---|---|
| export-fmt | 2 | 14 | 0 | importer, user-host | cai-chat damped, git-remote damped, other-agents damped |
| cai-chat | 2 | 13 | 0 | importer, user-host | git-remote damped, other-agents damped |
| cc-session | 2 | 13 | 0 | importer, user-host | cai-chat damped, git-remote damped, other-agents damped |
<!-- relay-model:end radius-confidentiality -->

Every egress is reached damped. Toward `git-remote` and `other-agents` the only damping control is C-REDACT: disable it and both become
unmitigated. Toward `cai-chat`, C-NMH-SCOPE (the native host serves only packs the user marked for claude.ai) is a second, independent
barrier. A test pins down both facts. An early draft of this document said "only C-REDACT, everywhere"; the model check caught the error.
Recommended second barrier for the other two paths, independent of the pattern list:

- (a) AGENTS.md and git exports render only structured fields (goal, decisions, next steps, file paths), never raw transcript excerpts.
- (b) An entropy and shape heuristic at egress that **blocks** on a hit and asks for review, instead of redacting silently.

## 6. Runtime: what fails when something fails *(generated)*

<!-- relay-model:begin radius-runtime -->
| Source | Unmitigated | Damped | Contained | Unmitigated reach | Agent / egress / host impact |
|---|---|---|---|---|---|
| store | 3 | 0 | 5 | hooks, mcp, nmh | cc-session contained, other-agents contained |
| chrome-nm | 1 | 0 | 1 | nmh | — |
| git-remote | 1 | 0 | 0 | repo-packs | — |
| node | 1 | 0 | 9 | nmh | cc-session contained, other-agents contained |
| shim | 1 | 0 | 8 | hooks | cc-session contained, other-agents contained |
| cc-hooks | 0 | 1 | 8 | — | cc-session contained, other-agents contained |
| cai-dom | 0 | 0 | 1 | — | — |
| cc-plugins | 0 | 0 | 9 | — | cc-session contained, other-agents contained |
| cli | 0 | 0 | 8 | — | cc-session contained, other-agents contained |
| export-fmt | 0 | 0 | 10 | — | cc-session contained, other-agents contained |
| hooks | 0 | 0 | 8 | — | cc-session contained, other-agents contained |
| importer | 0 | 0 | 9 | — | cc-session contained, other-agents contained |
| mcp | 0 | 0 | 8 | — | cc-session contained, other-agents contained |
| mcp-sdk | 0 | 0 | 9 | — | cc-session contained, other-agents contained |
| nmh | 0 | 0 | 1 | — | — |
| render | 0 | 0 | 9 | — | cc-session contained, other-agents contained |
| store-data | 0 | 0 | 8 | — | cc-session contained, other-agents contained |
<!-- relay-model:end radius-runtime -->

No runtime failure reaches a Claude Code session as harm, because C-FAILOPEN contains every path into `cc-session`. The cost of that
containment is **silence**: a missing Node runtime, a platform change or a corrupted store all just make continuity quietly stop. C-HEARTBEAT
(a local run log plus `ctxr doctor`) is what turns silent loss into detectable loss. Without it, detectability stays at D=5 for FM-01, FM-05 and FM-24.

## 7. Change: what a change breaks *(generated)*

<!-- relay-model:begin radius-change -->
| Source | Unmitigated | Damped | Contained | Unmitigated reach | Agent / egress / host impact |
|---|---|---|---|---|---|
| chrome-nm | 2 | 0 | 0 | extension, nmh | — |
| cai-skills | 1 | 0 | 0 | skill-zip | — |
| mcp | 1 | 0 | 0 | mcpb | — |
| spec | 1 | 0 | 18 | audit | — |
| schema | 0 | 9 | 2 | — | — |
| canon | 0 | 7 | 3 | — | — |
| store | 0 | 6 | 2 | — | — |
| wrap | 0 | 6 | 2 | — | — |
| redact | 0 | 5 | 1 | — | — |
| capture | 0 | 3 | 1 | — | — |
| cc-hooks | 0 | 3 | 0 | — | — |
| cc-plugins | 0 | 1 | 0 | — | — |
| hooks | 0 | 1 | 0 | — | — |
| importer | 0 | 1 | 1 | — | — |
| render | 0 | 1 | 7 | — | — |
| shim | 0 | 1 | 0 | — | — |
| skill | 0 | 1 | 1 | — | — |
| cai-dom | 0 | 0 | 1 | — | — |
| cli | 0 | 0 | 1 | — | — |
| export-fmt | 0 | 0 | 3 | — | — |
| mcp-sdk | 0 | 0 | 2 | — | — |
| nmh | 0 | 0 | 1 | — | — |
| wrapper-proto | 0 | 0 | 12 | — | — |
<!-- relay-model:end radius-change -->

- **Internal change radius is controlled by golden tests** (C-GOLDEN on the spec edges, C-TYPECHECK on core→surface edges).
- **The residual unmitigated edges are external contracts we can't CI-guard**: `chrome-nm`, `cai-skills`, and packaging to `mcpb`.
- **`cc-hooks` is damped only by C-HEARTBEAT.** If Claude Code changes hook semantics, fail-open hooks mask it. That makes a platform
  change the most likely *silent* break.
- **A non-obvious coupling: the wrapper markers are read in three places** (the wrap code, SKILL.md prose, and the echo-strip parser in
  capture). Change one without the others and two controls disarm at once: the wrapper on claude.ai, and echo-strip everywhere (FM-19).
  C-SYNC must cover the markers, not only the skill copies.
- **Canonicalization is a hidden hub.** Changing it changes every hash, so every seal fails and users are trained to ignore integrity
  warnings. Seals should record their algorithm (C-HASH-VERSION, FM-31).

## 8. Persistence loops *(generated)*

<!-- relay-model:begin cycles -->
**Added by Context Relay (baseline edges excluded):**

| Loop | Controls on loop | Status |
|---|---|---|
| capture → store-data → hooks → cc-session → capture | C-WRAP, C-ECHO-STRIP | accepted (D-008) |
| capture → store-data → mcp → cc-session → capture | C-WRAP, C-ECHO-STRIP | accepted (D-008) |
| capture → store-data → repo-packs → store-drafts → mcp → cc-session → capture | C-DRAFT-ISOLATION, C-WRAP, C-ECHO-STRIP | contained |
| capture → store-data → agents-md → cc-session → capture | C-INSTR-GATE, C-ECHO-STRIP | contained |
| capture → store-data → nmh → extension → cai-chat → importer → store-drafts → mcp → cc-session → capture | C-NMH-SCOPE, C-WRAP, C-DRAFT-ISOLATION, C-ECHO-STRIP | contained |
| importer → store-drafts → store-data → hooks → cc-session → importer | C-TRUST-GATE, C-WRAP, C-VIA-CLI-ONLY | contained |
| importer → store-drafts → store-data → mcp → cc-session → importer | C-TRUST-GATE, C-WRAP, C-VIA-CLI-ONLY | contained |
| importer → store-drafts → store-data → agents-md → cc-session → importer | C-TRUST-GATE, C-INSTR-GATE, C-VIA-CLI-ONLY | contained |
| importer → store-drafts → store-data → nmh → extension → cai-chat → importer | C-TRUST-GATE, C-NMH-SCOPE, C-WRAP | contained |
| importer → store-drafts → mcp → cc-session → importer | C-DRAFT-ISOLATION, C-WRAP, C-VIA-CLI-ONLY | contained |
| hooks → cc-session → store-drafts → store-data → hooks | C-WRAP, C-TRUST-GATE | contained |
| mcp → cc-session → store-drafts → store-data → mcp | C-WRAP, C-TRUST-GATE | contained |
| mcp → cc-session → store-drafts → mcp | C-WRAP, C-DRAFT-ISOLATION | contained |
| store-data → agents-md → cc-session → store-drafts → store-data | C-INSTR-GATE, C-TRUST-GATE | contained |

**Present only because of host capabilities (baseline edges), shown for completeness:**

| Loop | Controls on loop | Status |
|---|---|---|
| capture → store-data → cli → cc-session → capture | C-WRAP, C-ECHO-STRIP | OPEN |
| capture → store-data → repo-packs → store-drafts → cli → cc-session → capture | C-WRAP, C-ECHO-STRIP | OPEN |
| capture → store-data → nmh → extension → cai-chat → importer → store-drafts → cli → cc-session → capture | C-NMH-SCOPE, C-WRAP, C-ECHO-STRIP | OPEN |
| importer → store-drafts → store-data → cli → cc-session → importer | C-TRUST-GATE, C-WRAP, C-VIA-CLI-ONLY | contained |
| importer → store-drafts → mcp → cc-session → store-data → nmh → extension → cai-chat → importer | C-DRAFT-ISOLATION, C-WRAP, C-PERMISSION-BOUNDARY, C-NMH-SCOPE | contained |
| importer → store-drafts → cli → cc-session → importer | C-WRAP, C-VIA-CLI-ONLY | contained |
| importer → store-drafts → cli → cc-session → store-data → nmh → extension → cai-chat → importer | C-WRAP, C-PERMISSION-BOUNDARY, C-NMH-SCOPE | OPEN |
| hooks → cc-session → store-data → hooks | C-WRAP, C-PERMISSION-BOUNDARY | OPEN |
| cli → cc-session → store-drafts → store-data → cli | C-WRAP, C-TRUST-GATE | contained |
| cli → cc-session → store-drafts → cli | C-WRAP | OPEN |
| cli → cc-session → store-data → cli | C-WRAP, C-PERMISSION-BOUNDARY | OPEN |
| cli → cc-session → store-data → repo-packs → store-drafts → cli | C-WRAP, C-PERMISSION-BOUNDARY | OPEN |
| mcp → cc-session → store-data → mcp | C-WRAP, C-PERMISSION-BOUNDARY | OPEN |
| mcp → cc-session → store-data → repo-packs → store-drafts → mcp | C-WRAP, C-PERMISSION-BOUNDARY, C-DRAFT-ISOLATION | contained |
| store-data → agents-md → cc-session → store-data | C-INSTR-GATE, C-PERMISSION-BOUNDARY | contained |
<!-- relay-model:end cycles -->

Every "OPEN" loop in the second table passes through a baseline edge. Either the model writes the store directly, or it reads packs through
`ctxr` in the Bash tool. These loops exist because the host grants those capabilities, and the relay can only make them harder:

- Keep the store outside the project directory, so writes need a permission prompt in `default` and `acceptEdits` modes (C-PERMISSION-BOUNDARY).
- Require an interactive TTY confirmation for `approve`.

In `bypassPermissions` mode none of this holds, and the threat model says so explicitly (D-009).

## 9. Control value and build order *(generated)*

<!-- relay-model:begin controls -->
| Control | Strength | Status | Pairs protected | By view | Mechanism |
|---|---|---|---|---|---|
| C-TRUST-GATE | full | proposed | 41 | integrity 41 | Foreign and model-origin packs never auto-inject or reach instruction sinks. Only a human approval stamp, bound to the content hash, promotes them. |
| C-REDACT | partial | in-spec | 32 | confidentiality 32 | Pattern redaction at capture and again at every egress (idempotent), with counts reported. |
| C-TYPECHECK | partial | in-spec | 32 | change 32 | Monorepo typecheck and per-package tests on every commit. Catches interface breaks between core and surfaces, not behavioural drift. |
| C-ATOMIC | full | proposed | 22 | runtime 22 | One file per pack; write to temp and rename; index is a cache rebuilt from pack files; no cross-pack transactions. |
| C-WRAP | partial | in-spec | 16 | integrity 16 | Injected pack text sits inside an untrusted-data envelope with an instruction to treat it as reference, never as commands. |
| C-ECHO-STRIP | partial | proposed | 12 | integrity 12 | Capture drops every region that arrived inside a wrapper, so injected text cannot re-enter the next pack verbatim. A model paraphrase can still carry it forward. |
| C-FAIL-LOUD | full | in-spec | 10 | runtime 10 | The CLI and MCP server report failures with exit codes and actionable messages. They never fail silently. |
| C-DRAFT-ISOLATION | full | proposed | 9 | integrity 9 | MCP list/fetch returns only sealed session packs or approved packs, never drafts. |
| C-CWS | partial | proposed | 6 | integrity 6 | Chrome Web Store 2FA, minimal permissions (claude.ai host only), staged rollout where available. |
| C-EXT-DEGRADE | full | proposed | 6 | change 1, runtime 5 | Extension failures (selector miss, host missing, message over 1 MiB) degrade to copy-to-clipboard plus a notice; the page is never broken and the message is never sent. |
| C-GOLDEN | full | proposed | 6 | change 6 | Golden packs and golden renders per formatVersion; any byte change in a render, hash or parse fails CI unless the golden changes in the same PR. |
| C-INSTR-GATE | full | proposed | 6 | integrity 6 | Export to instruction files (AGENTS.md) requires effective trust 'user' and an explicit --i-reviewed flag in a TTY. |
| C-SYNC | full | in-spec | 6 | change 6 | CI sync check: skill copies, and wrapper markers in code, SKILL.md prose and the echo-strip parser, come from one source. |
| C-BUDGET | full | proposed | 5 | change 4, runtime 1 | Each render format has a char budget; render + wrapper stays under the sink's capacity. Counted in UTF-16 units and code points; the stricter wins. |
| C-PLUGIN-VALIDATE | partial | in-spec | 4 | change 4 | CI runs `claude plugin validate --strict` against the current Claude Code release. Catches manifest, path and userConfig breakage, not hook behaviour. |
| C-ZIP-SAFE | full | proposed | 4 | integrity 4 | Official-export import reads entries in memory, never extracts to disk, and caps entry count, sizes and expansion ratio. |
| C-FAILOPEN | full | in-spec | 3 | runtime 3 | Every hook catches all errors, bounds its own runtime, exits 0, and never emits exit 2 or decision:block. |
| C-FORMAT-VERSION | full | proposed | 3 | change 3 | formatVersion in every pack; readers accept N and N-1, preserve unknown fields, and migrate on read. |
| C-HEARTBEAT | partial | proposed | 3 | change 2, runtime 1 | Each hook run appends to a local run log; `ctxr doctor` reports the last success per event, which makes silent hook breakage visible. |
| C-PIN | full | proposed | 3 | change 2, runtime 1 | Exact dependency pins plus lockfile; upgrades only through reviewed PRs. |
| C-CONTRACT | full | in-spec | 2 | change 2 | CI runs every example adapter and protocol fixture (--via stdio, NMH frames). |
| C-FIXTURES | full | proposed | 2 | change 1, runtime 1 | Format sniffing plus versioned fixtures of the export format; an unknown shape is an actionable error, never a partial import. |
| C-NMH-ALLOWLIST | full | proposed | 2 | integrity 2 | The native host accepts a fixed read-only verb set (list, render-brief). allowed_origins is pinned to the one extension ID. |
| C-NMH-SCOPE | partial | proposed | 2 | integrity 2 | The native host serves only packs the user marked for claude.ai, and only as brief renders. |
| C-NODE-SHIM | full | proposed | 2 | runtime 2 | Shell-form hook command `node <script> <event> \|\| exit 0`; a missing runtime degrades to silence. |
| C-SUPPLY | partial | proposed | 2 | integrity 2 | Hooks ship as one pre-bundled file with no runtime installs; npm provenance, 2FA, SBOM, pnpm audit. |
| C-VIA-CLI-ONLY | full | proposed | 2 | integrity 2 | --via exists only as a CLI argv flag. It runs through execFile with an argv array and no shell. Pack fields, MCP arguments and NMH messages cannot supply it. |
| C-HASH-VERSION | full | proposed | 1 | change 1 | Seals record their algorithm (for example sha256 over canonical JSON, version 1); verifiers keep every released algorithm, so a canonicalization change never invalidates existing seals. |
| C-STORE-LOCATION | full | proposed | 1 | runtime 1 | Store lives in a tool-owned user data directory, not in CLAUDE_PLUGIN_DATA, which uninstall deletes. |
<!-- relay-model:end controls -->

Recommended implementation order, by protected pairs per unit of effort:

1. **C-TRUST-GATE, C-DRAFT-ISOLATION, C-INSTR-GATE.** Label checks in `core`, about a day, 56 pairs.
2. **C-ATOMIC, C-STORE-LOCATION, C-FAILOPEN, C-FAIL-LOUD.** The store and hook skeleton, needed anyway.
3. **C-GOLDEN, C-SYNC, C-CONTRACT.** CI; they freeze the hub once Checkpoint B lands.
4. **C-WRAP, C-ECHO-STRIP, C-REDACT, plus their θ harnesses.** Partial controls; the harness is part of the Definition of Done.
5. The rest.

## 10. Risk register *(generated)*

<!-- relay-model:begin risks -->
| Rank | ID | Failure mode | Kind | Inherent S·L·D | Residual S·L·D | RPN | Controls | Level |
|---|---|---|---|---|---|---|---|---|
| 1 | FM-29 | Compaction summarizes wrapped pack text into unwrapped summary text | trust | 4·3·5 = 60 | 4·3·4 = 48 | 48 | C-WRAP | L2 |
| 2 | FM-08 | Laundering loop: injected content re-captured and re-injected every session | trust | 5·3·5 = 75 | 5·2·4 = 40 | 40 | C-WRAP, C-ECHO-STRIP | L1 |
| 3 | FM-09 | Secret in an unsupported format leaves through an egress route | trust | 5·4·5 = 100 | 5·2·4 = 40 | 40 | C-REDACT | L0 |
| 4 | FM-30 | Model forges an approval by writing store files directly (host capability, not added by the relay) | trust | 4·2·5 = 40 | 4·2·4 = 32 | 32 | C-PERMISSION-BOUNDARY | L1 |
| 5 | FM-07 | Instructions inside a pack are acted on | trust | 5·3·4 = 60 | 5·2·3 = 30 | 30 | C-WRAP | L1 |
| 6 | FM-01 | Node absent on the host, so hooks cannot run | runtime | 3·3·4 = 36 | 3·3·3 = 27 | 27 | C-NODE-SHIM, C-HEARTBEAT | L0 |
| 7 | FM-22 | Contradictory packs injected together; the model follows a superseded decision | trust | 3·4·5 = 60 | 3·3·3 = 27 | 27 | C-AUDIT, C-LINEAGE | L2 |
| 8 | FM-11 | AGENTS.md export turns pack data into standing instructions for every agent | trust | 5·3·4 = 60 | 4·2·3 = 24 | 24 | C-INSTR-GATE | L1 |
| 9 | FM-21 | Integrity hash mistaken for provenance ('hash ok' read as 'safe') | trust | 4·3·5 = 60 | 3·2·4 = 24 | 24 | C-TRUST-GATE | L1 |
| 10 | FM-17 | Hijacked extension update reads claude.ai and every pack the host serves | trust | 5·1·5 = 25 | 5·1·4 = 20 | 20 | C-CWS, C-NMH-ALLOWLIST, C-NMH-SCOPE | L1 |
| 11 | FM-18 | Supply-chain compromise runs code at every SessionStart on every install | trust | 5·2·5 = 50 | 5·1·4 = 20 | 20 | C-SUPPLY, C-PIN | L0 |
| 12 | FM-24 | Cloud session store vanishes with its container | runtime | 3·4·5 = 60 | 2·3·3 = 18 | 18 | C-HEARTBEAT | L0 |
| 13 | FM-04 | SessionEnd capture overruns the shared 1.5 s budget and is killed mid-write | runtime | 3·3·5 = 45 | 2·2·3 = 12 | 12 | C-ATOMIC, C-HOOK-TIMEOUT | L0 |
| 14 | FM-05 | Parallel sessions corrupt a shared index; hooks fail open, so nobody notices | runtime | 4·3·5 = 60 | 2·2·3 = 12 | 12 | C-ATOMIC, C-HEARTBEAT, C-FAIL-LOUD | L1 |
| 15 | FM-10 | Any contributor's committed pack is auto-injected into every teammate's session | trust | 5·3·5 = 75 | 5·1·2 = 10 | 10 | C-TRUST-GATE | L1 |
| 16 | FM-12 | --via reachable from a non-human actor, giving command execution | trust | 5·2·4 = 40 | 5·1·2 = 10 | 10 | C-VIA-CLI-ONLY | L0 |
| 17 | FM-25 | Scope bleed: project A's pack injected into project B | trust | 4·3·4 = 48 | 3·1·3 = 9 | 9 | C-PROJECT-SCOPE | L1 |
| 18 | FM-14 | Undocumented export format changes and import breaks | change | 2·4·2 = 16 | 2·4·1 = 8 | 8 | C-FIXTURES | L0 |
| 19 | FM-20 | Format change strands stored packs and third-party adapters | change | 3·3·3 = 27 | 2·2·2 = 8 | 8 | C-FORMAT-VERSION, C-CONTRACT | L0 |
| 20 | FM-02 | Injected text over 10,000 chars is replaced by a file path and a 2,000-char preview | runtime | 3·3·5 = 45 | 3·1·2 = 6 | 6 | C-BUDGET | L0 |
| 21 | FM-23 | Windows without Git Bash: shell-form hook runs in PowerShell and errors every session | runtime | 3·3·1 = 9 | 3·2·1 = 6 | 6 | C-NODE-SHIM | L2 |
| 22 | FM-26 | Forked sessions overwrite each other's pack | runtime | 2·3·4 = 24 | 1·2·3 = 6 | 6 | C-LINEAGE | L1 |
| 23 | FM-15 | claude.ai UI change breaks composer insertion | change | 2·5·3 = 30 | 1·5·1 = 5 | 5 | C-EXT-DEGRADE | L0 |
| 24 | FM-03 | PreCompact exits 2 or returns decision:block, so compaction is blocked | runtime | 4·2·1 = 8 | 4·1·1 = 4 | 4 | C-FAILOPEN | L0 |
| 25 | FM-19 | Wrapper markers drift between code, SKILL.md and echo-strip, disarming two controls at once | change | 4·3·5 = 60 | 4·1·1 = 4 | 4 | C-SYNC | L1 |
| 26 | FM-13 | Crafted export zip (zip-slip, zip bomb, oversized JSON) | trust | 4·2·3 = 24 | 2·1·2 = 4 | 4 | C-ZIP-SAFE | L0 |
| 27 | FM-27 | Hung hook delays session start up to the 600 s default timeout | runtime | 4·2·2 = 16 | 2·1·2 = 4 | 4 | C-HOOK-TIMEOUT | L1 |
| 28 | FM-31 | Canonicalization change invalidates every stored seal; mass integrity warnings train users to ignore them | change | 4·2·2 = 16 | 2·1·1 = 2 | 2 | C-HASH-VERSION, C-GOLDEN | L1 |
| 29 | FM-28 | Stray stdout from the hook or a dependency is injected as context | runtime | 2·3·4 = 24 | 1·1·2 = 2 | 2 | C-STDOUT-DISCIPLINE | L0 |
| 30 | FM-06 | Uninstalling the plugin deletes every pack stored under CLAUDE_PLUGIN_DATA | runtime | 4·3·5 = 60 | 1·1·1 = 1 | 1 | C-STORE-LOCATION | L0 |
| 31 | FM-16 | Host-to-extension message over 1 MiB | runtime | 2·3·2 = 12 | 1·1·1 = 1 | 1 | C-BUDGET, C-EXT-DEGRADE | L0 |
<!-- relay-model:end risks -->

**Top five for Checkpoint A:**

| # | Risk | Residual RPN | Why it ranks | Next action |
|---|---|---|---|---|
| 1 | FM-29 compaction de-wraps pack text | 48 | Silent; outside our process; unmeasured | Verify how compaction handles `additionalContext`; delta-only re-inject on `source=compact` |
| 2 | FM-08 laundering loop | 40 | Inherent to W1; only partial controls | Echo-strip plus a planted-marker harness; D-008 acceptance test |
| 3 | FM-09 secret egress | 40 | One partial control toward git and other agents | Second, different egress barrier (§5) |
| 4 | FM-30 approval forged by direct write | 32 | Host capability; can't be fully closed | Store outside the project; TTY-bound approve; document the boundary |
| 5 | FM-07 instructions in a pack are acted on | 30 | Wrapper θ unknown | S6 per surface, measured on held-out seeds |

## 11. Measurement backlog *(generated)*

<!-- relay-model:begin backlog -->
| Control | θ measured | θ target | Harness |
|---|---|---|---|
| C-PERMISSION-BOUNDARY | unmeasured | to set | Manual check per permission mode |
| C-WRAP | unmeasured | 0.95 | Planted-injection harness (S6), per surface |
| C-ECHO-STRIP | unmeasured | 0.99 | Planted-marker harness over N chained sessions |
| C-NMH-SCOPE | unmeasured | to set | Compromised-extension tabletop |
| C-SUPPLY | unmeasured | to set | Release checklist |
| C-CWS | unmeasured | to set | Release checklist |
| C-HEARTBEAT | unmeasured | to set | Doctor test with hooks disabled |
| C-AUDIT | unmeasured | to set | Planted-contradiction harness |
| C-TYPECHECK | unmeasured | to set | CI job |
| C-PLUGIN-VALIDATE | unmeasured | to set | CI job (release checklist) |
| C-REDACT | unmeasured | 1 | Planted-secret harness |
<!-- relay-model:end backlog -->

---

## 12. Release blast radius

Distribution channels differ by orders of magnitude in reach and reversibility. Checkpoint C should approve them one at a time,
**lowest reach first**.

| Channel | Who receives a bad release | How fast | Reversible? | Controls | Level |
|---|---|---|---|---|---|
| claude.ai skill zip | Users who upload it by hand | Manual | Yes, re-upload | C-SYNC | L0 |
| `.mcpb` bundle | Users who install it by hand | Manual | Manual | — | L1 |
| npm (`ctxr`, core) | New installs and upgrades | On install | Only partially: registry unpublish rules are narrow, deprecation is not removal | provenance, 2FA, C-PIN | L1 |
| Plugin marketplace | Every user on update; `version` pins until bumped (PN-14) | On update | Revert + bump; already-updated users keep running it | C-SUPPLY, C-PLUGIN-VALIDATE | L0 |
| Hooks (inside plugin) | Everyone updated; runs **every SessionStart** with no user action | Immediate after update | As above | C-SUPPLY (bundled), C-FAILOPEN | L0 |
| Chrome Web Store | Every extension user, through auto-update | Hours | No recall; only a newer version | C-CWS, minimal host permissions | L1 |

## 13. Recommended changes to the plan

1. **Phase 1 spec additions.** Origin field; approval stamp stored *outside* the hashed content and never serialized into shared packs;
   `decisions[]` with `key`, `value`, `supersedes`; `parent` lineage ids; `formatVersion` and `hashAlg`; project-identity key; brief budget
   of 8,000 chars.
2. **Phase 0 facts to add before Checkpoint A** (WORKFLOWS §9):
   - which PowerShell version the fallback uses
   - whether the Bash tool has a TTY
   - whether SessionStart blocks the first prompt
   - how compaction treats `additionalContext`
   - re-verify PN-06, PN-08 and PN-09, which were not checked here
3. **Definition of Done for the hooks phase** includes the injection and echo harnesses with θ reported. **S6 is extended to 3 chained sessions.**
4. **The MCP server is read-only in v1** (D-012). A later write tool creates drafts only.
5. **D-007 (store location) must be settled before any code writes a pack.**
6. **CI runs `node tools/relay-model.mjs check` and `node --test tools/relay-model.test.mjs`.** The sync check covers wrapper markers.
7. **Auto-inject defaults to on for own-session, project-scoped packs only, conditional on D-008.** Repo packs never auto-inject.

## 14. Open questions

- Should `approve` also be reachable from the extension UI (a human click)? That would widen the gate's surface to the native host,
  which currently has a read-only verb set.
- Is a per-user HMAC key in the OS keychain worth its cost for approval stamps? It would make forging (FM-30) harder than a plain file edit.
  The model can still reach the keychain through Bash in some configurations. **(L2)**
- Can C-PLUGIN-VALIDATE run against Claude Code pre-releases, so platform changes are caught before users hit them?
