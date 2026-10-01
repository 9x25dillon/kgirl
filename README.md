# Context Relay: workflow and blast-radius model

Working name **Context Relay** (skill `context-relay`, CLI `ctxr`, packs `*.ctxpack.json` / `*.ctxpack.md`). The final name is chosen at Checkpoint A.
This directory is **Phase 0 design input** for Checkpoint A. It contains no product code.

It answers two questions with an executable model instead of prose alone:

1. **Emergent workflows.** Which workflows arise when 8 ways of creating a pack are composed with 9 ways of consuming one, and which of
   them are safe as they stand, need a human gate, or must never exist?
2. **Blast radius.** When a component fails, changes, or is compromised, what else does it reach, and which control stops it?

## Contents

| Path | What it is |
|---|---|
| [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md) | Primitive algebra, route matrix, 13 workflows, 11 rejected anti-workflows, continuity-loop dynamics, θ harness plan |
| [`docs/BLAST_RADIUS.md`](docs/BLAST_RADIUS.md) | Integrity, confidentiality, runtime and change radius; persistence loops; control value; ranked risk register; release blast radius; plan changes |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | D-006 to D-017 (proposed), continuing the Phase 0 log |
| [`docs/PLATFORM_NOTES_ADDENDUM.md`](docs/PLATFORM_NOTES_ADDENDUM.md) | PN-01 to PN-15: platform facts with source URLs and status, checked 2026-09-30 |
| [`model/relay-model.json`](model/relay-model.json) | The model: primitives, sinks, controls, components, edges, routes, workflows, failure modes |
| [`tools/relay-model.mjs`](tools/relay-model.mjs) | Checker and analyzer (Node ≥ 18, no dependencies) |
| [`tools/relay-model.test.mjs`](tools/relay-model.test.mjs) | 44 tests, including mutation tests that break the model on purpose |
| [`plugins/porter-blast-radius/`](plugins/porter-blast-radius) | Claude Code plugin for Porter: a read-only MCP connector (9 tools) and a skill. It builds coupling models from local repos or a kgirl Atlas and computes blast radius, loops and emergent use cases. The engine above lives here. |
| [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json) | Marketplace `porter`: `claude plugin marketplace add ./context-relay` |
| [`docs/KGIRL_HARNESS.md`](docs/KGIRL_HARNESS.md) | Inspection of the kgirl harness (Atlas, Soup, Hermes, Jev), the Atlas bridge, cross-checks against kgirl's own results, findings and proposed patches |

## Commands

Run from this directory:

```sh
node tools/relay-model.mjs check                   # validate the model, workflows and loops, and that generated docs are fresh
node --test tools/relay-model.test.mjs             # test suite
node tools/relay-model.mjs report routes           # print one table (see `report` usage for all section names)
node tools/relay-model.mjs sync-docs               # regenerate the tables in docs/*.md after editing the model
node --test plugins/porter-blast-radius/test/*.test.mjs   # connector suite (see the plugin README)
```

To change the design, edit `model/relay-model.json`, run `sync-docs`, then `check` and the tests. Don't edit the generated tables
between the `relay-model:begin` and `relay-model:end` markers by hand.

## Inputs

- The Context Relay build prompt and the Phase 0 status summary (F1–F15, D-001–D-005). Those files are not in this repository.
- Claude Code documentation, read on 2026-09-30 (PLATFORM_NOTES_ADDENDUM).
- Research the user provided, used for methodology and cited as synthetic, L2 evidence. It is not committed here:
  - the OmegaClaw `memory_audit` plugin and its θ benchmark
  - the planted-contradiction test
  - the LiMPS coherent-reintegration experiment
