# porter-blast-radius

A Claude Code plugin for **Porter**, the harness that couples several repos so their functional processes compose into new use cases. It answers "what else does this touch?" before a change crosses a repo boundary.

It contains:

- **An MCP connector (`blast-radius`)** with 9 read-only tools, written with no dependencies on Node's standard library.
- **A skill (`blast-radius`)** that tells Claude when to use the tools and how to read their results.

The connector analyzes a **coupling model**: a JSON graph of components and typed edges. You can get one in three ways:

| Source | Tool | What it sees |
|---|---|---|
| Local git repos | `porter_discover` | Git facts, manifest dependencies (npm, PyPI, Go, Cargo, including git-URL dependencies), and each repo's `porter.json` |
| A kgirl Atlas database | `atlas_import` | Resolved imports across repos and cloned function bodies, per repo or per file |
| A hand-written model | any tool, `model` argument | Anything, for example [`examples/porter-workspace.model.json`](examples/porter-workspace.model.json) |

## Install

From a clone of this repository:

```sh
claude plugin marketplace add ./context-relay        # the marketplace root holds .claude-plugin/marketplace.json
claude plugin install porter-blast-radius@porter
```

For development, `claude --plugin-dir context-relay/plugins/porter-blast-radius`.

Requirements:

- Node 18 or later.
- Node 22.13 or later for `atlas_import`, which uses the built-in `node:sqlite`.

### Settings (`/config`)

| Option | Meaning |
|---|---|
| `extra_roots` | More directories the connector may read, separated by `:` (`;` on Windows). The project directory is always allowed. |
| `atlas_db` | Path to a kgirl Atlas database, usually `~/.kgirl/atlas.db`. Only this one file is allowed, and it is opened read-only. |

## Tools

| Tool | Question it answers |
|---|---|
| `list_models` | Which models and `porter.json` manifests exist under the roots? |
| `check_model` | Is this model consistent? Are all its persistence loops contained? (`as_built` checks today's system) |
| `blast_radius` | What does a change set reach (`downstream`)? What can break it (`upstream`)? Per view: change, runtime, integrity, confidentiality |
| `compose` | Does this workflow type-check? Which entry/exit compositions need a human gate? |
| `emergent_use_cases` | Which chains of repo capabilities line up across repos, and how fragile is each? |
| `risk_register` | Ranked failure modes, control value, open loops (`as_built` ranks before proposed fixes) |
| `atlas_import` | Coupling model from a kgirl Atlas: repos or files, import and clone edges; bare-name guesses flagged |
| `merge_models` | One graph from several sources (for example `porter_discover` plus `atlas_import`); repos with the same remote become one node |
| `porter_discover` | Coupling model from local repos, with evidence for every edge |

Results report nodes at three levels:

- **unmitigated**: no control on the best path.
- **damped**: only partial controls, such as tests, redaction or wrappers.
- **contained**: a full control stops it.

## Declaring Porter couplings: `porter.json`

Drop this at a repo's root (see [`examples/porter.json`](examples/porter.json)):

```json
{
  "porter": 1,
  "name": "ingest",
  "capabilities": [{ "id": "parse", "in": "RawDoc", "out": "Doc", "effects": ["fs.read"] }],
  "dependsOn": [{ "target": "github.com/acme/schema", "kinds": ["change", "runtime"], "controls": ["C-CONTRACT"] }],
  "controls": { "C-CONTRACT": { "strength": "full", "mechanism": "contract tests against schema fixtures" } }
}
```

- **`capabilities`** are typed functional processes. A chain whose types line up across repos is an emergent use case.
- **`dependsOn`** declares couplings that manifests can't show: HTTP APIs, shared data, synced context. `target` is a remote (`host/owner/repo`), a repo name, or a `repo:` id.
- **`controls`** are what stops propagation along a coupling: `full` contains it, `partial` damps it.

## Security properties

The connector applies to itself the blast-radius rules it computes. [`model/connector.model.json`](model/connector.model.json) is its own model, and [`test/connector-model.test.mjs`](test/connector-model.test.mjs) checks it.

- **No execution.** Git facts are read from `.git` files, never by running `git`, so a hostile repo can't run code through git config such as `core.fsmonitor`. The shipped code imports no process, network or VM modules ([`test/hygiene.test.mjs`](test/hygiene.test.mjs)).
- **No writes.** No filesystem write calls. SQLite is opened with the read-only flag, and the SQLite engine itself refuses writes.
- **Confinement.** Every path resolves to a real path inside the allowed roots, or to the one configured Atlas file. Symlinks can't escape the roots, and directory walks don't follow them.
- **Bounded.** Reads are capped (2 MB for models, 1 MB for manifests). Results shrink array by array to fit 30,000 characters, and say so when they were truncated.
- **Data, not instructions.** Output strings have control and bidi characters removed and are length-capped. Server instructions tell the client that model text is data. Credentials in remote URLs are dropped.
- **Strict arguments.** Unknown keys, wrong types and out-of-range values are rejected before any handler runs.

The self-model's top residual risk isn't a security risk. It is **reading an incomplete model as "safe"** (CFM-05). A node missing from a result may simply be missing from the model.

## Tests

```sh
node --test test/*.test.mjs                                            # 66 tests; the kgirl contract test skips without KGIRL_SRC
KGIRL_SRC=/path/to/kgirl/src node --test test/atlas.test.mjs           # also runs kgirl's own indexer and blast radius
```

The protocol follows the MCP TypeScript SDK 1.31.0:

- newline-delimited JSON-RPC
- version negotiation that echoes a supported client version, otherwise answers `2025-11-25`
- tool failures returned as `isError` results

`claude plugin validate --strict` passes for the plugin and the marketplace. Claude Code 2.1.285 loads the plugin and reports the server as connected.
