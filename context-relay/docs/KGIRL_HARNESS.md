# kgirl harness: inspection and Porter connection

This document covers the kgirl harness at `src/kgirl/harness/` (branch `claude/dazzling-sagan-0uf4y7`, commit `26748c4`): Atlas, Soup, Hermes and Jev. §3 also covers what `80f8b85` added on kgirl `main` (`d2314f2`): Jev intuition, the skill forge, and the `ecl_*` / `evolve_*` tools.

Inspection was read-only; nothing was pushed to kgirl. The connection lives in [`plugins/porter-blast-radius`](../plugins/porter-blast-radius). Evidence levels: **L0** verified by running code, **L1** from reading code, **L2** needs measurement.

---

## 1. Summary

- **Connected.** Porter reads kgirl's Atlas database directly: `atlas_import` in the porter-blast-radius connector opens `atlas.db` read-only and turns it into a coupling model. Every Porter tool then works on kgirl's graph: blast radius in four views, upstream exposure, loops, and as-built versus designed comparisons. **(L0)**
- **The two implementations agree.** On a real index of kgirl, numbskull and this repo (made by kgirl's own `index` command):
  - Our file-level blast radius of `advanced_embedding_pipeline/__init__.py` is **47 transitive and 23 direct**. That matches kgirl's `blast` exactly, and the ATLAS_REPORT figure.
  - Cross-repo import counts match kgirl's `coupling`.
  - A contract test runs kgirl's indexer and compares results. **(L0)**
- **Most important finding: Soup's staging gate is bypassed over MCP.** `soup_remember` stores memory written by the model as `active`, labelled `source="user"`. It is recalled in every later session. The curator's staging (support gate, LCB, validator) never sees it. This is the same persistence loop Context Relay's D-008 and C-TRUST-GATE address. **(L0 for the code path, L1 for the impact)**
- **`jev_swarm_task` is code execution.** Its `verify` argument is a command string the model supplies, and it runs with the user's full environment, API keys included. `apply` writes into any `repo_path` the model names. **(L1)**
- **Some of Atlas's cross-repo couplings are guesses.** A bare `from db import x` resolves to a same-named file in another repo when the importer's repo has none. In our three-repo run, **both** cross-repo import edges were such guesses: this repo's `sister.py` was linked to kgirl's `src/kgirl/utils/db.py`. The real coupling between kgirl and numbskull is **602 shared function and class bodies**. **(L0)**

## 2. How Porter connects to Atlas

```
 kgirl                                            porter-blast-radius (Claude Code plugin)
 ─────                                            ────────────────────────────────────────
 python -m kgirl.harness index <repos>            atlas_import  (node:sqlite, readOnly: true)
        │                                                │  checks REQUIRED_COLUMNS, names what is missing
        ▼                                                ▼
 $KGIRL_HOME/atlas.db ─────────── read only ───▶  coupling model  ──▶ blast_radius / check_model /
   repos, files, symbols(body_hash),                repo or file nodes      risk_register (as_built) /
   imports(resolved_file_id), refs                  import: change+runtime  compose / emergent_use_cases
                                                    clone:  change, both ways
```

- **Why read the database instead of calling kgirl.** Running `python -m kgirl.harness blast` would give the connector code execution, which is what its self-model forbids (C-NO-EXEC). Reading SQLite with the engine's read-only flag keeps the connector unable to run programs or write files, and the hygiene test enforces it.
- **Why not chain the two MCP servers.** Chaining couples them at the model level, where every result passes through the agent's context. A file contract couples them at the data level and can be tested (C-SCHEMA-CHECK, contract test).
- **Edge semantics** follow Atlas's own docstring in `atlas/graph.py`:
  - **Import** edges point from provider to importer and are `change` plus `runtime`.
  - **Clone** edges are `change` in both directions: "the same defect lives in both places".
  - Atlas's symbol-level `ref` edges are a subset of import edges. They refine *which* importer uses a symbol; at file level the import edge already carries them.
- **Setup:** set the plugin's `atlas_db` option to `~/.kgirl/atlas.db`. That one file is then readable, without widening the roots. Then run `atlas_import` and pass its model to `blast_radius` as `model_inline`.

### Verification on real data

Indexing took 4.6 s with kgirl's own CLI.

| Check | kgirl (Python) | Porter (Node) |
|---|---|---|
| Files indexed / parse failures | kgirl 386 / 19, numbskull 110 / 4, myAssistant 28 / 0 | same, from `files.parse_error` |
| numbskull → kgirl imports | `imp->=1` | 1 edge, flagged bare-name |
| kgirl → myAssistant imports | `imp<-=1` | 1 edge, flagged bare-name |
| Blast radius of `advanced_embedding_pipeline/__init__.py` | 47 impacts, 23 at depth 1 | 47 transitive, 23 direct (runtime view); 51 with clone edges |
| kgirl ↔ numbskull clones | 884 kgirl *symbols* with a twin | 602 distinct shared *bodies* |

The clone counts measure different things: one body hash can cover several kgirl symbols. Both confirm the report's finding.

## 3. Findings

The model is [`examples/kgirl-harness.model.json`](../plugins/porter-blast-radius/examples/kgirl-harness.model.json). Controls marked `implemented` exist in kgirl today; controls marked `proposed` are the fixes below. [`test/kgirl-model.test.mjs`](../plugins/porter-blast-radius/test/kgirl-model.test.mjs) pins down every claim in this table.

| ID | Finding | Where | As built → with fix (RPN) |
|---|---|---|---|
| KFM-01 | Model-written memory is stored `active` with `source="user"` and recalled in every later session | `mcp_server.py` `remember` → `assistant.py` `remember` → `soup.add(status="active")` | 100 → 10 |
| KFM-02 | `verify` runs any program the model names, with the full environment | `mcp_server.py` `task`, `jev/code_env.py` `verify` | 60 → 10 |
| KFM-03 | `apply=true` writes into any `repo_path` | `jev/swarm.py` `run` | 24 → 8 |
| KFM-04 | Docstrings and signatures of indexed third-party repos reach the agent unframed | `atlas_search`, `atlas_outline`, `atlas_source`, `atlas_card` | 48 → 24 |
| KFM-05 | Signatures (including default values), docstrings and source lines go to the scout model unredacted | `assistant.py` `ask` | 48 → 24 |
| KFM-06 | Files that fail to parse drop their imports, so their dependents vanish from blast radius (19 in kgirl) | `atlas/parse.py` | 48 → 24 |
| KFM-07 | Bare module names resolve across repos by guess | `atlas/graph.py` `py_suffix` | 48 → 24 |
| KFM-08 | Cloned bodies: a fix in one repo leaves the defect in the other | Atlas clone edges | 60 → 30 |
| KFM-09 | `initialize` echoes any requested protocol version | `mcp_server.py` `handle` | 18 → 1 |
| KFM-10 | An Atlas schema change would break the bridge (contained: the columns are checked) | bridge | 36 → 9 |
| KFM-11 | A goal with newlines adds sections to an exported `SKILL.md`; staged trajectories count; MCP `skills_forge` thresholds are unbounded; forged skills are stored `active` | `skills.py` `forge`, `to_skill_md`; `jev/intuition.py` `from_soup` | 48 → 12 |
| KFM-12 | A `jev_swarm_task` goal persists in an `active` trajectory, outside the curator: recalled, replayed by intuition, and exported by the forge | `jev/swarm.py` `_learn` | 48 → 12 |
| KFM-13 | The generic MCP server imports domain packages; a domain change or a missing numpy shows up as a harness tool failure | `mcp_server.py` `ecl_*`, `evolve_assay` | 18 → 12 |

The model maps every one of the server's 15 tools to the component it reaches (`kgirl-mcp.tools`). With `KGIRL_SRC` set, a contract test compares that map with `mcp_server._tools()`, so the model can't drift from the code without a red test. **(L0)**

### What the analysis shows

- **As built, the agent and `soup.db` share an uncontained loop** (`agent, assistant, kgirl-mcp, soup, soup-db`). Enumerating elementary cycles misses it, because the loop passes the MCP server twice. Porter's check now also tests strongly connected components (D-016). With C-STAGE-MCP alone, the loop closes.
- **As built, an agent reaches code execution** on the host through `jev_swarm_task`. With a verify allowlist, that path is contained.
- **As built, secrets in repos reach the scout model unmitigated.** Redaction only damps this, and a second barrier would need to be a different mechanism (see BLAST_RADIUS §5).
- **The Porter bridge has one outflow**: sanitized output to the agent. It writes nothing into kgirl.
- **Trajectories bypass the curator.** The swarm stores a winner's record with `soup.add`, which defaults to `active`; only the rules distilled from it go through `curator.propose`. The model used to put C-CURATOR on that edge, which overstated it. It now separates `distiller → soup` (C-CURATOR) from `jev-swarm → trajectories` (C-VERIFIED-WIN, partial). **(L1)**
- **With #55 and #59 merged, one loop stays open**: `kgirl-mcp → jev-swarm → trajectories → skill-export → skills-dir → agent`. It is damped, not unmitigated, because only the person-run CLI writes `SKILL.md` (C-EXPORT-CLI). C-STAGE-TRAJECTORY (proposed) closes it: store MCP-started trajectories `staged` with `source="mcp"`. #59 already makes forge and intuition read only `active` trajectories, so that one change is enough.
- **`soup_remember` notes can't reach the forge.** They have no steps, so the model gives trajectory rows their own data node (`trajectories`) and routes only those into intuition, the forge and the export.

### Patches

The patches for KFM-01, KFM-02, KFM-03 and KFM-09, plus server instructions for KFM-04, are in
[9x25dillon/kgirl#55](https://github.com/9x25dillon/kgirl/pull/55), stacked on the harness branch. 44 kgirl tests pass, 6 of them
new. The model keeps these controls `proposed` until that PR merges, so `as_built` describes kgirl's default branch. The
sketches below show the approach; the PR has the tested code.

**KFM-01: stage model-written memory.** Don't route it through `curator.propose`. `propose` reinforces near-duplicates and credits every new fragment as "born from a verified success", which would let a model promote its own memory by repeating it.

```python
# mcp_server.py
def remember(p):
    fid = a.soup.add(p["text"], kind=p.get("kind", "note"), tags=p.get("tags", ""),
                     source="mcp", scope=p.get("scope", "*"), status="staged")
    return f"staged fragment #{fid}; recalled only after a human promotes it"

# soup/curator.py, in the promotion loop over staged fragments
if f.source == "mcp":
    continue  # promoted by a person (add `soup promote <id>` to the CLI), never automatically
```

**KFM-02: allowlist verify commands and scrub the environment.**

```python
# mcp_server.py
ALLOWED_VERIFY = [c.split() for c in os.environ.get("KGIRL_VERIFY_ALLOWLIST", "").split(";") if c.strip()]
def task(p):
    cmd = p["verify"].split() if isinstance(p.get("verify"), str) else p.get("verify")
    if cmd and cmd not in ALLOWED_VERIFY:
        raise PermissionError("verify command is not in KGIRL_VERIFY_ALLOWLIST")
    ...

# jev/code_env.py, verify()
env = {k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I)}
```

**KFM-09: negotiate the protocol version.**

```python
SUPPORTED = ("2025-06-18", "2025-03-26", "2024-11-05")
requested = params.get("protocolVersion")
result = {"protocolVersion": requested if requested in SUPPORTED else PROTOCOL_VERSION, ...}
```

**KFM-11: close the forge's trust boundary.** Fixed in [9x25dillon/kgirl#59](https://github.com/9x25dillon/kgirl/pull/59). `util.inline` collapses control, format and separator characters and caps length, and every caller-supplied string goes through it before grouping, storing or rendering. Forge and `Intuition.from_soup` read only `active` trajectories. Forged skills are stored `staged`. MCP `skills_forge` has floors (`min_support >= 2`, `min_utility >= 0.5`). It adds 5 tests, and merges cleanly with #55: 54 harness tests pass on the combined tree. **(L0)**

**KFM-07: record how each import was resolved.** Add a `resolution` column: `relative`, `exact`, `suffix`, or `bare-guess`. Consumers could then weight edges instead of inferring from the target's shape, which is what the bridge does today (`exclude_bare_names`).

## 4. Using it from Porter

1. Index the repos with kgirl (`python -m kgirl.harness index …`), and set the plugin's `atlas_db` option.
2. Run `atlas_import`. Read its warnings: parse failures and bare-name guesses are where the graph is weakest.
3. Run `blast_radius` with `model_inline` on the repo or file you are changing. Use `direction: "upstream"` for what can break it.
4. For file-level work, use `atlas_import` with `granularity: "file"`, `repo`, and optionally `path_prefix`. Oversized graphs are refused with guidance.
5. To weigh kgirl's fixes, run `risk_register` or `check_model` on `examples/kgirl-harness.model.json`, with and without `as_built`.

## 5. Open questions

- **Merging models** is done: `merge_models` unions an Atlas model with a `porter_discover` model, and folds components with the same remote into one id. Atlas names repos after directories, discovery after remotes.
- **Live model run.** The harness PR notes that live Ollama and Anthropic runs haven't been done. KFM-02 and KFM-05 matter most once they are.
- **Real-world rate of bare-name guesses.** How many of the ATLAS_REPORT's cross-repo importers (for example orwells-egg and carryon → `al_uls_client.py`) are bare-name guesses? Running `atlas_import` on the full 33-repo database answers it in one call.
