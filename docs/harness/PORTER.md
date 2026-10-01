# Porter in kgirl

`context-relay/` is a copy of the same directory in [9x25dillon/myAssistant](https://github.com/9x25dillon/myAssistant), which
stays canonical. Porter couples the user's repos, and this copy gives kgirl two things:

| Path | What it is |
|---|---|
| `context-relay/plugins/porter-blast-radius/` | A Claude Code plugin with a read-only MCP connector (9 tools). It reads `~/.kgirl/atlas.db` and answers blast-radius questions in four views: change, runtime, integrity and confidentiality. |
| `context-relay/skills/mcp-builder-hardened/` | Anthropic's `mcp-builder` skill plus a blast-radius review for MCP servers and a stdlib checker. It is linked as a project skill at `.claude/skills/mcp-builder-hardened`. |

## Connector

From the repository root:

```bash
claude plugin marketplace add ./context-relay
claude plugin install porter-blast-radius@porter      # set atlas_db to ~/.kgirl/atlas.db when asked
```

The connector and the `atlas_*` tools serve different questions:

- **kgirl's own `atlas_blast_radius` and the `kgirl-blast-radius` skill** answer "what imports, uses or copies this file or symbol". That is the question to ask before an edit.
- **Porter** works on a typed coupling graph built from the same database (`atlas_import`). It can also merge that graph with manifest discovery (`merge_models`). It adds:
  - the four views
  - controls that damp or contain an edge
  - persistence loops, found with SCC
  - `as_built` versus designed comparisons
  - cross-repo capability chains (`emergent_use_cases`)

Porter opens the database read-only and never runs kgirl's code.

The kgirl model is `context-relay/plugins/porter-blast-radius/examples/kgirl-harness.model.json`; the review behind it is
`context-relay/docs/KGIRL_HARNESS.md`.

- The model describes the harness as merged in #54.
- It does not yet cover `80f8b85`: Jev intuition, the skill forge and the new MCP tools.
- The controls added by [#55](https://github.com/9x25dillon/kgirl/pull/55) stay `proposed` until #55 merges.

## Tests that use kgirl itself

```bash
KGIRL_SRC=$PWD/src node --test context-relay/plugins/porter-blast-radius/test/*.test.mjs   # includes the Atlas contract test
python3 context-relay/skills/mcp-builder-hardened/scripts/blast_radius_check.py src/kgirl/harness
```

- The contract test indexes three fixture repos with kgirl's own `index` command. It checks that the bridge reads that database cleanly: the schema, a resolved cross-repo import, and clone edges. It then compares blast radius with kgirl's `blast`.
- The checker is a lint for MCP tool handlers.
  - On `main` it reports four findings: BR001–BR004 in `mcp_server.py` and `jev/code_env.py`.
  - With #55 applied it reports none.

## Keeping the copy in sync

```bash
sh context-relay/tools/sync-subtree.sh                                    # pull myAssistant main
sh context-relay/tools/sync-subtree.sh https://github.com/9x25dillon/myAssistant <branch>
```

- Each sync adds one squashed commit and a merge. Edits made here survive.
- Prefer making changes in myAssistant and syncing, so the copies in other repos stay the same.
- Don't use `git subtree` here. kgirl tracks a file named `HEAD` at its root (kept on purpose, see HANDOFF.md §3), which makes `git subtree` fail. The script avoids that.
