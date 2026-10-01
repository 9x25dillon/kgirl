---
name: kgirl-blast-radius
description: "Use before editing, moving, renaming or deleting any module, class or function in kgirl (or code copied between the user's repos): finds every dependent file across all indexed repos (imports, symbol references, cloned bodies) so a change does not silently break vendored copies."
---

# kgirl blast radius

kgirl shares a great deal of code with the user's other repos. For example, it holds 884 cloned
function/class bodies from numbskull, and orwells-egg and carryon import its
`src/chaos_llm/services/al_uls_client.py` by bare module name. Grepping inside kgirl alone misses
those. The Atlas sees them.

## Steps

1. Make sure the index exists. If `atlas_search` returns nothing, index first:
   `PYTHONPATH=src python -m kgirl.harness index . 9x25dillon/numbskull 9x25dillon/Eopiez` (add any repos involved).
2. Run `atlas_blast_radius` with the narrowest target that fits:
   - a symbol: `path::Qual.name`, e.g. `llm_adapters.py::LocalStubAdapter`
   - a file: `repo:path`, e.g. `kgirl:src/kgirl/llm/llm_adapters.py`
3. Read the result by edge type:
   - `ref` (d1): the file imports *and uses* the symbol. Edit or check every one.
   - `import` (d2+): transitive dependents. Run their tests or entrypoints.
   - `clone`: an identical body in another file or repo. Same bug, no runtime link. Tell the user which copies exist. Never delete copies; the user keeps them deliberately.
4. For cross-repo questions, `atlas_coupling` (hub `kgirl`) ranks repos by entanglement.
5. Mention the blast radius (counts per repo, any clones) in your summary of the change.

## CLI equivalents

```bash
PYTHONPATH=src python -m kgirl.harness blast "llm_adapters.py::LocalStubAdapter"
PYTHONPATH=src python -m kgirl.harness coupling --hub kgirl
```
