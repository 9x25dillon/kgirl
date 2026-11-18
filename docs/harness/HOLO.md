# holo: small overlapping matrix databases, with provenance, serving real-time inference

`src/kgirl/harness/holo/` wraps math that already exists in kgirl into one harness layer. It adds no new
formula and no new embedding: every number it stores is derived from the vectors `kgirl chunks` already
made, and every formula it applies is called from the module that defines it.

    python -m kgirl.harness.holo build      # derive the store from the chunk index (offline, no model)
    python -m kgirl.harness.holo status     # entities, storage, chain verification, last build
    python -m kgirl.harness.holo query --text "where is the router"   # embeds the question (one model call)

The same operations are available as the `kgirl holo` verb (`cli.py`, public holo API only), plus
`verify` (every entity's hash chain and unproven matrices), `overlaps` (shared digests with owners) and
`owners --digest <d>` (who recorded one matrix):

    kgirl holo build | status | query --text "..." | verify | overlaps | owners --digest <d>   # JSON; `holo -h`

## The bar (written before the code)

1. Byte-compiles; `PYTHONPATH=src python3 -c "import kgirl.harness.holo"` resolves and pulls in no torch.
2. The offline slice, build then query on synthetic data, is correct by reading: no model, no network.
3. Overlap is real: at least one matrix digest is referenced by two or more entities.
4. Provenance is real: every record this layer writes has a hash-chained ledger entry, and
   `MatrixDB.verify` passes for every entity.
5. Deterministic: two builds from the same index with the same chaos seed give the same digests and the
   same ledger heads.
6. Nothing outside `src/kgirl/harness/holo/**`, `tests/holo/**` and this file is edited.

## What it wraps

| Module | What the layer uses | Wrapped how |
|---|---|---|
| `vessel/matrixdb.py` | `MatrixDB`: content-addressed matrices, typed geometries, the hash-chained ledger (`chain_hash`) | all writes go through `put_matrix`, `record`, `append`; the store lives at `$KGIRL_HOME/holo.db` |
| `harness/atlas/vectors.py` | KGVM: `Matrix`, `encode_query`, `search` (sign-bit Hamming, then int8 dot) | inference calls `Matrix.search` unchanged; the build reads `Matrix.vals` |
| `tauls/holo.py` | `HoloMemory.project/distribution`, `entropy`, `gate` | optional prior, only when a trained memory file exists (below) |
| `evolve/chaos.py` | `ChaosRNG` | the only source of randomness: which chunk vectors a cell keeps |
| `harness/metabolism.py` | `Metabolism.observe` | regulation, when the caller hands over the organism |
| `harness/hermes/bus.py` | `Bus.publish` | `holo.built`, `holo.query`, `holo.regulate` |
| `quantum/holographic_memory_system.py` | not imported | it needs torch, scipy and matplotlib at import time; wrapping it would break bar 1 |

## The store

An **entity** is a named region of the vector space with its own ledger:

    repo:<name>     every chunk of one indexed repository
    kind:<kind>     every chunk of one symbol kind (function, class, preamble, ...), across repositories
    soup:<kind>     the active Soup fragments of one kind (an evidence table, not vectors; see below)
    holo            the store itself: its ledger is the build log

Each entity holds small matrices, recorded as MatrixDB geometries:

| Geometry | Shape, dtype | Contents |
|---|---|---|
| `centroid` | (dim,) float32 | mean of the entity's KGVM int8 vectors, cast to float32 |
| `spread` | (dim,) float32 | their standard deviation per dimension |
| `chunk`, label = chunk id | (dim,) int8 | one chunk's KGVM vector, byte for byte |
| `evidence` (Soup entities) | (n, 4) float32 | support, wins, losses, uses of each active fragment |

**Overlap** is content addressing doing its job. A chunk belongs to exactly one repository and one symbol
kind, so its vector is recorded under `repo:<r>` and under `kind:<k>`. Both records point at the same
digest: one stored matrix, two owners. `HoloStore.overlaps()` lists every digest with two or more owners,
and `HoloStore.owners(digest)` answers "which regions share this vector".

**Small** is kept by sampling. A (repository, kind) cell with more than `per_cell` chunks (default 16)
keeps a chaos-drawn sample of that size; centroid and spread always use every chunk. `per_cell=None`
records every chunk vector.

**Provenance.** `HoloStore.record` writes the geometry and then appends a `holo.record` event to the
entity's ledger naming the kind, label, digest, dtype and shape. The ledger is MatrixDB's:
`h_i = sha256(h_{i-1} ‖ entity ‖ seq ‖ event ‖ canonical-JSON payload)`. A record whose digest is already
the latest for its (entity, kind, label) is not written again, so a rebuild of an unchanged index adds
nothing to an entity's chain. An entity that changed also gets a `holo.built` event with its chunk count
and the repository head it was built from. `HoloStore.verify()` recomputes every chain and also reports
any geometry with no ledger entry.

Timestamps are not part of the chain and payloads carry none, which is what makes bar 5 hold. The root
entity `holo` is the exception by design: its `holo.built` event records the paths of the chunk index and
the matrix, so the store can reopen them for inference.

## Inference

`query(store, vector_or_text, k)`:

1. A vector is used as given. Text is embedded with the matrix's own model and the `search_query: `
   prefix; that is the one model call in this layer, and it does not happen when you pass a vector.
2. `Matrix.encode_query` then `Matrix.search`: coarse sign-bit Hamming shortlist, fine int8 dot.
3. Each hit gets its owning entities, each owner's centroid digest and current ledger head, and the
   content address of its own vector (the digest MatrixDB gives that int8 row). `stored` says whether
   the store holds that vector under its owners, which it does for the sampled ones.
4. `entities` is the guidance block: for each owning entity, its share of the positive hit score and the
   cosine between the query and its centroid.
5. Optional prior (below), then `holo.query` on the bus, then regulation when asked for.

A hit:

    {"id": 17, "score": 15872, "repo": "kgirl", "path": "src/kgirl/harness/hermes/router.py",
     "symbol": "Router.stream", "kind": "function", "line": 157, "cite": "kgirl:src/.../router.py:157",
     "entities": ["repo:kgirl", "kind:function"], "digest": "<sha256 of the vector>", "stored": true,
     "provenance": [{"entity": "repo:kgirl", "centroid": "<sha256>", "head": "<ledger head>"}, ...]}

### The HoloMemory prior

Off unless `$KGIRL_HOME/holo-memory.npz` exists. Training one is not part of this layer. When the file is
there, its `dim` must equal the matrix dimension and its `vocab` must equal the number of `repo:` and
`kind:` entities, in the order `HoloStore.vocabulary()` returns them (so "next token" means "the entity the
answer comes from"). If either does not match, the prior is skipped and the result says why.

With `p_ret` the retrieval's share per entity and `p_mem = HoloMemory.distribution(project(q))`:

    w = gate(entropy(p_ret), entropy(p_mem), prior, temp)        # tauls/holo.py, unchanged
    p = (1 - w) · p_ret + w · p_mem

The gate settings (`prior`, `temp`, `beta`, `eta`, `knn_eta`, `tau`) are read from
`$KGIRL_HOME/holo-memory.json` when present: the dict `tauls.holo.fit_gate` returns. Without it:
`prior=0.1`, `temp=1.0` and the memory's own defaults. The prior changes the guidance block only. It does
not reorder hits: whether it should is a measurement, and no bar for it has been written.

## Bus topics

| Topic | When | Payload |
|---|---|---|
| `holo.built` | after a build | entity counts, chunks, sampled, new records, overlaps, storage bytes, seed |
| `holo.query` | after each query | `q` (text or null), `k`, `ms`, hits as `{id, score, cite, entities}`, `top` entity, `prior` used |
| `holo.regulate` | when a metabolism observed the query | the observation and the organism's F, O, C, R after it |

Handlers run on the publisher's thread, so payloads are small plain dicts with no matrices in them.

## How the console consumes it

The console already bridges bus topics to its SSE stream. Nothing here imports the console.

- Matrix view: subscribe to `holo.query` and draw `entities` (share, cosine) beside the hits; a hit's
  `provenance` gives the digests to show, and `HoloStore.owners(digest)` the regions that share it.
- Vitals view: `holo.regulate` carries the organism's state after each observed query.
- A build button calls `build(home, bus=assistant.bus)`; `status(home)` fills a store panel.

Wiring those three calls into `console/` and adding a `kgirl holo` verb to `cli.py` are left for the
console lane: both files are outside this one.

## Decisions that are yours

1. **Regulation is opt-in and feeds no energy.** `query(..., metabolism=m)` calls
   `m.observe((success, latency, 0, 0), energy=0.0, work=0.0, kind="holo.query")`. Every `observe` is one
   life step. In the offline spot check one observed query moved the reserve F of a fresh organism from
   5.00 to 4.63; with `energy=1.0` (what a verb gets) it would grow instead. Neither is neutral for swarm
   sizing, so the layer never does it unless a metabolism is passed.
2. **Soup entities hold evidence tables, not vectors.** Soup fragments have no embeddings and this lane may
   not create any.
3. **`per_cell=16` by default.** Recording every chunk vector twice is several MatrixDB transactions per
   chunk; on a full index that is a large database and a long build.

## What has been checked, and what has not

Checked on 2026-11-15: every file byte-compiles; `import kgirl.harness.holo` resolves with no torch,
scipy or matplotlib loaded; and one offline spot check on synthetic data (24 chaos vectors, two
repositories, three symbol kinds, temporary homes) ran build then query: digests and ledger heads were
identical across two homes, 18 digests had two owners at `per_cell=3`, every chain verified, a rebuild
wrote nothing new, a chunk's own vector found that chunk first, and the gate reproduced
`(1 − w)·p_ret + w·p_mem` with a fixture memory.

Not checked: `tests/holo/` (49 tests) is written and has not been run, per the build-forward rule. Nothing
has been built from the real index in `~/.kgirl`, and no text query has been embedded.
