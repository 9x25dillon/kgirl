# ChaosRAGJulia (Single-File)

A compact Julia service that unifies a **KFP chaos router**, **HHT/EEMD** time–frequency analytics, and **OpenAI-based RAG** for crypto research.

**License: Apache 2.0** (full text below).

## Install & Run

```bash
export DATABASE_URL=postgres://user:pass@localhost:5432/chaos
# optional
export OPENAI_API_KEY=sk-...

julia --project -e 'using Pkg; Pkg.add.(["HTTP","JSON3","LibPQ","DSP","UUIDs","Interpolations"])'
julia server.jl
```

The server bootstraps the schema and tries to enable `pgvector`. If extensions can’t be installed by your DB role, pre-install them or ignore the warning; tables still create.

## Endpoints

- `POST /chaos/rag/index` — index docs `{docs:[{source,kind,content,meta}]}`
- `POST /chaos/telemetry` — push `asset, realized_vol, entropy, mod_intensity_grad`
- `POST /chaos/hht/ingest` — EEMD + Hilbert on window `{asset, ts[], x[], fs}`
- `POST /chaos/graph/entangle` — upsert edges `{pairs:[[src,dst],...], weight?, nesting_level?, attrs?}`
- `GET  /chaos/graph/:uuid` — fetch node + edges
- `POST /chaos/rag/query` — chaos-routed mixed retrieval + LLM answer `{q, k?}`

## Router (KFP-inspired)
`stress = σ(1.8·vol + 1.5·entropy + 0.8·|grad|)`  
`mix = { vector, graph, hht }` increase HHT/graph under stress, shift back to vector when calm. `top_k` shrinks as stress rises.

## HHT/EEMD
CPU-only minimalist EEMD (ensemble, noise_std, max_imfs). Hilbert features: instantaneous frequency & amplitude with burst flag by amplitude percentile threshold.

## Apache License 2.0
Copyright 2025 Your Name

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
