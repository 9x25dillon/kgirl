# kgirl console — one live surface over the whole harness

    kgirl console                      # http://127.0.0.1:8765, workspace = the current directory
    kgirl console --dir ~/proj --port 9000
    kgirl console --no-workspace       # indexed repos and memory only, no file tools

A single-page app served by the harness itself (stdlib only, nothing leaves the machine). It drives the
same objects the CLI uses — no parallel implementation. Eight views:

| View | What it does |
|---|---|
| Console | chat with streaming answers, tool chips, recalled-memory markers, good/bad feedback, `/apply` for a waiting swarm diff, workspace diff/undo/auto/verifier |
| Specialists | dispatch a Jev swarm on one goal; live agent table (steps, llm calls, tokens, verified), the winner's diff colored line by line, apply only after you read it |
| Memory | Soup fragments with utility/support/LCB, search and filters, promote/retire/credit, "remember …", the transition ledger |
| Atlas | indexed repos, hybrid keyword+matrix search with citations, numbered source reading (secret-guarded), re-index |
| Forge | spec→code builder (confidence, agreement, module/tests/report tabs, accept as exemplar), long-form writer, evolve, skill forge + SKILL.md export |
| Vitals | metabolism dashboard (A/C/O/F/R/P, predictions, affordable swarm) and the swarm plan |
| Matrix | a sampled spatial projection of the KGVM store (ChaosRNG random projection, colored by repo) and live inference: coarse sign-bit Hamming pass → fine int8 dot pass, with scores |
| Services | every MCP tool kgirl offers other platforms, callable from the browser with a JSON argument form |

A live rail streams every bus envelope (`jev.spawn/step/done`, `build.start/end`, `soup.propose`, …) and
job event. Workspace edits and unlisted commands raise an approval dialog; an unanswered approval times
out to "no".

## How it stays correct

- **One heavy lock.** Chat, task, build, write, evolve and index serialize: the metabolism keeps one
  episode slot and the GPU holds one model. Reads (memory, atlas, vitals, matrix) never wait on it.
- **The bus bridge only enqueues.** Bus handlers run on the publisher's thread (swarm workers), so the
  console's handler puts events on bounded queues; a slow browser drops events, it never stalls a run.
- **Jobs.** Long operations run in worker threads with bounded event logs; the browser watches deltas
  over SSE and fetches results by id. Reconnects get a hello snapshot (pending approvals, running jobs).
- **Localhost only.** Binds 127.0.0.1 and rejects foreign Host headers (DNS rebinding). Static paths are
  confined; source reads pass the same secret guard chat uses.
- **Approvals bridge.** `Workspace.confirm` becomes an HTTP round-trip with a 10-minute timeout that
  refuses — an unattended console never edits or runs anything silently.

## Layout

    console/core.py     Hub (SSE fan-out), JobManager, Approvals, Console (Assistant + Minds + workspace)
    console/server.py   ThreadingHTTPServer, JSON API, static serving
    console/static/     index.html · style.css · app.js (no framework, no build step)

The `console` verb lives in `cli_mind.py`; `cli.py` is untouched.