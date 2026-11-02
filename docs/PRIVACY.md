# Privacy

What the kgirl harness (`src/kgirl/harness`) does with your data, and how to limit it.
`tests/harness/test_privacy.py` enforces the parts marked **tested**.

## The short version

- The harness has no telemetry, analytics, advertising or payment code. **tested**
- Nothing is sold, and no usage data is sent to the project.
- Its state stays on your machine, in a folder only your account can open. **tested**
- Your code goes to a cloud provider only if you configure one. `KGIRL_LOCAL_ONLY=1` rules that out. **tested**

## What is stored

Everything is in `$KGIRL_HOME` (default `~/.kgirl`), set to mode 0700 when the harness starts:

| file | contents |
|---|---|
| `atlas.db` | signatures, docstrings, imports and references from indexed repos |
| `soup.db` | memory fragments, which can include text you asked it to remember |
| `trace.jsonl` | a log of harness events, which can include questions and task goals |
| `life.json` | the metabolism's counters |
| `repos/` | shallow clones of repos you index by URL |

To erase all of it, delete that folder.

## What leaves your machine

| destination | when | what is sent |
|---|---|---|
| Anthropic | only if `ANTHROPIC_API_KEY` is set, for the `scout` role; `smith` also needs `KGIRL_ALLOW_CLOUD_CODE=1` | your question and retrieved code |
| Ollama | always, for local models, at `OLLAMA_HOST` (default localhost) | prompts; if you point it at another machine, they go there |
| A router you set in `KGIRL_ROUTER_SSH` | only if you set it | search terms, embedding vectors; it stores your chunk text |
| Git hosts | only when you index a repo by URL or `owner/repo` | a clone request |

Set `KGIRL_LOCAL_ONLY=1` to remove every cloud model from the routing.

## Network servers

The API servers listen on `127.0.0.1` by default, so other machines can't reach them. They have no login.
`KGIRL_BIND_HOST` changes the address. Put authentication in front before setting it to anything else. **tested**

## Limits of this page

It covers the harness and the servers named in the test. The repository also holds older and
experimental code under `src/kgirl`, `research/` and the root folder, which hasn't been reviewed
line by line. Some of it can write local logs, such as `telemetry.qthz.jsonl`, which holds synthetic
signal metrics.

## Rules for new features

Any feature that handles other people's data must:

1. Be off by default, and ask for clear consent that can be withdrawn.
2. Never put personal, health or biometric data on a public or permanent ledger, even hashed.
3. Never pay people for personal data in a way that pressures them to share it.
4. Verify who a caller is before charging them; a request header alone isn't proof.
5. Let people export and delete their data.
