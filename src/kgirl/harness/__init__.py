"""kgirl harness: a repo-aware personal coding assistant.

Layers (each importable and testable on its own):

    atlas   structural index of all repos: symbols, imports, clones, blast radius
    soup    shared memory pool: just-in-time recall + staged, regularized curation
    hermes  message bus + router; every cross-layer call is an envelope on a topic
    llm     backends (Claude for retrieval/summaries, Ollama for code) + fallbacks
    jev     Mr. Meeseeks agents: tiny context, indexed action space, sacrificial swarms

`assistant.Assistant` wires them together; `python -m kgirl.harness --help` is the CLI
and `python -m kgirl.harness mcp` exposes it to Claude Code as an MCP server.
"""

__version__ = "0.1.0"
