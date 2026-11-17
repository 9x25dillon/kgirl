"""The kgirl console: one local surface over the whole harness.

`kgirl console` serves a single-page app on 127.0.0.1 that drives the same objects the CLI uses:
Mind conversations, Soup memory, the Atlas index, Jev specialist swarms, the builder, the writer,
evolve/skills, metabolism vitals and the KGVM matrix. Long operations run as jobs in worker threads;
everything that happens (bus envelopes, job deltas, workspace approvals) reaches the browser over one
SSE stream. Stdlib only; nothing leaves the machine.
"""

from .core import Console
from .server import serve

__all__ = ["Console", "serve"]