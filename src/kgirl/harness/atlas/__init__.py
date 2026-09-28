"""Atlas: the structural memory of every repository the assistant knows.

Stores architecture (symbols, signatures, imports, references, clone hashes)
rather than raw text, so "knowing a repo" costs kilobytes, not megabytes, and
cross-repo blast radius is a graph query.
"""

from .graph import (BlastReport, Coupling, Impact, architecture_card, blast_radius, coupling,
                    render_card)
from .sources import Source, resolve_source
from .store import Atlas, Hit, IngestStats

__all__ = ["Atlas", "Hit", "IngestStats", "BlastReport", "Impact", "Coupling", "Source",
           "architecture_card", "blast_radius", "coupling", "render_card", "resolve_source"]
