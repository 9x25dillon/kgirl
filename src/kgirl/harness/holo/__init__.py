"""holo: small overlapping matrix databases with provenance, serving real-time inference.

    store.py   HoloStore: entities (repo:…, kind:…, soup:…) over vessel.matrixdb.MatrixDB; centroid, spread
               and sampled chunk vectors derived from the KGVM matrix; overlap by content address; every
               record hash-chained on its entity's ledger
    infer.py   query(): KGVM coarse-to-fine search with owning entities and digests on every hit, the
               guidance block, the optional tauls HoloMemory prior, bus events and regulation
    build.py   build(home), status(home): the offline builder and its report

See docs/harness/HOLO.md. `python -m kgirl.harness.holo build | status | query`.
"""

from .build import build, status
from .infer import GateConfig, query
from .store import HoloError, HoloStore, Record

__all__ = ["HoloStore", "HoloError", "Record", "GateConfig", "query", "build", "status"]
