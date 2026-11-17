"""helix: orchestration where code keeps the books and models write the blocks.

    state.py     STATE: the versioned control object (vector, last re-steer, ledger totals, open conflicts,
                 invariants hash); axis rubrics; validation that clamps and fails closed
    store.py     StateStore: every version hash-chained in $KGIRL_HOME/helix.db, with diffs; guarded ingest
    ledger.py    Ledger: budgets split once at L1, calls charged from the provider's token counts; the BYPASS
                 predicate
    protocol.py  the ASCII trailers (TUNE, DEFECT, CONCUR, CAL, UNRESOLVED) and their quote checks
    prompt.py    the cache-stable layout: static doctrine, task, append-only log, STATE tail
    harness.py   Helix.run(): bypass or experts -> critic on another model family -> synthesis
    evaluate.py  an offline audit of a HELIX-9 transcript a model wrote on its own (not needed for Helix.run)

See docs/harness/HELIX.md. `python -m kgirl.harness.helix run | state | history | diff | verify | ingest | floor | audit`.
"""

from .evaluate import evaluate, propose_resteer, reconcile
from .harness import Helix, family, route
from .ledger import Ledger, bypass, estimate
from .state import AXES, DEFAULT_VECTOR, RUBRICS, State, Verdict, default_state, invariants_hash, validate
from .store import HelixError, StateStore

__all__ = ["Helix", "HelixError", "StateStore", "State", "Verdict", "Ledger", "AXES", "DEFAULT_VECTOR", "RUBRICS",
           "bypass", "estimate", "validate", "default_state", "invariants_hash", "family", "route", "evaluate",
           "reconcile", "propose_resteer"]
