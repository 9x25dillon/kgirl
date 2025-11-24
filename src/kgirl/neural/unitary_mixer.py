from __future__ import annotations
from typing import Dict

ROUTES = ("symbolic", "retrieval", "semantic")

def route_mixture(qgi: Dict) -> Dict[str, float]:
    # Heuristic mixture based on presence of features in qgi
    symbolic_weight = 0.6 if qgi.get("symbolic_calls") else 0.2
    retrieval_weight = 0.6 if qgi.get("retrieval_routes") else 0.2
    semantic_weight = 0.6 if qgi.get("entropy_scores") else 0.2
    total = symbolic_weight + retrieval_weight + semantic_weight
    return {
        "symbolic": symbolic_weight / total,
        "retrieval": retrieval_weight / total,
        "semantic": semantic_weight / total,
    }

def choose_route(mixture: Dict[str, float]) -> str:
    return max(mixture.items(), key=lambda kv: kv[1])[0]
