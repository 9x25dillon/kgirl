from typing import List
from ..models.memory_event import MemoryEvent
from .vector_index import semantic_scores
from .graph_store import degree
from ..config import settings

def rank_memories(query: str, mems: List[MemoryEvent], k: int = 12, entropy: float = 0.0) -> List[MemoryEvent]:
    texts = [m.data for m in mems]
    sem = semantic_scores(query, texts)
    deg_map = degree([m.subject for m in mems])
    out = []
    for s, m in zip(sem, mems):
        rec = m.recency_score
        gd = deg_map.get(m.subject, 0.0)
        # entropy modulates the effective k later; include as delta term
        score = settings.alpha*s + settings.beta*rec + settings.gamma*gd + settings.delta*entropy
        out.append((score, m))
    out.sort(key=lambda x: x[0], reverse=True)
    k_eff = max(4, int(k * (0.75 if entropy > 0.5 else 1.0)))
    return [m for _, m in out[:k_eff]]