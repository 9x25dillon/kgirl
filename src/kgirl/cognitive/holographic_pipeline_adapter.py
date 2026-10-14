"""Holographic pipeline adapter: the tool interface `scripts/demo/demo_integrated_system.py` expected.

Rebuilt from its call sites (the LiMp original was never pushed). A thin, async tool layer over
`kgirl.quantum.holographic_memory_system.EnhancedCognitiveMemoryOrchestrator`:

    STORE_HOLOGRAPHIC   [data JSON list, metadata JSON object]  → ok, memory_key, emergence_detected,
                                                                   cognitive_integration_level, memory_resilience
    RECALL_ASSOCIATIVE  [query JSON list, threshold]            → ok, match_count, integrated_confidence, matches
    ENCODE_FRACTAL      [data JSON list]                        → ok, fractal_dimension, self_similarity,
                                                                   emergence_level, scales
    MEMORY_ANALYZE      []                                      → ok, num_memory_traces, cognitive_integration_level,
                                                                   memory_resilience, metacognition

Every reply has `ok`; a failure is {"ok": False, "error": ...}, never an exception. Inputs are JSON strings
(as an LLM tool call would send them) or already-decoded Python values.

Not to be confused with `kgirl.llm.pytorch_components_numbskull_adapter.HolographicNumbskullAdapter`, which
pairs the memory with numbskull embeddings through a different (method-call) API.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

TOOLS = ("STORE_HOLOGRAPHIC", "RECALL_ASSOCIATIVE", "ENCODE_FRACTAL", "MEMORY_ANALYZE")


def _decode(v: Any) -> Any:
    return json.loads(v) if isinstance(v, (str, bytes)) else v


class HolographicNumbskullAdapter:
    def __init__(self, orchestrator=None):
        if orchestrator is None:
            from kgirl.quantum.holographic_memory_system import EnhancedCognitiveMemoryOrchestrator
            orchestrator = EnhancedCognitiveMemoryOrchestrator()
        self.memory = orchestrator
        self.calls: Dict[str, int] = {t: 0 for t in TOOLS}

    @property
    def tools(self) -> List[str]:
        return list(TOOLS)

    async def invoke(self, tool: str, args: List[Any]) -> Dict[str, Any]:
        try:
            fn = {"STORE_HOLOGRAPHIC": self._store, "RECALL_ASSOCIATIVE": self._recall,
                  "ENCODE_FRACTAL": self._fractal, "MEMORY_ANALYZE": self._analyze}[tool]
        except KeyError:
            return {"ok": False, "error": f"unknown tool {tool!r}; one of {', '.join(TOOLS)}"}
        try:
            out = fn(list(args or []))
            self.calls[tool] += 1
            return {"ok": True, **out}
        except Exception as e:                               # a tool reply, not a crash
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _vector(arg: Any) -> np.ndarray:
        v = np.asarray(_decode(arg), dtype=np.float64).ravel()
        if v.size == 0 or not np.isfinite(v).all():
            raise ValueError("data must be a non-empty list of finite numbers")
        return v

    def _store(self, args: List[Any]) -> Dict[str, Any]:
        data = self._vector(args[0])
        meta = _decode(args[1]) if len(args) > 1 else {}
        r = self.memory.integrated_memory_processing({"data": data}, dict(meta or {}))
        return {"memory_key": r["memory_integration"]["holographic"], "emergence_detected": bool(r["emergence_detected"]),
                "cognitive_integration_level": float(r["cognitive_integration_level"]),
                "memory_resilience": float(r["memory_resilience"])}

    def _recall(self, args: List[Any]) -> Dict[str, Any]:
        query = self._vector(args[0])
        thr = float(_decode(args[1])) if len(args) > 1 else 0.5
        r = self.memory.emergent_memory_recall({"data": query, "similarity_threshold": thr})
        integ = r.get("integrated", {})
        matches = [{"memory_key": m["memory_key"], "similarity": float(m["similarity"])} for m in r.get("holographic", [])]
        return {"match_count": len(matches), "matches": matches,
                "integrated_confidence": float(integ.get("recall_confidence", 0.0)),
                "fractal_confidence": float(r.get("fractal", {}).get("fractal_completion_confidence", 0.0))}

    def _fractal(self, args: List[Any]) -> Dict[str, Any]:
        enc = self.memory.fractal_encoder.encode(self._vector(args[0]))
        return {"fractal_dimension": float(enc["fractal_dimension"]), "self_similarity": float(enc["self_similarity"]),
                "emergence_level": float(enc.get("emergence_level", 0.0)), "scales": len(enc["scales"])}

    def _analyze(self, args: List[Any]) -> Dict[str, Any]:
        m = self.memory
        meta = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in getattr(m, "memory_metacognition", {}).items()}
        return {"num_memory_traces": len(m.holographic_memory.memory_traces),
                "cognitive_integration_level": float(m.cognitive_integration_level),
                "memory_resilience": float(m.memory_resilience), "metacognition": meta,
                "tool_calls": dict(self.calls)}


async def demo_holographic_adapter() -> Dict[str, Any]:
    """Store three real lattice-physics enrichment maps, recall one from a noisy copy, analyse."""
    from kgirl import telemetry
    a = HolographicNumbskullAdapter()
    stored = []
    for off in (0, 300, 600):
        stored.append(await a.invoke("STORE_HOLOGRAPHIC", [json.dumps(telemetry.lattice_matrix(1, 256, offset=off).ravel().tolist()),
                                                           json.dumps({"offset": off})]))
    q = telemetry.lattice_matrix(1, 256, offset=300).ravel() + 0.05 * telemetry.chaos(0.77).normal(0, 1, 256)
    recall = await a.invoke("RECALL_ASSOCIATIVE", [json.dumps(q.tolist()), "0.6"])
    fractal = await a.invoke("ENCODE_FRACTAL", [json.dumps(telemetry.lattice_signal(256).tolist())])
    analysis = await a.invoke("MEMORY_ANALYZE", [])
    return {"source": telemetry.source(), "stored": stored, "recall": recall, "fractal": fractal, "analysis": analysis}
