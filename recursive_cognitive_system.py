#!/usr/bin/env python3
"""
Recursive Cognitive Knowledge System (demo)

This script is adapted to run in the workspace with minimal external
dependencies by using `stub_modules` and `holographic_memory_system`.

Author: Assistant
License: MIT
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Local stubs
from stub_modules import HybridEmbeddingPipeline, HybridConfig, EnhancedVectorIndex, EnhancedGraphStore
from holographic_memory_system import HolographicMemorySystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    recursion_depth: int = 0
    total_insights: int = 0
    knowledge_nodes: int = 0
    pattern_reinforcements: int = 0
    hallucination_coherence: float = 0.0
    emergent_patterns: List[str] = field(default_factory=list)
    cognitive_loops: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecursiveInsight:
    content: str
    embedding: List[float]
    source_query: str
    recursion_level: int
    related_insights: List[str] = field(default_factory=list)
    reinforcement_count: int = 0
    coherence_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


class RecursiveCognitiveKnowledge:
    def __init__(self, max_recursion_depth: int = 5, hallucination_temperature: float = 0.8, coherence_threshold: float = 0.6):
        logger.info("RECURSIVE COGNITIVE KNOWLEDGE SYSTEM (demo)")
        self.max_recursion = max_recursion_depth
        self.hallucination_temp = hallucination_temperature
        self.coherence_threshold = coherence_threshold

        self.embeddings = None
        self.vector_index = None
        self.knowledge_graph = None
        self.holographic = None

        self.state = CognitiveState()
        self.insights: List[RecursiveInsight] = []
        self.interaction_history: deque = deque(maxlen=1000)
        self.emergent_patterns: Dict[str, int] = {}
        self.syntax_patterns: Dict[str, List[str]] = {}

    async def initialize(self):
        config = HybridConfig(use_fractal=True, use_semantic=True, use_mathematical=True, cache_embeddings=True)
        self.embeddings = HybridEmbeddingPipeline(config)
        self.vector_index = EnhancedVectorIndex(use_numbskull=True)
        self.knowledge_graph = EnhancedGraphStore(use_numbskull=True)
        self.holographic = HolographicMemorySystem(hologram_dim=32)
        logger.info("Subsystems initialized")

    async def recursive_analyze(self, content: str, current_depth: int = 0, source_query: str = None) -> Dict[str, Any]:
        if current_depth >= self.max_recursion:
            return {"stopped": "max_depth", "depth": current_depth}

        logger.info(f"{'  '*current_depth}Recursive analyze depth={current_depth}: {content[:60]}")

        analysis = {"content": content, "depth": current_depth, "embeddings": None, "similar_insights": [], "generated_insights": [], "emergent_patterns": [], "reinforcements": 0}

        emb_result = await self.embeddings.embed(content)
        embedding_vector = emb_result.get("embedding", [])
        analysis["embeddings"] = {"vector": embedding_vector, "components": emb_result.get("metadata", {}).get("components_used", []), "dimension": emb_result.get("metadata", {}).get("embedding_dim", len(embedding_vector))}

        similar = await self.vector_index.search(content, top_k=3)
        analysis["similar_insights"] = [{"id": entry.id, "text": entry.text, "similarity": score, "metadata": entry.metadata} for entry, score in similar]

        if current_depth < self.max_recursion - 1:
            variations = self._hallucinate_variations(content, analysis["similar_insights"])[:3]
            analysis["generated_insights"] = variations
            for variation in variations[:2]:
                if variation.get("coherence", 0) >= self.coherence_threshold:
                    sub = await self.recursive_analyze(variation["text"], current_depth+1, source_query or content)
                    variation["sub_analysis"] = sub
                    await self._store_insight(variation["text"], embedding_vector, source_query or content, current_depth+1)

        patterns = self._detect_emergent_patterns(content, analysis)
        analysis["emergent_patterns"] = patterns

        if self.holographic and analysis["similar_insights"]:
            reinf = self._holographic_reinforcement(content, analysis)
            analysis["reinforcements"] = reinf

        self.state.recursion_depth = max(self.state.recursion_depth, current_depth)
        self.state.total_insights += 1
        self.state.cognitive_loops.append({"depth": current_depth, "patterns": len(patterns), "timestamp": time.time()})

        return analysis

    def _hallucinate_variations(self, content: str, similar_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        variations = []
        words = [w.strip('.,') for w in content.split()]
        key_concepts = [w for w in words if len(w) > 4][:5]
        if len(key_concepts) >= 2:
            var1 = f"{key_concepts[0]} enables {key_concepts[1]}"
            variations.append({"text": var1, "type": "combination", "coherence": 0.7 + (len(similar_insights)*0.05)})
            if similar_insights:
                pattern = self._extract_pattern(similar_insights)
                var2 = f"{pattern} emerges with {key_concepts[0]}"
                variations.append({"text": var2, "type": "pattern", "coherence": 0.65 + (len(similar_insights)*0.04)})
            if len(key_concepts) >= 2:
                var3 = f"{key_concepts[1]} requires {key_concepts[0]}"
                variations.append({"text": var3, "type": "inverse", "coherence": 0.6})
        return variations

    def _extract_pattern(self, insights: List[Dict[str, Any]]) -> str:
        words = []
        for s in insights:
            words.extend(s.get("text", "").split())
        freq = {}
        for w in words:
            if len(w) > 4:
                freq[w] = freq.get(w, 0) + 1
        if freq:
            return f"Recursive {max(freq, key=freq.get)} pattern"
        return "Emergent pattern"

    def _detect_emergent_patterns(self, content: str, analysis: Dict[str, Any]) -> List[str]:
        patterns = []
        words = content.lower().split()
        for w in set(words):
            if words.count(w) > 1:
                self.emergent_patterns[w] = self.emergent_patterns.get(w, 0) + 1
                if self.emergent_patterns[w] >= 3:
                    patterns.append(f"reinforced:{w}")
        if len(analysis.get("similar_insights", [])) >= 2:
            patterns.append("archetype_formation")
        if analysis.get("depth", 0) >= 2:
            patterns.append("deep_emergence")
        return patterns

    def _holographic_reinforcement(self, content: str, analysis: Dict[str, Any]) -> int:
        reinf = 0
        for insight in analysis.get("similar_insights", []):
            if insight.get("similarity", 0) > 0.5:
                try:
                    self.holographic.store(insight["id"], insight["text"], metadata=insight.get("metadata", {}))
                    reinf += 1
                except Exception:
                    pass
        self.state.pattern_reinforcements += reinf
        return reinf

    async def _store_insight(self, content: str, embedding: List[float], source: str, depth: int):
        insight = RecursiveInsight(content=content, embedding=embedding, source_query=source or "", recursion_level=depth, coherence_score=0.7)
        self.insights.append(insight)
        await self.vector_index.add_entry(f"insight_{len(self.insights)}", content, {"depth": depth, "source": source})
        await self.knowledge_graph.add_node(f"insight_{len(self.insights)}", "recursive_insight", {"text": content, "depth": depth, "source": source})
        self.state.knowledge_nodes += 1
        self.state.total_insights += 1

    async def process_with_recursion(self, query: str) -> Dict[str, Any]:
        start = time.time()
        self.interaction_history.append({"type": "input", "content": query, "timestamp": time.time()})
        analysis = await self.recursive_analyze(query, current_depth=0, source_query=query)
        self.interaction_history.append({"type": "output", "content": analysis, "timestamp": time.time()})
        synthesis = self._synthesize_insights(analysis)
        syntax = self._learn_syntax_patterns(analysis)
        self.state.hallucination_coherence = self._calculate_coherence()
        processing_time = time.time() - start
        result = {"query": query, "analysis": analysis, "synthesis": synthesis, "syntax_learned": syntax, "cognitive_state": {"recursion_depth": self.state.recursion_depth, "total_insights": self.state.total_insights, "knowledge_nodes": self.state.knowledge_nodes, "hallucination_coherence": self.state.hallucination_coherence, "emergent_patterns": len(self.emergent_patterns)}, "processing_time": processing_time}
        return result

    def _synthesize_insights(self, analysis: Dict[str, Any]) -> str:
        collected = []
        def walk(node, depth=0):
            if isinstance(node, dict):
                for gi in node.get("generated_insights", []):
                    collected.append((gi.get("text", ""), depth))
                    if "sub_analysis" in gi:
                        walk(gi["sub_analysis"], depth+1)
        walk(analysis)
        if collected:
            deepest = max(collected, key=lambda x: x[1])
            return f"Emergent synthesis: {deepest[0]} (depth {deepest[1]})"
        return "Initial cognitive state"

    def _learn_syntax_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        learned = []
        for p in analysis.get("emergent_patterns", []):
            pt = p.split(":")[0]
            if pt not in self.syntax_patterns:
                self.syntax_patterns[pt] = []
                learned.append(f"new_syntax:{pt}")
            self.syntax_patterns[pt].append(p)
        if analysis.get("depth", 0) > 0:
            st = f"depth_{analysis['depth']}_structure"
            if st not in self.syntax_patterns:
                self.syntax_patterns[st] = []
                learned.append(f"new_structure:{st}")
        return learned

    def _calculate_coherence(self) -> float:
        if not self.insights:
            return 0.0
        total_reinf = sum(i.reinforcement_count for i in self.insights)
        avg = total_reinf / max(1, len(self.insights))
        return min(1.0, avg / 10.0)

    def get_cognitive_map(self) -> Dict[str, Any]:
        return {"cognitive_state": {"recursion_depth": self.state.recursion_depth, "total_insights": self.state.total_insights, "knowledge_nodes": self.state.knowledge_nodes, "pattern_reinforcements": self.state.pattern_reinforcements, "hallucination_coherence": self.state.hallucination_coherence, "emergent_patterns": len(self.emergent_patterns), "cognitive_loops": len(self.state.cognitive_loops)}, "knowledge_systems": {"vector_index": self.vector_index.get_stats(), "knowledge_graph": self.knowledge_graph.get_stats(), "holographic_available": self.holographic is not None}, "syntax_patterns": {k: len(v) for k, v in self.syntax_patterns.items()}, "interaction_history": len(self.interaction_history), "insights": [{"content": i.content[:50], "depth": i.recursion_level, "reinforcements": i.reinforcement_count} for i in self.insights[:10]]}

    async def close(self):
        if self.embeddings:
            await self.embeddings.close()
        if self.vector_index:
            await self.vector_index.close()
        if self.knowledge_graph:
            await self.knowledge_graph.close()


async def demo_recursive_cognition():
    system = RecursiveCognitiveKnowledge(max_recursion_depth=3, hallucination_temperature=0.8, coherence_threshold=0.6)
    await system.initialize()
    queries = [
        "Quantum computing uses superposition and entanglement",
        "Neural networks learn patterns from data",
        "Cognitive systems emerge from recursive processing"
    ]
    for q in queries:
        res = await system.process_with_recursion(q)
        print(json.dumps({"query": q, "cognitive_state": res["cognitive_state"], "synthesis": res["synthesis"]}, indent=2))

    print(json.dumps(system.get_cognitive_map(), indent=2))
    await system.close()


if __name__ == "__main__":
    asyncio.run(demo_recursive_cognition())
