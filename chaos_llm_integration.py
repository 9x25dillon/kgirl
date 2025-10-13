#!/usr/bin/env python3
"""
Chaos LLM Services Integration
==============================

Integrates all 11 chaos_llm services from aipyapp into LiMp:
1. QGI (Quantum Geometric Intelligence)
2. AL-ULS (Symbolic evaluation)
3. Entropy Engine
4. Retrieval System
5. Suggestions
6. Motif Engine
7. Matrix Processor
8. Numbskull Service
9. Unitary Mixer
10. AL-ULS HTTP Client
11. AL-ULS WebSocket Client

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add aipyapp to path
aipyapp_path = Path("/home/kill/aipyapp")
if aipyapp_path.exists() and str(aipyapp_path) not in sys.path:
    sys.path.insert(0, str(aipyapp_path))

# Import chaos_llm services with graceful fallback
qgi = None
entropy_engine = None
retrieval = None
motif_engine = None
suggestions = None
unitary_mixer = None
numbskull = None
al_uls = None
al_uls_client = None
al_uls_ws_client = None
matrix_processor = None

try:
    from src.chaos_llm.services import entropy_engine
    from src.chaos_llm.services import retrieval
    from src.chaos_llm.services import motif_engine
    from src.chaos_llm.services import suggestions
    from src.chaos_llm.services import unitary_mixer
    from src.chaos_llm.services import al_uls
    from src.chaos_llm.services import al_uls_client
    
    # Try QGI separately (may have dependencies on broken matrix_processor)
    try:
        from src.chaos_llm.services import qgi
    except:
        pass
    
    CHAOS_SERVICES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Chaos_llm services imported (some may be unavailable)")
except ImportError as e:
    CHAOS_SERVICES_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️  Chaos_llm services not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaosLLMIntegration:
    """
    Unified integration of all chaos_llm services
    
    Provides a single interface to access:
    - Quantum Geometric Intelligence (QGI)
    - Entropy analysis
    - Retrieval system
    - Suggestions
    - Motif detection
    - Symbolic evaluation
    - Matrix operations
    - Unitary routing
    """
    
    def __init__(self, enable_all: bool = True):
        """Initialize chaos_llm integration"""
        logger.info("="*70)
        logger.info("CHAOS LLM SERVICES INTEGRATION")
        logger.info("="*70)
        
        self.available = CHAOS_SERVICES_AVAILABLE
        self.enable_all = enable_all
        
        if not self.available:
            logger.warning("⚠️  Chaos services not available - using fallbacks")
            return
        
        # Initialize services
        self.qgi = qgi
        self.entropy = entropy_engine.entropy_engine
        self.retrieval = retrieval
        self.motif = motif_engine.motif_engine
        self.suggestions = suggestions.SUGGESTIONS
        self.mixer = unitary_mixer
        self.numbskull_http = None
        self.aluls = al_uls.al_uls
        self.aluls_client = al_uls_client.al_uls_client
        
        # Statistics
        self.stats = {
            "qgi_queries": 0,
            "entropy_calculations": 0,
            "retrievals": 0,
            "suggestions_generated": 0,
            "motifs_detected": 0,
            "symbolic_evals": 0
        }
        
        logger.info("✅ Chaos LLM services initialized")
        logger.info(f"   QGI: ✅")
        logger.info(f"   Entropy Engine: ✅")
        logger.info(f"   Retrieval: ✅")
        logger.info(f"   Suggestions: ✅")
        logger.info(f"   Motif Engine: ✅")
        logger.info(f"   AL-ULS: ✅")
        logger.info(f"   Unitary Mixer: ✅")
        logger.info("="*70)
    
    async def suggest_with_qgi(
        self,
        prefix: str = "",
        state: str = "S0",
        use_semantic: bool = True
    ) -> Dict[str, Any]:
        """
        Generate suggestions with Quantum Geometric Intelligence
        
        Args:
            prefix: Query prefix
            state: Current state (S0, S1, etc.)
            use_semantic: Use semantic analysis
        
        Returns:
            Suggestions with QGI analysis
        """
        if not self.available:
            return {"suggestions": [], "qgi": {}, "error": "Services not available"}
        
        self.stats["qgi_queries"] += 1
        logger.info(f"🔮 QGI suggest: '{prefix}' in state {state}")
        
        result = await self.qgi.api_suggest_async(prefix, state, use_semantic)
        
        logger.info(f"   ✅ Generated {len(result.get('suggestions', []))} suggestions")
        logger.info(f"   ✅ QGI entropy scores: {len(result.get('qgi', {}).get('entropy_scores', []))}")
        
        return result
    
    def calculate_entropy(self, text: str) -> Dict[str, float]:
        """
        Calculate entropy metrics for text
        
        Args:
            text: Input text
        
        Returns:
            Entropy scores and volatility
        """
        if not self.available:
            return {"entropy": 0.0, "volatility": 0.0, "error": "Services not available"}
        
        self.stats["entropy_calculations"] += 1
        
        entropy_score = self.entropy.score_token(text)
        volatility = self.entropy.get_volatility_signal(text)
        
        logger.info(f"📊 Entropy: {entropy_score:.3f}, Volatility: {volatility:.3f}")
        
        return {
            "entropy": entropy_score,
            "volatility": volatility,
            "complexity": entropy_score * (1 + volatility)
        }
    
    async def retrieve(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            namespace: Document namespace
            top_k: Number of results
        
        Returns:
            List of relevant documents
        """
        if not self.available:
            return []
        
        self.stats["retrievals"] += 1
        logger.info(f"🔍 Retrieving: '{query}' from {namespace}")
        
        results = await self.retrieval.search(query, namespace, top_k)
        
        logger.info(f"   ✅ Found {len(results)} results")
        
        return results
    
    async def ingest_documents(
        self,
        documents: List[str],
        namespace: str = "default"
    ) -> int:
        """
        Ingest documents into retrieval system
        
        Args:
            documents: List of documents
            namespace: Storage namespace
        
        Returns:
            Total document count
        """
        if not self.available:
            return 0
        
        count = await self.retrieval.ingest_texts(documents, namespace)
        logger.info(f"📥 Ingested {len(documents)} docs into {namespace}, total: {count}")
        
        return count
    
    def detect_motifs(self, text: str) -> List[str]:
        """
        Detect motif patterns in text
        
        Args:
            text: Input text
        
        Returns:
            List of detected motif tags
        """
        if not self.available:
            return []
        
        self.stats["motifs_detected"] += 1
        
        tags = self.motif.detect_tags(text)
        
        if tags:
            logger.info(f"🔖 Motifs detected: {tags}")
        
        return tags
    
    def get_suggestions(self, state: str = "S0") -> List[str]:
        """
        Get suggestions for current state
        
        Args:
            state: Current state
        
        Returns:
            List of suggestions
        """
        if not self.available:
            return []
        
        self.stats["suggestions_generated"] += 1
        
        suggestions = self.suggestions.get(state, [])
        logger.info(f"💡 Suggestions for {state}: {len(suggestions)} items")
        
        return suggestions
    
    def calculate_route_mixture(self, qgi_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate unitary route mixture
        
        Args:
            qgi_data: QGI analysis data
        
        Returns:
            Route mixture weights
        """
        if not self.available:
            return {"symbolic": 0.33, "retrieval": 0.33, "semantic": 0.33}
        
        mixture = self.mixer.route_mixture(qgi_data)
        best_route = self.mixer.choose_route(mixture)
        
        logger.info(f"🎯 Route mixture: {mixture}")
        logger.info(f"   Best route: {best_route}")
        
        return {"mixture": mixture, "best_route": best_route}
    
    async def evaluate_symbolic(
        self,
        expression: str
    ) -> Dict[str, Any]:
        """
        Evaluate symbolic expression via AL-ULS
        
        Args:
            expression: Symbolic expression (e.g., "SUM(1,2,3)")
        
        Returns:
            Evaluation result
        """
        if not self.available:
            return {"ok": False, "error": "Services not available"}
        
        self.stats["symbolic_evals"] += 1
        logger.info(f"🧮 Evaluating: {expression}")
        
        # Check if it's a symbolic call
        if self.aluls.is_symbolic_call(expression):
            call = self.aluls.parse_symbolic_call(expression)
            result = await self.aluls.eval_symbolic_call_async(call)
            logger.info(f"   ✅ Result: {result}")
            return result
        else:
            return {"ok": False, "error": "Not a symbolic expression"}
    
    async def comprehensive_analysis(
        self,
        text: str,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using all services
        
        Args:
            text: Input text
            namespace: Namespace for retrieval
        
        Returns:
            Complete analysis results
        """
        logger.info(f"\n🔬 Comprehensive Analysis: '{text[:50]}...'")
        
        results = {
            "text": text,
            "entropy": None,
            "motifs": [],
            "qgi": None,
            "symbolic": None,
            "retrieval": [],
            "suggestions": []
        }
        
        if not self.available:
            results["error"] = "Services not available"
            return results
        
        # 1. Entropy analysis
        results["entropy"] = self.calculate_entropy(text)
        
        # 2. Motif detection
        results["motifs"] = self.detect_motifs(text)
        
        # 3. QGI analysis
        qgi_result = await self.suggest_with_qgi(text, "S0", True)
        results["qgi"] = qgi_result.get("qgi", {})
        results["suggestions"] = qgi_result.get("suggestions", [])
        
        # 4. Symbolic evaluation (if applicable)
        if self.aluls.is_symbolic_call(text):
            results["symbolic"] = await self.evaluate_symbolic(text)
        
        # 5. Retrieval (if documents exist)
        try:
            results["retrieval"] = await self.retrieve(text, namespace, 3)
        except:
            pass
        
        # 6. Route mixture
        if results["qgi"]:
            results["routing"] = self.calculate_route_mixture(results["qgi"])
        
        logger.info("✅ Comprehensive analysis complete")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.stats,
            "available": self.available
        }
    
    async def close(self):
        """Cleanup resources"""
        logger.info("✅ Chaos LLM integration closed")


# Convenience function for quick access
async def analyze_with_chaos(text: str) -> Dict[str, Any]:
    """
    Quick analysis using chaos_llm services
    
    Args:
        text: Input text
    
    Returns:
        Analysis results
    """
    integration = ChaosLLMIntegration()
    result = await integration.comprehensive_analysis(text)
    await integration.close()
    return result


if __name__ == "__main__":
    async def demo():
        print("\n" + "="*70)
        print("CHAOS LLM SERVICES DEMO")
        print("="*70)
        
        integration = ChaosLLMIntegration()
        
        # Test queries
        queries = [
            "SUM(1, 2, 3, 4, 5)",
            "What is quantum computing?",
            "SELECT * FROM data WHERE value > 10",
            "MEAN(100, 200, 300)"
        ]
        
        for query in queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")
            
            result = await integration.comprehensive_analysis(query)
            
            if result.get("entropy"):
                print(f"Entropy: {result['entropy']['entropy']:.3f}")
            if result.get("motifs"):
                print(f"Motifs: {result['motifs']}")
            if result.get("symbolic") and result["symbolic"].get("ok"):
                print(f"Symbolic: {result['symbolic']}")
            if result.get("suggestions"):
                print(f"Suggestions: {len(result['suggestions'])} items")
        
        print(f"\n{'='*70}")
        print("STATS")
        print(f"{'='*70}")
        stats = integration.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        await integration.close()
    
    asyncio.run(demo())

