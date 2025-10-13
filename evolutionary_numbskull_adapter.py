#!/usr/bin/env python3
"""
Evolutionary Communicator + Numbskull Integration Adapter
=========================================================

Deep integration between Evolutionary Communicator and Numbskull:
- Embedding-driven evolution strategies
- Adaptive communication with embedding feedback
- Multi-modal signal generation
- Evolutionary optimization of embeddings

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

try:
    from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
    NUMBSKULL_AVAILABLE = True
except ImportError:
    NUMBSKULL_AVAILABLE = False

try:
    import signal_processing as dsp
    # Don't import EvolutionaryCommunicator directly due to dataclass issue
    # We'll work with signal processing directly
    EVOL_COMM_AVAILABLE = True
except ImportError:
    EVOL_COMM_AVAILABLE = False
    dsp = None
    logger = logging.getLogger(__name__)
    logger.warning("Signal processing not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionaryNumbskullAdapter:
    """
    Adapter integrating Evolutionary Communicator with Numbskull
    
    Provides:
    - Embedding-guided evolution
    - Adaptive strategy selection based on embeddings
    - Multi-modal communication optimization
    - Feedback-driven embedding improvement
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("EVOLUTIONARY + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Check signal processing availability
        self.evol_available = EVOL_COMM_AVAILABLE
        if self.evol_available:
            self.modulators = dsp.Modulators()
            logger.info("✅ Signal processing available for evolution")
        else:
            self.modulators = None
            logger.warning("⚠️  Signal processing not available")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        # Evolution metrics
        self.generation_count = 0
        self.fitness_history = []
        
        logger.info("=" * 70)
    
    async def evolve_with_embeddings(
        self,
        message: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evolutionary processing with embedding enhancement
        
        Args:
            message: Message to process
            context: Optional context
        
        Returns:
            Evolution results
        """
        logger.info(f"\n🧬 Evolutionary Processing: {message[:60]}...")
        
        results = {
            "message": message,
            "generation": self.generation_count,
            "embedding_analysis": None,
            "evolution_strategy": None,
            "fitness": 0.0
        }
        
        # Analyze with embeddings
        if self.numbskull:
            try:
                emb_result = await self.numbskull.embed(message)
                embedding = emb_result["fused_embedding"]
                
                # Calculate fitness based on embedding characteristics
                fitness = self._calculate_fitness(embedding, emb_result["metadata"])
                
                results["embedding_analysis"] = {
                    "components": emb_result["metadata"]["components_used"],
                    "dimension": emb_result["metadata"]["embedding_dim"],
                    "fitness": fitness
                }
                results["fitness"] = fitness
                
                # Select evolution strategy based on fitness
                if fitness > 0.8:
                    strategy = "exploit"  # High fitness, exploit current approach
                elif fitness > 0.5:
                    strategy = "balanced"  # Medium fitness, balance exploration/exploitation
                else:
                    strategy = "explore"  # Low fitness, explore new approaches
                
                results["evolution_strategy"] = strategy
                
                logger.info(f"  ✅ Fitness: {fitness:.3f}, Strategy: {strategy}")
                
                # Track evolution
                self.fitness_history.append(fitness)
                self.generation_count += 1
                
            except Exception as e:
                logger.warning(f"  ⚠️  Embedding analysis failed: {e}")
        
        # Use signal processing for evolution if available
        if self.modulators and self.evol_available:
            try:
                # Select modulation based on fitness
                if results["fitness"] > 0.7:
                    modulation = dsp.ModulationScheme.QAM16  # High efficiency for fit individuals
                elif results["fitness"] > 0.4:
                    modulation = dsp.ModulationScheme.QPSK  # Balanced
                else:
                    modulation = dsp.ModulationScheme.BFSK  # Robust for low fitness
                
                results["selected_modulation"] = modulation.name
                logger.info(f"  ✅ Modulation: {modulation.name} (fitness-based)")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Modulation selection failed: {e}")
        
        return results
    
    def _calculate_fitness(
        self,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate fitness score from embedding
        
        Args:
            embedding: Embedding vector
            metadata: Embedding metadata
        
        Returns:
            Fitness score (0-1)
        """
        # Fitness factors
        norm = float(np.linalg.norm(embedding))
        variance = float(np.var(embedding))
        num_components = len(metadata.get("components_used", []))
        
        # Calculate fitness
        # Higher norm = more information
        # Higher variance = more diverse features
        # More components = richer representation
        fitness = (
            0.3 * min(1.0, norm / 10.0) +
            0.3 * min(1.0, variance) +
            0.4 * (num_components / 3.0)
        )
        
        return min(1.0, fitness)
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        if not self.fitness_history:
            return {"generations": 0}
        
        return {
            "generations": self.generation_count,
            "avg_fitness": np.mean(self.fitness_history),
            "best_fitness": max(self.fitness_history),
            "worst_fitness": min(self.fitness_history),
            "fitness_trend": "improving" if len(self.fitness_history) > 1 and 
                           self.fitness_history[-1] > self.fitness_history[0] else "stable"
        }
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ Evolutionary adapter closed")


async def demo_evolutionary_adapter():
    """Demonstration of Evolutionary + Numbskull integration"""
    print("\n" + "=" * 70)
    print("EVOLUTIONARY + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = EvolutionaryNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_semantic": True,
            "use_mathematical": True,
            "use_fractal": True,
            "fusion_method": "attention"  # Use attention for evolution
        }
    )
    
    # Simulate evolution over generations
    messages = [
        "Simple message generation 1",
        "More complex message with additional context generation 2",
        "Advanced multi-modal message with rich semantic content generation 3",
        "Optimized message based on learned patterns generation 4"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n{'='*70}")
        print(f"GENERATION {i}")
        print(f"{'='*70}")
        
        result = await adapter.evolve_with_embeddings(message)
        print(f"Message: {message[:60]}...")
        print(f"Fitness: {result['fitness']:.3f}")
        print(f"Strategy: {result.get('evolution_strategy', 'N/A')}")
        print(f"Components: {result.get('embedding_analysis', {}).get('components', 'N/A')}")
    
    # Show evolution stats
    print(f"\n{'='*70}")
    print("EVOLUTION STATISTICS")
    print(f"{'='*70}")
    stats = adapter.get_evolution_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_evolutionary_adapter())

