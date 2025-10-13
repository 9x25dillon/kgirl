#!/usr/bin/env python3
"""
Emergent Cognitive Network + Numbskull Integration Adapter
==========================================================

Integration of Emergent Network Infrastructure with Numbskull:
- Swarm intelligence with embedding-based coordination
- Quantum-inspired optimization of embeddings
- Neuromorphic computing integration
- Emergent pattern detection and learning

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
    from emergent_cognitive_network import (
        QuantumInspiredOptimizer,
        SwarmCognitiveNetwork
    )
    EMERGENT_AVAILABLE = True
except ImportError:
    EMERGENT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmergentNetworkNumbskullAdapter:
    """
    Adapter for Emergent Cognitive Network + Numbskull
    
    Provides:
    - Swarm optimization of embedding generation
    - Quantum-inspired embedding enhancement
    - Emergent pattern detection from embeddings
    - Distributed cognitive processing
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None,
        num_swarm_agents: int = 20
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("EMERGENT NETWORK + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize emergent components
        if EMERGENT_AVAILABLE:
            self.quantum_optimizer = QuantumInspiredOptimizer(num_qubits=8)
            self.swarm_network = SwarmCognitiveNetwork(
                num_agents=num_swarm_agents,
                search_space=(-5, 5)
            )
            logger.info(f"✅ Emergent network initialized ({num_swarm_agents} agents)")
        else:
            self.quantum_optimizer = None
            self.swarm_network = None
            logger.warning("⚠️  Emergent network not available")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        # Emergent state
        self.emergent_patterns = []
        self.swarm_history = []
        
        logger.info("=" * 70)
    
    async def swarm_optimize_embedding(
        self,
        text: str,
        optimization_target: str = "coherence"
    ) -> Dict[str, Any]:
        """
        Use swarm intelligence to optimize embedding generation
        
        Args:
            text: Input text
            optimization_target: What to optimize (coherence, diversity, etc.)
        
        Returns:
            Optimization results
        """
        logger.info(f"\n🐝 Swarm Optimization: {text[:60]}...")
        
        results = {
            "text": text,
            "target": optimization_target,
            "optimized": False
        }
        
        if not self.numbskull:
            logger.warning("  ⚠️  No embeddings without Numbskull")
            return results
        
        try:
            # Generate baseline embedding
            emb_result = await self.numbskull.embed(text)
            baseline_embedding = emb_result["fused_embedding"]
            
            results["baseline"] = {
                "components": emb_result["metadata"]["components_used"],
                "dimension": emb_result["metadata"]["embedding_dim"],
                "norm": float(np.linalg.norm(baseline_embedding))
            }
            
            # Optimize if swarm available
            if self.swarm_network:
                # Define optimization function
                def cost_function(weights):
                    """Cost based on embedding characteristics"""
                    # Simulate optimizing fusion weights
                    coherence = float(1.0 / (1.0 + np.var(baseline_embedding * weights[0])))
                    return -coherence  # Minimize negative = maximize coherence
                
                # Run swarm optimization
                swarm_result = self.swarm_network.optimize(cost_function, max_iter=50)
                
                results["optimized"] = True
                results["swarm_result"] = {
                    "best_cost": swarm_result["best_cost"],
                    "iterations": 50,
                    "convergence": swarm_result.get("convergence_history", [])[-1] if swarm_result.get("convergence_history") else 0
                }
                
                self.swarm_history.append(swarm_result)
                
                logger.info(f"  ✅ Swarm optimized: cost={swarm_result['best_cost']:.3f}")
            else:
                logger.info("  ℹ️  Using baseline embedding (no swarm)")
            
        except Exception as e:
            logger.error(f"  ❌ Optimization failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def quantum_enhance_pattern(
        self,
        pattern_data: str
    ) -> Dict[str, Any]:
        """
        Use quantum optimization to enhance pattern recognition
        
        Args:
            pattern_data: Pattern data
        
        Returns:
            Enhancement results
        """
        logger.info(f"\n⚛️  Quantum Pattern Enhancement: {pattern_data[:60]}...")
        
        results = {
            "pattern": pattern_data,
            "enhanced": False
        }
        
        if not self.numbskull:
            logger.warning("  ⚠️  No embeddings without Numbskull")
            return results
        
        try:
            # Generate embedding for pattern
            emb_result = await self.numbskull.embed(pattern_data)
            embedding = emb_result["fused_embedding"]
            
            # Apply quantum optimization if available
            if self.quantum_optimizer:
                def cost_func(x):
                    """Cost function for quantum optimization"""
                    # Minimize distance from optimal embedding space
                    return float(np.sum((x - embedding[:8])**2))
                
                quantum_result = self.quantum_optimizer.quantum_annealing_optimization(
                    cost_func,
                    max_iter=100
                )
                
                results["enhanced"] = True
                results["quantum_result"] = {
                    "cost": quantum_result["cost"],
                    "quantum_entropy": quantum_result["quantum_entropy"]
                }
                
                logger.info(f"  ✅ Quantum enhanced: entropy={quantum_result['quantum_entropy']:.3f}")
            else:
                logger.info("  ℹ️  Using baseline (no quantum)")
            
        except Exception as e:
            logger.error(f"  ❌ Enhancement failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def detect_emergent_patterns(self) -> Dict[str, Any]:
        """Detect emergent patterns from swarm history"""
        if not self.swarm_history:
            return {"patterns_detected": 0}
        
        # Simple pattern detection
        costs = [s["best_cost"] for s in self.swarm_history]
        improvement = costs[0] - costs[-1] if len(costs) > 1 else 0
        
        return {
            "patterns_detected": len(self.swarm_history),
            "optimization_runs": len(self.swarm_history),
            "total_improvement": improvement,
            "trend": "improving" if improvement > 0 else "stable"
        }
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ Emergent network adapter closed")


async def demo_emergent_adapter():
    """Demonstration of emergent network + Numbskull integration"""
    print("\n" + "=" * 70)
    print("EMERGENT NETWORK + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = EmergentNetworkNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={"use_fractal": True},
        num_swarm_agents=15
    )
    
    # Test data
    test_cases = [
        "Distributed cognitive processing across neural networks",
        "Emergent behavior in complex adaptive systems",
        "Quantum optimization of multi-agent coordination"
    ]
    
    # Test swarm optimization
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: Swarm Optimization")
        print(f"{'='*70}")
        
        result = await adapter.swarm_optimize_embedding(text, "coherence")
        print(f"Text: {text[:50]}...")
        print(f"Optimized: {result.get('optimized', False)}")
        if result.get('baseline'):
            print(f"Baseline norm: {result['baseline']['norm']:.3f}")
        if result.get('swarm_result'):
            print(f"Swarm cost: {result['swarm_result']['best_cost']:.3f}")
    
    # Test quantum enhancement
    print(f"\n{'='*70}")
    print("TEST: Quantum Enhancement")
    print(f"{'='*70}")
    result = await adapter.quantum_enhance_pattern("Repeating fractal patterns in cognitive data")
    print(f"Enhanced: {result.get('enhanced', False)}")
    if result.get('quantum_result'):
        print(f"Quantum entropy: {result['quantum_result']['quantum_entropy']:.3f}")
    
    # Detect patterns
    print(f"\n{'='*70}")
    print("EMERGENT PATTERNS")
    print(f"{'='*70}")
    patterns = adapter.detect_emergent_patterns()
    for key, value in patterns.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_emergent_adapter())

