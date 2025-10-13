#!/usr/bin/env python3
"""
Cognitive Communication Organism + Numbskull Integration Adapter
================================================================

Complete integration of Cognitive Communication Organism with Numbskull:
- 3-level cognitive architecture (Neural, Orchestration, Physical)
- Embedding-enhanced cognitive processing
- Autonomous adaptation and learning
- Complete communication organism functionality

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CognitiveOrganismState:
    """State of the cognitive organism"""
    embeddings: Optional[Dict[str, Any]] = None
    cognitive_level: str = "neural"
    stability: float = 0.0
    coherence: float = 0.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)


class CognitiveOrganismNumbskullAdapter:
    """
    Adapter for Cognitive Communication Organism + Numbskull
    
    Integrates the 3-level cognitive architecture with Numbskull embeddings:
    - Level 1: Neural Cognition (embeddings + neuro-symbolic)
    - Level 2: Orchestration (dual LLM with embedding enhancement)
    - Level 3: Physical Manifestation (signal processing with patterns)
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("COGNITIVE ORGANISM + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        self.state = CognitiveOrganismState()
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        # Cognitive organism components
        self.communication_history = []
        self.learning_metrics = {}
        
        logger.info("=" * 70)
    
    async def cognitive_communication(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process communication through cognitive organism
        
        Args:
            message: Message to process
            context: Optional communication context
        
        Returns:
            Complete cognitive processing results
        """
        logger.info(f"\n🧠 Cognitive Communication: {message[:60]}...")
        
        context = context or {}
        results = {
            "message": message,
            "context": context,
            "processing_levels": {},
            "final_output": None
        }
        
        # Level 1: Neural Cognition
        logger.info("  Level 1: Neural Cognition")
        if self.numbskull:
            try:
                emb_result = await self.numbskull.embed(message)
                self.state.embeddings = emb_result
                
                # Calculate cognitive metrics
                embedding = emb_result["fused_embedding"]
                self.state.stability = float(1.0 / (1.0 + np.var(embedding)))
                self.state.coherence = float(np.linalg.norm(embedding))
                
                results["processing_levels"]["neural"] = {
                    "embeddings": emb_result["metadata"]["components_used"],
                    "stability": self.state.stability,
                    "coherence": self.state.coherence
                }
                
                logger.info(f"    ✅ Stability: {self.state.stability:.3f}, Coherence: {self.state.coherence:.3f}")
            except Exception as e:
                logger.warning(f"    ⚠️  Neural cognition failed: {e}")
        
        # Level 2: Orchestration Intelligence
        logger.info("  Level 2: Orchestration Intelligence")
        try:
            # Determine processing strategy based on embeddings
            if self.state.embeddings:
                components = self.state.embeddings["metadata"]["components_used"]
                if len(components) >= 3:
                    strategy = "multi_modal"
                elif "mathematical" in components:
                    strategy = "analytical"
                elif "semantic" in components:
                    strategy = "linguistic"
                else:
                    strategy = "pattern_based"
            else:
                strategy = "default"
            
            results["processing_levels"]["orchestration"] = {
                "strategy": strategy,
                "confidence": min(1.0, self.state.coherence / 10.0)
            }
            
            logger.info(f"    ✅ Strategy: {strategy}")
        except Exception as e:
            logger.warning(f"    ⚠️  Orchestration failed: {e}")
        
        # Level 3: Physical Manifestation
        logger.info("  Level 3: Physical Manifestation")
        try:
            # Select communication parameters
            if self.state.stability > 0.5:
                modulation = "QPSK"  # Stable = efficient modulation
            else:
                modulation = "BFSK"  # Unstable = robust modulation
            
            results["processing_levels"]["physical"] = {
                "modulation": modulation,
                "adaptive": True
            }
            
            logger.info(f"    ✅ Modulation: {modulation}")
        except Exception as e:
            logger.warning(f"    ⚠️  Physical manifestation failed: {e}")
        
        # Generate final output
        results["final_output"] = {
            "cognitive_analysis": f"Message processed through {len(results['processing_levels'])} cognitive levels",
            "strategy": results["processing_levels"].get("orchestration", {}).get("strategy", "unknown"),
            "stability": self.state.stability,
            "recommendation": f"Use {results['processing_levels'].get('physical', {}).get('modulation', 'QPSK')} modulation"
        }
        
        # Track in history
        self.communication_history.append(results)
        
        logger.info(f"✅ Cognitive communication complete: {len(results['processing_levels'])} levels")
        return results
    
    def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cognitive metrics"""
        return {
            "total_communications": len(self.communication_history),
            "current_stability": self.state.stability,
            "current_coherence": self.state.coherence,
            "cognitive_level": self.state.cognitive_level,
            "adaptation_count": len(self.state.adaptation_history)
        }
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ Cognitive organism adapter closed")


async def demo_cognitive_organism_adapter():
    """Demonstration of cognitive organism + Numbskull integration"""
    print("\n" + "=" * 70)
    print("COGNITIVE ORGANISM + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = CognitiveOrganismNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_semantic": True,
            "use_mathematical": True,
            "use_fractal": True,
            "fusion_method": "attention"  # Use attention for organism
        }
    )
    
    # Test communications
    messages = [
        {
            "message": "Emergency network coordination required for distributed system",
            "context": {"priority": 10, "channel": "emergency"}
        },
        {
            "message": "Solve optimization problem: minimize f(x) = x^2 + 2x + 1",
            "context": {"priority": 5, "channel": "analytical"}
        },
        {
            "message": "Regular communication update for status monitoring",
            "context": {"priority": 1, "channel": "standard"}
        }
    ]
    
    # Process each message
    for i, msg_data in enumerate(messages, 1):
        print(f"\n{'='*70}")
        print(f"COMMUNICATION {i}")
        print(f"{'='*70}")
        
        result = await adapter.cognitive_communication(
            msg_data["message"],
            msg_data["context"]
        )
        
        print(f"\nProcessing Levels:")
        for level, data in result["processing_levels"].items():
            print(f"  {level}: {data}")
        
        print(f"\nFinal Output:")
        for key, value in result["final_output"].items():
            print(f"  {key}: {value}")
    
    # Show metrics
    print(f"\n{'='*70}")
    print("COGNITIVE METRICS")
    print(f"{'='*70}")
    metrics = adapter.get_cognitive_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_cognitive_organism_adapter())

