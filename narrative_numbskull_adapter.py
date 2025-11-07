#!/usr/bin/env python3
"""
Narrative Agent + Numbskull Integration Adapter
===============================================

Integration of Narrative Intelligence with Numbskull embeddings:
- Embedding-guided narrative generation
- Emotional arc analysis with embeddings
- Thematic coherence tracking
- Multi-modal narrative understanding

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NarrativeNumbskullAdapter:
    """
    Adapter for Narrative Agent + Numbskull
    
    Provides embedding-enhanced narrative processing:
    - Emotional arc tracking with embeddings
    - Thematic coherence measurement
    - Narrative structure analysis
    - Multi-modal storytelling
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("NARRATIVE AGENT + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        # Narrative tracking
        self.narrative_history = []
        self.emotional_trajectory = []
        self.thematic_embeddings = []
        
        logger.info("=" * 70)
    
    async def analyze_narrative_with_embeddings(
        self,
        narrative_text: str
    ) -> Dict[str, Any]:
        """
        Analyze narrative structure using embeddings
        
        Args:
            narrative_text: Narrative text to analyze
        
        Returns:
            Comprehensive narrative analysis
        """
        logger.info(f"\n📖 Narrative Analysis: {narrative_text[:60]}...")
        
        results = {
            "text": narrative_text,
            "embeddings": None,
            "emotional_valence": 0.0,
            "thematic_coherence": 0.0,
            "narrative_structure": None
        }
        
        # Generate embeddings
        if self.numbskull:
            try:
                emb_result = await self.numbskull.embed(narrative_text)
                embedding = emb_result["fused_embedding"]
                
                # Analyze emotional content from embeddings
                # Positive dimensions vs negative dimensions
                positive_energy = float(np.sum(np.maximum(embedding, 0)))
                negative_energy = float(np.sum(np.abs(np.minimum(embedding, 0))))
                total_energy = positive_energy + negative_energy
                
                if total_energy > 0:
                    emotional_valence = (positive_energy - negative_energy) / total_energy
                else:
                    emotional_valence = 0.0
                
                results["emotional_valence"] = emotional_valence
                self.emotional_trajectory.append(emotional_valence)
                
                # Calculate thematic coherence
                # High coherence = low variance in embedding
                coherence = float(1.0 / (1.0 + np.var(embedding)))
                results["thematic_coherence"] = coherence
                
                # Store thematic embedding
                self.thematic_embeddings.append(embedding)
                
                results["embeddings"] = {
                    "components": emb_result["metadata"]["components_used"],
                    "dimension": emb_result["metadata"]["embedding_dim"]
                }
                
                logger.info(f"  ✅ Emotional: {emotional_valence:.3f}, Coherence: {coherence:.3f}")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Embedding analysis failed: {e}")
        
        # Analyze narrative structure
        sentences = narrative_text.split('. ')
        results["narrative_structure"] = {
            "sentence_count": len(sentences),
            "avg_sentence_length": len(narrative_text) / max(1, len(sentences)),
            "complexity": "high" if len(sentences) > 5 else "medium" if len(sentences) > 2 else "simple"
        }
        
        # Track in history
        self.narrative_history.append(results)
        
        logger.info(f"✅ Narrative analysis complete")
        return results
    
    async def generate_narrative_arc(
        self,
        theme: str,
        target_emotional_arc: str = "rise"
    ) -> Dict[str, Any]:
        """
        Generate narrative arc guided by embeddings
        
        Args:
            theme: Narrative theme
            target_emotional_arc: Target arc (rise, fall, rise_fall, fall_rise)
        
        Returns:
            Generated narrative arc
        """
        logger.info(f"\n✍️  Generating Narrative Arc: {theme}")
        
        results = {
            "theme": theme,
            "arc_type": target_emotional_arc,
            "story_beats": []
        }
        
        # Generate theme embedding
        if self.numbskull:
            try:
                theme_emb = await self.numbskull.embed(theme)
                theme_vector = theme_emb["fused_embedding"]
                
                # Generate story beats based on arc type
                num_beats = 5
                for i in range(num_beats):
                    position = i / (num_beats - 1)
                    
                    # Calculate target emotional value
                    if target_emotional_arc == "rise":
                        target_emotion = -0.5 + 1.5 * position
                    elif target_emotional_arc == "fall":
                        target_emotion = 0.5 - 1.5 * position
                    elif target_emotional_arc == "rise_fall":
                        target_emotion = np.sin(position * np.pi)
                    else:  # fall_rise
                        target_emotion = -np.sin(position * np.pi)
                    
                    beat = {
                        "position": position,
                        "target_emotion": target_emotion,
                        "intensity": abs(target_emotion),
                        "description": f"Beat {i+1}: Emotional level {target_emotion:.2f}"
                    }
                    results["story_beats"].append(beat)
                
                logger.info(f"  ✅ Generated {num_beats} story beats")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Arc generation failed: {e}")
        
        return results
    
    def get_narrative_metrics(self) -> Dict[str, Any]:
        """Get narrative processing metrics"""
        if not self.emotional_trajectory:
            return {"narratives_processed": 0}
        
        return {
            "narratives_processed": len(self.narrative_history),
            "avg_emotional_valence": np.mean(self.emotional_trajectory),
            "emotional_range": max(self.emotional_trajectory) - min(self.emotional_trajectory),
            "thematic_embeddings_stored": len(self.thematic_embeddings)
        }
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ Narrative adapter closed")


async def demo_narrative_adapter():
    """Demonstration of narrative + Numbskull integration"""
    print("\n" + "=" * 70)
    print("NARRATIVE AGENT + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = NarrativeNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_semantic": True,
            "use_fractal": True,
            "fusion_method": "weighted_average"
        }
    )
    
    # Test narratives
    narratives = [
        "Once upon a time, there was a brilliant scientist who discovered quantum entanglement. She revolutionized communication technology. The world changed forever.",
        "The algorithm failed repeatedly. Debug attempts proved futile. Finally, a breakthrough emerged. Success at last.",
        "In the beginning, chaos reigned. Order slowly emerged. Patterns became clear. Understanding dawned."
    ]
    
    # Analyze each narrative
    for i, narrative in enumerate(narratives, 1):
        print(f"\n{'='*70}")
        print(f"NARRATIVE {i}")
        print(f"{'='*70}")
        print(f"Text: {narrative[:60]}...")
        
        result = await adapter.analyze_narrative_with_embeddings(narrative)
        
        print(f"\nAnalysis:")
        print(f"  Emotional Valence: {result['emotional_valence']:.3f}")
        print(f"  Thematic Coherence: {result['thematic_coherence']:.3f}")
        print(f"  Complexity: {result['narrative_structure']['complexity']}")
        print(f"  Sentences: {result['narrative_structure']['sentence_count']}")
    
    # Generate arc
    print(f"\n{'='*70}")
    print("NARRATIVE ARC GENERATION")
    print(f"{'='*70}")
    arc = await adapter.generate_narrative_arc("Hero's journey through quantum realms", "rise_fall")
    print(f"Theme: {arc['theme']}")
    print(f"Arc Type: {arc['arc_type']}")
    print(f"Story Beats: {len(arc['story_beats'])}")
    for beat in arc['story_beats'][:3]:
        print(f"  - {beat['description']}")
    
    # Show metrics
    print(f"\n{'='*70}")
    print("NARRATIVE METRICS")
    print(f"{'='*70}")
    metrics = adapter.get_narrative_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_narrative_adapter())

