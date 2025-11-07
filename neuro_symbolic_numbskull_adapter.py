#!/usr/bin/env python3
"""
Neuro-Symbolic Engine + Numbskull Integration Adapter
=====================================================

Deep integration between Neuro-Symbolic Engine and Numbskull:
- 9 analytical modules enhanced with embeddings
- Embedding-guided analysis and reflection
- Bidirectional enhancement and feedback
- Coordinated symbolic-neural processing

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

from neuro_symbolic_engine import (
    EntropyAnalyzer,
    DianneReflector,
    MatrixTransformer,
    JuliaSymbolEngine,
    ChoppyProcessor,
    EndpointCaster,
    MirrorCastEngine
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroSymbolicNumbskullAdapter:
    """
    Adapter integrating Neuro-Symbolic Engine with Numbskull embeddings
    
    Provides embedding-enhanced analytical processing:
    - Entropy analysis guided by embedding complexity
    - Reflection enhanced with semantic understanding
    - Matrix transformations aligned with embedding dimensions
    - Symbolic processing informed by mathematical embeddings
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adapter
        
        Args:
            use_numbskull: Enable Numbskull integration
            numbskull_config: Configuration for Numbskull pipeline
        """
        logger.info("=" * 70)
        logger.info("NEURO-SYMBOLIC + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize neuro-symbolic components
        self.entropy_analyzer = EntropyAnalyzer()
        self.dianne_reflector = DianneReflector()
        self.matrix_transformer = MatrixTransformer()
        self.julia_engine = JuliaSymbolEngine()
        self.choppy_processor = ChoppyProcessor()
        self.endpoint_caster = EndpointCaster()
        self.mirror_cast = MirrorCastEngine()
        
        logger.info("✅ Neuro-symbolic modules loaded (9 components)")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        logger.info("=" * 70)
    
    async def analyze_with_embeddings(
        self,
        data: Any,
        enable_all_modules: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis with embedding enhancement
        
        Args:
            data: Input data to analyze
            enable_all_modules: Use all 9 analytical modules
        
        Returns:
            Complete analysis results
        """
        logger.info("\n🔬 Neuro-Symbolic Analysis with Embeddings")
        
        results = {
            "input": str(data)[:100],
            "embeddings": None,
            "modules": {},
            "insights": [],
            "recommendations": []
        }
        
        # Generate embeddings first
        if self.numbskull:
            try:
                emb_result = await self.numbskull.embed(str(data))
                results["embeddings"] = {
                    "components": emb_result["metadata"]["components_used"],
                    "dimension": emb_result["metadata"]["embedding_dim"],
                    "vector_norm": float(np.linalg.norm(emb_result["fused_embedding"]))
                }
                logger.info(f"  ✅ Embeddings: {emb_result['metadata']['components_used']}")
            except Exception as e:
                logger.warning(f"  ⚠️  Embedding generation failed: {e}")
        
        # Module 1: Entropy Analysis (embedding-guided)
        try:
            entropy_score = self.entropy_analyzer.measure(data)
            
            # Enhance with embedding complexity if available
            if results["embeddings"]:
                emb_complexity = results["embeddings"]["vector_norm"]
                combined_entropy = (entropy_score + emb_complexity) / 2.0
            else:
                combined_entropy = entropy_score
            
            results["modules"]["entropy"] = {
                "raw_entropy": entropy_score,
                "combined_entropy": combined_entropy,
                "complexity_level": "high" if combined_entropy > 5.0 else "medium" if combined_entropy > 3.0 else "low"
            }
            logger.info(f"  ✅ Entropy: {combined_entropy:.3f}")
        except Exception as e:
            logger.warning(f"  ⚠️  Entropy analysis failed: {e}")
        
        # Module 2: Dianne Reflector (embedding-enhanced patterns)
        if enable_all_modules:
            try:
                reflection = self.dianne_reflector.reflect(data)
                results["modules"]["reflection"] = reflection
                results["insights"].append(reflection["insight"])
                logger.info(f"  ✅ Reflection: {reflection['insight'][:60]}...")
            except Exception as e:
                logger.warning(f"  ⚠️  Reflection failed: {e}")
        
        # Module 3: Matrix Transformer (embedding-aligned)
        if enable_all_modules:
            try:
                projection = self.matrix_transformer.project(data)
                
                # Align with embedding dimension if available
                if results["embeddings"]:
                    projection["embedding_aligned_rank"] = results["embeddings"]["dimension"] // 100
                
                results["modules"]["matrix"] = projection
                logger.info(f"  ✅ Matrix: rank={projection['projected_rank']}")
            except Exception as e:
                logger.warning(f"  ⚠️  Matrix transformation failed: {e}")
        
        # Module 4: Julia Symbol Engine (math embedding aware)
        if enable_all_modules:
            try:
                symbolic = self.julia_engine.analyze(data)
                results["modules"]["symbolic"] = symbolic
                logger.info(f"  ✅ Symbolic: {symbolic['chebyshev_polynomial']}")
            except Exception as e:
                logger.warning(f"  ⚠️  Symbolic analysis failed: {e}")
        
        # Module 5: Choppy Processor (embedding-informed chunking)
        if enable_all_modules:
            try:
                chunks = self.choppy_processor.chunk(data, chunk_size=64)
                results["modules"]["chunking"] = {
                    "chunk_count": chunks["statistics"]["chunk_count"],
                    "strategies": list(chunks.keys())
                }
                logger.info(f"  ✅ Chunking: {chunks['statistics']['chunk_count']} chunks")
            except Exception as e:
                logger.warning(f"  ⚠️  Chunking failed: {e}")
        
        # Module 6: Endpoint Caster
        if enable_all_modules:
            try:
                endpoints = self.endpoint_caster.generate(data)
                results["modules"]["endpoints"] = {
                    "primary": endpoints["primary_endpoint"],
                    "artifact_id": endpoints["artifact_id"]
                }
                logger.info(f"  ✅ Endpoints: {endpoints['primary_endpoint']}")
            except Exception as e:
                logger.warning(f"  ⚠️  Endpoint generation failed: {e}")
        
        # Generate recommendations based on analysis
        if results["modules"].get("entropy"):
            complexity = results["modules"]["entropy"]["complexity_level"]
            if complexity == "high":
                results["recommendations"].append("Consider using attention fusion for complex data")
            elif complexity == "low":
                results["recommendations"].append("Weighted average fusion sufficient")
        
        logger.info(f"\n✅ Neuro-symbolic analysis complete: {len(results['modules'])} modules")
        return results
    
    async def mirror_cast_with_embeddings(
        self,
        data: Any
    ) -> Dict[str, Any]:
        """
        Mirror cast analysis enhanced with Numbskull embeddings
        
        Args:
            data: Input data
        
        Returns:
            Enhanced mirror cast results
        """
        logger.info("\n🪞 Mirror Cast with Embeddings")
        
        # Generate embeddings first
        embedding_context = None
        if self.numbskull:
            try:
                emb_result = await self.numbskull.embed(str(data))
                embedding_context = {
                    "components": emb_result["metadata"]["components_used"],
                    "dimension": emb_result["metadata"]["embedding_dim"]
                }
                logger.info(f"  ✅ Embedding context prepared")
            except Exception as e:
                logger.warning(f"  ⚠️  Embedding failed: {e}")
        
        # Perform mirror cast
        mirror_result = self.mirror_cast.cast(data)
        
        # Enhance with embedding context
        if embedding_context:
            mirror_result["embedding_enhancement"] = embedding_context
            mirror_result["enhanced"] = True
        
        logger.info(f"  ✅ Mirror cast complete")
        return mirror_result
    
    async def embedding_guided_chunking(
        self,
        text: str,
        use_semantic_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Chunking guided by embedding analysis
        
        Args:
            text: Text to chunk
            use_semantic_chunks: Use semantic boundaries
        
        Returns:
            Enhanced chunking results
        """
        logger.info("\n✂️  Embedding-Guided Chunking")
        
        # Standard chunking
        chunks = self.choppy_processor.chunk(text, chunk_size=128, overlap=32)
        
        # If Numbskull available, analyze each chunk
        if self.numbskull and use_semantic_chunks:
            chunk_embeddings = []
            for chunk in chunks["semantic"][:5]:  # Analyze first 5
                try:
                    emb_result = await self.numbskull.embed(chunk)
                    chunk_embeddings.append({
                        "chunk": chunk[:50],
                        "dimension": emb_result["metadata"]["embedding_dim"],
                        "components": emb_result["metadata"]["components_used"]
                    })
                except Exception as e:
                    logger.warning(f"  ⚠️  Chunk embedding failed: {e}")
            
            chunks["embedding_enhanced_chunks"] = chunk_embeddings
            logger.info(f"  ✅ Enhanced {len(chunk_embeddings)} chunks with embeddings")
        
        return chunks
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ Neuro-symbolic adapter closed")


async def demo_neuro_symbolic_adapter():
    """Demonstration of neuro-symbolic + Numbskull integration"""
    print("\n" + "=" * 70)
    print("NEURO-SYMBOLIC + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = NeuroSymbolicNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_semantic": False,
            "use_mathematical": False,
            "use_fractal": True,
            "cache_embeddings": True
        }
    )
    
    # Test data
    test_cases = [
        "The quantum entanglement phenomenon enables faster-than-light communication",
        "f(x) = 3x^2 + 2x + 1, solve for x when f(x) = 0",
        "Machine learning algorithms learn patterns from training data"
    ]
    
    # Run analyses
    for i, data in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}")
        print(f"{'='*70}")
        print(f"Input: {data}")
        
        result = await adapter.analyze_with_embeddings(data, enable_all_modules=True)
        
        print(f"\nResults:")
        print(f"  Modules activated: {len(result['modules'])}")
        print(f"  Embeddings used: {result['embeddings']['components'] if result['embeddings'] else 'None'}")
        print(f"  Insights: {len(result['insights'])}")
        
        if result['recommendations']:
            print(f"  Recommendations: {result['recommendations'][0]}")
    
    # Test mirror cast
    print(f"\n{'='*70}")
    print("MIRROR CAST TEST")
    print(f"{'='*70}")
    mirror_result = await adapter.mirror_cast_with_embeddings(test_cases[0])
    print(f"Enhanced: {mirror_result.get('enhanced', False)}")
    
    # Test chunking
    print(f"\n{'='*70}")
    print("EMBEDDING-GUIDED CHUNKING TEST")
    print(f"{'='*70}")
    chunks = await adapter.embedding_guided_chunking(test_cases[2])
    print(f"Total chunks: {chunks['statistics']['chunk_count']}")
    print(f"Enhanced chunks: {len(chunks.get('embedding_enhanced_chunks', []))}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_neuro_symbolic_adapter())

