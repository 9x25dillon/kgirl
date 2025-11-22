#!/usr/bin/env python3
"""
AL-ULS Symbolic + Numbskull Integration Adapter
===============================================

Deep integration between AL-ULS Symbolic Evaluation and Numbskull:
- Mathematical embedding preprocessing
- Symbolic expression analysis with embeddings
- Embedding-guided symbolic optimization
- Batch symbolic processing

Author: Assistant
License: MIT
"""

import asyncio
import logging
import re
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
    from src.chaos_llm.services.al_uls import al_uls
    ALULS_AVAILABLE = True
except ImportError:
    ALULS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("AL-ULS not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ALULSNumbskullAdapter:
    """
    Adapter integrating AL-ULS symbolic evaluation with Numbskull embeddings
    
    Provides:
    - Mathematical embedding preprocessing for symbolic calls
    - Embedding-enhanced symbolic evaluation
    - Batch processing with embedding context
    - Symbolic result integration with embeddings
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("AL-ULS SYMBOLIC + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Check AL-ULS availability
        self.aluls_available = ALULS_AVAILABLE
        if self.aluls_available:
            logger.info("✅ AL-ULS symbolic engine available")
        else:
            logger.warning("⚠️  AL-ULS not available (using mock)")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            # Prefer mathematical embeddings for symbolic work
            config_dict = numbskull_config or {}
            config_dict.setdefault("use_mathematical", True)
            config_dict.setdefault("use_semantic", False)
            config_dict.setdefault("use_fractal", True)
            
            config = HybridConfig(**config_dict)
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated (math + fractal)")
        else:
            logger.warning("⚠️  Operating without Numbskull embeddings")
        
        # Expression patterns
        self.expr_pattern = re.compile(r'[A-Za-z_]\w*\s*\([^)]*\)')
        
        logger.info("=" * 70)
    
    def is_symbolic_expression(self, text: str) -> bool:
        """Check if text contains symbolic expression"""
        return bool(self.expr_pattern.search(text))
    
    async def analyze_expression_with_embeddings(
        self,
        expression: str
    ) -> Dict[str, Any]:
        """
        Analyze symbolic expression with mathematical embeddings
        
        Args:
            expression: Symbolic expression (e.g., "SUM(1,2,3)")
        
        Returns:
            Analysis results
        """
        logger.info(f"\n🔢 Analyzing Expression: {expression}")
        
        results = {
            "expression": expression,
            "is_symbolic": self.is_symbolic_expression(expression),
            "embedding_analysis": None,
            "symbolic_result": None
        }
        
        # Generate mathematical embedding
        if self.numbskull:
            try:
                emb_result = await self.numbskull.embed(expression)
                results["embedding_analysis"] = {
                    "components": emb_result["metadata"]["components_used"],
                    "dimension": emb_result["metadata"]["embedding_dim"],
                    "mathematical_component": "mathematical" in emb_result["metadata"]["components_used"]
                }
                logger.info(f"  ✅ Embedding: {emb_result['metadata']['components_used']}")
            except Exception as e:
                logger.warning(f"  ⚠️  Embedding failed: {e}")
        
        # Evaluate symbolically if AL-ULS available
        if self.aluls_available and results["is_symbolic"]:
            try:
                # Parse call
                call = al_uls.parse_symbolic_call(expression)
                
                if call.get("name"):
                    # Evaluate
                    symbolic_result = await al_uls.eval_symbolic_call_async(call)
                    results["symbolic_result"] = symbolic_result
                    logger.info(f"  ✅ Symbolic evaluation complete")
            except Exception as e:
                logger.warning(f"  ⚠️  Symbolic evaluation failed: {e}")
                results["symbolic_result"] = {"error": str(e)}
        elif not results["is_symbolic"]:
            logger.info("  ℹ️  Not a symbolic expression")
        
        return results
    
    async def batch_symbolic_with_embeddings(
        self,
        expressions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch process symbolic expressions with embeddings
        
        Args:
            expressions: List of symbolic expressions
        
        Returns:
            List of analysis results
        """
        logger.info(f"\n📊 Batch Processing {len(expressions)} Expressions")
        
        results = []
        
        # Generate embeddings in parallel
        if self.numbskull:
            try:
                embedding_tasks = [self.numbskull.embed(expr) for expr in expressions]
                embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
                logger.info(f"  ✅ Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.warning(f"  ⚠️  Batch embedding failed: {e}")
                embeddings = [None] * len(expressions)
        else:
            embeddings = [None] * len(expressions)
        
        # Process each expression
        for expr, emb in zip(expressions, embeddings):
            result = {
                "expression": expr,
                "embedding": None,
                "symbolic_result": None
            }
            
            if emb and not isinstance(emb, Exception):
                result["embedding"] = {
                    "components": emb["metadata"]["components_used"],
                    "dimension": emb["metadata"]["embedding_dim"]
                }
            
            # Check if symbolic
            if self.is_symbolic_expression(expr) and self.aluls_available:
                try:
                    call = al_uls.parse_symbolic_call(expr)
                    if call.get("name"):
                        symbolic_result = await al_uls.eval_symbolic_call_async(call)
                        result["symbolic_result"] = symbolic_result
                except Exception as e:
                    result["symbolic_result"] = {"error": str(e)}
            
            results.append(result)
        
        logger.info(f"  ✅ Processed {len(results)} expressions")
        return results
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()
        logger.info("✅ AL-ULS adapter closed")


async def demo_aluls_adapter():
    """Demonstration of AL-ULS + Numbskull integration"""
    print("\n" + "=" * 70)
    print("AL-ULS SYMBOLIC + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Create adapter
    adapter = ALULSNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_mathematical": True,  # Prefer math for symbolic
            "use_fractal": True,
            "cache_embeddings": True
        }
    )
    
    # Test expressions
    test_expressions = [
        "SUM(1, 2, 3, 4, 5)",
        "MEAN(10, 20, 30)",
        "This is not a symbolic expression",
        "VAR(1, 2, 3, 4)",
    ]
    
    # Test individual analysis
    for i, expr in enumerate(test_expressions[:2], 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: Individual Analysis")
        print(f"{'='*70}")
        
        result = await adapter.analyze_expression_with_embeddings(expr)
        print(f"Expression: {expr}")
        print(f"Is Symbolic: {result['is_symbolic']}")
        if result.get('embedding_analysis'):
            print(f"Embeddings: {result['embedding_analysis']['components']}")
        if result.get('symbolic_result'):
            print(f"Result: {result['symbolic_result']}")
    
    # Test batch processing
    print(f"\n{'='*70}")
    print("TEST: Batch Processing")
    print(f"{'='*70}")
    batch_results = await adapter.batch_symbolic_with_embeddings(test_expressions)
    print(f"Processed: {len(batch_results)} expressions")
    for i, result in enumerate(batch_results, 1):
        emb_info = result.get('embedding', {})
        components = emb_info.get('components', 'None')
        print(f"  {i}. {result['expression'][:40]:<40} | Embeddings: {components}")
    
    # Cleanup
    await adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(demo_aluls_adapter())

