"""
Numbskull - Advanced AI Embedding Pipeline
==========================================

A sophisticated multi-modal embedding system that integrates:
- Semantic vectorization (Eopiez)
- Mathematical optimization (LIMPS/Matrix Orchestrator)
- Entropy-based token processing
- Sparse embedding with distributed optimization
- Hybrid search capabilities
- Fractal cascade simulation

Components:
- SemanticEmbedder: Advanced semantic vectorization
- MathematicalEmbedder: Mathematical/symbolic embedding
- FractalCascadeEmbedder: Fractal-based embedding simulation
- HybridEmbeddingPipeline: Unified embedding orchestration
- EmbeddingOptimizer: Performance optimization
"""

from .semantic_embedder import SemanticEmbedder, SemanticConfig
from .mathematical_embedder import MathematicalEmbedder, MathematicalConfig
from .fractal_cascade_embedder import FractalCascadeEmbedder, FractalConfig
from .hybrid_pipeline import HybridEmbeddingPipeline, HybridConfig
from .optimizer import EmbeddingOptimizer, OptimizationConfig

__version__ = "1.0.0"
__author__ = "9x25dillon"
__email__ = "your.email@example.com"
__description__ = "Advanced AI Embedding Pipeline with Multi-Modal Fusion"

__all__ = [
    "SemanticEmbedder",
    "SemanticConfig",
    "MathematicalEmbedder", 
    "MathematicalConfig",
    "FractalCascadeEmbedder",
    "FractalConfig",
    "HybridEmbeddingPipeline",
    "HybridConfig",
    "EmbeddingOptimizer",
    "OptimizationConfig"
]