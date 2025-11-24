#!/usr/bin/env python3
"""
PyTorch Components + Numbskull Integration Adapter
==================================================

Integration for PyTorch-based LiMp components with Numbskull:
- TA ULS Transformer (with KFP layers)
- Holographic Memory System
- Quantum Cognitive Processor

Provides fallback implementations when PyTorch not available.

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

# Try importing PyTorch components
try:
    import torch
    import torch.nn as nn
    from tauls_transformer import TAULSControlUnit, KFPLayer, EntropyRegulator
    TAULS_AVAILABLE = True
except ImportError:
    TAULS_AVAILABLE = False
    torch = None

try:
    from holographic_memory_system import (
        HolographicAssociativeMemory,
        FractalEncoder,
        QuantumEnhancedMemory
    )
    HOLOGRAPHIC_AVAILABLE = True
except ImportError:
    HOLOGRAPHIC_AVAILABLE = False

try:
    from quantum_cognitive_processor import (
        QuantumNeuralNetwork,
        QuantumWalkOptimizer
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TAULSNumbskullAdapter:
    """
    Adapter for TA ULS Transformer + Numbskull
    
    Provides stability control and optimization for embeddings
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None,
        input_dim: int = 768
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("TA ULS TRANSFORMER + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize TA ULS if available
        self.tauls_unit = None
        if TAULS_AVAILABLE:
            try:
                self.tauls_unit = TAULSControlUnit(
                    input_dim=input_dim,
                    hidden_dim=512,
                    control_dim=256
                )
                logger.info("✅ TA ULS transformer initialized")
            except Exception as e:
                logger.warning(f"⚠️  TA ULS init failed: {e}")
        else:
            logger.warning("⚠️  TA ULS not available (PyTorch needed)")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        
        logger.info("=" * 70)
    
    async def stabilize_embedding(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Apply TA ULS stabilization to embedding
        
        Args:
            text: Input text
        
        Returns:
            Stabilization results
        """
        logger.info(f"\n⚖️  TA ULS Stabilization: {text[:60]}...")
        
        results = {
            "text": text,
            "embedding": None,
            "stabilized": False,
            "stability_metrics": None
        }
        
        if not self.numbskull:
            logger.warning("  ⚠️  No embeddings without Numbskull")
            return results
        
        try:
            # Generate embedding
            emb_result = await self.numbskull.embed(text)
            embedding = emb_result["fused_embedding"]
            results["embedding"] = {
                "dimension": len(embedding),
                "components": emb_result["metadata"]["components_used"]
            }
            
            # Apply TA ULS if available
            if self.tauls_unit and torch:
                # Convert to tensor
                if len(embedding) < 768:
                    embedding = np.pad(embedding, (0, 768 - len(embedding)))
                elif len(embedding) > 768:
                    embedding = embedding[:768]
                
                tensor_input = torch.from_numpy(embedding).float().unsqueeze(0)
                control_state = torch.zeros(1, 256)
                
                # Apply TA ULS transformation
                with torch.no_grad():
                    control_output, stability_metrics = self.tauls_unit(tensor_input, control_state)
                
                results["stabilized"] = True
                results["stability_metrics"] = {
                    "mean": float(stability_metrics.mean()),
                    "std": float(stability_metrics.std())
                }
                
                logger.info(f"  ✅ TA ULS applied, stability: {results['stability_metrics']['mean']:.3f}")
            else:
                logger.info("  ℹ️  Using embedding without TA ULS stabilization")
            
        except Exception as e:
            logger.error(f"  ❌ Stabilization failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()


class HolographicNumbskullAdapter:
    """
    Adapter for Holographic Memory + Numbskull
    
    Provides memory-augmented embeddings and pattern storage
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("HOLOGRAPHIC MEMORY + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize holographic memory if available
        self.holographic = None
        if HOLOGRAPHIC_AVAILABLE:
            try:
                self.holographic = HolographicAssociativeMemory(
                    memory_size=1024,
                    hologram_dim=256
                )
                logger.info("✅ Holographic memory initialized")
            except Exception as e:
                logger.warning(f"⚠️  Holographic init failed: {e}")
        else:
            logger.warning("⚠️  Holographic memory not available (PyTorch needed)")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        
        logger.info("=" * 70)
    
    async def store_with_embeddings(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store text in holographic memory with embeddings
        
        Args:
            text: Text to store
            metadata: Optional metadata
        
        Returns:
            Storage results
        """
        logger.info(f"\n💾 Holographic Storage: {text[:60]}...")
        
        results = {
            "text": text,
            "stored": False,
            "memory_key": None
        }
        
        if not self.numbskull:
            logger.warning("  ⚠️  No embeddings without Numbskull")
            return results
        
        try:
            # Generate embedding
            emb_result = await self.numbskull.embed(text)
            embedding = emb_result["fused_embedding"]
            
            # Store in holographic memory if available
            if self.holographic:
                memory_key = self.holographic.store(embedding, metadata or {})
                results["stored"] = True
                results["memory_key"] = memory_key
                logger.info(f"  ✅ Stored in holographic memory: {memory_key}")
            else:
                logger.info("  ℹ️  Holographic memory not available, embedding generated only")
            
            results["embedding"] = {
                "dimension": len(embedding),
                "components": emb_result["metadata"]["components_used"]
            }
            
        except Exception as e:
            logger.error(f"  ❌ Storage failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()


class QuantumNumbskullAdapter:
    """
    Adapter for Quantum Processor + Numbskull
    
    Provides quantum-enhanced embedding processing
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None,
        num_qubits: int = 4
    ):
        """Initialize adapter"""
        logger.info("=" * 70)
        logger.info("QUANTUM PROCESSOR + NUMBSKULL ADAPTER")
        logger.info("=" * 70)
        
        # Initialize quantum processor if available
        self.quantum = None
        if QUANTUM_AVAILABLE and torch:
            try:
                self.quantum = QuantumNeuralNetwork(
                    num_qubits=num_qubits,
                    num_layers=2
                )
                logger.info(f"✅ Quantum processor initialized ({num_qubits} qubits)")
            except Exception as e:
                logger.warning(f"⚠️  Quantum init failed: {e}")
        else:
            logger.warning("⚠️  Quantum processor not available (PyTorch needed)")
        
        # Initialize Numbskull
        self.numbskull = None
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Numbskull pipeline integrated")
        
        logger.info("=" * 70)
    
    async def quantum_enhance_embedding(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Quantum-enhance embedding
        
        Args:
            text: Input text
        
        Returns:
            Quantum enhancement results
        """
        logger.info(f"\n⚛️  Quantum Enhancement: {text[:60]}...")
        
        results = {
            "text": text,
            "quantum_enhanced": False,
            "quantum_metrics": None
        }
        
        if not self.numbskull:
            logger.warning("  ⚠️  No embeddings without Numbskull")
            return results
        
        try:
            # Generate embedding
            emb_result = await self.numbskull.embed(text)
            embedding = emb_result["fused_embedding"]
            
            results["embedding"] = {
                "dimension": len(embedding),
                "components": emb_result["metadata"]["components_used"]
            }
            
            # Apply quantum processing if available
            if self.quantum and torch:
                # Prepare input (take first 16 dims or pad)
                if len(embedding) >= 16:
                    quantum_input = embedding[:16]
                else:
                    quantum_input = np.pad(embedding, (0, 16 - len(embedding)))
                
                tensor_input = torch.from_numpy(quantum_input).float().unsqueeze(0)
                
                # Process through quantum network
                with torch.no_grad():
                    quantum_output = self.quantum(tensor_input)
                
                results["quantum_enhanced"] = True
                results["quantum_metrics"] = {
                    "entropy": float(quantum_output["quantum_entropy"]),
                    "coherence": float(quantum_output["quantum_coherence"])
                }
                
                logger.info(f"  ✅ Quantum enhanced: entropy={results['quantum_metrics']['entropy']:.3f}")
            else:
                logger.info("  ℹ️  Quantum processing not available")
            
        except Exception as e:
            logger.error(f"  ❌ Quantum enhancement failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()


async def demo_pytorch_adapters():
    """Demonstration of PyTorch component adapters"""
    print("\n" + "=" * 70)
    print("PYTORCH COMPONENTS + NUMBSKULL ADAPTER DEMO")
    print("=" * 70)
    
    # Test TA ULS adapter
    print("\n--- TA ULS ADAPTER ---")
    tauls_adapter = TAULSNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={"use_fractal": True}
    )
    
    result = await tauls_adapter.stabilize_embedding("Test message for TA ULS stabilization")
    print(f"Stabilized: {result.get('stabilized', False)}")
    if result.get('stability_metrics'):
        print(f"Stability: {result['stability_metrics']['mean']:.3f}")
    
    await tauls_adapter.close()
    
    # Test Holographic adapter
    print("\n--- HOLOGRAPHIC MEMORY ADAPTER ---")
    holo_adapter = HolographicNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={"use_fractal": True}
    )
    
    result = await holo_adapter.store_with_embeddings(
        "Knowledge to store in holographic memory",
        {"category": "knowledge", "importance": 0.9}
    )
    print(f"Stored: {result.get('stored', False)}")
    if result.get('memory_key'):
        print(f"Memory Key: {result['memory_key']}")
    
    await holo_adapter.close()
    
    # Test Quantum adapter
    print("\n--- QUANTUM PROCESSOR ADAPTER ---")
    quantum_adapter = QuantumNumbskullAdapter(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={"use_fractal": True},
        num_qubits=4
    )
    
    result = await quantum_adapter.quantum_enhance_embedding(
        "Quantum-enhanced cognitive processing"
    )
    print(f"Quantum Enhanced: {result.get('quantum_enhanced', False)}")
    if result.get('quantum_metrics'):
        print(f"Quantum Entropy: {result['quantum_metrics']['entropy']:.3f}")
        print(f"Quantum Coherence: {result['quantum_metrics']['coherence']:.3f}")
    
    await quantum_adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")
    print("\nNOTE: Some components may not be active without PyTorch.")
    print("Install PyTorch: pip install torch")


if __name__ == "__main__":
    asyncio.run(demo_pytorch_adapters())

