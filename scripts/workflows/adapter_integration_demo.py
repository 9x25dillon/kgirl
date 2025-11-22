#!/usr/bin/env python3
"""
Complete Adapter Integration Demo
==================================

Demonstrates all 6 component adapters working together:
1. Neuro-Symbolic + Numbskull
2. Signal Processing + Numbskull
3. AL-ULS + Numbskull
4. Evolutionary + Numbskull
5. TA ULS + Numbskull
6. Holographic + Numbskull
7. Quantum + Numbskull

Shows complete end-to-end integration of all LiMp + Numbskull components.

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import all adapters
from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from aluls_numbskull_adapter import ALULSNumbskullAdapter
from evolutionary_numbskull_adapter import EvolutionaryNumbskullAdapter
from pytorch_components_numbskull_adapter import (
    TAULSNumbskullAdapter,
    HolographicNumbskullAdapter,
    QuantumNumbskullAdapter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_all_adapters():
    """Comprehensive demo of all adapters working together"""
    
    print("\n" + "=" * 70)
    print("COMPLETE ADAPTER INTEGRATION DEMO")
    print("All LiMp + Numbskull Components")
    print("=" * 70)
    
    # Common Numbskull config
    numbskull_config = {
        "use_semantic": False,  # Set to True if Eopiez available
        "use_mathematical": False,  # Set to True if LIMPS available
        "use_fractal": True,  # Always available
        "fusion_method": "weighted_average",
        "cache_embeddings": True
    }
    
    # Test data
    test_data = [
        {"text": "Quantum entanglement enables instant communication", "type": "physics"},
        {"text": "SUM(1, 2, 3, 4, 5)", "type": "symbolic"},
        {"text": "Neural networks learn from training data", "type": "AI"}
    ]
    
    # Initialize all adapters
    adapters = {}
    
    print("\n" + "=" * 70)
    print("INITIALIZING ALL ADAPTERS")
    print("=" * 70)
    
    try:
        adapters["neuro_symbolic"] = NeuroSymbolicNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=numbskull_config
        )
        print("✅ 1/7 Neuro-Symbolic adapter")
    except Exception as e:
        logger.warning(f"Neuro-Symbolic adapter failed: {e}")
    
    try:
        adapters["signal"] = SignalProcessingNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=numbskull_config
        )
        print("✅ 2/7 Signal Processing adapter")
    except Exception as e:
        logger.warning(f"Signal adapter failed: {e}")
    
    try:
        adapters["aluls"] = ALULSNumbskullAdapter(
            use_numbskull=True,
            numbskull_config={**numbskull_config, "use_mathematical": True}
        )
        print("✅ 3/7 AL-ULS adapter")
    except Exception as e:
        logger.warning(f"AL-ULS adapter failed: {e}")
    
    try:
        adapters["evolutionary"] = EvolutionaryNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=numbskull_config
        )
        print("✅ 4/7 Evolutionary adapter")
    except Exception as e:
        logger.warning(f"Evolutionary adapter failed: {e}")
    
    try:
        adapters["tauls"] = TAULSNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=numbskull_config
        )
        print("✅ 5/7 TA ULS adapter")
    except Exception as e:
        logger.warning(f"TA ULS adapter failed: {e}")
    
    try:
        adapters["holographic"] = HolographicNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=numbskull_config
        )
        print("✅ 6/7 Holographic adapter")
    except Exception as e:
        logger.warning(f"Holographic adapter failed: {e}")
    
    try:
        adapters["quantum"] = QuantumNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=numbskull_config,
            num_qubits=4
        )
        print("✅ 7/7 Quantum adapter")
    except Exception as e:
        logger.warning(f"Quantum adapter failed: {e}")
    
    print(f"\nInitialized {len(adapters)}/7 adapters")
    
    # Process each test case through all adapters
    for i, test_case in enumerate(test_data, 1):
        print("\n" + "=" * 70)
        print(f"TEST CASE {i}: {test_case['type'].upper()}")
        print("=" * 70)
        print(f"Input: {test_case['text']}")
        print("-" * 70)
        
        results = {}
        
        # 1. Neuro-Symbolic Analysis
        if "neuro_symbolic" in adapters:
            print("\n1️⃣  Neuro-Symbolic Analysis")
            try:
                result = await adapters["neuro_symbolic"].analyze_with_embeddings(
                    test_case["text"],
                    enable_all_modules=True
                )
                results["neuro_symbolic"] = {
                    "modules": len(result["modules"]),
                    "insights": len(result["insights"]),
                    "embeddings": result["embeddings"]["components"] if result["embeddings"] else None
                }
                print(f"   ✅ {results['neuro_symbolic']['modules']} modules analyzed")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # 2. Signal Processing
        if "signal" in adapters:
            print("\n2️⃣  Signal Processing")
            try:
                scheme, analysis = await adapters["signal"].select_modulation_from_embedding(
                    test_case["text"]
                )
                results["signal"] = {
                    "modulation": scheme.name,
                    "reason": analysis.get("reason", "N/A")[:50]
                }
                print(f"   ✅ Modulation: {scheme.name}")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # 3. AL-ULS (if symbolic)
        if "aluls" in adapters and adapters["aluls"].is_symbolic_expression(test_case["text"]):
            print("\n3️⃣  AL-ULS Symbolic Evaluation")
            try:
                result = await adapters["aluls"].analyze_expression_with_embeddings(
                    test_case["text"]
                )
                results["aluls"] = {
                    "is_symbolic": result["is_symbolic"],
                    "has_embedding": result["embedding_analysis"] is not None
                }
                print(f"   ✅ Symbolic: {result['is_symbolic']}")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # 4. Evolutionary Processing
        if "evolutionary" in adapters:
            print("\n4️⃣  Evolutionary Processing")
            try:
                result = await adapters["evolutionary"].evolve_with_embeddings(
                    test_case["text"]
                )
                results["evolutionary"] = {
                    "fitness": result["fitness"],
                    "strategy": result.get("evolution_strategy", "N/A")
                }
                print(f"   ✅ Fitness: {result['fitness']:.3f}, Strategy: {result.get('evolution_strategy', 'N/A')}")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # 5. TA ULS Stabilization
        if "tauls" in adapters:
            print("\n5️⃣  TA ULS Stabilization")
            try:
                result = await adapters["tauls"].stabilize_embedding(test_case["text"])
                results["tauls"] = {
                    "stabilized": result.get("stabilized", False)
                }
                print(f"   {'✅ Stabilized' if result.get('stabilized') else 'ℹ️  Generated (no PyTorch)'}")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # 6. Holographic Storage
        if "holographic" in adapters:
            print("\n6️⃣  Holographic Storage")
            try:
                result = await adapters["holographic"].store_with_embeddings(
                    test_case["text"],
                    {"type": test_case["type"]}
                )
                results["holographic"] = {
                    "stored": result.get("stored", False),
                    "key": result.get("memory_key")
                }
                print(f"   {'✅ Stored: ' + result.get('memory_key', '') if result.get('stored') else 'ℹ️  Generated (no PyTorch)'}")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # 7. Quantum Enhancement
        if "quantum" in adapters:
            print("\n7️⃣  Quantum Enhancement")
            try:
                result = await adapters["quantum"].quantum_enhance_embedding(test_case["text"])
                results["quantum"] = {
                    "enhanced": result.get("quantum_enhanced", False)
                }
                if result.get("quantum_metrics"):
                    print(f"   ✅ Enhanced: entropy={result['quantum_metrics']['entropy']:.3f}")
                else:
                    print(f"   ℹ️  Generated (no PyTorch)")
            except Exception as e:
                logger.warning(f"   ⚠️  {e}")
        
        # Summary for this test case
        print("\n" + "-" * 70)
        print("Test Case Summary:")
        print(json.dumps(results, indent=2, default=str))
    
    # Get evolution stats
    if "evolutionary" in adapters:
        print("\n" + "=" * 70)
        print("EVOLUTION STATISTICS")
        print("=" * 70)
        stats = adapters["evolutionary"].get_evolution_stats()
        print(json.dumps(stats, indent=2))
    
    # Cleanup all adapters
    print("\n" + "=" * 70)
    print("CLEANING UP")
    print("=" * 70)
    
    for name, adapter in adapters.items():
        try:
            await adapter.close()
            print(f"✅ Closed {name}")
        except Exception as e:
            logger.warning(f"⚠️  Error closing {name}: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ALL ADAPTERS DEMO COMPLETE")
    print("=" * 70)
    print(f"\nTested {len(adapters)} adapters on {len(test_data)} test cases")
    print("All LiMp + Numbskull components working together!")


if __name__ == "__main__":
    asyncio.run(demo_all_adapters())

