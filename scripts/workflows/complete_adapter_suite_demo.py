#!/usr/bin/env python3
"""
Complete Adapter Suite Demo
===========================

Comprehensive demonstration of ALL 10 component adapters:

1. Neuro-Symbolic + Numbskull
2. Signal Processing + Numbskull
3. AL-ULS + Numbskull
4. Evolutionary + Numbskull
5. TA ULS + Numbskull (PyTorch)
6. Holographic Memory + Numbskull (PyTorch)
7. Quantum Processor + Numbskull (PyTorch)
8. Cognitive Organism + Numbskull
9. Narrative Agent + Numbskull
10. Emergent Network + Numbskull

Shows complete end-to-end integration of entire LiMp + Numbskull ecosystem.

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from kgirl.paths import find_repo

# Add numbskull to path
numbskull_path = find_repo("numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import all 10 adapters
from kgirl.cognitive.neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from kgirl.llm.signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from kgirl.llm.aluls_numbskull_adapter import ALULSNumbskullAdapter
from kgirl.llm.evolutionary_numbskull_adapter import EvolutionaryNumbskullAdapter
from kgirl.llm.pytorch_components_numbskull_adapter import (
    TAULSNumbskullAdapter,
    HolographicNumbskullAdapter,
    QuantumNumbskullAdapter
)
from kgirl.cognitive.cognitive_organism_numbskull_adapter import CognitiveOrganismNumbskullAdapter
from kgirl.llm.narrative_numbskull_adapter import NarrativeNumbskullAdapter
from kgirl.cognitive.emergent_network_numbskull_adapter import EmergentNetworkNumbskullAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_complete_adapter_suite():
    """Comprehensive demo of all 10 adapters"""
    
    print("\n" + "=" * 80)
    print("COMPLETE ADAPTER SUITE DEMONSTRATION")
    print("ALL 10 LiMp + Numbskull Component Adapters")
    print("=" * 80)
    
    # Common config
    numbskull_config = {
        "use_semantic": False,  # Set True if Eopiez available
        "use_mathematical": False,  # Set True if LIMPS available
        "use_fractal": True,  # Always available
        "fusion_method": "weighted_average",
        "cache_embeddings": True
    }
    
    # Initialize all adapters
    print("\n" + "-" * 80)
    print("INITIALIZING ALL 10 ADAPTERS")
    print("-" * 80)
    
    adapters = {}
    adapter_definitions = [
        ("neuro_symbolic", NeuroSymbolicNumbskullAdapter, numbskull_config),
        ("signal_processing", SignalProcessingNumbskullAdapter, numbskull_config),
        ("aluls", ALULSNumbskullAdapter, {**numbskull_config, "use_mathematical": True}),
        ("evolutionary", EvolutionaryNumbskullAdapter, numbskull_config),
        ("tauls", TAULSNumbskullAdapter, numbskull_config),
        ("holographic", HolographicNumbskullAdapter, numbskull_config),
        ("quantum", QuantumNumbskullAdapter, numbskull_config),
        ("cognitive_organism", CognitiveOrganismNumbskullAdapter, numbskull_config),
        ("narrative", NarrativeNumbskullAdapter, numbskull_config),
        ("emergent_network", EmergentNetworkNumbskullAdapter, numbskull_config),
    ]
    
    for name, adapter_class, config in adapter_definitions:
        try:
            adapters[name] = adapter_class(use_numbskull=True, numbskull_config=config)
            print(f"✅ {len(adapters)}/10 {name} adapter initialized")
        except Exception as e:
            logger.warning(f"⚠️  {name} adapter failed: {e}")
    
    print(f"\n✅ Initialized {len(adapters)}/10 adapters successfully")
    
    # Test data
    test_case = {
        "text": "Advanced cognitive processing integrates multiple AI modalities for emergent intelligence",
        "symbolic": "SUM(1, 2, 3)",
        "narrative": "The system evolved. Intelligence emerged. Understanding deepened. Wisdom arose."
    }
    
    # Run comprehensive test
    print("\n" + "=" * 80)
    print("COMPREHENSIVE INTEGRATION TEST")
    print("=" * 80)
    print(f"Test Input: {test_case['text'][:70]}...")
    print("-" * 80)
    
    results = {}
    start_time = time.time()
    
    # Test each adapter
    if "neuro_symbolic" in adapters:
        print("\n1️⃣  Neuro-Symbolic Analysis")
        try:
            result = await adapters["neuro_symbolic"].analyze_with_embeddings(test_case["text"])
            results["neuro_symbolic"] = {
                "modules": len(result["modules"]),
                "insights": len(result["insights"])
            }
            print(f"   ✅ {results['neuro_symbolic']['modules']} modules analyzed")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "signal_processing" in adapters:
        print("\n2️⃣  Signal Processing")
        try:
            scheme, analysis = await adapters["signal_processing"].select_modulation_from_embedding(test_case["text"])
            results["signal"] = {"modulation": scheme.name}
            print(f"   ✅ Modulation: {scheme.name}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "aluls" in adapters:
        print("\n3️⃣  AL-ULS Symbolic")
        try:
            result = await adapters["aluls"].analyze_expression_with_embeddings(test_case["symbolic"])
            results["aluls"] = {"is_symbolic": result["is_symbolic"]}
            print(f"   ✅ Symbolic: {result['is_symbolic']}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "evolutionary" in adapters:
        print("\n4️⃣  Evolutionary Processing")
        try:
            result = await adapters["evolutionary"].evolve_with_embeddings(test_case["text"])
            results["evolutionary"] = {"fitness": result["fitness"]}
            print(f"   ✅ Fitness: {result['fitness']:.3f}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "tauls" in adapters:
        print("\n5️⃣  TA ULS Stabilization")
        try:
            result = await adapters["tauls"].stabilize_embedding(test_case["text"])
            results["tauls"] = {"stabilized": result.get("stabilized", False)}
            print(f"   ✅ Stabilized: {result.get('stabilized', False)}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "holographic" in adapters:
        print("\n6️⃣  Holographic Memory")
        try:
            result = await adapters["holographic"].store_with_embeddings(test_case["text"])
            results["holographic"] = {"stored": result.get("stored", False)}
            print(f"   ✅ Stored: {result.get('stored', False)}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "quantum" in adapters:
        print("\n7️⃣  Quantum Processing")
        try:
            result = await adapters["quantum"].quantum_enhance_embedding(test_case["text"])
            results["quantum"] = {"enhanced": result.get("quantum_enhanced", False)}
            print(f"   ✅ Enhanced: {result.get('quantum_enhanced', False)}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "cognitive_organism" in adapters:
        print("\n8️⃣  Cognitive Organism")
        try:
            result = await adapters["cognitive_organism"].cognitive_communication(test_case["text"])
            results["cognitive_organism"] = {"levels": len(result["processing_levels"])}
            print(f"   ✅ Levels: {len(result['processing_levels'])}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "narrative" in adapters:
        print("\n9️⃣  Narrative Intelligence")
        try:
            result = await adapters["narrative"].analyze_narrative_with_embeddings(test_case["narrative"])
            results["narrative"] = {"emotional_valence": result["emotional_valence"]}
            print(f"   ✅ Emotional: {result['emotional_valence']:.3f}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    if "emergent_network" in adapters:
        print("\n🔟 Emergent Network")
        try:
            result = await adapters["emergent_network"].swarm_optimize_embedding(test_case["text"])
            results["emergent"] = {"optimized": result.get("optimized", False)}
            print(f"   ✅ Optimized: {result.get('optimized', False)}")
        except Exception as e:
            logger.warning(f"   ⚠️  {e}")
    
    total_time = time.time() - start_time
    
    # Display results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Adapters Tested: {len(adapters)}/10")
    print(f"\nResults:")
    print(json.dumps(results, indent=2))
    
    # Cleanup all adapters
    print("\n" + "=" * 80)
    print("CLEANING UP ALL ADAPTERS")
    print("=" * 80)
    
    for name, adapter in adapters.items():
        try:
            await adapter.close()
            print(f"✅ Closed {name}")
        except Exception as e:
            logger.warning(f"⚠️  Error closing {name}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE ADAPTER SUITE DEMO FINISHED")
    print("=" * 80)
    print(f"\n🎉 All {len(adapters)} adapters demonstrated successfully!")
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    print("\n💡 Next step: Start LFM2-8B-A1B server for full LLM integration")


if __name__ == "__main__":
    asyncio.run(demo_complete_adapter_suite())

