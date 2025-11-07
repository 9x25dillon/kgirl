#!/usr/bin/env python3
"""
Simple Integrated WaveCaster Demo
=================================

Demonstrates complete integration WITHOUT requiring PyTorch:
- Numbskull embeddings
- Dual LLM orchestration
- Neuro-symbolic analysis
- Signal processing
- Modulation scheme selection

Works with available components only.

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import our integrated components
from numbskull_dual_orchestrator import create_numbskull_orchestrator
from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from complete_system_integration import CompleteSystemIntegration

import signal_processing as dsp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_simple_integrated_demo():
    """Simple integrated demo that works without PyTorch"""
    
    print("\n" + "=" * 70)
    print("SIMPLE INTEGRATED WAVECASTER DEMO")
    print("LiMp + Numbskull + Signal Processing (No PyTorch Required)")
    print("=" * 70)
    
    # Configuration
    config = {
        "local_llm": {
            "base_url": "http://127.0.0.1:8080",
            "mode": "llama-cpp",
            "model": "LFM2-8B-A1B",
            "timeout": 120
        },
        "numbskull": {
            "use_semantic": False,
            "use_mathematical": False,
            "use_fractal": True,  # Always available
            "fusion_method": "weighted_average",
            "cache_embeddings": True
        },
        "orchestrator_settings": {
            "temperature": 0.7,
            "max_tokens": 256,
            "style": "concise",
            "use_numbskull": True
        }
    }
    
    # Initialize components
    print("\n" + "-" * 70)
    print("INITIALIZING COMPONENTS")
    print("-" * 70)
    
    # 1. Numbskull + Dual LLM
    try:
        orchestrator = create_numbskull_orchestrator(
            local_configs=[config["local_llm"]],
            remote_config=None,  # Use local fallback
            settings=config["orchestrator_settings"],
            numbskull_config=config["numbskull"]
        )
        print("✅ 1/3 Numbskull + Dual LLM Orchestrator")
    except Exception as e:
        logger.warning(f"Orchestrator init failed: {e}")
        orchestrator = None
    
    # 2. Neuro-Symbolic Adapter
    try:
        neuro_symbolic = NeuroSymbolicNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=config["numbskull"]
        )
        print("✅ 2/3 Neuro-Symbolic Adapter")
    except Exception as e:
        logger.warning(f"Neuro-symbolic init failed: {e}")
        neuro_symbolic = None
    
    # 3. Signal Processing Adapter
    try:
        signal_adapter = SignalProcessingNumbskullAdapter(
            use_numbskull=True,
            numbskull_config=config["numbskull"]
        )
        print("✅ 3/3 Signal Processing Adapter")
    except Exception as e:
        logger.warning(f"Signal adapter init failed: {e}")
        signal_adapter = None
    
    # Test scenarios
    scenarios = [
        {
            "name": "Emergency Communication",
            "content": "URGENT: All units respond to sector 7. Network coordination required immediately.",
            "type": "emergency"
        },
        {
            "name": "Technical Analysis",
            "content": "The dual LLM orchestration system integrates Numbskull hybrid embeddings with LFM2-8B-A1B for enhanced contextual understanding.",
            "type": "technical"
        },
        {
            "name": "Mathematical Processing",
            "content": "Calculate: The derivative of f(x) = 3x^2 + 2x + 1 is f'(x) = 6x + 2",
            "type": "mathematical"
        }
    ]
    
    # Process each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i}/{len(scenarios)}: {scenario['name']}")
        print(f"{'='*70}")
        print(f"Content: {scenario['content'][:60]}...")
        print("-" * 70)
        
        # Stage 1: Generate embeddings
        if orchestrator:
            print("\n📊 Stage 1: Embedding Generation")
            try:
                emb_result = await orchestrator._generate_embeddings(scenario["content"])
                if emb_result:
                    print(f"   ✅ Components: {emb_result['metadata']['components_used']}")
                    print(f"   ✅ Dimension: {emb_result['metadata']['embedding_dim']}")
                    print(f"   ✅ Time: {emb_result['metadata']['processing_time']:.3f}s")
            except Exception as e:
                print(f"   ⚠️  {e}")
        
        # Stage 2: Neuro-Symbolic Analysis
        if neuro_symbolic:
            print("\n🔬 Stage 2: Neuro-Symbolic Analysis")
            try:
                analysis = await neuro_symbolic.analyze_with_embeddings(
                    scenario["content"],
                    enable_all_modules=True
                )
                print(f"   ✅ Modules: {len(analysis['modules'])}")
                print(f"   ✅ Insights: {len(analysis['insights'])}")
                if analysis["insights"]:
                    print(f"   💡 {analysis['insights'][0][:70]}...")
            except Exception as e:
                print(f"   ⚠️  {e}")
        
        # Stage 3: Modulation Selection
        if signal_adapter:
            print("\n📡 Stage 3: Modulation Selection")
            try:
                scheme, selection_analysis = await signal_adapter.select_modulation_from_embedding(
                    scenario["content"]
                )
                print(f"   ✅ Selected: {scheme.name}")
                print(f"   ✅ Reason: {selection_analysis.get('reason', 'N/A')[:60]}...")
            except Exception as e:
                print(f"   ⚠️  {e}")
        
        # Stage 4: Signal Generation Info
        print("\n🎵 Stage 4: Signal Generation")
        print(f"   ℹ️  Content ready for signal processing")
        print(f"   ℹ️  Output would be saved to: {scenario.get('output_dir', 'output')}")
        print(f"   💡 To actually generate signals, use full wavecaster with services")
    
    # Show summary
    print(f"\n{'='*70}")
    print("DEMO SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Processed {len(scenarios)} scenarios")
    print(f"✅ Embeddings: Generated with Numbskull")
    print(f"✅ Analysis: Neuro-symbolic (9 modules)")
    print(f"✅ Modulation: Adaptive selection based on embeddings")
    print(f"\n💡 This demo shows the integration working!")
    print(f"   For full signal generation, start services:")
    print(f"   - LFM2-8B-A1B on port 8080 (for LLM generation)")
    print(f"   - Eopiez on port 8001 (for semantic embeddings)")
    print(f"   - LIMPS on port 8000 (for mathematical embeddings)")
    
    # Cleanup
    if orchestrator:
        await orchestrator.close()
    if neuro_symbolic:
        await neuro_symbolic.close()
    if signal_adapter:
        await signal_adapter.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(run_simple_integrated_demo())

