#!/usr/bin/env python3
"""Verify ALL components are working together"""
import sys
from kgirl.paths import find_repo
sys.path.insert(0, str(find_repo("numbskull")))
import asyncio
import requests

print("╔══════════════════════════════════════════════════════════════════════╗")
print("║              🔍 VERIFYING ALL COMPONENTS                             ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print()

# Check services
print("Services:")
print("─"*70)

try:
    r = requests.get('http://localhost:11434/api/tags', timeout=2)
    print("✅ Ollama LLM (port 11434) - RUNNING")
except:
    print("❌ Ollama LLM (port 11434) - NOT RUNNING")

try:
    r = requests.get('http://localhost:8000/health', timeout=2)
    print("✅ LIMPS Mathematical (port 8000) - RUNNING")
except:
    print("❌ LIMPS Mathematical (port 8000) - NOT RUNNING")

print()
print("Components:")
print("─"*70)

# Test each component
async def test_all():
    # 1. AL-ULS
    try:
        from kgirl.llm.enable_aluls_and_qwen import LocalALULSEvaluator
        aluls = LocalALULSEvaluator()
        result = aluls.evaluate(aluls.parse_call("SUM(1,2,3)"))
        print(f"✅ AL-ULS Symbolic: {result['result']}")
    except Exception as e:
        print(f"❌ AL-ULS: {e}")
    
    # 2. Embeddings
    try:
        from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
        config = HybridConfig(use_fractal=True, use_mathematical=True)
        pipeline = HybridEmbeddingPipeline(config)
        result = await pipeline.embed("test")
        print(f"✅ Numbskull Embeddings: {result.get('metadata', {}).get('components_used', [])}")
        await pipeline.close()
    except Exception as e:
        print(f"❌ Embeddings: {e}")
    
    # 3. Matrix processor
    try:
        from kgirl.core.matrix_processor_adapter import matrix_processor
        matrix = matrix_processor.encode_to_matrix([[1,2,3],[4,5,6]])
        print(f"✅ Matrix Processor: shape {matrix.shape}")
    except Exception as e:
        print(f"❌ Matrix Processor: {e}")
    
    # 4. Recursive cognition
    try:
        from kgirl.cognitive.recursive_cognitive_knowledge import RecursiveCognitiveKnowledge
        print("✅ Recursive Cognition: Available")
    except Exception as e:
        print(f"❌ Recursive Cognition: {e}")
    
    # 5. Holographic
    try:
        from kgirl.quantum.holographic_memory_system import HolographicMemorySystem
        print("✅ Holographic Memory: Available")
    except Exception as e:
        print(f"❌ Holographic Memory: {e}")
    
    # 6. CoCo
    try:
        from kgirl.cognitive.CoCo_0rg import CognitiveCommunicationOrganism
        print("✅ CoCo Organism: Available")
    except Exception as e:
        print(f"❌ CoCo Organism: {e}")

asyncio.run(test_all())

print()
print("─"*70)
print("Verification complete!")
