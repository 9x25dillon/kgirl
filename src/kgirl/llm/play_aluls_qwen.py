#!/usr/bin/env python3
"""
Interactive AL-ULS + Multi-LLM Playground
==========================================

Play with:
- AL-ULS symbolic evaluation (SUM, MEAN, VAR, STD, MIN, MAX, PROD)
- Numbskull embeddings (fractal, semantic, mathematical)
- Multi-LLM inference (LFM2, Qwen, Qwen-Coder)

Usage:
  python play_aluls_qwen.py

Then edit this file to try different queries!
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/kill/numbskull')))

from enable_aluls_and_qwen import MultiLLMOrchestrator

async def quick_demo():
    """Quick interactive demo"""
    
    print("\n" + "="*70)
    print("🎮 AL-ULS + MULTI-LLM PLAYGROUND")
    print("="*70)
    
    # Configure LLMs (add/remove/change ports as needed)
    llm_configs = [
        {
            "base_url": "http://127.0.0.1:8080",
            "mode": "llama-cpp",
            "model": "LFM2-8B-A1B",
            "timeout": 60
        },
        {
            "base_url": "http://127.0.0.1:8081",
            "mode": "openai-chat",
            "model": "Qwen2.5-7B",
            "timeout": 60
        }
    ]
    
    # Initialize system
    system = MultiLLMOrchestrator(
        llm_configs=llm_configs,
        enable_aluls=True,
        numbskull_config={'use_fractal': True}
    )
    
    # =========================================================================
    # 🎯 EDIT THESE TO TRY DIFFERENT QUERIES!
    # =========================================================================
    
    queries = [
        # Symbolic math expressions
        "SUM(100, 200, 300, 400, 500)",
        "MEAN(5, 10, 15, 20, 25)",
        "STD(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)",
        
        # Regular text queries (will use LLM if server is running)
        "Explain neural networks in simple terms",
        "What is the difference between AI and ML?",
    ]
    
    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        
        result = await system.process_with_symbolic(query)
        
        # Show symbolic result
        if result.get("symbolic_result"):
            sr = result["symbolic_result"]
            if sr.get("ok"):
                print(f"✅ Symbolic: {sr['function']}(...) = {sr['result']:.2f}")
            else:
                print(f"⚠️  Symbolic error: {sr.get('error', 'unknown')}")
        
        # Show embeddings
        if result.get("embeddings"):
            emb = result["embeddings"]
            print(f"✅ Embeddings: {emb['components']} (dim: {emb['dimension']})")
        
        # Show LLM response
        if result.get("llm_response"):
            resp = result["llm_response"]
            if len(resp) > 100:
                print(f"🤖 LLM: {resp[:100]}...")
            else:
                print(f"🤖 LLM: {resp}")
    
    # Cleanup
    await system.close()
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n💡 TO PLAY MORE:")
    print("   1. Edit queries list in play_aluls_qwen.py")
    print("   2. Run: python play_aluls_qwen.py")
    print("\n🚀 TO ENABLE LLM INFERENCE:")
    print("   • Terminal 1: bash start_lfm2.sh  (configure first!)")
    print("   • Terminal 2: bash start_qwen.sh  (configure first!)")
    print()


async def custom_query(query: str, context: str = None):
    """
    Run a single custom query
    
    Usage:
        asyncio.run(custom_query("SUM(1,2,3,4,5)"))
    """
    system = MultiLLMOrchestrator(
        llm_configs=[{"base_url": "http://127.0.0.1:8080", "mode": "llama-cpp", "model": "LFM2"}],
        enable_aluls=True
    )
    
    result = await system.process_with_symbolic(query, context)
    
    print("\n" + "="*70)
    print(f"Query: {query}")
    print("="*70)
    
    if result.get("symbolic_result") and result["symbolic_result"].get("ok"):
        print(f"✅ Result: {result['symbolic_result']['result']}")
    
    if result.get("embeddings"):
        print(f"✅ Embeddings: {result['embeddings']['components']}")
    
    if result.get("llm_response"):
        print(f"🤖 Response: {result['llm_response'][:200]}...")
    
    await system.close()
    return result


if __name__ == "__main__":
    # Run the quick demo
    asyncio.run(quick_demo())
    
    # Or uncomment to run a custom query:
    # asyncio.run(custom_query("What is quantum computing?"))

