#!/usr/bin/env python3
import asyncio, sys
from pathlib import Path
from kgirl.paths import find_repo
sys.path.insert(0, str(find_repo("numbskull")))

from kgirl.cognitive.neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from kgirl.llm.signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from kgirl.neural.enhanced_vector_index import EnhancedVectorIndex

async def main():
    print('\n🎮 Quick Playground - Ready to Use!\n')
    
    # Init
    neuro = NeuroSymbolicNumbskullAdapter(use_numbskull=True, numbskull_config={'use_fractal': True})
    signal = SignalProcessingNumbskullAdapter(use_numbskull=True, numbskull_config={'use_fractal': True})
    vector = EnhancedVectorIndex(use_numbskull=True)
    
    print('✅ Systems loaded\n')
    print('='*70)
    print('TRY THESE EXAMPLES (modify the text to play!):')
    print('='*70)
    
    # Example 1: Analyze text
    print('\n1️⃣  Analyzing: "Quantum computing uses superposition"')
    result = await neuro.analyze_with_embeddings("Quantum computing uses superposition", enable_all_modules=True)
    print(f'   Modules: {len(result["modules"])}')
    print(f'   Insight: {result["insights"][0] if result["insights"] else "N/A"}')
    
    # Example 2: Signal modulation
    print('\n2️⃣  Selecting modulation for: "Emergency alert message"')
    scheme, analysis = await signal.select_modulation_from_embedding("Emergency alert message")
    print(f'   Scheme: {scheme.name}')
    print(f'   Reason: {analysis.get("reason", "N/A")[:50]}...')
    
    # Example 3: Knowledge base
    print('\n3️⃣  Building knowledge base...')
    await vector.add_entry("ai1", "Artificial intelligence transforms technology", {"tag": "AI"})
    await vector.add_entry("ml1", "Machine learning learns from data", {"tag": "ML"})
    await vector.add_entry("dl1", "Deep learning uses neural networks", {"tag": "DL"})
    results = await vector.search("neural networks and AI", top_k=2)
    print(f'   Added 3 docs, searched, found {len(results)} matches')
    for entry, score in results:
        print(f'   [{score:.3f}] {entry.text[:50]}')
    
    print('\n'+'='*70)
    print('✅ PLAYGROUND WORKING!')
    print('='*70)
    print('\n💡 To play more: Edit play.py and change the text!')
    print('   Then run: python play.py')
    print('\n📊 Stats:', vector.get_stats())
    
    await neuro.close()
    await signal.close()
    await vector.close()

asyncio.run(main())
