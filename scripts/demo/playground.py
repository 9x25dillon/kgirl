#!/usr/bin/env python3
"""
Interactive Playground
======================

Play with your integrated LiMp + Numbskull system!

Quick commands you can try:
  analyze("your text") - Neuro-symbolic analysis with embeddings
  embed("text") - Generate embeddings
  search("query") - Search knowledge base
  add_knowledge("text", "tag") - Add to knowledge base
  modulate("message") - Select modulation scheme
  
Author: Assistant
"""

import asyncio
import sys
from pathlib import Path

# Setup
numbskull_path = Path('/home/kill/numbskull')
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

from numbskull_dual_orchestrator import create_numbskull_orchestrator
from neuro_symbolic_numbskull_adapter import NeuroSymbolicNumbskullAdapter
from signal_processing_numbskull_adapter import SignalProcessingNumbskullAdapter
from enhanced_vector_index import EnhancedVectorIndex
from enhanced_graph_store import EnhancedGraphStore

print('\n' + '='*70)
print('🎮 LIMP + NUMBSKULL PLAYGROUND 🎮')
print('='*70)
print('\nInitializing...\n')

# Global components
orchestrator = None
neuro = None
signal_proc = None
vector_index = None
graph = None

async def init_playground():
    global orchestrator, neuro, signal_proc, vector_index, graph
    
    # Simple config without description fields
    orchestrator = create_numbskull_orchestrator(
        local_configs=[{
            'base_url': 'http://127.0.0.1:8080',
            'mode': 'llama-cpp',
            'model': 'LFM2-8B-A1B'
        }],
        settings={'use_numbskull': True, 'use_fractal': True},
        numbskull_config={'use_fractal': True, 'cache_embeddings': True}
    )
    
    neuro = NeuroSymbolicNumbskullAdapter(
        use_numbskull=True,
        numbskull_config={'use_fractal': True}
    )
    
    signal_proc = SignalProcessingNumbskullAdapter(
        use_numbskull=True,
        numbskull_config={'use_fractal': True}
    )
    
    vector_index = EnhancedVectorIndex(use_numbskull=True)
    graph = EnhancedGraphStore(use_numbskull=True)
    
    print('✅ All systems ready!\n')

asyncio.run(init_playground())

# Helper functions for interactive use
async def embed(text):
    """Generate embeddings for text"""
    return await orchestrator._generate_embeddings(text)

async def analyze(text):
    """Neuro-symbolic analysis with embeddings"""
    return await neuro.analyze_with_embeddings(text, enable_all_modules=True)

async def modulate(text):
    """Select modulation scheme based on embeddings"""
    return await signal_proc.select_modulation_from_embedding(text)

async def add_knowledge(text, tag="general"):
    """Add text to knowledge base"""
    doc_id = f"doc_{hash(text) % 10000}"
    await vector_index.add_entry(doc_id, text, {"tag": tag})
    return f"Added as {doc_id}"

async def search(query, k=5):
    """Search knowledge base"""
    results = await vector_index.search(query, top_k=k)
    return [(entry.text[:60], score) for entry, score in results]

async def add_concept(id, label, content):
    """Add concept to knowledge graph"""
    await graph.add_node(id, label, content)
    return f"Added node: {id}"

async def find_related(query, k=3):
    """Find related concepts in graph"""
    results = await graph.find_similar_nodes(query, top_k=k)
    return [(node.id, node.label, score) for node, score in results]

# Show examples
print('='*70)
print('🎮 PLAYGROUND READY! Try these:')
print('='*70)
print()
print('# Generate embeddings:')
print('  result = await embed("Quantum computing is revolutionary")')
print()
print('# Analyze text:')
print('  analysis = await analyze("Machine learning learns from data")')
print('  print(analysis["insights"])')
print()
print('# Select modulation:')
print('  scheme, info = await modulate("Emergency message")')
print('  print(f"Use {scheme.name}")')
print()
print('# Build knowledge base:')
print('  await add_knowledge("AI is transforming technology", "AI")')
print('  results = await search("artificial intelligence")')
print('  print(results)')
print()
print('# Build knowledge graph:')
print('  await add_concept("ai", "Technology", "Artificial intelligence")')
print('  await add_concept("ml", "Technology", "Machine learning")')
print('  related = await find_related("deep learning")')
print('  print(related)')
print()
print('='*70)
print()
print('💡 Copy and paste these into a Python async REPL!')
print('   Or use: python -m asyncio')
print()
print('Or just run commands one at a time:')
print('  python -c "import asyncio; exec(open(\'playground.py\').read()); print(asyncio.run(embed(\'test\')))"')
print()

