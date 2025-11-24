#!/usr/bin/env python3
"""
Recursive Cognitive Playground - Interactive
============================================

Interactive playground for recursive self-improving AI system.

Your goal: "Recursive cognitions emerge from each addition to your knowledge base"
- Constant creative hallucination
- Holographic memory reinforcement
- LIMPS mathematical optimization
- Real-time syntax learning
- Self-evolving intelligence

Author: Assistant
License: MIT
"""

import asyncio
import json
import sys
import warnings
from pathlib import Path

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add paths
sys.path.insert(0, str(Path("/home/kill/numbskull")))

from recursive_cognitive_knowledge import RecursiveCognitiveKnowledge

import logging
logging.getLogger('advanced_embedding_pipeline').setLevel(logging.ERROR)
logging.getLogger('enhanced_vector_index').setLevel(logging.ERROR)
logging.getLogger('enhanced_graph_store').setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)


async def interactive_recursive_cognition():
    """
    Interactive mode for recursive cognitive system
    """
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        🧠 RECURSIVE COGNITIVE KNOWLEDGE - INTERACTIVE                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("Goal: Recursive cognitions emerge from each addition to knowledge base")
    print()
    print("Features:")
    print("  • Constant creative generation (controlled hallucination)")
    print("  • Holographic memory reinforcement")
    print("  • Self-evolving knowledge base")
    print("  • Emergent pattern detection")
    print("  • Real-time syntax learning")
    print()
    print("Commands:")
    print("  • Type your input (adds to knowledge base)")
    print("  • 'map' - View cognitive map")
    print("  • 'insights' - Show recent insights")
    print("  • 'patterns' - Show emergent patterns")
    print("  • 'stats' - System statistics")
    print("  • 'exit' - Quit")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    
    # Initialize system
    print("Initializing recursive cognitive system...")
    system = RecursiveCognitiveKnowledge(
        max_recursion_depth=4,  # Deep recursion for emergence
        hallucination_temperature=0.85,  # High creativity
        coherence_threshold=0.55  # Allow more variations
    )
    
    await system.initialize()
    
    print("✅ System ready! Each input triggers recursive cognition...\n")
    
    iteration = 0
    
    try:
        while True:
            print("─" * 70)
            query = input(f"\n🧠 Input [{iteration}]: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Shutting down recursive cognition...")
                break
            
            if query.lower() == 'map':
                cognitive_map = system.get_cognitive_map()
                print("\n🗺️  COGNITIVE MAP:")
                print("━"*70)
                print(json.dumps(cognitive_map, indent=2))
                continue
            
            if query.lower() == 'insights':
                print(f"\n💡 RECENT INSIGHTS ({len(system.insights)} total):")
                print("━"*70)
                for i, insight in enumerate(system.insights[-10:], 1):
                    print(f"{i}. [{insight.recursion_level}] {insight.content[:60]}...")
                    print(f"   Reinforcements: {insight.reinforcement_count}")
                continue
            
            if query.lower() == 'patterns':
                print(f"\n✨ EMERGENT PATTERNS:")
                print("━"*70)
                for pattern, count in system.emergent_patterns.items():
                    print(f"  {pattern}: {count} occurrences")
                if not system.emergent_patterns:
                    print("  (None yet - keep adding inputs!)")
                continue
            
            if query.lower() == 'stats':
                stats = system.get_cognitive_map()
                print(f"\n📊 SYSTEM STATISTICS:")
                print("━"*70)
                print(f"  Recursion depth: {stats['cognitive_state']['recursion_depth']}")
                print(f"  Total insights: {stats['cognitive_state']['total_insights']}")
                print(f"  Knowledge nodes: {stats['cognitive_state']['knowledge_nodes']}")
                print(f"  Pattern reinforcements: {stats['cognitive_state']['pattern_reinforcements']}")
                print(f"  Hallucination coherence: {stats['cognitive_state']['hallucination_coherence']:.1%}")
                print(f"  Emergent patterns: {stats['cognitive_state']['emergent_patterns']}")
                print(f"  Cognitive loops: {stats['cognitive_state']['cognitive_loops']}")
                continue
            
            # PROCESS WITH RECURSIVE COGNITION
            print(f"\n🌀 Processing recursively...")
            
            result = await system.process_with_recursion(query)
            
            # Display results
            print(f"\n📊 RECURSIVE RESULTS:")
            print("━"*70)
            
            state = result['cognitive_state']
            print(f"✅ Recursion depth reached: {state['recursion_depth']}")
            print(f"✅ Total insights generated: {state['total_insights']}")
            print(f"✅ Knowledge nodes created: {state['knowledge_nodes']}")
            print(f"✅ Hallucination coherence: {state['hallucination_coherence']:.1%}")
            
            if result['synthesis']:
                print(f"\n💡 Emergent Synthesis:")
                print(f"   {result['synthesis']}")
            
            if result['syntax_learned']:
                print(f"\n🧠 Syntax Learned:")
                for learned in result['syntax_learned']:
                    print(f"   • {learned}")
            
            # Show what emerged
            if result['analysis'].get('generated_insights'):
                print(f"\n💭 Generated Variations:")
                for var in result['analysis']['generated_insights'][:3]:
                    print(f"   [{var['coherence']:.2f}] {var['text']}")
            
            if result['analysis'].get('emergent_patterns'):
                print(f"\n✨ Emergent Patterns Detected:")
                for pattern in result['analysis']['emergent_patterns']:
                    print(f"   • {pattern}")
            
            print(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
            
            iteration += 1
            
            # Show evolution
            if iteration % 5 == 0:
                print(f"\n🌀 SYSTEM EVOLUTION (after {iteration} inputs):")
                print(f"   Total knowledge: {state['knowledge_nodes']} nodes")
                print(f"   System coherence: {state['hallucination_coherence']:.1%}")
                print(f"   The system is evolving! Keep adding inputs...")
    
    finally:
        await system.close()
        print(f"\n✅ Final State:")
        print(f"   {system.state.total_insights} total insights")
        print(f"   {system.state.knowledge_nodes} knowledge nodes")
        print(f"   {system.state.hallucination_coherence:.1%} coherence")
        print(f"\n🌀 Recursive cognition session complete!")


if __name__ == "__main__":
    try:
        asyncio.run(interactive_recursive_cognition())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")

