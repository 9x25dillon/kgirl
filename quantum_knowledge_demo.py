#!/usr/bin/env python3
"""
Quick Demo of Quantum Holographic Knowledge Synthesis System
=============================================================

This demo shows how to:
1. Ingest various data sources
2. Generate quantum knowledge representations
3. Query the database
4. Explore emergent patterns
5. Analyze qualia encodings
"""

import asyncio
from quantum_knowledge_database import QuantumHolographicKnowledgeDatabase

async def demo():
    print("\n" + "="*80)
    print("🌌 QUANTUM HOLOGRAPHIC KNOWLEDGE SYNTHESIS - DEMO 🌌")
    print("="*80 + "\n")

    # Create database
    print("📦 Initializing quantum database...")
    db = QuantumHolographicKnowledgeDatabase(
        db_path="demo_quantum_knowledge.db",
        enable_numbskull=False,  # Set to True if Numbskull is available
        embedding_dimension=768,
        quantum_dimension=256
    )

    print("\n" + "="*80)
    print("TEST 1: Ingesting Text Input")
    print("="*80)

    # Test 1: Ingest text
    text1 = """
    Quantum entanglement is a physical phenomenon that occurs when a group of
    particles are generated, interact, or share spatial proximity in a way such
    that the quantum state of each particle of the group cannot be described
    independently of the state of the others, including when the particles are
    separated by a large distance.
    """

    quantum1 = await db.ingest_and_process(text1, source_type="text")
    print(f"\n✅ Created Quantum: {quantum1.quantum_id}")
    print(f"   Qualia Type: {quantum1.qualia_encoding.qualia_type.value}")
    print(f"   Consciousness Level: {quantum1.qualia_encoding.consciousness_level:.3f}")
    print(f"   Coherence Resonance: {quantum1.coherence_resonance:.3f}")
    print(f"   Chaos Attractor: {quantum1.chaos_ragged_state.attractor_basin}")
    print(f"   Emergent Patterns: {len(quantum1.emergent_patterns)}")

    print("\n" + "="*80)
    print("TEST 2: Ingesting Algorithmic Equation")
    print("="*80)

    # Test 2: Ingest equation
    equation = "E = mc^2"
    context = "Einstein's mass-energy equivalence relation"

    quantum2 = await db.ingest_and_process(
        f"EQUATION: {equation}\nCONTEXT: {context}",
        source_type="equation"
    )

    print(f"\n✅ Created Quantum: {quantum2.quantum_id}")
    print(f"   Qualia Type: {quantum2.qualia_encoding.qualia_type.value}")
    print(f"   Phenomenal Properties:")
    for key, value in list(quantum2.qualia_encoding.phenomenal_properties.items())[:3]:
        print(f"     - {key}: {value}")

    print("\n" + "="*80)
    print("TEST 3: Ingesting Complex Text with Fractals")
    print("="*80)

    text3 = """
    Fractals are infinitely complex patterns that are self-similar across
    different scales. They are created by repeating a simple process over
    and over in an ongoing feedback loop. The Mandelbrot set, perhaps the
    most famous fractal, exhibits this self-similarity at all scales.
    """

    quantum3 = await db.ingest_and_process(text3, source_type="text")

    print(f"\n✅ Created Quantum: {quantum3.quantum_id}")
    print(f"   Holographic Fractal Dimension: {quantum3.holographic_encoding.fractal_dimension:.3f}")
    print(f"   Reconstruction Fidelity: {quantum3.holographic_encoding.reconstruction_fidelity:.3f}")
    print(f"   Orwellian Nested Layers: {len(quantum3.orwells_egged_structure.nested_layers)}")

    print("\n" + "="*80)
    print("TEST 4: Querying the Database")
    print("="*80)

    query = "quantum and fractal patterns"
    results = await db.query_knowledge(query, top_k=3)

    print(f"\n🔍 Query: '{query}'")
    print(f"   Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.quantum_id}")
        print(f"      Source: {result.source_type.value}")
        print(f"      Coherence: {result.coherence_resonance:.3f}")
        print(f"      Qualia: {result.qualia_encoding.qualia_type.value}")
        print(f"      Content: {result.raw_content[:100]}...\n")

    print("=" * 80)
    print("TEST 5: Exploring Emergent Patterns")
    print("=" * 80)

    all_patterns = []
    for quantum in db.knowledge_quanta.values():
        all_patterns.extend(quantum.emergent_patterns)

    print(f"\n🌟 Total Emergent Patterns Detected: {len(all_patterns)}")

    for i, pattern in enumerate(all_patterns[:3], 1):
        print(f"\n   Pattern {i}: {pattern.pattern_id}")
        print(f"      Type: {pattern.pattern_type}")
        print(f"      Emergence Score: {pattern.emergence_score:.3f}")
        print(f"      Complexity: {pattern.complexity_measure:.3f}")
        print(f"      Coherence: {pattern.coherence_score:.3f}")
        print(f"      Properties: {', '.join(pattern.emergent_properties[:2])}")

    print("\n" + "="*80)
    print("TEST 6: Qualia Analysis")
    print("="*80)

    print(f"\nAnalyzing subjective experience of {quantum1.quantum_id}:\n")
    qualia = quantum1.qualia_encoding

    print(f"   Qualia Type: {qualia.qualia_type.value}")
    print(f"   Intentionality: {qualia.intentionality}")
    print(f"   Consciousness Level: {qualia.consciousness_level:.3f}")
    print(f"\n   Phenomenal Properties (How it 'feels'):")
    for key, value in qualia.phenomenal_properties.items():
        print(f"      - {key}: {value}")

    print(f"\n   Emergent Phenomenal Properties:")
    for prop in qualia.emergent_properties:
        print(f"      - {prop}")

    print("\n" + "="*80)
    print("TEST 7: Chaos_Ragged State Analysis")
    print("="*80)

    chaos_state = quantum1.chaos_ragged_state
    print(f"\n   Chaos Entropy: {chaos_state.chaos_entropy:.3f}")
    print(f"   Attractor Basin: {chaos_state.attractor_basin}")
    print(f"   Edge of Chaos: {chaos_state.edge_of_chaos:.3f}")
    print(f"   Bifurcation Points: {len(chaos_state.bifurcation_points)}")
    print(f"   Ragged Boundaries: {len(chaos_state.ragged_boundaries)}")

    print("\n" + "="*80)
    print("TEST 8: Database Statistics")
    print("="*80)

    print(f"\n   Total Knowledge Quanta: {len(db.knowledge_quanta)}")

    coherences = [q.coherence_resonance for q in db.knowledge_quanta.values()]
    import numpy as np
    print(f"   Average Coherence: {np.mean(coherences):.3f}")
    print(f"   Max Coherence: {np.max(coherences):.3f}")

    qualia_types = {}
    for q in db.knowledge_quanta.values():
        qt = q.qualia_encoding.qualia_type.value
        qualia_types[qt] = qualia_types.get(qt, 0) + 1

    print(f"\n   Qualia Types Distribution:")
    for qtype, count in qualia_types.items():
        print(f"      - {qtype}: {count}")

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Run: python quantum_llm_interface.py")
    print("  2. Try: ingest <your_file_or_directory>")
    print("  3. Try: query <your_question>")
    print("  4. Try: analyze <quantum_id>")
    print("  5. Read: QUANTUM_KNOWLEDGE_README.md")
    print("\n" + "="*80 + "\n")

    await db.close()


if __name__ == "__main__":
    asyncio.run(demo())
