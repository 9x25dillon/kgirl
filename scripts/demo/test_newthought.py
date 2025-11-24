"""
Quick test script for NewThought integration
"""

import asyncio
import sys


async def test_newthought():
    """Test NewThought service integration."""
    print("=" * 60)
    print("NewThought Integration Test")
    print("=" * 60)

    try:
        # Test imports
        print("\n1. Testing imports...")
        from src.chaos_llm.services.newthought import (
            newthought_service,
            QuantumCoherenceEngine,
            SpatialThoughtEncoder,
            RecursiveThoughtGenerator,
            IntegrityValidator,
            HolographicThoughtMemory,
            NewThoughtService,
            Thought,
            ThoughtCascade,
        )
        print("   ✓ All imports successful")

        # Test component initialization
        print("\n2. Testing component initialization...")
        qce = QuantumCoherenceEngine(num_qubits=8, temperature=0.3)
        ste = SpatialThoughtEncoder(embedding_dim=768, spatial_resolution=32)
        rtg = RecursiveThoughtGenerator(max_depth=5, branching_factor=3, coherence_threshold=0.6)
        iv = IntegrityValidator(min_coherence=0.5, max_entropy=0.8)
        htm = HolographicThoughtMemory(memory_size=1000, hologram_dim=768)
        print("   ✓ All components initialized")

        # Test service health
        print("\n3. Testing service health check...")
        health = newthought_service.health_check()
        print(f"   ✓ Service status: {health['status']}")
        print(f"   ✓ Components active: {len(health['components'])}")

        # Test thought generation
        print("\n4. Testing thought generation...")
        result = await newthought_service.generate_new_thought(
            seed_text="Quantum computing leverages superposition for parallel computation",
            depth=3,
            store_in_memory=True,
        )
        print(f"   ✓ Generated {result['thoughts_validated']} validated thoughts")
        print(f"   ✓ Cascade coherence: {result['cascade_coherence']:.3f}")
        print(f"   ✓ Depth reached: {result['depth_reached']}")
        print(f"   ✓ Emergence patterns: {result['emergence_patterns']}")
        print(f"   ✓ Processing time: {result['processing_time']:.3f}s")

        # Test quantum superposition
        print("\n5. Testing quantum superposition...")
        superpose_result = await newthought_service.quantum_superpose_thoughts(
            thought_texts=[
                "Neural networks learn hierarchical representations",
                "Quantum systems exhibit superposition states",
            ]
        )
        print(f"   ✓ Superposed thought generated")
        print(f"   ✓ Coherence: {superpose_result['superposed_thought']['coherence_score']:.3f}")

        # Test memory recall
        print("\n6. Testing memory recall...")
        recall_result = await newthought_service.recall_similar_thoughts(
            query_text="quantum mechanics superposition",
            top_k=3,
            similarity_threshold=0.3,
        )
        print(f"   ✓ Recalled {len(recall_result)} similar thoughts")

        # Test entanglement measurement
        print("\n7. Testing entanglement measurement...")
        entangle_result = await newthought_service.measure_thought_entanglement(
            thought_text_a="Quantum coherence maintains superposition states",
            thought_text_b="Decoherence causes quantum state collapse",
        )
        print(f"   ✓ Quantum entanglement: {entangle_result['quantum_entanglement']:.3f}")
        print(f"   ✓ Classical similarity: {entangle_result['classical_similarity']:.3f}")
        print(f"   ✓ Interpretation: {entangle_result['interpretation']}")

        # Test statistics
        print("\n8. Testing service statistics...")
        stats = newthought_service.get_statistics()
        print(f"   ✓ Total thoughts: {stats['service_stats']['total_thoughts_generated']}")
        print(f"   ✓ Total cascades: {stats['service_stats']['total_cascades']}")
        print(f"   ✓ Avg coherence: {stats['service_stats']['avg_coherence']:.3f}")
        print(f"   ✓ Memory utilization: {stats['memory_stats']['memory_utilization']:.1%}")

        print("\n" + "=" * 60)
        print("✨ All tests passed successfully!")
        print("=" * 60)

        # Display sample thought
        if result['generated_thoughts']:
            print("\nSample Generated Thought:")
            print("-" * 60)
            sample = result['generated_thoughts'][0]
            print(f"Content: {sample['content']}")
            print(f"Depth: {sample['depth']}")
            print(f"Coherence: {sample['coherence_score']:.3f}")
            print(f"Entropy: {sample['entropy']:.3f}")
            print("-" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_newthought())
    sys.exit(0 if success else 1)
