"""
Standalone test for NewThought module (without full dependencies)
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


async def test_newthought_standalone():
    """Test NewThought service without full dependency chain."""
    print("=" * 60)
    print("NewThought Standalone Test")
    print("=" * 60)

    try:
        # Test direct import
        print("\n1. Testing direct import...")
        from chaos_llm.services.newthought import (
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
        print("   ✓ Module import successful")

        # Test component initialization
        print("\n2. Testing component initialization...")
        qce = QuantumCoherenceEngine(num_qubits=8, temperature=0.3)
        print("   ✓ QuantumCoherenceEngine initialized")

        ste = SpatialThoughtEncoder(embedding_dim=768, spatial_resolution=32)
        print("   ✓ SpatialThoughtEncoder initialized")

        rtg = RecursiveThoughtGenerator(max_depth=5, branching_factor=3, coherence_threshold=0.6)
        print("   ✓ RecursiveThoughtGenerator initialized")

        iv = IntegrityValidator(min_coherence=0.5, max_entropy=0.8)
        print("   ✓ IntegrityValidator initialized")

        htm = HolographicThoughtMemory(memory_size=1000, hologram_dim=768)
        print("   ✓ HolographicThoughtMemory initialized")

        # Test service initialization
        print("\n3. Testing service initialization...")
        service = NewThoughtService(
            embedding_dim=768,
            max_recursion_depth=5,
            coherence_threshold=0.6,
            memory_size=1000,
        )
        print("   ✓ NewThoughtService initialized")

        # Test service health
        print("\n4. Testing service health check...")
        health = service.health_check()
        print(f"   ✓ Service status: {health['status']}")
        print(f"   ✓ Components: {', '.join(health['components'].keys())}")
        print(f"   ✓ Memory utilization: {health['memory_utilization']:.1%}")

        # Test quantum coherence engine
        print("\n5. Testing QuantumCoherenceEngine...")
        import numpy as np
        from kgirl import telemetry  # real NuRea_sim lattice data; chaos fallback

        test_vector = telemetry.lattice_signal(768)
        test_vector = test_vector / np.linalg.norm(test_vector)

        recovered = qce.coherence_recovery(test_vector, noise_threshold=0.1)
        print(f"   ✓ Coherence recovery applied")
        print(f"   ✓ Output vector shape: {recovered.shape}")

        thoughts = ["Quantum computing uses qubits", "Neural networks use weights"]
        superposed = qce.quantum_superposition(thoughts)
        print(f"   ✓ Quantum superposition: '{superposed[:50]}...'")

        vector_a = telemetry.lattice_signal(768)
        vector_b = telemetry.lattice_signal(768, target="pppf", offset=768)
        vector_a /= np.linalg.norm(vector_a)
        vector_b /= np.linalg.norm(vector_b)
        entanglement = qce.entanglement_measure(vector_a, vector_b)
        print(f"   ✓ Entanglement measure: {entanglement:.3f}")

        # Test spatial encoder
        print("\n6. Testing SpatialThoughtEncoder...")
        embedding = ste.spatial_encode("Quantum computing leverages superposition", preserve_locality=True)
        print(f"   ✓ Spatial encoding shape: {embedding.shape}")
        print(f"   ✓ Embedding norm: {np.linalg.norm(embedding):.3f}")

        locality_score = ste.locality_preservation_score(embedding, recovered)
        print(f"   ✓ Locality preservation score: {locality_score:.3f}")

        projected = ste.dimensional_projection(embedding, target_dim=256)
        print(f"   ✓ Dimensional projection: {embedding.shape} -> {projected.shape}")

        # Test thought generation
        print("\n7. Testing thought generation...")
        result = await service.generate_new_thought(
            seed_text="Quantum entanglement connects particles across space",
            depth=3,
            store_in_memory=True,
        )
        print(f"   ✓ Generated {result['thoughts_validated']} validated thoughts")
        print(f"   ✓ Cascade coherence: {result['cascade_coherence']:.3f}")
        print(f"   ✓ Depth reached: {result['depth_reached']}")
        print(f"   ✓ Emergence patterns: {result['emergence_patterns']}")
        print(f"   ✓ Processing time: {result['processing_time']:.3f}s")
        print(f"   ✓ Total entropy: {result['total_entropy']:.3f}")

        # Test superposition
        print("\n8. Testing quantum superposition...")
        superpose_result = await service.quantum_superpose_thoughts(
            thought_texts=[
                "Quantum mechanics describes probabilistic behavior",
                "Classical physics follows deterministic laws",
                "Quantum computing exploits superposition states",
            ],
            weights=[0.5, 0.3, 0.2],
        )
        print(f"   ✓ Superposed thought: '{superpose_result['superposed_thought']['content'][:60]}...'")
        print(f"   ✓ Coherence: {superpose_result['superposed_thought']['coherence_score']:.3f}")

        # Test memory recall
        print("\n9. Testing holographic memory recall...")
        recall_result = await service.recall_similar_thoughts(
            query_text="quantum physics superposition entanglement",
            top_k=5,
            similarity_threshold=0.3,
        )
        print(f"   ✓ Recalled {len(recall_result)} similar thoughts")
        if recall_result:
            print(f"   ✓ Top similarity: {recall_result[0]['similarity']:.3f}")

        # Test entanglement measurement
        print("\n10. Testing entanglement measurement...")
        entangle_result = await service.measure_thought_entanglement(
            thought_text_a="Quantum coherence preserves superposition states",
            thought_text_b="Quantum decoherence destroys superposition states",
        )
        print(f"   ✓ Quantum entanglement: {entangle_result['quantum_entanglement']:.3f}")
        print(f"   ✓ Classical similarity: {entangle_result['classical_similarity']:.3f}")
        print(f"   ✓ Interpretation: {entangle_result['interpretation']}")

        # Test statistics
        print("\n11. Testing service statistics...")
        stats = service.get_statistics()
        print(f"   ✓ Total thoughts: {stats['service_stats']['total_thoughts_generated']}")
        print(f"   ✓ Total cascades: {stats['service_stats']['total_cascades']}")
        print(f"   ✓ Avg coherence: {stats['service_stats']['avg_coherence']:.3f}")
        print(f"   ✓ Emergence patterns: {stats['service_stats']['emergence_patterns_detected']}")
        print(f"   ✓ Memory utilization: {stats['memory_stats']['memory_utilization']:.1%}")

        print("\n" + "=" * 60)
        print("✨ All tests passed successfully!")
        print("=" * 60)

        # Display sample thoughts
        if result['generated_thoughts']:
            print("\nSample Generated Thoughts:")
            print("-" * 60)
            for i, thought in enumerate(result['generated_thoughts'][:3], 1):
                print(f"\n[Thought {i}]")
                print(f"Content: {thought['content']}")
                print(f"Depth: {thought['depth']}")
                print(f"Coherence: {thought['coherence_score']:.3f}")
                print(f"Entropy: {thought['entropy']:.3f}")
            print("-" * 60)

        print("\n🎉 NewThought is ready for Hugging Face integration!")
        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMissing dependency. Install with:")
        print("  pip install numpy")
        return False

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_newthought_standalone())
    sys.exit(0 if success else 1)
