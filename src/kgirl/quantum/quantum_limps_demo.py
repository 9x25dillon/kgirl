#!/usr/bin/env python3
"""
Quantum-LIMPS Integration Demonstration
Shows comprehensive usage of the integrated Quantum Knowledge + LIMPS system
"""

import asyncio
import logging
import sys
from pathlib import Path
import numpy as np
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from quantum_limps_integration import (
    QuantumLIMPSIntegration,
    QuantumLIMPSConfig,
    optimize_quantum_knowledge,
    query_quantum_limps,
    create_quantum_limps_integration
)


async def demo_1_system_status():
    """Demo 1: Check system status and capabilities"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 1: System Status and Capabilities")
    logger.info("=" * 70)

    integration = create_quantum_limps_integration(use_gpu=True, debug=False)
    status = integration.get_system_status()

    logger.info("\nSystem Configuration:")
    logger.info(f"  Quantum DB: {'✓' if status['quantum_db_initialized'] else '✗'}")
    logger.info(f"  LIMPS Framework: {'✓' if status['limps_available'] else '✗'}")
    logger.info(f"  Matrix Optimization: {'✓' if status['matrix_optimization_enabled'] else '✗'}")
    logger.info(f"  Entropy Analysis: {'✓' if status['entropy_analysis_enabled'] else '✗'}")
    logger.info(f"  GPU Acceleration: {'✓' if status['gpu_available'] else '✗'}")

    if status['gpu_device']:
        logger.info(f"  GPU Device: {status['gpu_device']}")

    if 'gpu_memory' in status:
        memory = status['gpu_memory']
        if 'gpu_allocated_gb' in memory:
            logger.info(f"  GPU Memory: {memory['gpu_allocated_gb']:.2f} GB allocated")


async def demo_2_text_optimization():
    """Demo 2: Optimize text-based knowledge"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 2: Text Knowledge Optimization")
    logger.info("=" * 70)

    # Create sample quantum physics text
    quantum_text = """
    Quantum Mechanics: Fundamental Principles

    The quantum realm operates under fundamentally different rules than classical physics.
    Key principles include:

    1. Wave-Particle Duality: Particles exhibit both wave and particle characteristics.
       The famous double-slit experiment demonstrates this counterintuitive behavior.

    2. Uncertainty Principle: Formulated by Heisenberg, this principle states that
       certain pairs of properties (like position and momentum) cannot be simultaneously
       measured with arbitrary precision. Δx·Δp ≥ ℏ/2

    3. Quantum Superposition: A quantum system can exist in multiple states simultaneously
       until measured. Schrödinger's cat thought experiment illustrates this concept.

    4. Quantum Entanglement: Einstein called it "spooky action at a distance." When
       particles become entangled, measuring one instantly affects the other, regardless
       of the distance separating them.

    5. Quantum Tunneling: Particles can pass through energy barriers that classical
       physics would deem impenetrable. This phenomenon is crucial for nuclear fusion
       in stars and modern electronics.

    These principles form the foundation of quantum computing, quantum cryptography,
    and our understanding of the universe at its most fundamental level.
    """

    # Write to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(quantum_text)
        temp_file = f.name

    try:
        logger.info(f"\nIngesting quantum physics knowledge from text file...")
        logger.info(f"Text length: {len(quantum_text)} characters")

        # Optimize using convenience function
        optimized_state = await optimize_quantum_knowledge(
            temp_file,
            config=QuantumLIMPSConfig(use_gpu=True, debug=False)
        )

        logger.info("\n📊 Optimization Results:")
        logger.info(f"  ├─ Compression Ratio: {optimized_state.compression_ratio:.2%}")
        logger.info(f"  ├─ Complexity Score: {optimized_state.complexity_score:.4f}")
        logger.info(f"  ├─ Optimization Time: {optimized_state.optimization_time:.2f}s")
        logger.info(f"  ├─ Optimized Embeddings: {len(optimized_state.optimized_embeddings)}")
        logger.info(f"  └─ Coherence Resonance: {optimized_state.original_quantum.coherence_resonance:.3f}")

        # Show entropy metrics
        if optimized_state.entropy_metrics:
            logger.info("\n🔬 Entropy Analysis:")
            agg = optimized_state.entropy_metrics.get("aggregate", {})
            logger.info(f"  ├─ Mean Complexity: {agg.get('mean_complexity', 0):.4f}")
            logger.info(f"  ├─ Mean Entropy: {agg.get('mean_entropy', 0):.4f}")
            logger.info(f"  └─ Total Dimensions: {agg.get('total_dimensionality', 0)}")

        # Show emergent patterns
        quantum = optimized_state.original_quantum
        if quantum.emergent_patterns:
            logger.info(f"\n✨ Emergent Patterns Detected: {len(quantum.emergent_patterns)}")
            for i, pattern in enumerate(quantum.emergent_patterns[:3], 1):
                logger.info(f"  Pattern {i}: {pattern.pattern_type.value} (emergence: {pattern.emergence_score:.3f})")

        return optimized_state

    finally:
        import os
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def demo_3_code_optimization():
    """Demo 3: Optimize code-based knowledge"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 3: Code Knowledge Optimization")
    logger.info("=" * 70)

    # Create sample Python code
    python_code = """
#!/usr/bin/env python3
\"\"\"
Fractal Generator: Mandelbrot Set Implementation
Demonstrates chaos theory and fractal mathematics
\"\"\"

import numpy as np
import matplotlib.pyplot as plt

class MandelbrotGenerator:
    def __init__(self, width=800, height=600, max_iterations=256):
        self.width = width
        self.height = height
        self.max_iterations = max_iterations

    def compute_point(self, c):
        \"\"\"Compute Mandelbrot iterations for complex point c\"\"\"
        z = 0
        for n in range(self.max_iterations):
            if abs(z) > 2:
                return n
            z = z*z + c
        return self.max_iterations

    def generate(self, x_min=-2.5, x_max=1.5, y_min=-2, y_max=2):
        \"\"\"Generate full Mandelbrot set\"\"\"
        x = np.linspace(x_min, x_max, self.width)
        y = np.linspace(y_min, y_max, self.height)

        mandelbrot = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                c = complex(x[j], y[i])
                mandelbrot[i, j] = self.compute_point(c)

        return mandelbrot

    def visualize(self, mandelbrot_set):
        \"\"\"Visualize the fractal\"\"\"
        plt.figure(figsize=(12, 9))
        plt.imshow(mandelbrot_set, extent=[-2.5, 1.5, -2, 2],
                   cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Iterations to escape')
        plt.title('Mandelbrot Set Fractal')
        plt.xlabel('Real axis')
        plt.ylabel('Imaginary axis')
        return plt

# Golden ratio Fibonacci spiral
def fibonacci_spiral(n_terms=10):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    fibonacci = [0, 1]

    for i in range(2, n_terms):
        fibonacci.append(fibonacci[i-1] + fibonacci[i-2])

    # Calculate spiral coordinates
    angles = np.linspace(0, 2*np.pi*n_terms, 1000)
    r = phi ** (angles / (2*np.pi))

    x = r * np.cos(angles)
    y = r * np.sin(angles)

    return x, y, fibonacci

if __name__ == "__main__":
    generator = MandelbrotGenerator()
    fractal = generator.generate()
    generator.visualize(fractal)

    print("Fractal dimensions:", fractal.shape)
    print("Complexity metric:", np.std(fractal))
"""

    # Write to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        temp_file = f.name

    try:
        logger.info(f"\nIngesting fractal generator code...")
        logger.info(f"Code lines: {len(python_code.splitlines())}")

        config = QuantumLIMPSConfig(
            use_gpu=True,
            optimization_method="polynomial",  # Best for mathematical code
            debug=False
        )

        optimized_state = await optimize_quantum_knowledge(temp_file, config=config)

        logger.info("\n📊 Code Optimization Results:")
        logger.info(f"  ├─ Compression: {optimized_state.compression_ratio:.2%}")
        logger.info(f"  ├─ Mathematical Complexity: {optimized_state.complexity_score:.4f}")
        logger.info(f"  ├─ Processing Time: {optimized_state.optimization_time:.2f}s")
        logger.info(f"  └─ Fractal Completion: {optimized_state.original_quantum.fractal_completion:.3f}")

        # Show qualia encoding (how the code "feels" to the AI)
        quantum = optimized_state.original_quantum
        if quantum.qualia_encoding:
            qualia = quantum.qualia_encoding
            logger.info(f"\n🧠 Qualia Encoding (Subjective Experience):")
            logger.info(f"  ├─ Type: {qualia.qualia_type.value}")
            logger.info(f"  ├─ Intensity: {qualia.phenomenal_properties.get('intensity', 0):.3f}")
            logger.info(f"  ├─ Clarity: {qualia.phenomenal_properties.get('clarity', 0):.3f}")
            logger.info(f"  └─ Consciousness Level: {qualia.consciousness_level:.3f}")

        return optimized_state

    finally:
        import os
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def demo_4_query_system():
    """Demo 4: Query the knowledge base with natural language"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 4: Natural Language Querying")
    logger.info("=" * 70)

    queries = [
        "quantum entanglement and superposition",
        "fractal patterns in chaos theory",
        "golden ratio and Fibonacci sequences",
        "Heisenberg uncertainty principle"
    ]

    config = QuantumLIMPSConfig(use_gpu=True, debug=False)

    for query in queries:
        logger.info(f"\n🔍 Query: '{query}'")

        try:
            results = await query_quantum_limps(query, config=config)

            logger.info(f"  └─ Found {results['num_results']} relevant knowledge quanta")

            for i, result in enumerate(results['results'][:2], 1):
                logger.info(f"      Result {i}:")
                logger.info(f"        ├─ Type: {result['source_type']}")
                logger.info(f"        ├─ Coherence: {result['coherence_resonance']:.3f}")
                logger.info(f"        ├─ Complexity: {result['complexity']:.4f}")
                logger.info(f"        └─ Emergent Patterns: {result['emergent_patterns']}")

        except Exception as e:
            logger.error(f"  └─ Query failed: {e}")


async def demo_5_batch_processing():
    """Demo 5: Batch process multiple knowledge sources"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 5: Batch Knowledge Optimization")
    logger.info("=" * 70)

    # Create multiple sample files
    samples = {
        "physics.txt": "Einstein's theory of relativity revolutionized our understanding of space and time.",
        "math.txt": "The Riemann hypothesis concerns the distribution of prime numbers.",
        "ai.txt": "Machine learning algorithms learn patterns from data without explicit programming."
    }

    temp_files = []
    import tempfile

    try:
        # Create temporary files
        for filename, content in samples.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_files.append(f.name)

        logger.info(f"\n📦 Batch processing {len(temp_files)} knowledge sources...")

        # Batch optimize
        config = QuantumLIMPSConfig(use_gpu=True, debug=False)
        integration = QuantumLIMPSIntegration(config)

        optimized_states = integration.batch_optimize(temp_files)

        logger.info("\n📊 Batch Results:")
        for i, state in enumerate(optimized_states, 1):
            logger.info(f"  Source {i}:")
            logger.info(f"    ├─ Compression: {state.compression_ratio:.2%}")
            logger.info(f"    ├─ Complexity: {state.complexity_score:.4f}")
            logger.info(f"    └─ Time: {state.optimization_time:.2f}s")

        total_time = sum(s.optimization_time for s in optimized_states)
        avg_compression = np.mean([s.compression_ratio for s in optimized_states])

        logger.info(f"\n  Total Time: {total_time:.2f}s")
        logger.info(f"  Average Compression: {avg_compression:.2%}")

    finally:
        # Cleanup
        import os
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


async def demo_6_export_reports():
    """Demo 6: Generate and export optimization reports"""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 6: Optimization Report Generation")
    logger.info("=" * 70)

    # Create sample data
    sample_text = "Chaos theory studies the behavior of dynamical systems that are highly sensitive to initial conditions."

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name

    try:
        logger.info("\nGenerating comprehensive optimization report...")

        config = QuantumLIMPSConfig(use_gpu=True, debug=False)
        integration = QuantumLIMPSIntegration(config)

        optimized_state = await integration.ingest_and_optimize(temp_file)

        # Export report
        report_path = "quantum_limps_report.json"
        report = integration.export_optimization_report(optimized_state, report_path)

        logger.info(f"\n✅ Report exported to: {report_path}")
        logger.info(f"\n📄 Report Summary:")
        logger.info(f"  ├─ Quantum ID: {report['quantum_id'][:16]}...")
        logger.info(f"  ├─ Source Type: {report['source_type']}")
        logger.info(f"  ├─ Compression: {report['compression_ratio']:.2%}")
        logger.info(f"  ├─ Complexity: {report['complexity_score']:.4f}")
        logger.info(f"  └─ Optimized Embeddings: {len(report['optimized_embeddings'])}")

        # Show embedding details
        logger.info(f"\n  Embedding Details:")
        for key, details in report['optimized_embeddings'].items():
            logger.info(f"    {key}:")
            logger.info(f"      ├─ Shape: {details['shape']}")
            logger.info(f"      ├─ Size: {details['size']}")
            logger.info(f"      └─ Mean: {details['mean']:.4f}")

        return report_path

    finally:
        import os
        if os.path.exists(temp_file):
            os.unlink(temp_file)


async def run_all_demos():
    """Run all demonstration scenarios"""
    logger.info("\n" + "=" * 70)
    logger.info("QUANTUM-LIMPS INTEGRATION: COMPREHENSIVE DEMONSTRATION")
    logger.info("=" * 70)

    try:
        # Demo 1: System Status
        await demo_1_system_status()

        # Demo 2: Text Optimization
        await demo_2_text_optimization()

        # Demo 3: Code Optimization
        await demo_3_code_optimization()

        # Demo 4: Querying
        await demo_4_query_system()

        # Demo 5: Batch Processing
        await demo_5_batch_processing()

        # Demo 6: Report Generation
        await demo_6_export_reports()

        logger.info("\n" + "=" * 70)
        logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\nDemo failed with error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_demos())
    sys.exit(exit_code)
