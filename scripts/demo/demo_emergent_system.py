#!/usr/bin/env python3
"""
Demonstration of Emergent Cognitive Network Infrastructure
=========================================================
This script demonstrates the capabilities of the emergent cognitive network
system through various scenarios and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from emergent_cognitive_network import (
    QuantumInspiredOptimizer,
    SwarmCognitiveNetwork,
    NeuromorphicProcessor,
    HolographicDataEngine,
    MorphogeneticSystem,
    EmergentTechnologyOrchestrator
)
import time

def demonstrate_quantum_optimization():
    """Demonstrate quantum-inspired optimization capabilities"""
    print("🔮 Quantum-Inspired Optimization Demo")
    print("=" * 50)
    
    optimizer = QuantumInspiredOptimizer(num_qubits=8)
    
    # Define a complex optimization problem (Rastrigin function)
    def rastrigin_function(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    print("Optimizing Rastrigin function (global minimum at origin)...")
    start_time = time.time()
    result = optimizer.quantum_annealing_optimization(rastrigin_function, max_iter=500)
    end_time = time.time()
    
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Best solution: {result['solution']}")
    print(f"Best cost: {result['cost']:.4f}")
    print(f"Quantum entropy: {result['quantum_entropy']:.4f}")
    print()

def demonstrate_swarm_intelligence():
    """Demonstrate swarm intelligence and emergent behavior"""
    print("🐝 Swarm Intelligence Demo")
    print("=" * 50)
    
    swarm = SwarmCognitiveNetwork(num_agents=30, search_space=(-5, 5))
    
    # Define a multimodal optimization problem
    def multimodal_function(x):
        return np.sum(x**2) + 10 * np.sum(np.cos(2 * np.pi * x))
    
    print("Running swarm optimization with emergent behavior detection...")
    start_time = time.time()
    result = swarm.optimize_swarm(multimodal_function, max_iterations=100)
    end_time = time.time()
    
    print(f"Swarm optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Global best position: {result['global_best']['position']}")
    print(f"Global best cost: {result['global_best']['cost']:.4f}")
    print(f"Final swarm intelligence: {result['swarm_intelligence'][-1]:.4f}")
    print(f"Emergent behaviors detected: {len(result['emergent_behaviors'])}")
    
    if result['emergent_behaviors']:
        print("Emergent behavior patterns:")
        for i, behavior in enumerate(result['emergent_behaviors'][:3]):  # Show first 3
            print(f"  Pattern {i+1}: {behavior['pattern_type']} (coordination: {behavior['coordination_level']:.3f})")
    print()

def demonstrate_neuromorphic_processing():
    """Demonstrate neuromorphic computing capabilities"""
    print("🧠 Neuromorphic Processing Demo")
    print("=" * 50)
    
    processor = NeuromorphicProcessor(num_neurons=500)
    
    # Generate input spikes (simulating sensory input)
    input_spikes = np.random.poisson(0.1, processor.num_neurons)
    
    print("Processing spiking input through neuromorphic network...")
    start_time = time.time()
    result = processor.process_spiking_input(input_spikes, timesteps=200)
    end_time = time.time()
    
    print(f"Neuromorphic processing completed in {end_time - start_time:.2f} seconds")
    print(f"Output activity range: {min(result['output_activity']):.4f} - {max(result['output_activity']):.4f}")
    print(f"Network entropy: {result['network_entropy']:.4f}")
    print(f"Criticality measure: {result['criticality_measure']:.4f}")
    print(f"Total spikes generated: {sum(len(spikes) for spikes in result['spike_trains'])}")
    print()

def demonstrate_holographic_encoding():
    """Demonstrate holographic data processing"""
    print("🌊 Holographic Data Processing Demo")
    print("=" * 50)
    
    engine = HolographicDataEngine(data_dim=64)
    
    # Create test data (image-like pattern)
    test_data = np.random.random((64, 64))
    test_data[20:40, 20:40] = 1.0  # Add a square pattern
    
    print("Encoding data into holographic representation...")
    start_time = time.time()
    hologram = engine.encode_holographic(test_data)
    end_time = time.time()
    
    print(f"Holographic encoding completed in {end_time - start_time:.2f} seconds")
    print(f"Hologram shape: {hologram.shape}")
    print(f"Hologram magnitude range: {np.min(np.abs(hologram)):.4f} - {np.max(np.abs(hologram)):.4f}")
    
    # Test partial recall
    partial_input = test_data.copy()
    partial_input[30:50, 30:50] = np.nan  # Remove a portion
    
    print("Testing holographic recall from partial input...")
    recalled = engine.recall_holographic(partial_input, iterations=5)
    
    # Calculate reconstruction accuracy
    known_mask = ~np.isnan(partial_input)
    reconstruction_error = np.mean((recalled[known_mask] - test_data[known_mask])**2)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Test associative recall
    query = test_data[25:35, 25:35]  # Small query pattern
    associations = engine.associative_recall(query, similarity_threshold=0.3)
    print(f"Associative recall found {len(associations)} similar patterns")
    if associations:
        print(f"Best match similarity: {associations[0]['similarity']:.4f}")
    print()

def demonstrate_morphogenetic_growth():
    """Demonstrate morphogenetic system growth"""
    print("🌱 Morphogenetic System Demo")
    print("=" * 50)
    
    system = MorphogeneticSystem(grid_size=50)
    
    # Create a target pattern (spiral)
    target_pattern = np.zeros((50, 50))
    center = 25
    for i in range(50):
        for j in range(50):
            angle = np.arctan2(i - center, j - center)
            radius = np.sqrt((i - center)**2 + (j - center)**2)
            if 5 < radius < 20 and (angle * 3) % (2 * np.pi) < np.pi:
                target_pattern[i, j] = 1
    
    print("Growing self-organizing structure...")
    start_time = time.time()
    result = system.grow_structure(target_pattern, iterations=500)
    end_time = time.time()
    
    print(f"Morphogenetic growth completed in {end_time - start_time:.2f} seconds")
    print(f"Convergence iteration: {result['convergence_iteration']}")
    print(f"Final active cells: {np.sum(result['final_pattern'])}")
    print(f"Pattern evolution steps: {len(result['pattern_evolution'])}")
    
    if result['pattern_evolution']:
        final_similarity = result['pattern_evolution'][-1]['pattern_similarity']
        print(f"Final pattern similarity: {final_similarity:.4f}")
    print()

def demonstrate_integrated_orchestration():
    """Demonstrate integrated emergent technology orchestration"""
    print("🎭 Integrated Emergent Technology Orchestration")
    print("=" * 60)
    
    orchestrator = EmergentTechnologyOrchestrator()
    
    # Simulate a communication scenario
    message = "Hello, emergent cognitive network!"
    context = {
        'bandwidth_limit': 1000,
        'latency_requirement': 0.1,
        'reliability_threshold': 0.95
    }
    
    print("Orchestrating emergent communication technologies...")
    start_time = time.time()
    result = orchestrator.orchestrate_emergent_communication(message, context)
    end_time = time.time()
    
    print(f"Orchestration completed in {end_time - start_time:.2f} seconds")
    print(f"Quantum optimization cost: {result['quantum_optimized']['optimization_cost']:.4f}")
    print(f"Swarm intelligence level: {result['transmission_plan']['swarm_intelligence']:.4f}")
    print(f"Neural entropy: {result['adaptive_signals']['neural_entropy']:.4f}")
    print(f"Holographic encoding entropy: {result['holographic_encoding']['encoding_entropy']:.4f}")
    print(f"Emergent protocol complexity: {np.sum(result['emergent_protocol']['emergent_protocol'])}")
    print(f"Emergence metrics: {result['emergence_metrics']}")
    print()

def demonstrate_holographic_memory():
    """Demonstrate holographic associative memory"""
    print("🧠 Holographic Associative Memory Demo")
    print("=" * 50)
    
    orchestrator = EmergentTechnologyOrchestrator()
    
    # Create test data with different emotional contexts
    test_data = [
        {
            'data': np.random.random(64) + 0.5,  # Positive pattern
            'emotional_valence': 0.8,
            'category': 'positive_experience',
            'metadata': {'importance': 0.9, 'temporal_context': 'recent'}
        },
        {
            'data': np.random.random(64) - 0.3,  # Negative pattern
            'emotional_valence': 0.2,
            'category': 'negative_experience',
            'metadata': {'importance': 0.7, 'temporal_context': 'past'}
        },
        {
            'data': np.random.random(64) * 0.1,  # Neutral pattern
            'emotional_valence': 0.5,
            'category': 'neutral_experience',
            'metadata': {'importance': 0.3, 'temporal_context': 'present'}
        },
        {
            'data': np.random.random(64) + 0.2,  # Similar to first
            'emotional_valence': 0.7,
            'category': 'positive_experience',
            'metadata': {'importance': 0.6, 'temporal_context': 'recent'}
        }
    ]
    
    print("Testing holographic associative memory with emotional contexts...")
    start_time = time.time()
    memory_result = orchestrator.demonstrate_holographic_memory(test_data)
    end_time = time.time()
    
    print(f"Holographic memory demo completed in {end_time - start_time:.2f} seconds")
    print(f"Stored {len(memory_result['stored_memories'])} memories")
    print(f"Found {len(memory_result['recall_results'])} similar memories in recall")
    print()

def demonstrate_cognitive_evolution():
    """Demonstrate cognitive network evolution"""
    print("🧬 Cognitive Network Evolution Demo")
    print("=" * 50)
    
    orchestrator = EmergentTechnologyOrchestrator()
    
    # Simulate learning experiences
    experiences = [
        {'type': 'communication', 'success': True, 'complexity': 0.7},
        {'type': 'optimization', 'success': True, 'complexity': 0.5},
        {'type': 'pattern_recognition', 'success': False, 'complexity': 0.9},
        {'type': 'emergent_behavior', 'success': True, 'complexity': 0.8},
        {'type': 'adaptation', 'success': True, 'complexity': 0.6}
    ]
    
    print("Evolving cognitive network through experiential learning...")
    start_time = time.time()
    evolution_result = orchestrator.evolve_cognitive_network(experiences, generations=5)
    end_time = time.time()
    
    print(f"Cognitive evolution completed in {end_time - start_time:.2f} seconds")
    print(f"Evolutionary trajectory length: {len(evolution_result['evolutionary_trajectory'])}")
    print(f"Final cognitive state: {evolution_result['final_cognitive_state']}")
    print(f"Emergent cognitions captured: {len(evolution_result['emergent_cognitions'])}")
    
    if evolution_result['evolutionary_trajectory']:
        print("Evolution progression:")
        for i, metrics in enumerate(evolution_result['evolutionary_trajectory'][:3]):
            print(f"  Generation {i+1}: Quantum={metrics['quantum_complexity']}, "
                  f"Swarm={metrics['swarm_intelligence']}, Neural={metrics['neural_capacity']}")
    print()

def create_visualizations():
    """Create visualizations of the emergent system"""
    print("📊 Creating Visualizations")
    print("=" * 30)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Emergent Cognitive Network Infrastructure', fontsize=16, fontweight='bold')
    
    # 1. Quantum optimization convergence
    optimizer = QuantumInspiredOptimizer(num_qubits=6)
    costs = []
    for i in range(100):
        def test_func(x):
            return np.sum(x**2) + 10 * np.sum(np.cos(2 * np.pi * x))
        result = optimizer.quantum_annealing_optimization(test_func, max_iter=10)
        costs.append(result['cost'])
    
    axes[0, 0].plot(costs)
    axes[0, 0].set_title('Quantum Optimization Convergence')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Cost')
    
    # 2. Swarm intelligence evolution
    swarm = SwarmCognitiveNetwork(num_agents=20)
    def test_func(x):
        return np.sum(x**2)
    
    result = swarm.optimize_swarm(test_func, max_iterations=50)
    axes[0, 1].plot(result['swarm_intelligence'])
    axes[0, 1].set_title('Swarm Intelligence Evolution')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Intelligence Metric')
    
    # 3. Neuromorphic spike patterns
    processor = NeuromorphicProcessor(num_neurons=100)
    input_spikes = np.random.poisson(0.1, 100)
    result = processor.process_spiking_input(input_spikes, timesteps=50)
    
    spike_matrix = np.array(result['spike_trains']).T
    axes[0, 2].imshow(spike_matrix[:50, :], cmap='hot', aspect='auto')
    axes[0, 2].set_title('Neuromorphic Spike Patterns')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Neuron Index')
    
    # 4. Holographic encoding
    engine = HolographicDataEngine(data_dim=32)
    test_data = np.random.random((32, 32))
    hologram = engine.encode_holographic(test_data)
    
    im = axes[1, 0].imshow(np.abs(hologram), cmap='viridis')
    axes[1, 0].set_title('Holographic Encoding')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Morphogenetic pattern
    system = MorphogeneticSystem(grid_size=40)
    target = np.random.random((40, 40)) > 0.5
    result = system.grow_structure(target, iterations=200)
    
    axes[1, 1].imshow(result['final_pattern'], cmap='binary')
    axes[1, 1].set_title('Morphogenetic Pattern')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    
    # 6. Emergent behavior timeline
    orchestrator = EmergentTechnologyOrchestrator()
    message = "Test message"
    context = {'test': True}
    
    # Run multiple orchestrations to build emergence timeline
    emergence_timeline = []
    for i in range(10):
        result = orchestrator.orchestrate_emergent_communication(message, context)
        emergence_timeline.append(result['emergence_metrics']['emergence_level'])
    
    axes[1, 2].plot(emergence_timeline)
    axes[1, 2].set_title('Emergent Behavior Timeline')
    axes[1, 2].set_xlabel('Orchestration Event')
    axes[1, 2].set_ylabel('Emergence Level')
    
    plt.tight_layout()
    plt.savefig('emergent_cognitive_network_demo.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'emergent_cognitive_network_demo.png'")
    print()

def main():
    """Run the complete demonstration"""
    print("🚀 Emergent Cognitive Network Infrastructure Demonstration")
    print("=" * 70)
    print("This demonstration showcases advanced emergent communication technologies")
    print("including quantum optimization, swarm intelligence, neuromorphic computing,")
    print("holographic data processing, and morphogenetic system growth.")
    print()
    
    try:
        # Run individual component demonstrations
        demonstrate_quantum_optimization()
        demonstrate_swarm_intelligence()
        demonstrate_neuromorphic_processing()
        demonstrate_holographic_encoding()
        demonstrate_holographic_memory()
        demonstrate_morphogenetic_growth()
        demonstrate_integrated_orchestration()
        demonstrate_cognitive_evolution()
        
        # Create visualizations
        create_visualizations()
        
        print("✅ Demonstration completed successfully!")
        print("The emergent cognitive network infrastructure is ready for advanced")
        print("communication and cognitive processing applications.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()