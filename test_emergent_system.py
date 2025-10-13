#!/usr/bin/env python3
"""
Test Suite for Emergent Cognitive Network Infrastructure
=======================================================
Comprehensive tests for all components of the emergent cognitive network system.
"""

import unittest
import numpy as np
import time
from emergent_cognitive_network import (
    QuantumInspiredOptimizer,
    SwarmCognitiveNetwork,
    NeuromorphicProcessor,
    HolographicDataEngine,
    HolographicAssociativeMemory,
    MorphogeneticSystem,
    EmergentTechnologyOrchestrator
)

class TestQuantumInspiredOptimizer(unittest.TestCase):
    """Test cases for QuantumInspiredOptimizer"""
    
    def setUp(self):
        self.optimizer = QuantumInspiredOptimizer(num_qubits=6)
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.num_qubits, 6)
        self.assertEqual(len(self.optimizer.quantum_state), 2**6)
        self.assertAlmostEqual(np.sum(np.abs(self.optimizer.quantum_state)**2), 1.0, places=10)
    
    def test_quantum_annealing_optimization(self):
        """Test quantum annealing optimization"""
        def simple_cost_function(x):
            return np.sum(x**2)
        
        result = self.optimizer.quantum_annealing_optimization(simple_cost_function, max_iter=50)
        
        self.assertIn('solution', result)
        self.assertIn('cost', result)
        self.assertIn('quantum_entropy', result)
        self.assertEqual(len(result['solution']), self.optimizer.num_qubits)
        self.assertIsInstance(result['cost'], float)
        self.assertIsInstance(result['quantum_entropy'], float)
    
    def test_quantum_entropy_calculation(self):
        """Test quantum entropy calculation"""
        entropy = self.optimizer._calculate_quantum_entropy()
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)

class TestSwarmCognitiveNetwork(unittest.TestCase):
    """Test cases for SwarmCognitiveNetwork"""
    
    def setUp(self):
        self.swarm = SwarmCognitiveNetwork(num_agents=10, search_space=(-5, 5))
    
    def test_initialization(self):
        """Test swarm initialization"""
        self.assertEqual(self.swarm.num_agents, 10)
        self.assertEqual(self.swarm.search_space, (-5, 5))
        self.assertEqual(len(self.swarm.agents), 10)
        
        for agent in self.swarm.agents:
            self.assertIn('id', agent)
            self.assertIn('position', agent)
            self.assertIn('velocity', agent)
            self.assertIn('personal_best', agent)
            self.assertIn('personal_best_cost', agent)
            self.assertEqual(len(agent['position']), 10)
            self.assertEqual(len(agent['velocity']), 10)
    
    def test_swarm_optimization(self):
        """Test swarm optimization"""
        def simple_objective(x):
            return np.sum(x**2)
        
        result = self.swarm.optimize_swarm(simple_objective, max_iterations=20)
        
        self.assertIn('global_best', result)
        self.assertIn('swarm_intelligence', result)
        self.assertIn('emergent_behaviors', result)
        self.assertIn('final_swarm_state', result)
        
        self.assertIsInstance(result['swarm_intelligence'], list)
        self.assertEqual(len(result['swarm_intelligence']), 20)
        self.assertIsInstance(result['emergent_behaviors'], list)
    
    def test_emergent_behavior_detection(self):
        """Test emergent behavior detection"""
        # Test with highly coordinated agents
        for agent in self.swarm.agents:
            agent['position'] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.assertTrue(self.swarm._detect_emergent_behavior())
        
        # Test with dispersed agents
        for i, agent in enumerate(self.swarm.agents):
            agent['position'] = np.random.uniform(-10, 10, 10)
        
        # Should not detect emergence with random positions
        self.assertFalse(self.swarm._detect_emergent_behavior())

class TestNeuromorphicProcessor(unittest.TestCase):
    """Test cases for NeuromorphicProcessor"""
    
    def setUp(self):
        self.processor = NeuromorphicProcessor(num_neurons=100)
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.num_neurons, 100)
        self.assertIn('membrane_potentials', self.processor.neuron_states)
        self.assertIn('recovery_variables', self.processor.neuron_states)
        self.assertIn('firing_rates', self.processor.neuron_states)
        self.assertEqual(self.processor.synaptic_weights.shape, (100, 100))
    
    def test_spiking_processing(self):
        """Test spiking input processing"""
        input_spikes = np.random.poisson(0.1, 100)
        result = self.processor.process_spiking_input(input_spikes, timesteps=50)
        
        self.assertIn('output_activity', result)
        self.assertIn('spike_trains', result)
        self.assertIn('network_entropy', result)
        self.assertIn('criticality_measure', result)
        
        self.assertEqual(len(result['output_activity']), 50)
        self.assertEqual(len(result['spike_trains']), 50)
        self.assertIsInstance(result['network_entropy'], float)
        self.assertIsInstance(result['criticality_measure'], float)
    
    def test_neuron_dynamics(self):
        """Test neuron dynamics update"""
        initial_potentials = self.processor.neuron_states['membrane_potentials'].copy()
        input_currents = np.random.normal(0, 1, 100)
        
        self.processor._update_neuron_dynamics(input_currents)
        
        # Potentials should have changed
        self.assertFalse(np.array_equal(
            initial_potentials, 
            self.processor.neuron_states['membrane_potentials']
        ))

class TestHolographicDataEngine(unittest.TestCase):
    """Test cases for HolographicDataEngine"""
    
    def setUp(self):
        self.engine = HolographicDataEngine(data_dim=32)
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.data_dim, 32)
        self.assertEqual(self.engine.holographic_memory.shape, (32, 32))
        self.assertEqual(self.engine.holographic_memory.dtype, complex)
    
    def test_holographic_encoding(self):
        """Test holographic encoding"""
        test_data = np.random.random(32 * 32)
        hologram = self.engine.encode_holographic(test_data)
        
        self.assertEqual(hologram.shape, (32, 32))
        self.assertEqual(hologram.dtype, complex)
        self.assertNotEqual(np.sum(np.abs(hologram)), 0)
    
    def test_holographic_recall(self):
        """Test holographic recall"""
        # Store some data
        original_data = np.random.random(32 * 32)
        self.engine.encode_holographic(original_data)
        
        # Create partial input
        partial_input = original_data.copy()
        partial_input[100:200] = np.nan
        
        # Recall
        recalled = self.engine.recall_holographic(partial_input, iterations=5)
        
        self.assertEqual(recalled.shape, original_data.shape)
        self.assertFalse(np.any(np.isnan(recalled)))
    
    def test_associative_recall(self):
        """Test associative recall"""
        # Store some patterns
        pattern1 = np.random.random(32 * 32)
        pattern2 = np.random.random(32 * 32)
        
        self.engine.encode_holographic(pattern1)
        self.engine.encode_holographic(pattern2)
        
        # Query with similar pattern
        query = pattern1 + np.random.normal(0, 0.1, pattern1.shape)
        associations = self.engine.associative_recall(query, similarity_threshold=0.1)
        
        self.assertIsInstance(associations, list)

class TestHolographicAssociativeMemory(unittest.TestCase):
    """Test cases for HolographicAssociativeMemory"""
    
    def setUp(self):
        self.memory = HolographicAssociativeMemory(memory_size=100, hologram_dim=32)
    
    def test_initialization(self):
        """Test memory initialization"""
        self.assertEqual(self.memory.memory_size, 100)
        self.assertEqual(self.memory.hologram_dim, 32)
        self.assertEqual(self.memory.holographic_memory.shape, (32, 32))
        self.assertEqual(len(self.memory.memory_traces), 0)
        self.assertEqual(len(self.memory.associative_links), 0)
    
    def test_store_holographic(self):
        """Test holographic storage"""
        test_data = np.random.random(32 * 32)
        metadata = {'emotional_valence': 0.8, 'category': 'test'}
        
        memory_key = self.memory.store_holographic(test_data, metadata)
        
        self.assertIsInstance(memory_key, str)
        self.assertTrue(memory_key.startswith('mem_'))
        self.assertEqual(len(self.memory.memory_traces), 1)
        self.assertIn(memory_key, self.memory.associative_links)
    
    def test_recall_associative(self):
        """Test associative recall"""
        # Store some memories
        data1 = np.random.random(32 * 32)
        data2 = np.random.random(32 * 32)
        
        key1 = self.memory.store_holographic(data1, {'emotional_valence': 0.8})
        key2 = self.memory.store_holographic(data2, {'emotional_valence': 0.2})
        
        # Query with similar data
        query = data1 + np.random.normal(0, 0.1, data1.shape)
        recalled = self.memory.recall_associative(query, similarity_threshold=0.1)
        
        self.assertIsInstance(recalled, list)
        if recalled:
            self.assertIn('memory_key', recalled[0])
            self.assertIn('similarity', recalled[0])
            self.assertIn('reconstructed_data', recalled[0])
    
    def test_memory_statistics(self):
        """Test memory statistics"""
        # Store some memories
        for i in range(5):
            data = np.random.random(32 * 32)
            self.memory.store_holographic(data, {'emotional_valence': i * 0.2})
        
        stats = self.memory.get_memory_statistics()
        
        self.assertIn('total_memories', stats)
        self.assertIn('memory_utilization', stats)
        self.assertIn('associative_links', stats)
        self.assertIn('average_emotional_valence', stats)
        self.assertIn('memory_diversity', stats)
        
        self.assertEqual(stats['total_memories'], 5)
        self.assertIsInstance(stats['memory_utilization'], float)
        self.assertIsInstance(stats['average_emotional_valence'], float)
    
    def test_memory_consolidation(self):
        """Test memory consolidation"""
        # Store similar memories
        for i in range(3):
            data = np.random.random(32 * 32)
            self.memory.store_holographic(data, {'emotional_valence': 0.8})  # Similar valence
        
        consolidation_result = self.memory.consolidate_memories(consolidation_threshold=0.7)
        
        self.assertIn('consolidated_memories', consolidation_result)
        self.assertIn('remaining_memories', consolidation_result)
        self.assertIn('consolidation_ratio', consolidation_result)
        
        self.assertIsInstance(consolidation_result['consolidated_memories'], int)
        self.assertIsInstance(consolidation_result['remaining_memories'], int)
        self.assertIsInstance(consolidation_result['consolidation_ratio'], float)

class TestMorphogeneticSystem(unittest.TestCase):
    """Test cases for MorphogeneticSystem"""
    
    def setUp(self):
        self.system = MorphogeneticSystem(grid_size=20)
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertEqual(self.system.grid_size, 20)
        self.assertIn('activator', self.system.morphogen_fields)
        self.assertIn('inhibitor', self.system.morphogen_fields)
        self.assertIn('growth_factor', self.system.morphogen_fields)
        self.assertEqual(self.system.cell_states.shape, (20, 20))
    
    def test_morphogen_field_initialization(self):
        """Test morphogen field initialization"""
        fields = self.system.morphogen_fields
        
        for field_name, field_data in fields.items():
            self.assertEqual(field_data.shape, (20, 20))
            if field_name == 'growth_factor':
                self.assertEqual(np.sum(field_data), 0)  # Should be initialized to zeros
            else:
                self.assertTrue(np.all(field_data >= 0) and np.all(field_data <= 1))
    
    def test_structure_growth(self):
        """Test structure growth"""
        pattern_template = np.random.random((20, 20)) > 0.5
        result = self.system.grow_structure(pattern_template, iterations=50)
        
        self.assertIn('final_pattern', result)
        self.assertIn('pattern_evolution', result)
        self.assertIn('morphogen_final_state', result)
        self.assertIn('convergence_iteration', result)
        
        self.assertEqual(result['final_pattern'].shape, (20, 20))
        self.assertIsInstance(result['pattern_evolution'], list)
        self.assertIsInstance(result['convergence_iteration'], int)
    
    def test_reaction_diffusion(self):
        """Test reaction-diffusion update"""
        initial_activator = self.system.morphogen_fields['activator'].copy()
        initial_inhibitor = self.system.morphogen_fields['inhibitor'].copy()
        
        self.system._update_reaction_diffusion()
        
        # Fields should have changed
        self.assertFalse(np.array_equal(
            initial_activator, 
            self.system.morphogen_fields['activator']
        ))
        self.assertFalse(np.array_equal(
            initial_inhibitor, 
            self.system.morphogen_fields['inhibitor']
        ))

class TestEmergentTechnologyOrchestrator(unittest.TestCase):
    """Test cases for EmergentTechnologyOrchestrator"""
    
    def setUp(self):
        self.orchestrator = EmergentTechnologyOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsInstance(self.orchestrator.quantum_optimizer, QuantumInspiredOptimizer)
        self.assertIsInstance(self.orchestrator.swarm_network, SwarmCognitiveNetwork)
        self.assertIsInstance(self.orchestrator.neuromorphic_processor, NeuromorphicProcessor)
        self.assertIsInstance(self.orchestrator.holographic_engine, HolographicDataEngine)
        self.assertIsInstance(self.orchestrator.holographic_memory, HolographicAssociativeMemory)
        self.assertIsInstance(self.orchestrator.morphogenetic_system, MorphogeneticSystem)
        
        self.assertEqual(len(self.orchestrator.emergent_behaviors), 0)
        self.assertEqual(len(self.orchestrator.cognitive_evolution), 0)
    
    def test_emergent_communication_orchestration(self):
        """Test emergent communication orchestration"""
        message = "Test message"
        context = {'bandwidth_limit': 1000, 'latency_requirement': 0.1}
        
        result = self.orchestrator.orchestrate_emergent_communication(message, context)
        
        self.assertIn('quantum_optimized', result)
        self.assertIn('transmission_plan', result)
        self.assertIn('adaptive_signals', result)
        self.assertIn('holographic_encoding', result)
        self.assertIn('emergent_protocol', result)
        self.assertIn('emergence_metrics', result)
        
        # Check that all phases produced results
        self.assertIsInstance(result['quantum_optimized'], dict)
        self.assertIsInstance(result['transmission_plan'], dict)
        self.assertIsInstance(result['adaptive_signals'], dict)
        self.assertIsInstance(result['holographic_encoding'], dict)
        self.assertIsInstance(result['emergent_protocol'], dict)
        self.assertIsInstance(result['emergence_metrics'], dict)
    
    def test_cognitive_evolution(self):
        """Test cognitive network evolution"""
        experiences = [
            {'type': 'test', 'success': True, 'complexity': 0.5},
            {'type': 'test2', 'success': False, 'complexity': 0.8}
        ]
        
        result = self.orchestrator.evolve_cognitive_network(experiences, generations=3)
        
        self.assertIn('evolutionary_trajectory', result)
        self.assertIn('final_cognitive_state', result)
        self.assertIn('emergent_cognitions', result)
        
        self.assertEqual(len(result['evolutionary_trajectory']), 3)
        self.assertIsInstance(result['final_cognitive_state'], dict)
        self.assertIsInstance(result['emergent_cognitions'], list)
    
    def test_holographic_memory_demonstration(self):
        """Test holographic memory demonstration"""
        test_data = [
            {
                'data': np.random.random(64),
                'emotional_valence': 0.8,
                'category': 'positive',
                'metadata': {'importance': 0.9}
            },
            {
                'data': np.random.random(64),
                'emotional_valence': 0.2,
                'category': 'negative',
                'metadata': {'importance': 0.3}
            }
        ]
        
        result = self.orchestrator.demonstrate_holographic_memory(test_data)
        
        self.assertIn('stored_memories', result)
        self.assertIn('recall_results', result)
        self.assertIn('memory_statistics', result)
        self.assertIn('consolidation_result', result)
        
        self.assertEqual(len(result['stored_memories']), 2)
        self.assertIsInstance(result['memory_statistics'], dict)
        self.assertIsInstance(result['consolidation_result'], dict)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        orchestrator = EmergentTechnologyOrchestrator()
        
        # Test communication orchestration
        message = "Integration test message"
        context = {'test': True, 'bandwidth': 500}
        
        start_time = time.time()
        result = orchestrator.orchestrate_emergent_communication(message, context)
        end_time = time.time()
        
        # Should complete without errors
        self.assertIsInstance(result, dict)
        self.assertLess(end_time - start_time, 10.0)  # Should complete quickly
        
        # Test cognitive evolution
        experiences = [{'type': 'integration', 'success': True, 'complexity': 0.6}]
        evolution_result = orchestrator.evolve_cognitive_network(experiences, generations=2)
        
        self.assertIsInstance(evolution_result, dict)
        self.assertEqual(len(evolution_result['evolutionary_trajectory']), 2)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        orchestrator = EmergentTechnologyOrchestrator()
        
        # Benchmark quantum optimization
        start_time = time.time()
        def test_func(x):
            return np.sum(x**2)
        result = orchestrator.quantum_optimizer.quantum_annealing_optimization(test_func, max_iter=100)
        quantum_time = time.time() - start_time
        
        # Benchmark swarm optimization
        start_time = time.time()
        swarm_result = orchestrator.swarm_network.optimize_swarm(test_func, max_iterations=50)
        swarm_time = time.time() - start_time
        
        # Benchmark neuromorphic processing
        start_time = time.time()
        input_spikes = np.random.poisson(0.1, 100)
        neuromorphic_result = orchestrator.neuromorphic_processor.process_spiking_input(input_spikes, timesteps=50)
        neuromorphic_time = time.time() - start_time
        
        # All should complete in reasonable time
        self.assertLess(quantum_time, 5.0)
        self.assertLess(swarm_time, 5.0)
        self.assertLess(neuromorphic_time, 5.0)
        
        print(f"\nPerformance Benchmarks:")
        print(f"  Quantum optimization: {quantum_time:.3f}s")
        print(f"  Swarm optimization: {swarm_time:.3f}s")
        print(f"  Neuromorphic processing: {neuromorphic_time:.3f}s")

def run_performance_tests():
    """Run performance tests and benchmarks"""
    print("🚀 Running Performance Tests")
    print("=" * 40)
    
    # Test with different system sizes
    sizes = [50, 100, 200]
    
    for size in sizes:
        print(f"\nTesting with {size} agents/neurons:")
        
        # Swarm test
        swarm = SwarmCognitiveNetwork(num_agents=size)
        start_time = time.time()
        def test_func(x):
            return np.sum(x**2)
        swarm.optimize_swarm(test_func, max_iterations=20)
        swarm_time = time.time() - start_time
        print(f"  Swarm ({size} agents): {swarm_time:.3f}s")
        
        # Neuromorphic test
        processor = NeuromorphicProcessor(num_neurons=size)
        start_time = time.time()
        input_spikes = np.random.poisson(0.1, size)
        processor.process_spiking_input(input_spikes, timesteps=20)
        neuromorphic_time = time.time() - start_time
        print(f"  Neuromorphic ({size} neurons): {neuromorphic_time:.3f}s")

if __name__ == '__main__':
    # Run unit tests
    print("🧪 Running Unit Tests")
    print("=" * 30)
    
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n✅ All tests completed!")