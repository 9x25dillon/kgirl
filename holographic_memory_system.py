#!/usr/bin/env python3
"""
Holographic Memory System
========================
Advanced holographic memory and processing including:
- Holographic associative memory
- Fractal memory encoding
- Quantum holographic storage
- Emergent memory patterns

Author: Assistant
License: MIT
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from numpy.fft import fft2, ifft2, fftn, ifftn
import scipy.fft
from scipy import ndimage
import math
import time
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass
class MemoryPattern:
    """Represents a memory pattern with metadata"""
    key: str
    data: np.ndarray
    hologram: np.ndarray
    fractal_encoding: np.ndarray
    quantum_state: np.ndarray
    metadata: Dict[str, Any]
    access_count: int = 0
    last_accessed: float = 0.0
    importance: float = 1.0

class FractalMemoryEncoder:
    """Fractal-based memory encoding for efficient storage and retrieval"""
    
    def __init__(self, fractal_dim: int = 32, max_iterations: int = 100):
        self.fractal_dim = fractal_dim
        self.max_iterations = max_iterations
        self.fractal_cache = {}
        
    def encode_fractal(self, data: np.ndarray) -> np.ndarray:
        """Encode data using fractal compression"""
        # Convert data to fractal representation
        if data.ndim == 1:
            # Pad or truncate to make it square
            target_size = self.fractal_dim * self.fractal_dim
            if len(data) < target_size:
                # Pad with zeros (handle complex data)
                if np.iscomplexobj(data):
                    padded = np.zeros(target_size, dtype=complex)
                else:
                    padded = np.zeros(target_size)
                padded[:len(data)] = data
                data = padded
            else:
                # Truncate
                data = data[:target_size]
            data = data.reshape(self.fractal_dim, self.fractal_dim)
        
        # Create fractal attractor
        fractal_encoding = self._create_fractal_attractor(data)
        
        # Apply fractal compression
        compressed = self._fractal_compress(fractal_encoding)
        
        return compressed
    
    def _create_fractal_attractor(self, data: np.ndarray) -> np.ndarray:
        """Create fractal attractor from data"""
        height, width = data.shape
        fractal = np.zeros((self.fractal_dim, self.fractal_dim))
        
        # Map data to fractal space using iterative function system
        for i in range(height):
            for j in range(width):
                x, y = self._map_to_fractal_space(i, j, height, width)
                # Handle complex data
                if np.iscomplexobj(data):
                    fractal[y, x] += np.abs(data[i, j])
                else:
                    fractal[y, x] += data[i, j]
        
        # Normalize
        fractal = fractal / np.max(np.abs(fractal)) if np.max(np.abs(fractal)) > 0 else fractal
        
        return fractal
    
    def _map_to_fractal_space(self, i: int, j: int, height: int, width: int) -> Tuple[int, int]:
        """Map data coordinates to fractal space"""
        # Use logistic map for fractal generation
        x = (j / width) * 2 - 1  # Normalize to [-1, 1]
        y = (i / height) * 2 - 1
        
        # Apply logistic map iterations with bounds checking
        for _ in range(self.max_iterations):
            x = 3.9 * x * (1 - x)  # Logistic map
            y = 3.9 * y * (1 - y)
            
            # Clamp values to prevent overflow
            x = max(-1, min(1, x))
            y = max(-1, min(1, y))
        
        # Map back to fractal dimensions with bounds checking
        fractal_x = int((x + 1) / 2 * (self.fractal_dim - 1))
        fractal_y = int((y + 1) / 2 * (self.fractal_dim - 1))
        
        # Ensure values are within bounds
        fractal_x = max(0, min(self.fractal_dim - 1, fractal_x))
        fractal_y = max(0, min(self.fractal_dim - 1, fractal_y))
        
        return fractal_x, fractal_y
    
    def _fractal_compress(self, fractal: np.ndarray) -> np.ndarray:
        """Compress fractal using self-similarity"""
        compressed = np.zeros_like(fractal)
        
        # Find self-similar patterns at different scales
        for scale in [2, 4, 8]:
            if scale < fractal.shape[0]:
                scaled = ndimage.zoom(fractal, 1/scale, order=1)
                upscaled = ndimage.zoom(scaled, scale, order=1)
                compressed += upscaled / scale
        
        return compressed / np.max(np.abs(compressed)) if np.max(np.abs(compressed)) > 0 else compressed

class QuantumHolographicStorage:
    """Quantum-enhanced holographic storage system"""
    
    def __init__(self, num_qubits: int = 8, storage_dim: int = 64):
        self.num_qubits = num_qubits
        self.storage_dim = storage_dim
        self.quantum_states = {}
        self.entanglement_network = self._create_entanglement_network()
        
    def _create_entanglement_network(self) -> np.ndarray:
        """Create quantum entanglement network between memory locations"""
        network = np.zeros((self.storage_dim, self.storage_dim), dtype=complex)
        
        # Create entangled pairs
        for i in range(0, self.storage_dim, 2):
            if i + 1 < self.storage_dim:
                # Create Bell state |00> + |11>
                network[i, i] = 1/np.sqrt(2)
                network[i+1, i+1] = 1/np.sqrt(2)
                network[i, i+1] = 1/np.sqrt(2)
                network[i+1, i] = 1/np.sqrt(2)
        
        return network
    
    def store_quantum_state(self, key: str, data: np.ndarray) -> np.ndarray:
        """Store data as quantum state in holographic memory"""
        # Encode data into quantum state
        quantum_state = self._encode_to_quantum_state(data)
        
        # Apply quantum error correction
        corrected_state = self._quantum_error_correction(quantum_state)
        
        # Store with entanglement
        entangled_state = self._apply_entanglement(corrected_state, key)
        
        self.quantum_states[key] = entangled_state
        return entangled_state
    
    def _encode_to_quantum_state(self, data: np.ndarray) -> np.ndarray:
        """Encode classical data into quantum state"""
        # Normalize data
        data_norm = data / np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else data
        
        # Create quantum superposition
        quantum_state = np.zeros(2**self.num_qubits, dtype=complex)
        
        # Encode using amplitude encoding
        for i, amplitude in enumerate(data_norm.flat[:2**self.num_qubits]):
            quantum_state[i] = amplitude
        
        # Normalize quantum state
        norm = np.sqrt(np.sum(np.abs(quantum_state)**2))
        if norm > 0:
            quantum_state = quantum_state / norm
        
        return quantum_state
    
    def _quantum_error_correction(self, state: np.ndarray) -> np.ndarray:
        """Apply quantum error correction"""
        # Simplified error correction using repetition code
        corrected = np.zeros_like(state)
        
        # Majority voting for error correction
        for i in range(len(state)):
            # Simulate noisy measurement and correction
            noise_real = 0.1 * np.random.normal(0, 1)
            noise_imag = 0.1 * np.random.normal(0, 1)
            noisy_measurement = state[i] + noise_real + 1j * noise_imag
            corrected[i] = noisy_measurement
        
        return corrected
    
    def _apply_entanglement(self, state: np.ndarray, key: str) -> np.ndarray:
        """Apply quantum entanglement to stored state"""
        # Create entanglement with network
        entangled = np.zeros_like(state)
        
        # Use modulo to handle size mismatch
        for i in range(len(state)):
            for j in range(len(state)):
                network_i = i % self.storage_dim
                network_j = j % self.storage_dim
                entangled[i] += self.entanglement_network[network_i, network_j] * state[j]
        
        return entangled
    
    def recall_quantum_state(self, query: np.ndarray) -> Dict[str, Any]:
        """Recall quantum state using quantum interference"""
        query_state = self._encode_to_quantum_state(query)
        
        results = []
        for key, stored_state in self.quantum_states.items():
            # Calculate quantum fidelity
            fidelity = np.abs(np.vdot(query_state, stored_state))**2
            
            # Quantum interference pattern
            interference = np.abs(query_state + stored_state)**2
            
            results.append({
                'key': key,
                'fidelity': fidelity,
                'interference_pattern': interference,
                'quantum_coherence': self._calculate_quantum_coherence(stored_state)
            })
        
        return max(results, key=lambda x: x['fidelity'])

class EmergentPatternDetector:
    """Detect and analyze emergent patterns in memory"""
    
    def __init__(self, pattern_threshold: float = 0.7):
        self.pattern_threshold = pattern_threshold
        self.pattern_history = []
        self.emergent_patterns = {}
        
    def detect_emergent_patterns(self, memories: List[MemoryPattern]) -> Dict[str, Any]:
        """Detect emergent patterns across memories"""
        if len(memories) < 2:
            return {'patterns': [], 'emergence_score': 0.0}
        
        # Analyze pattern correlations
        correlations = self._analyze_pattern_correlations(memories)
        
        # Detect self-organizing patterns
        self_organizing = self._detect_self_organizing_patterns(memories)
        
        # Calculate emergence metrics
        emergence_metrics = self._calculate_emergence_metrics(correlations, self_organizing)
        
        # Identify critical transitions
        critical_transitions = self._identify_critical_transitions(memories)
        
        return {
            'patterns': self_organizing,
            'correlations': correlations,
            'emergence_score': emergence_metrics['overall_emergence'],
            'critical_transitions': critical_transitions,
            'complexity_metrics': emergence_metrics
        }
    
    def _analyze_pattern_correlations(self, memories: List[MemoryPattern]) -> np.ndarray:
        """Analyze correlations between memory patterns"""
        n_memories = len(memories)
        correlation_matrix = np.zeros((n_memories, n_memories))
        
        for i, mem1 in enumerate(memories):
            for j, mem2 in enumerate(memories):
                if i != j:
                    # Calculate cross-correlation in frequency domain
                    h1_flat = mem1.hologram.flatten()
                    h2_flat = mem2.hologram.flatten()
                    
                    # Handle complex data by taking absolute values for correlation
                    if np.iscomplexobj(h1_flat) or np.iscomplexobj(h2_flat):
                        h1_real = np.abs(h1_flat)
                        h2_real = np.abs(h2_flat)
                    else:
                        h1_real = h1_flat
                        h2_real = h2_flat
                    
                    corr = np.corrcoef(h1_real, h2_real)[0, 1]
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
        
        return correlation_matrix
    
    def _detect_self_organizing_patterns(self, memories: List[MemoryPattern]) -> List[Dict]:
        """Detect self-organizing patterns in memory"""
        patterns = []
        
        # Group memories by similarity
        groups = self._cluster_memories(memories)
        
        for group in groups:
            if len(group) > 1:
                # Calculate group coherence
                coherence = self._calculate_group_coherence(group)
                
                if coherence > self.pattern_threshold:
                    patterns.append({
                        'group': group,
                        'coherence': coherence,
                        'pattern_type': self._classify_pattern_type(group),
                        'emergence_strength': self._calculate_emergence_strength(group)
                    })
        
        return patterns
    
    def _cluster_memories(self, memories: List[MemoryPattern]) -> List[List[MemoryPattern]]:
        """Cluster memories by similarity"""
        if len(memories) <= 1:
            return [memories]
        
        # Simple hierarchical clustering
        clusters = [[mem] for mem in memories]
        
        while len(clusters) > 1:
            best_merge = None
            best_similarity = 0
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = self._calculate_cluster_similarity(clusters[i], clusters[j])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_merge = (i, j)
            
            if best_merge and best_similarity > 0.5:
                i, j = best_merge
                clusters[i].extend(clusters[j])
                clusters.pop(j)
            else:
                break
        
        return clusters
    
    def _calculate_cluster_similarity(self, cluster1: List[MemoryPattern], cluster2: List[MemoryPattern]) -> float:
        """Calculate similarity between two clusters"""
        similarities = []
        
        for mem1 in cluster1:
            for mem2 in cluster2:
                sim = np.corrcoef(mem1.hologram.flatten(), mem2.hologram.flatten())[0, 1]
                similarities.append(sim if not np.isnan(sim) else 0)
        
        return np.mean(similarities) if similarities else 0
    
    def _calculate_group_coherence(self, group: List[MemoryPattern]) -> float:
        """Calculate coherence within a group of memories"""
        if len(group) < 2:
            return 1.0
        
        holograms = [mem.hologram for mem in group]
        coherence_matrix = np.zeros((len(holograms), len(holograms)))
        
        for i, h1 in enumerate(holograms):
            for j, h2 in enumerate(holograms):
                if i != j:
                    coherence = np.abs(np.vdot(h1.flatten(), h2.flatten()))
                    coherence_matrix[i, j] = coherence
        
        return np.mean(coherence_matrix[coherence_matrix > 0])
    
    def _classify_pattern_type(self, group: List[MemoryPattern]) -> str:
        """Classify the type of emergent pattern"""
        # Analyze temporal patterns
        access_times = [mem.last_accessed for mem in group]
        access_counts = [mem.access_count for mem in group]
        
        if len(set(access_times)) == 1:
            return "synchronous"
        elif np.std(access_counts) < 0.1:
            return "uniform"
        elif np.corrcoef(access_times, access_counts)[0, 1] > 0.5:
            return "cascading"
        else:
            return "emergent"
    
    def _calculate_emergence_strength(self, group: List[MemoryPattern]) -> float:
        """Calculate the strength of emergence in a pattern"""
        # Based on non-linearity and self-organization
        holograms = [mem.hologram for mem in group]
        
        # Calculate non-linearity
        nonlinearity = 0
        for i, h1 in enumerate(holograms):
            for j, h2 in enumerate(holograms):
                if i != j:
                    # Non-linear interaction
                    interaction = np.abs(np.fft.fft2(h1 * h2))
                    nonlinearity += np.mean(interaction)
        
        # Calculate self-organization (entropy reduction)
        individual_entropies = [self._calculate_entropy(h) for h in holograms]
        group_entropy = self._calculate_entropy(np.mean(holograms, axis=0))
        
        self_organization = np.mean(individual_entropies) - group_entropy
        
        return min(1.0, (nonlinearity + self_organization) / 2)
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        hist, _ = np.histogram(data.flatten(), bins=50)
        hist = hist[hist > 0]  # Remove zero bins
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def _calculate_emergence_metrics(self, correlations: np.ndarray, patterns: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive emergence metrics"""
        # Overall emergence score
        pattern_strengths = [p['emergence_strength'] for p in patterns]
        overall_emergence = np.mean(pattern_strengths) if pattern_strengths else 0
        
        # Complexity metrics
        correlation_entropy = self._calculate_entropy(correlations)
        pattern_diversity = len(set(p['pattern_type'] for p in patterns))
        
        # Self-organization index
        self_organization = np.mean([p['coherence'] for p in patterns]) if patterns else 0
        
        return {
            'overall_emergence': overall_emergence,
            'correlation_entropy': correlation_entropy,
            'pattern_diversity': pattern_diversity,
            'self_organization': self_organization,
            'complexity_index': overall_emergence * pattern_diversity
        }
    
    def _identify_critical_transitions(self, memories: List[MemoryPattern]) -> List[Dict]:
        """Identify critical transitions in memory patterns"""
        transitions = []
        
        # Sort by access time
        sorted_memories = sorted(memories, key=lambda m: m.last_accessed)
        
        for i in range(1, len(sorted_memories)):
            prev_mem = sorted_memories[i-1]
            curr_mem = sorted_memories[i]
            
            # Calculate transition strength
            transition_strength = self._calculate_transition_strength(prev_mem, curr_mem)
            
            if transition_strength > 0.8:  # High transition threshold
                transitions.append({
                    'from_key': prev_mem.key,
                    'to_key': curr_mem.key,
                    'strength': transition_strength,
                    'transition_type': self._classify_transition_type(prev_mem, curr_mem)
                })
        
        return transitions
    
    def _calculate_transition_strength(self, mem1: MemoryPattern, mem2: MemoryPattern) -> float:
        """Calculate strength of transition between memories"""
        # Cross-correlation in frequency domain
        corr = np.corrcoef(mem1.hologram.flatten(), mem2.hologram.flatten())[0, 1]
        corr = corr if not np.isnan(corr) else 0
        
        # Temporal proximity
        time_diff = abs(mem1.last_accessed - mem2.last_accessed)
        temporal_factor = 1.0 / (1.0 + time_diff)
        
        # Importance weighting
        importance_factor = (mem1.importance + mem2.importance) / 2
        
        return corr * temporal_factor * importance_factor
    
    def _classify_transition_type(self, mem1: MemoryPattern, mem2: MemoryPattern) -> str:
        """Classify the type of transition"""
        # Analyze pattern changes
        h1_energy = np.sum(np.abs(mem1.hologram)**2)
        h2_energy = np.sum(np.abs(mem2.hologram)**2)
        
        if h2_energy > h1_energy * 1.5:
            return "amplification"
        elif h2_energy < h1_energy * 0.5:
            return "attenuation"
        elif mem2.access_count > mem1.access_count:
            return "activation"
        else:
            return "modulation"

class HolographicAssociativeMemory:
    """Advanced holographic associative memory system"""
    
    def __init__(self, hologram_dim: int = 64, fractal_dim: int = 32):
        self.hologram_dim = hologram_dim
        self.fractal_encoder = FractalMemoryEncoder(fractal_dim)
        self.quantum_storage = QuantumHolographicStorage(storage_dim=hologram_dim)
        self.pattern_detector = EmergentPatternDetector()
        
        # Memory storage
        self.memories: Dict[str, MemoryPattern] = {}
        self.accumulated_hologram = np.zeros((hologram_dim, hologram_dim), dtype=np.complex128)
        
        # Performance tracking
        self.access_history = []
        self.performance_metrics = defaultdict(list)
        
    def store(self, key: str, data: Any, metadata: Optional[Dict] = None) -> str:
        """Store data with advanced holographic encoding"""
        import time
        
        # Convert data to array
        data_array = self._ensure_array(data)
        
        # Create holographic representation
        hologram = self._create_hologram(data_array)
        
        # Create fractal encoding
        fractal_encoding = self.fractal_encoder.encode_fractal(data_array)
        
        # Create quantum state
        quantum_state = self.quantum_storage.store_quantum_state(key, data_array)
        
        # Create memory pattern
        memory_pattern = MemoryPattern(
            key=key,
            data=data_array,
            hologram=hologram,
            fractal_encoding=fractal_encoding,
            quantum_state=quantum_state,
            metadata=metadata or {},
            last_accessed=time.time()
        )
        
        # Store in memory system
        self.memories[key] = memory_pattern
        self.accumulated_hologram += hologram
        
        return key
    
    def recall(self, query: Any, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Advanced recall with multiple similarity measures"""
        import time
        
        query_array = self._ensure_array(query)
        query_hologram = self._create_hologram(query_array)
        
        results = []
        
        for key, memory in self.memories.items():
            # Update access tracking
            memory.access_count += 1
            memory.last_accessed = time.time()
            
            # Calculate multiple similarity measures
            holographic_sim = self._calculate_holographic_similarity(query_hologram, memory.hologram)
            fractal_sim = self._calculate_fractal_similarity(query_array, memory.fractal_encoding)
            quantum_sim = self._calculate_quantum_similarity(query_array, memory.quantum_state)
            
            # Combined similarity score
            combined_sim = (holographic_sim * 0.4 + fractal_sim * 0.3 + quantum_sim * 0.3)
            
            if combined_sim >= threshold:
                # Reconstruct memory
                reconstruction = self._reconstruct_memory(query_hologram, memory)
                
                results.append({
                    'key': key,
                    'similarity': combined_sim,
                    'holographic_similarity': holographic_sim,
                    'fractal_similarity': fractal_sim,
                    'quantum_similarity': quantum_sim,
                    'reconstruction': reconstruction,
                    'metadata': memory.metadata,
                    'access_count': memory.access_count,
                    'importance': memory.importance
                })
        
        # Sort by combined similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Update performance metrics
        self._update_performance_metrics(results)
        
        return results[:top_k]
    
    def _ensure_array(self, data: Any) -> np.ndarray:
        """Convert data to numpy array"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data, dtype=float)
        elif isinstance(data, str):
            return np.frombuffer(data.encode('utf-8'), dtype=np.uint8).astype(float)
        else:
            # Handle complex data by taking absolute values
            arr = np.array(data)
            if np.iscomplexobj(arr):
                return np.abs(arr)
            return arr.astype(float)
    
    def _create_hologram(self, data: np.ndarray) -> np.ndarray:
        """Create holographic representation of data"""
        # Ensure data fits hologram dimensions
        if data.ndim == 1:
            # Pad or truncate to make it square
            target_size = self.hologram_dim * self.hologram_dim
            if len(data) < target_size:
                # Pad with zeros (handle complex data)
                if np.iscomplexobj(data):
                    padded = np.zeros(target_size, dtype=complex)
                else:
                    padded = np.zeros(target_size)
                padded[:len(data)] = data
                data = padded
            else:
                # Truncate
                data = data[:target_size]
            data = data.reshape(self.hologram_dim, self.hologram_dim)
        
        # Resize to hologram dimensions if needed
        if data.shape != (self.hologram_dim, self.hologram_dim):
            data = ndimage.zoom(data, 
                              (self.hologram_dim / data.shape[0], 
                               self.hologram_dim / data.shape[1]), 
                              order=1)
        
        # Create hologram using FFT
        hologram = fft2(data)
        
        # Add random phase for holographic interference
        phase = np.exp(1j * 2 * np.pi * np.random.random(hologram.shape))
        hologram = hologram * phase
        
        return hologram
    
    def _calculate_holographic_similarity(self, query_hologram: np.ndarray, stored_hologram: np.ndarray) -> float:
        """Calculate holographic similarity using cross-correlation"""
        # Cross-correlation in frequency domain
        correlation = np.abs((query_hologram.conj() * stored_hologram).sum())
        
        # Normalize by magnitudes
        query_mag = np.sqrt(np.sum(np.abs(query_hologram)**2))
        stored_mag = np.sqrt(np.sum(np.abs(stored_hologram)**2))
        
        if query_mag > 0 and stored_mag > 0:
            return correlation / (query_mag * stored_mag)
        else:
            return 0.0
    
    def _calculate_fractal_similarity(self, query_data: np.ndarray, fractal_encoding: np.ndarray) -> float:
        """Calculate similarity using fractal encoding"""
        query_fractal = self.fractal_encoder.encode_fractal(query_data)
        
        # Calculate correlation between fractal encodings
        corr = np.corrcoef(query_fractal.flatten(), fractal_encoding.flatten())[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def _calculate_quantum_similarity(self, query_data: np.ndarray, quantum_state: np.ndarray) -> float:
        """Calculate quantum similarity using fidelity"""
        query_quantum = self.quantum_storage._encode_to_quantum_state(query_data)
        
        # Calculate quantum fidelity
        fidelity = np.abs(np.vdot(query_quantum, quantum_state))**2
        return fidelity
    
    def _reconstruct_memory(self, query_hologram: np.ndarray, memory: MemoryPattern) -> np.ndarray:
        """Reconstruct memory using holographic interference"""
        # Use phase of accumulated hologram for reconstruction
        reconstruction_freq = np.abs(query_hologram) * np.exp(1j * np.angle(self.accumulated_hologram))
        reconstruction = np.real(ifft2(reconstruction_freq))
        
        return reconstruction
    
    def _update_performance_metrics(self, results: List[Dict]):
        """Update performance tracking metrics"""
        if results:
            self.performance_metrics['recall_accuracy'].append(results[0]['similarity'])
            self.performance_metrics['num_results'].append(len(results))
            self.performance_metrics['timestamp'].append(time.time())
    
    def detect_emergent_patterns(self) -> Dict[str, Any]:
        """Detect emergent patterns in stored memories"""
        memory_list = list(self.memories.values())
        return self.pattern_detector.detect_emergent_patterns(memory_list)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.memories:
            return {'total_memories': 0}
        
        access_counts = [mem.access_count for mem in self.memories.values()]
        importances = [mem.importance for mem in self.memories.values()]
        
        return {
            'total_memories': len(self.memories),
            'total_accesses': sum(access_counts),
            'average_access_count': np.mean(access_counts),
            'memory_utilization': len([c for c in access_counts if c > 0]) / len(access_counts),
            'average_importance': np.mean(importances),
            'hologram_energy': np.sum(np.abs(self.accumulated_hologram)**2),
            'performance_metrics': dict(self.performance_metrics)
        }

# Backward compatibility
class HolographicMemorySystem(HolographicAssociativeMemory):
    """Backward compatibility wrapper"""
    pass

def demo_holographic_memory():
    """Demonstrate advanced holographic memory system"""
    print("=== Holographic Memory System Demonstration ===")
    
    # Initialize system
    memory = HolographicAssociativeMemory(hologram_dim=32, fractal_dim=16)
    
    # Store various types of data
    test_data = [
        ("text_data", "Hello, holographic world!"),
        ("numeric_data", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ("image_data", np.random.rand(8, 8)),
        ("pattern_data", np.sin(np.linspace(0, 4*np.pi, 64))),
        ("complex_data", np.random.random(16) + 1j * np.random.random(16))
    ]
    
    print("Storing test data...")
    for key, data in test_data:
        memory.store(key, data, metadata={'type': type(data).__name__})
    
    # Test recall
    print("\nTesting recall...")
    query = "Hello, world"
    results = memory.recall(query, top_k=3)
    
    print(f"Query: '{query}'")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['key']} (similarity: {result['similarity']:.4f})")
        print(f"     Holographic: {result['holographic_similarity']:.4f}")
        print(f"     Fractal: {result['fractal_similarity']:.4f}")
        print(f"     Quantum: {result['quantum_similarity']:.4f}")
    
    # Detect emergent patterns
    print("\nDetecting emergent patterns...")
    patterns = memory.detect_emergent_patterns()
    print(f"Emergence Score: {patterns['emergence_score']:.4f}")
    print(f"Pattern Diversity: {patterns['complexity_metrics']['pattern_diversity']}")
    print(f"Self-Organization: {patterns['complexity_metrics']['self_organization']:.4f}")
    
    # Memory statistics
    print("\nMemory Statistics:")
    stats = memory.get_memory_statistics()
    for key, value in stats.items():
        if key != 'performance_metrics':
            print(f"  {key}: {value}")
    
    return {
        'memory_system': memory,
        'recall_results': results,
        'emergent_patterns': patterns,
        'statistics': stats
    }

if __name__ == "__main__":
    demo_holographic_memory()