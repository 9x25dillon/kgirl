#!/usr/bin/env python3
"""
Emergent Cognitive Network Infrastructure
========================================
Advanced infrastructure for emergent communication technologies including:
- Swarm intelligence for distributed cognitive networks
- Quantum-inspired optimization algorithms
- Neuromorphic computing interfaces
- Holographic data representations
- Morphogenetic system growth

Author: Assistant
License: MIT
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import signal as sp_signal


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization for cognitive network parameters"""

    def __init__(self, num_qubits: int = 10):
        self.num_qubits = num_qubits
        self.quantum_state = self._initialize_quantum_state()

    def _initialize_quantum_state(self) -> np.ndarray:
        """Initialize in superposition state"""
        state = np.ones(2**self.num_qubits, dtype=np.complex128) / np.sqrt(2**self.num_qubits)
        return state

    def _estimate_gradient(self, cost_function, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        grad = np.zeros_like(x, dtype=np.float64)
        base = float(cost_function(x))
        for i in range(x.size):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            grad[i] = (float(cost_function(xp)) - float(cost_function(xm))) / (2 * eps)
        # Add small noise to simulate quantum fluctuation in estimation
        grad += np.random.normal(0.0, 1e-4, size=grad.shape)
        return grad

    def quantum_annealing_optimization(self, cost_function, max_iter: int = 200) -> Dict[str, Any]:
        """Quantum annealing style search with stochastic tunneling."""
        best_solution: Optional[np.ndarray] = None
        best_cost = float("inf")
        current = np.random.normal(0, 1, self.num_qubits)

        for iteration in range(max_iter):
            tunneling_prob = math.exp(-iteration / max(1, max_iter))
            if np.random.random() < tunneling_prob:
                candidate = self._quantum_tunneling()
            else:
                candidate = self._quantum_gradient_step(cost_function)

            c = float(cost_function(candidate))
            if c < best_cost:
                best_cost = c
                best_solution = candidate
                current = candidate
            else:
                # Metropolis-like accept with small probability
                if np.random.random() < math.exp(-(c - best_cost)):
                    current = candidate

        return {
            "solution": best_solution,
            "cost": best_cost,
            "quantum_entropy": self._calculate_quantum_entropy(),
        }

    def _quantum_tunneling(self) -> np.ndarray:
        """Quantum tunneling to escape local minima"""
        return np.random.normal(0, 1, self.num_qubits)

    def _quantum_gradient_step(self, cost_function) -> np.ndarray:
        """Gradient step with quantum fluctuations"""
        current = np.random.normal(0, 1, self.num_qubits)
        gradient = self._estimate_gradient(cost_function, current)
        quantum_noise = np.random.normal(0, 0.05, self.num_qubits)
        return current - 0.01 * gradient + quantum_noise

    def _calculate_quantum_entropy(self) -> float:
        """Calculate quantum entropy of the system"""
        probabilities = np.abs(self.quantum_state) ** 2
        p = probabilities / max(1e-12, probabilities.sum())
        return float(-(p * np.log(p + 1e-12)).sum())


class SwarmCognitiveNetwork:
    """Swarm intelligence for emergent network behavior"""

    def __init__(self, num_agents: int = 50, search_space: Tuple[float, float] = (-10, 10)):
        self.num_agents = num_agents
        self.search_space = search_space
        self.dim = 10
        self.agents = self._initialize_agents()
        self.global_best: Optional[Dict[str, Any]] = None
        self.emergence_threshold = 0.7
        # PSO coefficients
        self.w_inertia = 0.7
        self.c_cog = 1.5
        self.c_soc = 1.5

    def _initialize_agents(self) -> List[Dict[str, Any]]:
        agents: List[Dict[str, Any]] = []
        low, high = self.search_space
        for i in range(self.num_agents):
            position = np.random.uniform(low, high, self.dim)
            velocity = np.random.uniform(-1, 1, self.dim)
            agents.append({
                "id": i,
                "position": position,
                "velocity": velocity,
                "personal_best": position.copy(),
                "personal_best_cost": float("inf"),
                "cognitive_memory": [],
                "social_influence": 0.5,
            })
        return agents

    def optimize_swarm(self, objective_function, max_iterations: int = 100) -> Dict[str, Any]:
        swarm_intelligence: List[float] = []
        emergent_behaviors: List[Dict[str, Any]] = []

        for _ in range(max_iterations):
            for agent in self.agents:
                cost = float(objective_function(agent["position"]))
                if cost < agent["personal_best_cost"]:
                    agent["personal_best"] = agent["position"].copy()
                    agent["personal_best_cost"] = cost
                if self.global_best is None or cost < self.global_best["cost"]:
                    self.global_best = {"position": agent["position"].copy(), "cost": cost, "agent_id": agent["id"]}

            if self._detect_emergent_behavior():
                emergent_behaviors.append(self._capture_emergent_pattern())

            self._update_swarm_dynamics()
            swarm_intelligence.append(self._calculate_swarm_intelligence())

        return {
            "global_best": self.global_best,
            "swarm_intelligence": swarm_intelligence,
            "emergent_behaviors": emergent_behaviors,
            "final_swarm_state": self._analyze_swarm_state(),
        }

    def _detect_emergent_behavior(self) -> bool:
        positions = np.array([a["position"] for a in self.agents])
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        coordination = 1.0 / (np.std(distances) + 1e-12)
        return coordination > self.emergence_threshold

    def _capture_emergent_pattern(self) -> Dict[str, Any]:
        positions = np.array([a["position"] for a in self.agents])
        return {
            "pattern_type": self._classify_pattern(positions),
            "coordination_level": float(np.std(positions)),
            "swarm_entropy": self._calculate_swarm_entropy(positions),
            "topology": self._analyze_swarm_topology(positions),
        }

    def _update_swarm_dynamics(self) -> None:
        if self.global_best is None:
            return
        gbest = self.global_best["position"]
        for agent in self.agents:
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            cognitive = self.c_cog * r1 * (agent["personal_best"] - agent["position"])
            social = self.c_soc * r2 * (gbest - agent["position"])
            agent["velocity"] = self.w_inertia * agent["velocity"] + cognitive + social
            agent["position"] = agent["position"] + agent["velocity"]

    def _calculate_swarm_intelligence(self) -> float:
        positions = np.array([a["position"] for a in self.agents])
        diversity = float(np.mean(np.std(positions, axis=0)))
        convergence = 1.0 / (1.0 + diversity)
        return diversity * convergence

    def _classify_pattern(self, positions: np.ndarray) -> str:
        stds = np.std(positions, axis=0)
        if float(stds.mean()) < 0.5:
            return "converged-cluster"
        if float(stds.max()) - float(stds.min()) > 2.0:
            return "elongated-swarm"
        return "dispersed"

    def _calculate_swarm_entropy(self, positions: np.ndarray) -> float:
        d = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
        hist, _ = np.histogram(d, bins=10, density=True)
        hist = hist + 1e-12
        hist = hist / hist.sum()
        return float(-(hist * np.log(hist)).sum())

    def _analyze_swarm_topology(self, positions: np.ndarray) -> Dict[str, float]:
        # Build k-NN graph (k=5)
        k = min(5, len(positions) - 1)
        G = nx.Graph()
        for idx, p in enumerate(positions):
            dists = np.linalg.norm(positions - p, axis=1)
            neighbors = list(np.argsort(dists))[1 : k + 1]
            for nb in neighbors:
                G.add_edge(idx, int(nb), weight=float(dists[int(nb)]))
        clustering = nx.average_clustering(G, weight=None) if len(G) else 0.0
        degrees = [deg for _, deg in G.degree()]
        avg_deg = float(np.mean(degrees)) if degrees else 0.0
        return {"avg_degree": avg_deg, "clustering": float(clustering)}

    def _analyze_swarm_state(self) -> Dict[str, Any]:
        if self.global_best is None:
            return {"status": "initializing"}
        return {
            "best_cost": float(self.global_best["cost"]),
            "best_agent": int(self.global_best["agent_id"]),
        }


class NeuromorphicProcessor:
    """Neuromorphic computing interface for cognitive tasks"""

    def __init__(self, num_neurons: int = 1000):
        self.num_neurons = num_neurons
        self.neuron_states = self._initialize_neurons()
        self.synaptic_weights = self._initialize_synapses()
        self.spike_history: List[np.ndarray] = []

    def _initialize_neurons(self) -> Dict[str, np.ndarray]:
        return {
            "membrane_potentials": np.random.uniform(-70, -50, self.num_neurons),
            "recovery_variables": np.zeros(self.num_neurons),
            "firing_rates": np.zeros(self.num_neurons),
            "adaptation_currents": np.zeros(self.num_neurons),
        }

    def _initialize_synapses(self) -> np.ndarray:
        weights = np.random.normal(0, 0.1, (self.num_neurons, self.num_neurons))
        for i in range(self.num_neurons):
            neighbors = [(i + j) % self.num_neurons for j in range(-5, 6) if j != 0]
            for neighbor in neighbors:
                weights[i, neighbor] = np.random.normal(0.5, 0.1)
        return weights

    def process_spiking_input(self, input_spikes: np.ndarray, timesteps: int = 100) -> Dict[str, Any]:
        outputs: List[float] = []
        spike_trains: List[np.ndarray] = []
        for _ in range(timesteps):
            self._update_neuron_dynamics(input_spikes)
            spikes = self._detect_spikes()
            spike_trains.append(spikes)
            outputs.append(float(np.mean(spikes[-100:])))
            self._update_synaptic_plasticity(spikes)
        self.spike_history.extend(spike_trains)
        return {
            "output_activity": outputs,
            "spike_trains": spike_trains,
            "network_entropy": self._calculate_network_entropy(),
            "criticality_measure": self._assess_criticality(),
        }

    def _update_neuron_dynamics(self, input_currents: np.ndarray) -> None:
        v = self.neuron_states["membrane_potentials"]
        u = self.neuron_states["recovery_variables"]
        dv = 0.04 * v**2 + 5 * v + 140 - u + input_currents
        v_new = v + dv * 0.5
        du = 0.02 * (0.2 * v - u)
        u_new = u + du * 0.5
        spiked = v_new >= 30
        v_new[spiked] = -65
        u_new[spiked] = u[spiked] + 8
        self.neuron_states["membrane_potentials"] = v_new
        self.neuron_states["recovery_variables"] = u_new
        self.neuron_states["firing_rates"][spiked] += 1

    def _detect_spikes(self) -> np.ndarray:
        return self.neuron_states["membrane_potentials"] >= 30

    def _update_synaptic_plasticity(self, spikes: np.ndarray) -> None:
        # Simple STDP-like rule on a subset for stability
        active = np.where(spikes)[0]
        if active.size == 0:
            return
        idx = active[: min(10, active.size)]
        self.synaptic_weights[idx[:, None], idx] += 0.001
        self.synaptic_weights *= 0.9999

    def _calculate_network_entropy(self) -> float:
        rates = self.neuron_states["firing_rates"]
        if rates.sum() <= 0:
            return 0.0
        p = rates / rates.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def _assess_criticality(self) -> float:
        last = self.spike_history[-1] if self.spike_history else np.zeros(self.num_neurons, dtype=bool)
        return float(last.mean())


class HolographicDataEngine:
    """Holographic data representation and processing"""

    def __init__(self, data_dim: int = 256):
        self.data_dim = data_dim
        self.holographic_memory = np.zeros((data_dim, data_dim), dtype=np.complex128)

    def encode_holographic(self, data: np.ndarray) -> np.ndarray:
        data2d = data.reshape(self.data_dim, self.data_dim)
        data_freq = np.fft.fft2(data2d)
        random_phase = np.exp(1j * 2 * np.pi * np.random.random((self.data_dim, self.data_dim)))
        hologram = data_freq * random_phase
        self.holographic_memory += hologram
        return hologram

    def recall_holographic(self, partial_input: np.ndarray, iterations: int = 10) -> np.ndarray:
        current = partial_input.copy()
        for _ in range(iterations):
            estimate_freq = np.fft.fft2(current)
            correction = np.exp(1j * np.angle(self.holographic_memory + 1e-12))
            updated_freq = np.abs(estimate_freq) * correction
            current = np.fft.ifft2(updated_freq).real
            known_mask = ~np.isnan(partial_input)
            current[known_mask] = partial_input[known_mask]
        return current

    def associative_recall(self, query: np.ndarray, threshold: float = 0.8) -> List[Dict[str, Any]]:
        similarities: List[Dict[str, Any]] = []
        q = query.flatten()
        for i in range(self.data_dim):
            pattern = self.holographic_memory[i, :].real
            if pattern.size == 0:
                continue
            sim = float(np.corrcoef(q[: pattern.size], pattern.flatten())[0, 1])
            if sim > threshold:
                similarities.append({"pattern_index": i, "similarity": sim})
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)


class MorphogeneticSystem:
    """Morphogenetic system for self-organizing structure growth"""

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.morphogen_fields = self._initialize_morphogen_fields()
        self.cell_states = self._initialize_cell_states()

    def _initialize_morphogen_fields(self) -> Dict[str, np.ndarray]:
        return {
            "activator": np.random.random((self.grid_size, self.grid_size)),
            "inhibitor": np.random.random((self.grid_size, self.grid_size)),
            "growth_factor": np.zeros((self.grid_size, self.grid_size)),
        }

    def _initialize_cell_states(self) -> np.ndarray:
        return np.random.choice([0, 1], (self.grid_size, self.grid_size))

    def grow_structure(self, pattern_template: np.ndarray, iterations: int = 200) -> Dict[str, Any]:
        pattern_evolution: List[Dict[str, Any]] = []
        for iteration in range(iterations):
            self._update_reaction_diffusion()
            self._update_cell_states(pattern_template)
            if iteration % 50 == 0:
                pattern_evolution.append(self._analyze_pattern_formation(pattern_template))
            if self._pattern_converged(pattern_template):
                break
        return {
            "final_pattern": self.cell_states,
            "pattern_evolution": pattern_evolution,
            "morphogen_final_state": self.morphogen_fields,
            "convergence_iteration": iteration,
        }

    def _update_reaction_diffusion(self) -> None:
        a = self.morphogen_fields["activator"]
        b = self.morphogen_fields["inhibitor"]
        da = 0.1 * a - a * b**2 + 0.01
        db = 0.1 * b + a * b**2 - 0.12 * b
        diffusion_a = 0.01 * self._laplacian(a)
        diffusion_b = 0.1 * self._laplacian(b)
        self.morphogen_fields["activator"] = np.clip(a + da + diffusion_a, 0, 1)
        self.morphogen_fields["inhibitor"] = np.clip(b + db + diffusion_b, 0, 1)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        return (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            - 4 * field
        )

    def _update_cell_states(self, pattern_template: np.ndarray) -> None:
        a = self.morphogen_fields["activator"]
        threshold = np.median(a)
        self.cell_states = (a > threshold).astype(int)

    def _analyze_pattern_formation(self, pattern_template: np.ndarray) -> Dict[str, float]:
        current = self.cell_states.astype(float)
        tpl = pattern_template.astype(float)
        overlap = float((current * tpl).mean())
        energy = float((current**2).mean())
        return {"overlap": overlap, "energy": energy}

    def _pattern_converged(self, pattern_template: np.ndarray) -> bool:
        metrics = self._analyze_pattern_formation(pattern_template)
        return metrics["overlap"] > 0.9


class EmergentTechnologyOrchestrator:
    """Orchestrator for emergent technology integration"""

    def __init__(self):
        self.quantum_optimizer = QuantumInspiredOptimizer()
        self.swarm_network = SwarmCognitiveNetwork()
        self.neuromorphic_processor = NeuromorphicProcessor()
        self.holographic_engine = HolographicDataEngine()
        self.morphogenetic_system = MorphogeneticSystem()
        self.emergent_behaviors: List[Dict[str, Any]] = []
        self.cognitive_evolution: List[Dict[str, Any]] = []

    def orchestrate_emergent_communication(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        quantum_optimized = self._quantum_optimize_content(message)
        transmission_plan = self._swarm_optimize_transmission(quantum_optimized, context)
        adaptive_signals = self._neuromorphic_processing(transmission_plan)
        holographic_encoding = self._holographic_encode(adaptive_signals)
        emergent_protocol = self._grow_emergent_protocol(holographic_encoding)
        self._track_emergence(emergent_protocol)
        return {
            "quantum_optimized": quantum_optimized,
            "transmission_plan": transmission_plan,
            "adaptive_signals": adaptive_signals,
            "holographic_encoding": holographic_encoding,
            "emergent_protocol": emergent_protocol,
            "emergence_metrics": self._calculate_emergence_metrics(),
        }

    def _quantum_optimize_content(self, content: str) -> Dict[str, Any]:
        def cost(params: np.ndarray) -> float:
            complexity = float(np.sum(np.abs(params)))
            clarity = float(1.0 / (1.0 + np.var(params)))
            return complexity - clarity

        res = self.quantum_optimizer.quantum_annealing_optimization(cost)
        return {
            "optimized_parameters": res["solution"],
            "quantum_entropy": res["quantum_entropy"],
            "optimization_cost": res["cost"],
        }

    def _swarm_optimize_transmission(self, content: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        def objective(strategy_params: np.ndarray) -> float:
            bandwidth_eff = 1.0 / (1.0 + np.sum(np.abs(strategy_params[:3])))
            reliability = float(np.mean(strategy_params[3:6]))
            latency = float(np.sum(np.abs(strategy_params[6:])))
            return bandwidth_eff - reliability + 0.001 * latency

        res = self.swarm_network.optimize_swarm(objective)
        return {
            "optimal_strategy": res["global_best"],
            "swarm_intelligence": res["swarm_intelligence"][-1] if res["swarm_intelligence"] else 0.0,
            "emergent_behaviors_detected": len(res["emergent_behaviors"]),
        }

    def _neuromorphic_processing(self, transmission_plan: Dict[str, Any]) -> Dict[str, Any]:
        # Convert strategy vector into spike currents
        vec = transmission_plan.get("optimal_strategy", {}).get("position", np.zeros(10))
        currents = np.tanh(np.pad(vec, (0, max(0, self.neuromorphic_processor.num_neurons - vec.size)))[: self.neuromorphic_processor.num_neurons])
        currents = 5 * currents  # scale into excitatory current range
        out = self.neuromorphic_processor.process_spiking_input(currents, timesteps=50)
        return out

    def _holographic_encode(self, adaptive_signals: Dict[str, Any]) -> Dict[str, Any]:
        # Use last spike train to form an image-like vector
        spikes = adaptive_signals.get("spike_trains", [])
        if spikes:
            sig = spikes[-1].astype(float)
        else:
            sig = np.zeros(self.holographic_engine.data_dim * self.holographic_engine.data_dim)
        sig = np.pad(sig, (0, max(0, self.holographic_engine.data_dim**2 - sig.size)))[: self.holographic_engine.data_dim**2]
        hologram = self.holographic_engine.encode_holographic(sig)
        return {"mean_phase": float(np.angle(hologram).mean()), "energy": float(np.abs(hologram).mean())}

    def _grow_emergent_protocol(self, holographic_encoding: Dict[str, Any]) -> Dict[str, Any]:
        template = (np.random.random((self.morphogenetic_system.grid_size, self.morphogenetic_system.grid_size)) > 0.5).astype(int)
        growth = self.morphogenetic_system.grow_structure(template, iterations=150)
        return {"overlap": growth["pattern_evolution"][-1]["overlap"] if growth["pattern_evolution"] else 0.0}

    def _track_emergence(self, emergent_protocol: Dict[str, Any]) -> None:
        self.emergent_behaviors.append({"protocol": emergent_protocol})

    def _calculate_emergence_metrics(self) -> Dict[str, float]:
        return {"events": float(len(self.emergent_behaviors))}

    def evolve_cognitive_network(self, experiences: List[Dict[str, Any]], generations: int = 5) -> Dict[str, Any]:
        traj: List[Dict[str, Any]] = []
        for _ in range(generations):
            learning = self._learn_from_experiences(experiences)
            self._adapt_network_structures(learning)
            metrics = self._measure_cognitive_evolution()
            traj.append(metrics)
            if self._detect_cognitive_emergence(metrics):
                self.cognitive_evolution.append(self._capture_emergent_cognition())
        return {
            "evolutionary_trajectory": traj,
            "final_cognitive_state": self._analyze_cognitive_state(),
            "emergent_cognitions": self.cognitive_evolution,
        }

    def _learn_from_experiences(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        return {"avg_reward": float(np.mean([e.get("reward", 0.0) for e in experiences])) if experiences else 0.0}

    def _adapt_network_structures(self, learning: Dict[str, float]) -> None:
        # Placeholder for structural plasticity
        _ = learning

    def _measure_cognitive_evolution(self) -> Dict[str, float]:
        return {"complexity": float(np.random.random()), "robustness": float(np.random.random())}

    def _detect_cognitive_emergence(self, metrics: Dict[str, float]) -> bool:
        return metrics.get("complexity", 0.0) > 0.8 and metrics.get("robustness", 0.0) > 0.5

    def _capture_emergent_cognition(self) -> Dict[str, Any]:
        return {"signature": np.random.rand(4).tolist(), "time": float(np.random.random())}

    def _analyze_cognitive_state(self) -> Dict[str, Any]:
        return {"emergent_events": len(self.cognitive_evolution)}


def demo_emergent_technologies() -> Dict[str, Any]:
    orchestrator = EmergentTechnologyOrchestrator()
    test_message = "Emergent cognitive communication test"
    test_context = {"channel_conditions": {"snr": 25, "bandwidth": 1000}, "priority_level": "high", "content_type": "cognitive_directive"}
    result = orchestrator.orchestrate_emergent_communication(test_message, test_context)
    print("=== Emergent Technology Demonstration ===")
    print(f"Quantum Optimization Entropy: {result['quantum_optimized']['quantum_entropy']:.4f}")
    print(f"Swarm Intelligence: {result['transmission_plan']['swarm_intelligence']:.4f}")
    print(f"Emergent Behaviors: {result['transmission_plan']['emergent_behaviors_detected']}")
    print(f"Emergence Metrics: {result['emergence_metrics']}")
    return result


if __name__ == "__main__":
    demo_emergent_technologies()


