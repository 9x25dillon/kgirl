#!/usr/bin/env python3
"""
Neuro-Symbolic Adaptive Reflective Engine
==========================================

This module implements a comprehensive neuro-symbolic system that combines:
- Multiple analytical modules (entropy, reflection, matrix transformation, etc.)
- Feature extraction and neural-symbolic fusion
- Reinforcement learning for adaptive decision making
- Reflective database for self-tuning and memory

Author: Assistant
License: MIT
"""

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= Core Analytics Modules ============================

class EntropyAnalyzer:
    """Measures information entropy of data"""
    
    def measure(self, data: Any) -> float:
        s = str(data)
        if not s:
            return 0.0
            
        counts: Dict[str, int] = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
            
        n = len(s)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
                
        return entropy

class DianneReflector:
    """Reflective analysis system for pattern detection and insight generation"""
    
    def reflect(self, data: Any) -> Dict[str, Any]:
        patterns = self._detect_patterns(data)
        head = str(data)[:40].replace("\n", " ")
        
        if "high_repetition" in patterns:
            insight = f"Cyclical resonance detected in: {head}..."
        elif "hierarchical_structure" in patterns:
            insight = f"Nested reality layers within: {head}..."
        else:
            insight = f"Linear transformation potential in: {head}..."
            
        return {
            "insight": insight, 
            "patterns": patterns, 
            "symbolic_depth": self._depth(data)
        }
    
    def _detect_patterns(self, data: Any) -> List[str]:
        s = str(data)
        patterns = []
        
        # High repetition pattern
        if len(s) > 100 and len(set(s)) < 20:
            patterns.append("high_repetition")
            
        # Hierarchical structure pattern
        if s.count('\n') > 5 and any(c in s for c in ['{', '[', '(', '<']):
            patterns.append("hierarchical_structure")
            
        # Numerical pattern
        if sum(c.isdigit() for c in s) > len(s) * 0.3:
            patterns.append("numerical_dominant")
            
        return patterns
    
    def _depth(self, data: Any) -> int:
        s = str(data)
        return min(10, len(s) // 100)

class MatrixTransformer:
    """Projects data into matrix space for dimensional analysis"""
    
    def project(self, data: Any) -> Dict[str, Any]:
        dims = self._analyze(data)
        h = hash(str(data)) & 0xFFFFFFFF
        rank = int(dims["rank"])
        
        eigenvalues = [math.sin(h * 0.001 * i) for i in range(max(1, min(3, rank)))]
        
        return {
            "projected_rank": dims["rank"],
            "structure": dims["structure"],
            "eigenvalues": eigenvalues,
            "determinant": math.cos(h * 0.0001),
            "trace": math.tan(h * 0.00001) if (h % 100) else 0.0,
        }
    
    def _analyze(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        unique_chars = len(set(s))
        
        return {
            "rank": min(10, len(s) // 50),
            "structure": "sparse" if unique_chars < 20 else "dense"
        }

class JuliaSymbolEngine:
    """Symbolic computation engine with polynomial analysis"""
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        coeffs = self._coeffs(data)
        return {
            "chebyshev_polynomial": self._poly(coeffs),
            "coefficients": coeffs,
            "derivatives": self._derivs(coeffs),
            "critical_points": self._crit(coeffs),
        }
    
    def _coeffs(self, data: Any) -> List[float]:
        s = str(data)
        return [
            math.sin(hash(s[i:i+4]) % 100) if i < len(s) else 0.0 
            for i in range(5)
        ]
    
    def _poly(self, coeffs: List[float]) -> str:
        return f"{coeffs[0]:.3f} + {coeffs[1]:.3f}x + {coeffs[2]:.3f}x²"
    
    def _derivs(self, coeffs: List[float]) -> List[float]:
        return [coeffs[1], 2*coeffs[2], 0.0, 0.0, 0.0]
    
    def _crit(self, coeffs: List[float]) -> List[float]:
        if abs(coeffs[2]) > 1e-6:
            return [-coeffs[1]/(2*coeffs[2])]
        return []

class ChoppyProcessor:
    """Advanced chunking processor with multiple strategies"""
    
    def chunk(self, data: Any, chunk_size: int = 64, overlap: int = 16) -> Dict[str, Any]:
        s = str(data)
        step = max(1, chunk_size - overlap)
        
        # Standard chunking
        standard_chunks = [s[i:i + chunk_size] for i in range(0, len(s), step)]
        
        # Semantic chunking
        words = s.split()
        word_chunk_size = max(1, chunk_size // 5)
        semantic_chunks = [
            " ".join(words[i:i + word_chunk_size]) 
            for i in range(0, len(words), word_chunk_size)
        ]
        
        return {
            "standard": standard_chunks,
            "semantic": semantic_chunks,
            "fibonacci": self._fibonacci_chunk(s),
            "statistics": {
                "total_length": len(s), 
                "chunk_count": len(standard_chunks), 
                "average_chunk_size": len(s) / max(1, len(standard_chunks))
            },
        }
    
    def _fibonacci_chunk(self, s: str) -> List[str]:
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        chunks = []
        pos = 0
        
        for f in fib:
            if pos >= len(s):
                break
            chunks.append(s[pos:pos+f])
            pos += f
            
        return chunks

class EndpointCaster:
    """Generates API endpoints and metadata for data artifacts"""
    
    def generate(self, data: Any) -> Dict[str, Any]:
        sig = hashlib.sha256(
            json.dumps(data, default=str, sort_keys=True).encode()
        ).hexdigest()[:12]
        base = uuid.uuid4().hex[:6]
        
        return {
            "primary_endpoint": f"/api/v1/cast/{base}",
            "versioned_endpoints": [
                f"/api/v1/cast/{base}/reflect",
                f"/api/v1/cast/{base}/transform",
                f"/api/v1/cast/{base}/metadata",
                f"/api/v2/mirror/{sig}",
            ],
            "artifact_id": f"art-{uuid.uuid4().hex[:8]}",
            "metadata": {
                "content_type": self._content_type(data), 
                "estimated_size": len(str(data)), 
                "complexity": self._complexity(data)
            },
        }
    
    def _content_type(self, data: Any) -> str:
        s = str(data)
        if len(s) < 100:
            return "text/plain"
        if any(c in s for c in ['{', '[', '(']):
            return "application/json"
        return "text/plain"
    
    def _complexity(self, data: Any) -> float:
        s = str(data)
        return min(1.0, len(set(s)) / max(1, len(s)))

class CarryOnManager:
    """Memory management system with access tracking"""
    
    def __init__(self, max_history: int = 200):
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.access: Dict[str, int] = {}
    
    def store(self, key: str, value: Any) -> None:
        self.memory[key] = value
        self.access[key] = int(time.time())
        
        self.history.append({
            "key": key, 
            "value": str(value)[:100], 
            "time": time.time()
        })
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def retrieve(self, key: str) -> Optional[Any]:
        if key in self.memory:
            self.access[key] = int(time.time())
            return self.memory[key]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "memory_items": len(self.memory),
            "history_length": len(self.history),
            "most_accessed": max(self.access.items(), key=lambda x: x[1]) if self.access else None
        }

class SemanticMapper:
    """Maps text to semantic networks and categories"""
    
    def __init__(self):
        self.semantic_networks = {
            "reflection": ["mirror", "echo", "reverberation", "contemplation", "introspection"],
            "transformation": ["metamorphosis", "mutation", "evolution", "adaptation", "transmutation"],
            "analysis": ["examination", "scrutiny", "dissection", "investigation", "exploration"],
            "synthesis": ["combination", "fusion", "amalgamation", "integration", "unification"],
        }
    
    def map(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {}
        
        for category, words in self.semantic_networks.items():
            score = sum(1 for word in words if word in text_lower)
            scores[category] = score / len(words)
            
        return scores

class LoveReflector:
    """Emotional and poetic analysis system"""
    
    def infuse(self, data: Any) -> Dict[str, Any]:
        text = str(data)
        return {
            "poetic": self._poem(text), 
            "emotional_resonance": self._emotional_resonance(text), 
            "love_quotient": self._love_quotient(text), 
            "harmony_index": self._harmony_index(text)
        }
    
    def _poem(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        return f"{words[0]} {words[1]} {words[-1]}"
    
    def _emotional_resonance(self, text: str) -> float:
        emotional_words = ['love', 'hate', 'joy', 'sad', 'happy', 'angry', 'peace', 'war', 'hope', 'fear']
        return sum(1 for word in emotional_words if word in text.lower()) / len(emotional_words)
    
    def _love_quotient(self, text: str) -> float:
        love_words = ['love', 'heart', 'soul', 'beauty', 'harmony', 'unity']
        return sum(text.lower().count(word) for word in love_words) / max(1, len(text.split()))
    
    def _harmony_index(self, text: str) -> float:
        # Simple harmony measure based on character distribution
        if not text:
            return 0.0
        char_counts = {}
        for c in text.lower():
            if c.isalpha():
                char_counts[c] = char_counts.get(c, 0) + 1
        
        if not char_counts:
            return 0.0
            
        # Calculate variance of character frequencies
        frequencies = list(char_counts.values())
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        
        # Lower variance = higher harmony
        return 1.0 / (1.0 + variance)

class FractalResonator:
    """Fractal analysis system for recursive pattern detection"""
    
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
    
    def cascade(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        layers = []
        
        for depth in range(1, min(self.max_depth + 1, len(s) // 10 + 1)):
            chunk = s[:depth * 10]
            entropy = EntropyAnalyzer().measure(chunk)
            
            layers.append({
                "depth": depth, 
                "entropy": entropy, 
                "content": chunk[:50] + "..." if len(chunk) > 50 else chunk
            })
            
        return {
            "layers": layers, 
            "max_depth_reached": len(layers),
            "fractal_dimension": self._estimate_fractal_dimension(layers)
        }
    
    def _estimate_fractal_dimension(self, layers: List[Dict[str, Any]]) -> float:
        if len(layers) < 2:
            return 1.0
            
        # Simple box-counting approximation
        entropies = [layer["entropy"] for layer in layers]
        depths = [layer["depth"] for layer in layers]
        
        # Linear regression on log-log plot (simplified)
        if len(entropies) > 1:
            return abs(entropies[-1] - entropies[0]) / abs(depths[-1] - depths[0])
        return 1.0

# ===================== Neuro-Symbolic Control & Memory =======================

class FeatureExtractor:
    """Lightweight local features + optional imported embedding"""
    
    def __init__(self, dim: int = 64, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram
    
    def extract(self, text: str) -> List[float]:
        """Extract n-gram hash features"""
        s = text.lower()
        features = [0.0] * self.dim
        
        for i in range(len(s) - self.ngram + 1):
            ngram = s[i:i+self.ngram]
            idx = hash(ngram) % self.dim
            features[idx] += 1.0
        
        # Normalize
        total = sum(features)
        if total > 0:
            features = [f / total for f in features]
            
        return features

class NeuroSymbolicFusion:
    """Fuse neural features + symbolic metrics"""
    
    def __init__(self):
        # Learned (static) weights for demo; could be trained via RL
        self.w_neuro = 0.55
        self.w_symbol = 0.45
    
    def fuse(self, neuro_features: List[float], symbolic_metrics: Dict[str, float]) -> Dict[str, Any]:
        neuro_score = sum(neuro_features) / len(neuro_features) if neuro_features else 0.0
        symbol_score = sum(symbolic_metrics.values()) / len(symbolic_metrics) if symbolic_metrics else 0.0
        
        fused = self.w_neuro * neuro_score + self.w_symbol * symbol_score
        
        return {
            "neuro_score": neuro_score,
            "symbol_score": symbol_score,
            "fused_score": fused,
            "decision": "transmit" if fused > 0.5 else "hold"
        }

class DecisionLogger:
    """Logs decision events for analysis"""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
    
    def log(self, event: Dict[str, Any]) -> None:
        self.events.append({**event, "timestamp": time.time()})
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.events[-n:]
    
    def clear(self) -> None:
        self.events.clear()

class ReflectiveDB:
    """JSON file for self-tuning memory of configs & outcomes"""
    
    def __init__(self, path: str = "reflective_db.json"):
        self.path = path
        self._data: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load reflective DB: {e}")
                self._data = []
    
    def save(self) -> None:
        try:
            with open(self.path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reflective DB: {e}")
    
    def add_record(self, record: Dict[str, Any]) -> None:
        self._data.append(record)
        self.save()
    
    def query(self, filter_func: callable) -> List[Dict[str, Any]]:
        return [record for record in self._data if filter_func(record)]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_records": len(self._data),
            "latest_timestamp": max((r.get("timestamp", 0) for r in self._data), default=0)
        }

class RLAgent:
    """Tiny contextual bandit for adaptive decision making"""
    
    def __init__(self, actions: List[str] = None, eps: float = 0.1):
        self.actions = actions or ["bpsk", "qpsk", "ofdm"]
        self.eps = eps
        # state -> action -> {q, n}
        self.q: Dict[Tuple[int, int, int], Dict[str, Dict[str, float]]] = {}
    
    def choose_action(self, state: Tuple[int, int, int]) -> str:
        if np.random.random() < self.eps or state not in self.q:
            return np.random.choice(self.actions)
        
        action_values = {
            action: self.q[state][action]["q"] 
            for action in self.actions 
            if action in self.q[state]
        }
        
        if not action_values:
            return np.random.choice(self.actions)
            
        return max(action_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple[int, int, int], action: str, reward: float) -> None:
        if state not in self.q:
            self.q[state] = {a: {"q": 0.0, "n": 0} for a in self.actions}
        
        if action not in self.q[state]:
            self.q[state][action] = {"q": 0.0, "n": 0}
        
        self.q[state][action]["n"] += 1
        n = self.q[state][action]["n"]
        old_q = self.q[state][action]["q"]
        
        # Incremental mean update
        self.q[state][action]["q"] = old_q + (reward - old_q) / n
    
    def get_stats(self) -> Dict[str, Any]:
        total_states = len(self.q)
        total_updates = sum(
            sum(action_data["n"] for action_data in state_actions.values())
            for state_actions in self.q.values()
        )
        
        return {
            "total_states": total_states,
            "total_updates": total_updates,
            "epsilon": self.eps
        }

# ======================= Mirror Cast + Adaptive Planner =======================

class MirrorCastEngine:
    """Main engine that coordinates all analytical modules"""
    
    def __init__(self):
        self.entropy = EntropyAnalyzer()
        self.reflector = DianneReflector()
        self.matrix = MatrixTransformer()
        self.symbols = JuliaSymbolEngine()
        self.choppy = ChoppyProcessor()
        self.endpoints = EndpointCaster()
        self.memory = CarryOnManager()
        self.semantic = SemanticMapper()
        self.love = LoveReflector()
        self.fractal = FractalResonator()
    
    def cast(self, data: Any) -> Dict[str, Any]:
        """Perform comprehensive analysis of input data"""
        start_time = time.time()
        
        result = {
            "entropy": self.entropy.measure(data),
            "reflection": self.reflector.reflect(data),
            "matrix": self.matrix.project(data),
            "symbolic": self.symbols.analyze(data),
            "chunks": self.choppy.chunk(data),
            "endpoints": self.endpoints.generate(data),
            "semantic": self.semantic.map(str(data)),
            "love": self.love.infuse(data),
            "fractal": self.fractal.cascade(data),
            "timestamp": time.time(),
            "processing_time": time.time() - start_time
        }
        
        # Store in memory
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()[:8]
        self.memory.store(f"cast_{data_hash}", result)
        
        return result

class AdaptiveLinkPlanner:
    """Neuro-Symbolic + RL planner for adaptive system configuration"""
    
    def __init__(self, db_path: str = "reflective_db.json"):
        self.extractor = FeatureExtractor()
        self.fusion = NeuroSymbolicFusion()
        self.agent = RLAgent(actions=["bpsk", "qpsk", "ofdm"], eps=0.1)
        self.db = ReflectiveDB(db_path)
        self.log = DecisionLogger()
    
    def plan(self, text: str, analysis: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], str]:
        """Generate adaptive configuration plan"""
        
        # Extract features
        features = self.extractor.extract(text)
        
        # Create symbolic metrics from analysis
        symbolic_metrics = {
            "entropy": analysis.get("entropy", 0.0),
            "complexity": analysis.get("endpoints", {}).get("metadata", {}).get("complexity", 0.5),
            "semantic_density": sum(analysis.get("semantic", {}).values()) / max(1, len(analysis.get("semantic", {}))),
            "harmony": analysis.get("love", {}).get("harmony_index", 0.5),
            "fractal_dimension": analysis.get("fractal", {}).get("fractal_dimension", 1.0)
        }
        
        # Fuse neuro-symbolic
        fusion_result = self.fusion.fuse(features, symbolic_metrics)
        
        # Create state representation (discretize continuous values)
        entropy_bin = min(9, int(analysis.get("entropy", 0.0) * 2))
        complexity_bin = min(9, int(symbolic_metrics["complexity"] * 10))
        harmony_bin = min(9, int(symbolic_metrics["harmony"] * 10))
        state = (entropy_bin, complexity_bin, harmony_bin)
        
        # Choose action
        action = self.agent.choose_action(state)
        
        # Generate configuration
        config = self._action_to_config(action, symbolic_metrics)
        
        explanation = (
            f"Neuro-symbolic score: {fusion_result['fused_score']:.3f}, "
            f"chose {action.upper()} for state {state}, "
            f"entropy: {analysis.get('entropy', 0):.2f}, "
            f"harmony: {symbolic_metrics['harmony']:.2f}"
        )
        
        # Log decision
        self.log.log({
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:8],
            "state": state,
            "action": action,
            "fusion_result": fusion_result,
            "explanation": explanation
        })
        
        return config, explanation
    
    def _action_to_config(self, action: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Convert action to system configuration"""
        base_config = {
            "modulation": action,
            "sample_rate": 48000,
            "symbol_rate": 1200,
            "amplitude": 0.7
        }
        
        # Adjust based on action and metrics
        if action == "bpsk":
            base_config["symbol_rate"] = 1200
        elif action == "qpsk":
            base_config["symbol_rate"] = int(2400 * metrics.get("harmony", 0.5))
        elif action == "ofdm":
            base_config["symbol_rate"] = int(4800 * metrics.get("complexity", 0.5))
        
        return base_config
    
    def reward_and_record(self, text: str, config: Dict[str, Any], explanation: str, 
                         success: bool, **kwargs) -> None:
        """Update RL agent and record results"""
        
        # Simple reward function
        reward = 1.0 if success else -1.0
        
        # Adjust reward based on additional metrics
        harmony = kwargs.get("harmony", 0.5)
        reward *= harmony
        
        # Reconstruct state (this should match the state used in plan())
        entropy = kwargs.get("entropy", 0.0)
        complexity = kwargs.get("complexity", 0.5)
        
        entropy_bin = min(9, int(entropy * 2))
        complexity_bin = min(9, int(complexity * 10))
        harmony_bin = min(9, int(harmony * 10))
        state = (entropy_bin, complexity_bin, harmony_bin)
        
        action = config.get("modulation", "bpsk")
        
        # Update Q-values
        self.agent.update(state, action, reward)
        
        # Record to database
        self.db.add_record({
            "timestamp": time.time(),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:8],
            "state": state,
            "action": action,
            "reward": reward,
            "success": success,
            "config": config,
            "explanation": explanation,
            **kwargs
        })

# =============================== Visualization ===============================

def plot_fractal_layers(fractal_data: Dict[str, Any], save_path: str = "fractal_layers.png"):
    """Plot fractal analysis layers"""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping plot")
        return
    
    layers = fractal_data.get("layers", [])
    if not layers:
        return
    
    depths = [layer["depth"] for layer in layers]
    entropies = [layer["entropy"] for layer in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, entropies, 'o-', linewidth=2, markersize=8)
    plt.title("Fractal Entropy vs Depth")
    plt.xlabel("Depth")
    plt.ylabel("Entropy")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_decision_timeline(decisions: List[Dict[str, Any]], save_path: str = "decisions.png"):
    """Plot decision timeline"""
    if not HAS_MATPLOTLIB or not decisions:
        return
    
    timestamps = [d.get("timestamp", 0) for d in decisions]
    actions = [d.get("action", "unknown") for d in decisions]
    
    # Convert to relative time
    if timestamps:
        start_time = min(timestamps)
        rel_times = [(t - start_time) / 60 for t in timestamps]  # minutes
        
        plt.figure(figsize=(12, 6))
        
        # Create action mapping for colors
        unique_actions = list(set(actions))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_actions)))
        action_colors = {action: colors[i] for i, action in enumerate(unique_actions)}
        
        for i, (time, action) in enumerate(zip(rel_times, actions)):
            plt.scatter(time, i, c=[action_colors[action]], s=100, alpha=0.7)
            plt.text(time, i + 0.1, action, fontsize=8, ha='center')
        
        plt.title("Decision Timeline")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Decision Index")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def demo_neuro_symbolic_engine():
    """Demonstration of the neuro-symbolic engine"""
    
    # Create engine
    engine = MirrorCastEngine()
    planner = AdaptiveLinkPlanner()
    
    # Test data
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a hole in the ground there lived a hobbit",
        "To be or not to be, that is the question",
        "E=mc² represents the mass-energy equivalence",
        "Love is the bridge between two hearts"
    ]
    
    results = []
    
    for i, text in enumerate(test_texts):
        logger.info(f"Processing text {i+1}: {text[:30]}...")
        
        # Perform analysis
        analysis = engine.cast(text)
        
        # Generate plan
        config, explanation = planner.plan(text, analysis)
        
        # Simulate success/failure
        success = np.random.random() > 0.3  # 70% success rate
        
        # Update planner
        planner.reward_and_record(
            text, config, explanation, success,
            entropy=analysis["entropy"],
            complexity=analysis["endpoints"]["metadata"]["complexity"],
            harmony=analysis["love"]["harmony_index"]
        )
        
        results.append({
            "text": text,
            "analysis": analysis,
            "config": config,
            "explanation": explanation,
            "success": success
        })
    
    # Generate visualizations
    if results:
        # Plot fractal analysis for first result
        plot_fractal_layers(results[0]["analysis"]["fractal"])
        
        # Plot decision timeline
        plot_decision_timeline(planner.log.events)
    
    # Print summary
    logger.info("=== Neuro-Symbolic Engine Demo Complete ===")
    logger.info(f"Processed {len(results)} texts")
    logger.info(f"Success rate: {sum(r['success'] for r in results) / len(results) * 100:.1f}%")
    logger.info(f"RL Agent stats: {planner.agent.get_stats()}")
    logger.info(f"Memory stats: {engine.memory.get_stats()}")
    
    return results

if __name__ == "__main__":
    demo_neuro_symbolic_engine()