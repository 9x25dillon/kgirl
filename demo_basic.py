#!/usr/bin/env python3
"""
Basic Demo without External Dependencies
=======================================

Demonstrates core concepts and architecture without requiring
numpy, scipy, torch, or other external libraries.

This shows the system design and key algorithms in pure Python.
"""

import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

class BasicEntropyAnalyzer:
    """Pure Python entropy analysis"""
    
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

class BasicReflector:
    """Pure Python reflective analysis"""
    
    def reflect(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        patterns = []
        
        # Detect patterns
        if len(s) > 100 and len(set(s)) < 20:
            patterns.append("high_repetition")
        if s.count('\n') > 5:
            patterns.append("hierarchical_structure")
        if sum(c.isdigit() for c in s) > len(s) * 0.3:
            patterns.append("numerical_dominant")
            
        return {
            "insight": f"Analyzed {len(s)} characters with {len(patterns)} patterns",
            "patterns": patterns,
            "symbolic_depth": min(10, len(s) // 100)
        }

class BasicModulator:
    """Pure Python modulation concepts"""
    
    @staticmethod
    def to_bits(data: bytes) -> List[int]:
        """Convert bytes to bit list"""
        return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]
    
    @staticmethod
    def from_bits(bits: List[int]) -> bytes:
        """Convert bit list to bytes"""
        if len(bits) % 8 != 0:
            bits = bits + [0] * (8 - len(bits) % 8)
        
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i:i+8]:
                byte = (byte << 1) | (1 if b else 0)
            result.append(byte)
        
        return bytes(result)
    
    @staticmethod
    def hamming74_encode(data_bits: List[int]) -> List[int]:
        """Hamming (7,4) encoding"""
        if len(data_bits) % 4 != 0:
            data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
        
        encoded = []
        for i in range(0, len(data_bits), 4):
            d0, d1, d2, d3 = data_bits[i:i+4]
            p1 = d0 ^ d1 ^ d3
            p2 = d0 ^ d2 ^ d3
            p3 = d1 ^ d2 ^ d3
            encoded.extend([p1, p2, d0, p3, d1, d2, d3])
        
        return encoded
    
    @staticmethod
    def simulate_bfsk(bits: List[int], sample_rate: int = 8000, symbol_rate: int = 1000) -> List[float]:
        """Simulate BFSK modulation (returns sample points)"""
        samples_per_bit = sample_rate // symbol_rate
        f0, f1 = 1200.0, 2200.0  # Frequencies for 0 and 1
        
        signal = []
        for bit in bits:
            freq = f1 if bit else f0
            for sample in range(samples_per_bit):
                t = sample / sample_rate
                amplitude = 0.7 * math.sin(2 * math.pi * freq * t)
                signal.append(amplitude)
        
        return signal

class BasicAdaptivePlanner:
    """Pure Python adaptive planning"""
    
    def __init__(self):
        self.q_values: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.actions = ["bpsk", "qpsk", "ofdm"]
        self.epsilon = 0.1
    
    def choose_action(self, state: Tuple[int, int]) -> str:
        """Choose action using epsilon-greedy policy"""
        import random
        
        if random.random() < self.epsilon or state not in self.q_values:
            return random.choice(self.actions)
        
        action_values = self.q_values[state]
        return max(action_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple[int, int], action: str, reward: float):
        """Update Q-values"""
        if state not in self.q_values:
            self.q_values[state] = {a: 0.0 for a in self.actions}
        
        # Simple Q-learning update
        alpha = 0.1
        old_q = self.q_values[state][action]
        self.q_values[state][action] = old_q + alpha * (reward - old_q)

class BasicWaveCaster:
    """Main system demonstration"""
    
    def __init__(self):
        self.entropy_analyzer = BasicEntropyAnalyzer()
        self.reflector = BasicReflector()
        self.modulator = BasicModulator()
        self.planner = BasicAdaptivePlanner()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        return {
            "entropy": self.entropy_analyzer.measure(text),
            "reflection": self.reflector.reflect(text),
            "length": len(text),
            "unique_chars": len(set(text)),
            "timestamp": time.time()
        }
    
    def encode_and_modulate(self, text: str) -> Dict[str, Any]:
        """Encode text and simulate modulation"""
        # Convert to bytes and bits
        data_bytes = text.encode('utf-8')
        data_bits = self.modulator.to_bits(data_bytes)
        
        # Apply FEC
        encoded_bits = self.modulator.hamming74_encode(data_bits)
        
        # Simulate modulation
        signal_samples = self.modulator.simulate_bfsk(encoded_bits)
        
        return {
            "original_bytes": len(data_bytes),
            "data_bits": len(data_bits),
            "encoded_bits": len(encoded_bits),
            "signal_samples": len(signal_samples),
            "code_rate": len(data_bits) / len(encoded_bits),
            "signal_duration": len(signal_samples) / 8000.0  # seconds at 8kHz
        }
    
    def adaptive_planning_demo(self, texts: List[str], episodes: int = 10) -> Dict[str, Any]:
        """Demonstrate adaptive planning"""
        results = []
        
        for episode in range(episodes):
            text = texts[episode % len(texts)]
            analysis = self.analyze_text(text)
            
            # Create state from analysis
            entropy_bin = min(9, int(analysis["entropy"]))
            length_bin = min(9, len(text) // 10)
            state = (entropy_bin, length_bin)
            
            # Choose action
            action = self.planner.choose_action(state)
            
            # Simulate success (70% success rate)
            import random
            success = random.random() > 0.3
            reward = 1.0 if success else -1.0
            
            # Update planner
            self.planner.update(state, action, reward)
            
            results.append({
                "episode": episode + 1,
                "text_length": len(text),
                "entropy": analysis["entropy"],
                "state": state,
                "action": action,
                "success": success,
                "reward": reward
            })
        
        success_rate = sum(r["success"] for r in results) / len(results)
        
        return {
            "episodes": results,
            "success_rate": success_rate,
            "q_table_size": len(self.planner.q_values)
        }
    
    def demonstrate_system(self) -> Dict[str, Any]:
        """Complete system demonstration"""
        print("🚀 Enhanced WaveCaster Basic Demo")
        print("=" * 50)
        
        # Test texts
        test_texts = [
            "Hello, World! This is a basic test.",
            "The quick brown fox jumps over the lazy dog.",
            "In the realm of digital signal processing, modulation schemes transform data into waveforms.",
            "Artificial intelligence and machine learning are revolutionizing communication systems.",
            "E=mc² represents the mass-energy equivalence in Einstein's theory of relativity."
        ]
        
        results = {}
        
        # 1. Text Analysis Demo
        print("\n1. Text Analysis Demo")
        print("-" * 30)
        
        analysis_results = []
        for i, text in enumerate(test_texts):
            analysis = self.analyze_text(text)
            analysis_results.append(analysis)
            print(f"Text {i+1}: Entropy={analysis['entropy']:.2f}, "
                  f"Length={analysis['length']}, "
                  f"Unique={analysis['unique_chars']}")
        
        results["text_analysis"] = analysis_results
        
        # 2. Encoding and Modulation Demo
        print("\n2. Encoding and Modulation Demo")
        print("-" * 35)
        
        encoding_results = []
        for i, text in enumerate(test_texts[:3]):  # First 3 for brevity
            encoding = self.encode_and_modulate(text)
            encoding_results.append(encoding)
            print(f"Text {i+1}: {encoding['original_bytes']} bytes → "
                  f"{encoding['data_bits']} bits → "
                  f"{encoding['encoded_bits']} encoded bits → "
                  f"{encoding['signal_samples']} samples "
                  f"({encoding['signal_duration']:.2f}s)")
        
        results["encoding_modulation"] = encoding_results
        
        # 3. Adaptive Planning Demo
        print("\n3. Adaptive Planning Demo")
        print("-" * 30)
        
        planning_results = self.adaptive_planning_demo(test_texts, episodes=15)
        print(f"Completed {len(planning_results['episodes'])} episodes")
        print(f"Success rate: {planning_results['success_rate']:.1%}")
        print(f"Q-table size: {planning_results['q_table_size']} states")
        
        # Show last few episodes
        print("\nLast 5 episodes:")
        for ep in planning_results['episodes'][-5:]:
            print(f"  Episode {ep['episode']}: {ep['action']} → "
                  f"{'✓' if ep['success'] else '✗'} "
                  f"(entropy={ep['entropy']:.2f})")
        
        results["adaptive_planning"] = planning_results
        
        # 4. System Integration Demo
        print("\n4. System Integration Summary")
        print("-" * 35)
        
        total_texts = len(test_texts)
        avg_entropy = sum(a["entropy"] for a in analysis_results) / len(analysis_results)
        total_samples = sum(e["signal_samples"] for e in encoding_results)
        
        integration_summary = {
            "total_texts_processed": total_texts,
            "average_entropy": avg_entropy,
            "total_signal_samples": total_samples,
            "adaptive_success_rate": planning_results['success_rate'],
            "system_components": [
                "Entropy Analysis",
                "Reflective Analysis", 
                "Hamming FEC Encoding",
                "BFSK Modulation Simulation",
                "Adaptive Q-Learning"
            ]
        }
        
        print(f"Processed {total_texts} texts")
        print(f"Average entropy: {avg_entropy:.2f} bits")
        print(f"Generated {total_samples} signal samples")
        print(f"Adaptive success rate: {planning_results['success_rate']:.1%}")
        print(f"System components: {len(integration_summary['system_components'])}")
        
        results["integration_summary"] = integration_summary
        
        print("\n✅ Demo completed successfully!")
        print("\nThis demonstrates the core concepts of the Enhanced WaveCaster system:")
        print("• Neuro-symbolic analysis (entropy, reflection)")
        print("• Signal processing (FEC, modulation)")
        print("• Adaptive learning (Q-learning)")
        print("• System integration")
        print("\nFor full functionality, install the required dependencies and use the complete system.")
        
        return results

def main():
    """Run the basic demonstration"""
    wavecaster = BasicWaveCaster()
    results = wavecaster.demonstrate_system()
    
    # Save results
    with open("demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: demo_results.json")
    return results

if __name__ == "__main__":
    main()