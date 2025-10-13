#!/usr/bin/env python3
"""
Fractal Resonance Simulation
Implements constructive interference patterns and resonance field analysis
"""

import asyncio
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import time
from scipy import signal
from scipy.fft import fft, fftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResonanceConfig:
    """Configuration for fractal resonance simulation"""
    resonance_frequency: float = 1.0
    damping_factor: float = 0.1
    harmonic_orders: int = 5
    fractal_depth: int = 3
    interference_threshold: float = 0.7
    visualization_resolution: int = 100

@dataclass
class ResonanceField:
    """Represents a resonance field with interference patterns"""
    field_matrix: np.ndarray
    frequency_spectrum: np.ndarray
    interference_patterns: List[Dict[str, Any]]
    resonance_strength: float
    coherence_measure: float
    timestamp: float = field(default_factory=time.time)

class FractalGenerator:
    """Generates fractal patterns for resonance simulation"""
    
    def __init__(self, config: ResonanceConfig):
        self.config = config
        
    def generate_mandelbrot_fractal(self, width: int, height: int, max_iter: int = 100) -> np.ndarray:
        """Generate Mandelbrot fractal pattern"""
        x = np.linspace(-2.5, 1.5, width)
        y = np.linspace(-2.0, 2.0, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        Z = np.zeros_like(C)
        fractal = np.zeros(C.shape)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            fractal[mask] = i
        
        return fractal / max_iter
    
    def generate_julia_fractal(self, width: int, height: int, c: complex, max_iter: int = 100) -> np.ndarray:
        """Generate Julia fractal pattern"""
        x = np.linspace(-2.0, 2.0, width)
        y = np.linspace(-2.0, 2.0, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        fractal = np.zeros(Z.shape)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            fractal[mask] = i
        
        return fractal / max_iter
    
    def generate_sierpinski_triangle(self, width: int, height: int, depth: int = 6) -> np.ndarray:
        """Generate Sierpinski triangle fractal"""
        triangle = np.zeros((height, width))
        
        def draw_triangle(x, y, size, level):
            if level == 0:
                # Draw filled triangle
                for i in range(int(size)):
                    for j in range(int(size - i)):
                        if 0 <= y + i < height and 0 <= x + j < width:
                            triangle[y + i, x + j] = 1
            else:
                # Recursively draw smaller triangles
                new_size = size / 2
                draw_triangle(x, y, new_size, level - 1)
                draw_triangle(x + new_size, y, new_size, level - 1)
                draw_triangle(x + new_size/2, y + new_size, new_size, level - 1)
        
        draw_triangle(0, 0, min(width, height), depth)
        return triangle

class ResonanceSimulator:
    """Simulates resonance and interference patterns"""
    
    def __init__(self, config: ResonanceConfig):
        self.config = config
        self.fractal_generator = FractalGenerator(config)
        
    def fractal_resonance(self, vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """
        Create fractal resonance between two vector sets
        Implements: combined = vectors_a + sin(vectors_b) with fractal modulation
        """
        if vectors_a.shape != vectors_b.shape:
            # Pad smaller array to match larger
            max_len = max(len(vectors_a), len(vectors_b))
            if len(vectors_a) < max_len:
                vectors_a = np.pad(vectors_a, (0, max_len - len(vectors_a)), mode='constant')
            if len(vectors_b) < max_len:
                vectors_b = np.pad(vectors_b, (0, max_len - len(vectors_b)), mode='constant')
        
        # Apply fractal modulation
        fractal_modulation = self._generate_fractal_modulation(len(vectors_a))
        
        # Create resonance: vectors_a + sin(vectors_b) * fractal_modulation
        sin_b = np.sin(vectors_b)
        combined = vectors_a + sin_b * fractal_modulation
        
        # Apply hyperbolic tangent for bounded output
        combined = np.tanh(combined)
        
        return combined
    
    def _generate_fractal_modulation(self, length: int) -> np.ndarray:
        """Generate fractal modulation pattern"""
        # Create a fractal pattern for modulation
        fractal = self.fractal_generator.generate_mandelbrot_fractal(
            int(np.sqrt(length)), int(np.sqrt(length))
        )
        
        # Flatten and resize to match input length
        fractal_flat = fractal.flatten()
        if len(fractal_flat) > length:
            fractal_flat = fractal_flat[:length]
        elif len(fractal_flat) < length:
            fractal_flat = np.pad(fractal_flat, (0, length - len(fractal_flat)), mode='edge')
        
        return fractal_flat
    
    def simulate_constructive_interference(self, vectors: List[np.ndarray]) -> ResonanceField:
        """Simulate constructive interference from multiple vector sources"""
        if not vectors:
            return self._empty_resonance_field()
        
        # Combine all vectors with resonance
        combined = vectors[0]
        interference_patterns = []
        
        for i, vector in enumerate(vectors[1:], 1):
            # Create resonance with previous combined result
            combined = self.fractal_resonance(combined, vector)
            
            # Analyze interference pattern
            pattern = self._analyze_interference_pattern(combined, vector, i)
            interference_patterns.append(pattern)
        
        # Calculate resonance strength
        resonance_strength = self._calculate_resonance_strength(combined)
        
        # Calculate coherence measure
        coherence_measure = self._calculate_coherence_measure(combined, interference_patterns)
        
        # Generate frequency spectrum
        frequency_spectrum = self._generate_frequency_spectrum(combined)
        
        return ResonanceField(
            field_matrix=combined,
            frequency_spectrum=frequency_spectrum,
            interference_patterns=interference_patterns,
            resonance_strength=resonance_strength,
            coherence_measure=coherence_measure
        )
    
    def _empty_resonance_field(self) -> ResonanceField:
        """Create empty resonance field"""
        return ResonanceField(
            field_matrix=np.array([]),
            frequency_spectrum=np.array([]),
            interference_patterns=[],
            resonance_strength=0.0,
            coherence_measure=0.0
        )
    
    def _analyze_interference_pattern(self, combined: np.ndarray, new_vector: np.ndarray, iteration: int) -> Dict[str, Any]:
        """Analyze interference pattern between combined and new vector"""
        # Calculate correlation
        correlation = np.corrcoef(combined, new_vector)[0, 1] if len(combined) > 1 else 0
        
        # Calculate constructive vs destructive interference
        constructive = np.sum(np.abs(combined + new_vector)) / np.sum(np.abs(combined) + np.abs(new_vector))
        destructive = 1 - constructive
        
        # Calculate harmonic content
        harmonics = self._analyze_harmonics(combined)
        
        return {
            "iteration": iteration,
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "constructive_ratio": float(constructive),
            "destructive_ratio": float(destructive),
            "harmonics": harmonics,
            "amplitude": float(np.mean(np.abs(combined))),
            "phase_shift": float(np.angle(np.mean(combined)))
        }
    
    def _analyze_harmonics(self, vector: np.ndarray) -> Dict[str, float]:
        """Analyze harmonic content of vector"""
        if len(vector) < 2:
            return {"fundamental": 0.0, "harmonics": []}
        
        # FFT analysis
        fft_result = fft(vector)
        freqs = fftfreq(len(vector))
        
        # Find peaks in frequency domain
        power_spectrum = np.abs(fft_result)**2
        peaks, _ = signal.find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.1)
        
        # Extract harmonic frequencies and amplitudes
        harmonic_freqs = freqs[peaks]
        harmonic_amplitudes = power_spectrum[peaks]
        
        # Sort by amplitude
        sorted_indices = np.argsort(harmonic_amplitudes)[::-1]
        
        return {
            "fundamental": float(harmonic_freqs[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0,
            "harmonics": [
                {"frequency": float(freq), "amplitude": float(amp)}
                for freq, amp in zip(harmonic_freqs[sorted_indices], harmonic_amplitudes[sorted_indices])
            ]
        }
    
    def _calculate_resonance_strength(self, vector: np.ndarray) -> float:
        """Calculate overall resonance strength"""
        if len(vector) == 0:
            return 0.0
        
        # Calculate energy in the signal
        energy = np.sum(np.abs(vector)**2)
        
        # Calculate peak-to-average ratio
        peak_avg_ratio = np.max(np.abs(vector)) / np.mean(np.abs(vector)) if np.mean(np.abs(vector)) > 0 else 0
        
        # Calculate resonance strength as combination of energy and peak ratio
        resonance_strength = np.sqrt(energy) * peak_avg_ratio / len(vector)
        
        return float(resonance_strength)
    
    def _calculate_coherence_measure(self, vector: np.ndarray, patterns: List[Dict[str, Any]]) -> float:
        """Calculate coherence measure of the resonance field"""
        if len(vector) == 0:
            return 0.0
        
        # Base coherence from vector consistency
        vector_coherence = 1.0 - np.std(vector) / (np.mean(np.abs(vector)) + 1e-10)
        
        # Pattern coherence from interference patterns
        if patterns:
            correlations = [p.get("correlation", 0) for p in patterns]
            pattern_coherence = np.mean(np.abs(correlations))
        else:
            pattern_coherence = 0.0
        
        # Combined coherence measure
        coherence = (vector_coherence + pattern_coherence) / 2
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def _generate_frequency_spectrum(self, vector: np.ndarray) -> np.ndarray:
        """Generate frequency spectrum of the vector"""
        if len(vector) < 2:
            return np.array([])
        
        fft_result = fft(vector)
        return np.abs(fft_result)

class ResonanceVisualizer:
    """Visualizes resonance patterns and fractal structures"""
    
    def __init__(self, config: ResonanceConfig):
        self.config = config
        
    def visualize_resonance_field(self, field: ResonanceField, save_path: str = None) -> None:
        """Visualize resonance field with multiple plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fractal Resonance Field Analysis', fontsize=16)
        
        # Plot 1: Field matrix
        if field.field_matrix.size > 0:
            im1 = axes[0, 0].imshow(field.field_matrix.reshape(-1, int(np.sqrt(len(field.field_matrix)))), 
                                   cmap='viridis', aspect='auto')
            axes[0, 0].set_title('Resonance Field Matrix')
            axes[0, 0].set_xlabel('Spatial Dimension')
            axes[0, 0].set_ylabel('Temporal Dimension')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Frequency spectrum
        if field.frequency_spectrum.size > 0:
            axes[0, 1].plot(field.frequency_spectrum)
            axes[0, 1].set_title('Frequency Spectrum')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].grid(True)
        
        # Plot 3: Interference patterns
        if field.interference_patterns:
            iterations = [p['iteration'] for p in field.interference_patterns]
            correlations = [p['correlation'] for p in field.interference_patterns]
            axes[1, 0].plot(iterations, correlations, 'o-')
            axes[1, 0].set_title('Interference Correlation')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Correlation')
            axes[1, 0].grid(True)
        
        # Plot 4: Resonance metrics
        metrics = ['Resonance Strength', 'Coherence Measure']
        values = [field.resonance_strength, field.coherence_measure]
        axes[1, 1].bar(metrics, values, color=['blue', 'green'])
        axes[1, 1].set_title('Resonance Metrics')
        axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Resonance visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_fractal_patterns(self, save_path: str = None) -> None:
        """Visualize various fractal patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fractal Patterns for Resonance Simulation', fontsize=16)
        
        # Mandelbrot fractal
        mandelbrot = self.fractal_generator.generate_mandelbrot_fractal(200, 200)
        axes[0, 0].imshow(mandelbrot, cmap='hot')
        axes[0, 0].set_title('Mandelbrot Fractal')
        axes[0, 0].axis('off')
        
        # Julia fractal
        julia = self.fractal_generator.generate_julia_fractal(200, 200, -0.7 + 0.27015j)
        axes[0, 1].imshow(julia, cmap='plasma')
        axes[0, 1].set_title('Julia Fractal')
        axes[0, 1].axis('off')
        
        # Sierpinski triangle
        sierpinski = self.fractal_generator.generate_sierpinski_triangle(200, 200, 6)
        axes[1, 0].imshow(sierpinski, cmap='binary')
        axes[1, 0].set_title('Sierpinski Triangle')
        axes[1, 0].axis('off')
        
        # Combined fractal pattern
        combined = (mandelbrot + julia + sierpinski) / 3
        axes[1, 1].imshow(combined, cmap='viridis')
        axes[1, 1].set_title('Combined Fractal Pattern')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fractal patterns saved to {save_path}")
        
        plt.show()

# Main Fractal Resonance System
class FractalResonanceSystem:
    """Main system for fractal resonance simulation"""
    
    def __init__(self, config: ResonanceConfig = None):
        self.config = config or ResonanceConfig()
        self.simulator = ResonanceSimulator(self.config)
        self.visualizer = ResonanceVisualizer(self.config)
        self.resonance_history: List[ResonanceField] = []
        
    async def process_resonance(self, vectors: List[np.ndarray]) -> ResonanceField:
        """Process vectors through fractal resonance simulation"""
        logger.info(f"Processing {len(vectors)} vectors through fractal resonance")
        
        # Simulate constructive interference
        resonance_field = self.simulator.simulate_constructive_interference(vectors)
        
        # Store in history
        self.resonance_history.append(resonance_field)
        
        # Log results
        logger.info(f"Resonance strength: {resonance_field.resonance_strength:.3f}")
        logger.info(f"Coherence measure: {resonance_field.coherence_measure:.3f}")
        logger.info(f"Interference patterns: {len(resonance_field.interference_patterns)}")
        
        return resonance_field
    
    def visualize_latest_resonance(self, save_path: str = None):
        """Visualize the latest resonance field"""
        if self.resonance_history:
            latest = self.resonance_history[-1]
            self.visualizer.visualize_resonance_field(latest, save_path)
        else:
            logger.warning("No resonance fields to visualize")
    
    def get_resonance_statistics(self) -> Dict[str, Any]:
        """Get statistics about resonance processing"""
        if not self.resonance_history:
            return {"message": "No resonance fields processed"}
        
        strengths = [f.resonance_strength for f in self.resonance_history]
        coherences = [f.coherence_measure for f in self.resonance_history]
        
        return {
            "total_fields": len(self.resonance_history),
            "avg_resonance_strength": float(np.mean(strengths)),
            "max_resonance_strength": float(np.max(strengths)),
            "avg_coherence_measure": float(np.mean(coherences)),
            "max_coherence_measure": float(np.max(coherences)),
            "resonance_trend": "increasing" if len(strengths) > 1 and strengths[-1] > strengths[0] else "stable"
        }

# Demo function
async def demo_fractal_resonance():
    """Demonstrate fractal resonance capabilities"""
    config = ResonanceConfig(
        resonance_frequency=1.0,
        harmonic_orders=5,
        fractal_depth=3
    )
    
    system = FractalResonanceSystem(config)
    
    # Generate sample vectors
    vectors = [
        np.random.randn(100) * 0.5 + 1.0,  # Base signal
        np.sin(np.linspace(0, 4*np.pi, 100)) * 0.3,  # Harmonic signal
        np.cos(np.linspace(0, 6*np.pi, 100)) * 0.2,  # Another harmonic
    ]
    
    # Process through resonance simulation
    resonance_field = await system.process_resonance(vectors)
    
    # Print results
    print("Fractal Resonance Results:")
    print(f"Resonance strength: {resonance_field.resonance_strength:.3f}")
    print(f"Coherence measure: {resonance_field.coherence_measure:.3f}")
    print(f"Interference patterns: {len(resonance_field.interference_patterns)}")
    
    # Get statistics
    stats = system.get_resonance_statistics()
    print(f"\nResonance Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Visualize (uncomment to show plots)
    # system.visualize_latest_resonance("resonance_field.png")
    # system.visualizer.visualize_fractal_patterns("fractal_patterns.png")

if __name__ == "__main__":
    asyncio.run(demo_fractal_resonance())