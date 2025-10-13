#!/usr/bin/env python3
"""
Fractal Cascade Embedder - Fractal-based embedding simulation
Implements hierarchical fractal structures for embedding generation
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import hashlib
import random
from scipy import sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FractalConfig:
    """Configuration for fractal cascade embedding"""
    max_depth: int = 6
    branching_factor: int = 3
    embedding_dim: int = 1024
    fractal_type: str = "mandelbrot"  # "mandelbrot", "julia", "sierpinski", "custom"
    use_entropy: bool = True
    cache_fractals: bool = True
    visualization: bool = False


class FractalCascadeEmbedder:
    """
    Fractal cascade embedder that generates embeddings based on fractal structures
    and hierarchical patterns.
    """
    
    def __init__(self, config: Optional[FractalConfig] = None):
        self.config = config or FractalConfig()
        self.fractal_cache = {}
        self.entropy_cache = {}
        
    def embed_text_with_fractal(self, text: str) -> np.ndarray:
        """
        Generate fractal-based embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Fractal embedding vector
        """
        try:
            # Generate fractal structure from text
            fractal_structure = self._text_to_fractal_structure(text)
            
            # Create embedding from fractal
            embedding = self._fractal_to_embedding(fractal_structure)
            
            # Apply entropy-based modifications if enabled
            if self.config.use_entropy:
                embedding = self._apply_entropy_modifications(embedding, text)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Fractal embedding failed for text: {e}")
            return self._generate_fallback_embedding(text)
    
    def _text_to_fractal_structure(self, text: str) -> Dict[str, Any]:
        """Convert text to fractal structure"""
        try:
            # Use text hash as seed for deterministic fractal generation
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            random.seed(seed)
            np.random.seed(seed)
            
            # Generate fractal based on type
            if self.config.fractal_type == "mandelbrot":
                structure = self._generate_mandelbrot_structure(text)
            elif self.config.fractal_type == "julia":
                structure = self._generate_julia_structure(text)
            elif self.config.fractal_type == "sierpinski":
                structure = self._generate_sierpinski_structure(text)
            else:
                structure = self._generate_custom_fractal_structure(text)
            
            return structure
            
        except Exception as e:
            logger.warning(f"Fractal structure generation failed: {e}")
            return self._generate_simple_structure(text)
    
    def _generate_mandelbrot_structure(self, text: str) -> Dict[str, Any]:
        """Generate Mandelbrot set-based fractal structure"""
        structure = {
            "type": "mandelbrot",
            "nodes": [],
            "connections": [],
            "properties": {}
        }
        
        # Extract text properties for fractal parameters
        text_length = len(text)
        char_diversity = len(set(text))
        
        # Mandelbrot parameters based on text
        c_real = (text_length % 100) / 100.0 - 0.5
        c_imag = (char_diversity % 100) / 100.0 - 0.5
        
        # Generate fractal nodes
        for depth in range(self.config.max_depth):
            for branch in range(self.config.branching_factor ** depth):
                # Calculate position based on Mandelbrot iteration
                z_real, z_imag = 0.0, 0.0
                iterations = 0
                max_iter = 50
                
                for i in range(max_iter):
                    if z_real * z_real + z_imag * z_imag > 4:
                        break
                    z_real, z_imag = z_real * z_real - z_imag * z_imag + c_real, 2 * z_real * z_imag + c_imag
                    iterations += 1
                
                # Create node based on iteration count
                node = {
                    "id": f"node_{depth}_{branch}",
                    "depth": depth,
                    "branch": branch,
                    "position": (z_real, z_imag),
                    "iterations": iterations,
                    "converged": iterations < max_iter
                }
                structure["nodes"].append(node)
        
        # Create connections between nodes
        for i, node in enumerate(structure["nodes"]):
            if node["depth"] < self.config.max_depth - 1:
                # Connect to child nodes
                for j in range(self.config.branching_factor):
                    child_idx = i * self.config.branching_factor + j + 1
                    if child_idx < len(structure["nodes"]):
                        structure["connections"].append((i, child_idx))
        
        structure["properties"] = {
            "c_real": c_real,
            "c_imag": c_imag,
            "total_nodes": len(structure["nodes"]),
            "total_connections": len(structure["connections"])
        }
        
        return structure
    
    def _generate_julia_structure(self, text: str) -> Dict[str, Any]:
        """Generate Julia set-based fractal structure"""
        structure = {
            "type": "julia",
            "nodes": [],
            "connections": [],
            "properties": {}
        }
        
        # Julia set parameters from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        c_real = (int(text_hash[0:2], 16) - 128) / 128.0
        c_imag = (int(text_hash[2:4], 16) - 128) / 128.0
        
        # Generate Julia set structure
        for depth in range(self.config.max_depth):
            for branch in range(self.config.branching_factor ** depth):
                # Julia set iteration
                z_real = (branch % 10) / 10.0 - 0.5
                z_imag = (depth % 10) / 10.0 - 0.5
                
                iterations = 0
                max_iter = 50
                
                for i in range(max_iter):
                    if z_real * z_real + z_imag * z_imag > 4:
                        break
                    z_real, z_imag = z_real * z_real - z_imag * z_imag + c_real, 2 * z_real * z_imag + c_imag
                    iterations += 1
                
                node = {
                    "id": f"julia_node_{depth}_{branch}",
                    "depth": depth,
                    "branch": branch,
                    "position": (z_real, z_imag),
                    "iterations": iterations,
                    "converged": iterations < max_iter
                }
                structure["nodes"].append(node)
        
        # Create hierarchical connections
        for i, node in enumerate(structure["nodes"]):
            if node["depth"] < self.config.max_depth - 1:
                for j in range(self.config.branching_factor):
                    child_idx = i * self.config.branching_factor + j + 1
                    if child_idx < len(structure["nodes"]):
                        structure["connections"].append((i, child_idx))
        
        structure["properties"] = {
            "c_real": c_real,
            "c_imag": c_imag,
            "total_nodes": len(structure["nodes"]),
            "total_connections": len(structure["connections"])
        }
        
        return structure
    
    def _generate_sierpinski_structure(self, text: str) -> Dict[str, Any]:
        """Generate Sierpinski triangle-based fractal structure"""
        structure = {
            "type": "sierpinski",
            "nodes": [],
            "connections": [],
            "properties": {}
        }
        
        # Sierpinski triangle generation
        def sierpinski_triangle(depth, x, y, size):
            if depth == 0:
                return [{"x": x, "y": y, "size": size, "depth": 0}]
            
            triangles = []
            new_size = size / 2
            
            # Three sub-triangles
            triangles.extend(sierpinski_triangle(depth - 1, x, y, new_size))
            triangles.extend(sierpinski_triangle(depth - 1, x + new_size, y, new_size))
            triangles.extend(sierpinski_triangle(depth - 1, x + new_size/2, y + new_size, new_size))
            
            return triangles
        
        # Generate triangle nodes
        triangles = sierpinski_triangle(self.config.max_depth, 0, 0, 1.0)
        
        for i, triangle in enumerate(triangles):
            node = {
                "id": f"sierpinski_node_{i}",
                "depth": triangle["depth"],
                "position": (triangle["x"], triangle["y"]),
                "size": triangle["size"],
                "type": "triangle"
            }
            structure["nodes"].append(node)
        
        # Create connections (simplified)
        for i in range(len(structure["nodes"]) - 1):
            structure["connections"].append((i, i + 1))
        
        structure["properties"] = {
            "total_triangles": len(triangles),
            "max_depth": self.config.max_depth
        }
        
        return structure
    
    def _generate_custom_fractal_structure(self, text: str) -> Dict[str, Any]:
        """Generate custom fractal structure based on text patterns"""
        structure = {
            "type": "custom",
            "nodes": [],
            "connections": [],
            "properties": {}
        }
        
        # Analyze text patterns
        words = text.split()
        unique_words = set(words)
        
        # Create nodes based on word patterns
        for i, word in enumerate(unique_words):
            if i >= self.config.max_depth * self.config.branching_factor:
                break
            
            # Calculate node properties based on word
            word_hash = hashlib.md5(word.encode()).hexdigest()
            position_x = (int(word_hash[0:4], 16) % 1000) / 1000.0
            position_y = (int(word_hash[4:8], 16) % 1000) / 1000.0
            
            node = {
                "id": f"custom_node_{i}",
                "depth": i % self.config.max_depth,
                "word": word,
                "position": (position_x, position_y),
                "frequency": words.count(word),
                "length": len(word)
            }
            structure["nodes"].append(node)
        
        # Create connections based on word relationships
        for i, node1 in enumerate(structure["nodes"]):
            for j, node2 in enumerate(structure["nodes"]):
                if i != j:
                    # Connect nodes with similar properties
                    if abs(node1["length"] - node2["length"]) <= 2:
                        structure["connections"].append((i, j))
        
        structure["properties"] = {
            "total_words": len(unique_words),
            "total_connections": len(structure["connections"])
        }
        
        return structure
    
    def _generate_simple_structure(self, text: str) -> Dict[str, Any]:
        """Generate simple fallback structure"""
        return {
            "type": "simple",
            "nodes": [{"id": "simple_node", "depth": 0, "position": (0, 0)}],
            "connections": [],
            "properties": {"total_nodes": 1}
        }
    
    def _fractal_to_embedding(self, structure: Dict[str, Any]) -> np.ndarray:
        """Convert fractal structure to embedding vector"""
        try:
            embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
            
            # Extract features from fractal structure
            features = self._extract_fractal_features(structure)
            
            # Convert features to embedding
            feature_idx = 0
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if feature_idx < self.config.embedding_dim:
                        embedding[feature_idx] = float(value)
                        feature_idx += 1
                elif isinstance(value, list) and len(value) > 0:
                    # Use first few elements of lists
                    for i, v in enumerate(value[:5]):  # Limit to 5 elements
                        if feature_idx < self.config.embedding_dim:
                            embedding[feature_idx] = float(v)
                            feature_idx += 1
            
            # Fill remaining dimensions with fractal-specific values
            while feature_idx < self.config.embedding_dim:
                # Use position-based values
                node_idx = feature_idx % len(structure["nodes"])
                if node_idx < len(structure["nodes"]):
                    node = structure["nodes"][node_idx]
                    if "position" in node:
                        pos = node["position"]
                        embedding[feature_idx] = (pos[0] + pos[1]) / 2.0
                    else:
                        embedding[feature_idx] = 0.0
                else:
                    embedding[feature_idx] = 0.0
                feature_idx += 1
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Fractal to embedding conversion failed: {e}")
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
    
    def _extract_fractal_features(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Extract numerical features from fractal structure"""
        features = {}
        
        # Basic structure features
        features["total_nodes"] = len(structure["nodes"])
        features["total_connections"] = len(structure["connections"])
        features["max_depth"] = max([node.get("depth", 0) for node in structure["nodes"]], default=0)
        
        # Node-based features
        if structure["nodes"]:
            positions = [node.get("position", (0, 0)) for node in structure["nodes"]]
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                features["mean_x"] = np.mean(x_coords)
                features["mean_y"] = np.mean(y_coords)
                features["std_x"] = np.std(x_coords)
                features["std_y"] = np.std(y_coords)
        
        # Depth distribution
        depths = [node.get("depth", 0) for node in structure["nodes"]]
        if depths:
            features["mean_depth"] = np.mean(depths)
            features["std_depth"] = np.std(depths)
        
        # Iteration-based features (for Mandelbrot/Julia)
        iterations = [node.get("iterations", 0) for node in structure["nodes"] if "iterations" in node]
        if iterations:
            features["mean_iterations"] = np.mean(iterations)
            features["max_iterations"] = np.max(iterations)
            features["converged_ratio"] = sum([node.get("converged", False) for node in structure["nodes"]]) / len(structure["nodes"])
        
        return features
    
    def _apply_entropy_modifications(self, embedding: np.ndarray, text: str) -> np.ndarray:
        """Apply entropy-based modifications to embedding"""
        try:
            # Calculate text entropy
            text_entropy = self._calculate_text_entropy(text)
            
            # Apply entropy-based scaling
            entropy_factor = 1.0 + (text_entropy - 0.5) * 0.2  # Scale by ±10%
            
            # Apply modifications
            modified_embedding = embedding * entropy_factor
            
            # Add entropy-based noise
            noise_scale = text_entropy * 0.01  # Small noise based on entropy
            noise = np.random.normal(0, noise_scale, embedding.shape)
            modified_embedding += noise
            
            # Renormalize
            norm = np.linalg.norm(modified_embedding)
            if norm > 0:
                modified_embedding = modified_embedding / norm
            
            return modified_embedding
            
        except Exception as e:
            logger.warning(f"Entropy modification failed: {e}")
            return embedding
    
    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        try:
            if not text:
                return 0.0
            
            # Character frequency
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calculate entropy
            total_chars = len(text)
            entropy = 0.0
            
            for count in char_counts.values():
                probability = count / total_chars
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return 0.5  # Default entropy
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate fallback embedding when fractal processing fails"""
        # Hash-based fallback
        hash_val = hashlib.sha256(text.encode()).digest()
        embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
        
        for i in range(0, min(len(hash_val), self.config.embedding_dim // 4)):
            seed = int.from_bytes(hash_val[i:i+4], 'big')
            np.random.seed(seed)
            
            for j in range(4):
                if i * 4 + j < self.config.embedding_dim:
                    embedding[i * 4 + j] = np.random.normal(0, 0.1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def visualize_fractal(self, structure: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize fractal structure"""
        if not self.config.visualization:
            return
        
        try:
            plt.figure(figsize=(10, 10))
            
            # Plot nodes
            for node in structure["nodes"]:
                if "position" in node:
                    x, y = node["position"]
                    plt.scatter(x, y, c='blue', s=20, alpha=0.6)
                    
                    # Add depth-based coloring
                    depth = node.get("depth", 0)
                    color_intensity = depth / max([n.get("depth", 0) for n in structure["nodes"]], default=1)
                    plt.scatter(x, y, c=plt.cm.viridis(color_intensity), s=20)
            
            # Plot connections
            for connection in structure["connections"]:
                if len(connection) == 2:
                    node1_idx, node2_idx = connection
                    if node1_idx < len(structure["nodes"]) and node2_idx < len(structure["nodes"]):
                        node1 = structure["nodes"][node1_idx]
                        node2 = structure["nodes"][node2_idx]
                        if "position" in node1 and "position" in node2:
                            x1, y1 = node1["position"]
                            x2, y2 = node2["position"]
                            plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
            
            plt.title(f"Fractal Structure: {structure['type']}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Fractal visualization failed: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text_with_fractal(text)
            embeddings.append(embedding)
        return embeddings
