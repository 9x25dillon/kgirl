#!/usr/bin/env python3
"""
Matrix Processor Adapter
========================

Provides matrix processing capabilities for the recursive cognitive system.
Helps compile the database with mathematical transformations.

Author: Assistant
License: MIT
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixProcessor:
    """
    Matrix processor for recursive cognitive database compilation
    
    Features:
    - Matrix transformations for knowledge encoding
    - Eigenvalue decomposition for pattern extraction
    - Singular value decomposition for dimensionality
    - Matrix operations for database optimization
    """
    
    def __init__(self):
        """Initialize matrix processor"""
        logger.info("✅ Matrix processor initialized")
        self.cache = {}
    
    def encode_to_matrix(
        self,
        embeddings: List[List[float]]
    ) -> np.ndarray:
        """
        Encode embeddings as matrix for processing
        
        Args:
            embeddings: List of embedding vectors
        
        Returns:
            Matrix representation
        """
        if not embeddings:
            return np.array([[]])
        
        matrix = np.array(embeddings)
        logger.info(f"📊 Encoded matrix: {matrix.shape}")
        
        return matrix
    
    def extract_patterns(
        self,
        matrix: np.ndarray,
        num_patterns: int = 5
    ) -> Dict[str, Any]:
        """
        Extract patterns using eigenvalue decomposition
        
        Args:
            matrix: Input matrix
            num_patterns: Number of patterns to extract
        
        Returns:
            Extracted patterns and eigenvalues
        """
        if matrix.size == 0:
            return {"patterns": [], "eigenvalues": []}
        
        try:
            # Compute covariance for pattern extraction
            if matrix.shape[0] > 1:
                cov = np.cov(matrix.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                
                # Sort by importance
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Extract top patterns
                patterns = eigenvectors[:, :num_patterns].T.tolist()
                
                logger.info(f"✨ Extracted {len(patterns)} patterns")
                logger.info(f"   Top eigenvalue: {eigenvalues[0]:.3f}")
                
                return {
                    "patterns": patterns,
                    "eigenvalues": eigenvalues[:num_patterns].tolist(),
                    "variance_explained": (eigenvalues[:num_patterns].sum() / eigenvalues.sum() * 100)
                }
            else:
                return {"patterns": matrix.tolist(), "eigenvalues": [1.0]}
        
        except Exception as e:
            logger.error(f"❌ Pattern extraction failed: {e}")
            return {"patterns": [], "eigenvalues": [], "error": str(e)}
    
    def decompose_svd(
        self,
        matrix: np.ndarray,
        rank: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Singular value decomposition for dimensionality reduction
        
        Args:
            matrix: Input matrix
            rank: Target rank (None for full)
        
        Returns:
            SVD components
        """
        if matrix.size == 0:
            return {"U": [], "S": [], "Vt": []}
        
        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            if rank:
                U = U[:, :rank]
                S = S[:rank]
                Vt = Vt[:rank, :]
            
            logger.info(f"🔬 SVD: U{U.shape}, S={len(S)}, Vt{Vt.shape}")
            
            return {
                "U": U.tolist(),
                "S": S.tolist(),
                "Vt": Vt.tolist(),
                "rank": len(S),
                "explained_variance": (S**2).sum()
            }
        
        except Exception as e:
            logger.error(f"❌ SVD failed: {e}")
            return {"U": [], "S": [], "Vt": [], "error": str(e)}
    
    def optimize_database_structure(
        self,
        knowledge_vectors: List[List[float]],
        target_dimension: int = 256
    ) -> Dict[str, Any]:
        """
        Optimize database structure using matrix operations
        
        Args:
            knowledge_vectors: Knowledge base vectors
            target_dimension: Target dimensionality
        
        Returns:
            Optimized structure
        """
        logger.info(f"🔧 Optimizing {len(knowledge_vectors)} vectors to {target_dimension}D")
        
        if not knowledge_vectors:
            return {"optimized": [], "compression_ratio": 0}
        
        matrix = self.encode_to_matrix(knowledge_vectors)
        
        # Use SVD for dimensionality reduction
        svd_result = self.decompose_svd(matrix, rank=min(target_dimension, min(matrix.shape)))
        
        # Reconstruct in lower dimension
        if svd_result.get("U") and svd_result.get("S") and svd_result.get("Vt"):
            U = np.array(svd_result["U"])
            S = np.array(svd_result["S"])
            Vt = np.array(svd_result["Vt"])
            
            optimized = (U @ np.diag(S)).tolist()
            
            compression = len(optimized[0]) / len(knowledge_vectors[0]) if knowledge_vectors else 0
            
            logger.info(f"   ✅ Optimized to {len(optimized[0])}D (compression: {compression:.1%})")
            
            return {
                "optimized": optimized,
                "original_dim": len(knowledge_vectors[0]),
                "optimized_dim": len(optimized[0]),
                "compression_ratio": compression,
                "quality_retained": svd_result.get("explained_variance", 0)
            }
        
        return {"optimized": knowledge_vectors, "error": "Optimization failed"}
    
    def create_fractal_resonance(
        self,
        primary_matrix: np.ndarray,
        secondary_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create fractal resonance between redundant pathways
        
        Args:
            primary_matrix: Primary processing pathway
            secondary_matrix: Secondary (redundant) pathway
        
        Returns:
            Resonance patterns
        """
        logger.info("🌀 Creating fractal resonance between pathways...")
        
        try:
            # Compute interference pattern
            if primary_matrix.shape == secondary_matrix.shape:
                interference = primary_matrix + secondary_matrix
                resonance_strength = np.linalg.norm(interference) / (
                    np.linalg.norm(primary_matrix) + np.linalg.norm(secondary_matrix)
                )
            else:
                # Handle different shapes
                min_shape = min(primary_matrix.shape[0], secondary_matrix.shape[0])
                interference = primary_matrix[:min_shape] + secondary_matrix[:min_shape]
                resonance_strength = 0.5
            
            logger.info(f"   ✨ Resonance strength: {resonance_strength:.3f}")
            
            return {
                "interference_pattern": interference.tolist(),
                "resonance_strength": resonance_strength,
                "fractal_dimension": 1.0 + resonance_strength,
                "emergence_detected": resonance_strength > 0.7
            }
        
        except Exception as e:
            logger.error(f"❌ Resonance calculation failed: {e}")
            return {"error": str(e)}
    
    def compile_database_matrix(
        self,
        knowledge_base: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compile complete database using matrix operations
        
        Args:
            knowledge_base: Complete knowledge base
        
        Returns:
            Compiled matrix database
        """
        logger.info(f"💾 Compiling database from {len(knowledge_base)} entries...")
        
        # Extract all embeddings
        embeddings = []
        for entry in knowledge_base:
            if "embedding" in entry:
                embeddings.append(entry["embedding"])
        
        if not embeddings:
            return {"compiled": None, "error": "No embeddings found"}
        
        # Create matrix
        matrix = self.encode_to_matrix(embeddings)
        
        # Extract patterns
        patterns = self.extract_patterns(matrix)
        
        # Optimize structure
        optimized = self.optimize_database_structure(embeddings)
        
        compilation = {
            "total_entries": len(knowledge_base),
            "matrix_shape": matrix.shape,
            "patterns_extracted": len(patterns.get("patterns", [])),
            "top_eigenvalues": patterns.get("eigenvalues", []),
            "optimized_dimension": optimized.get("optimized_dim", 0),
            "compression_ratio": optimized.get("compression_ratio", 0),
            "compilation_success": True
        }
        
        logger.info(f"   ✅ Database compiled: {compilation['matrix_shape']}")
        logger.info(f"   ✅ Patterns: {compilation['patterns_extracted']}")
        logger.info(f"   ✅ Optimized: {compilation['optimized_dimension']}D")
        
        return compilation


# Global instance
matrix_processor = MatrixProcessor()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MATRIX PROCESSOR DEMO")
    print("="*70)
    
    # Test data
    vectors = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0]
    ]
    
    # Test matrix encoding
    matrix = matrix_processor.encode_to_matrix(vectors)
    print(f"\n✅ Matrix shape: {matrix.shape}")
    
    # Test pattern extraction
    patterns = matrix_processor.extract_patterns(matrix, num_patterns=2)
    print(f"✅ Patterns extracted: {len(patterns['patterns'])}")
    print(f"   Variance explained: {patterns.get('variance_explained', 0):.1f}%")
    
    # Test database compilation
    knowledge_base = [
        {"id": "1", "embedding": [1, 2, 3, 4]},
        {"id": "2", "embedding": [2, 3, 4, 5]},
        {"id": "3", "embedding": [3, 4, 5, 6]}
    ]
    
    compilation = matrix_processor.compile_database_matrix(knowledge_base)
    print(f"\n✅ Database compiled: {compilation['matrix_shape']}")
    print(f"✅ Patterns: {compilation['patterns_extracted']}")
    
    print(f"\n{'='*70}")
    print("Matrix processor ready for recursive cognition!")
    print("="*70)

