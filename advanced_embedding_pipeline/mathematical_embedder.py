#!/usr/bin/env python3
"""
Mathematical Embedder - Symbolic and mathematical vectorization
Integrates LIMPS matrix processing with mathematical optimization
"""

import asyncio
import logging
import numpy as np
import httpx
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import re
import sympy as sp
from sympy import symbols, Matrix, simplify, expand
import ast
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MathematicalConfig:
    """Configuration for mathematical embedding"""
    limps_url: str = "http://localhost:8000"
    max_dimension: int = 1024
    polynomial_degree: int = 3
    use_matrix_optimization: bool = True
    timeout: float = 30.0
    cache_matrices: bool = True


class MathematicalEmbedder:
    """
    Mathematical embedder that processes symbolic expressions and mathematical content
    using LIMPS matrix processing and optimization.
    """
    
    def __init__(self, config: Optional[MathematicalConfig] = None):
        self.config = config or MathematicalConfig()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
        self.symbol_cache = {}
        self.matrix_cache = {}
        
    async def embed_mathematical_expression(self, expression: str) -> np.ndarray:
        """
        Embed a mathematical expression
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Mathematical embedding vector
        """
        try:
            # Parse and simplify expression
            parsed_expr = self._parse_expression(expression)
            
            # Generate matrix representation
            matrix = self._expression_to_matrix(parsed_expr)
            
            # Optimize matrix using LIMPS if available
            if self.config.use_matrix_optimization:
                optimized_matrix = await self._optimize_matrix(matrix)
            else:
                optimized_matrix = matrix
            
            # Convert to embedding vector
            embedding = self._matrix_to_embedding(optimized_matrix)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Mathematical embedding failed for '{expression}': {e}")
            # Fallback to simple hash-based embedding
            return self._generate_fallback_embedding(expression)
    
    def _parse_expression(self, expression: str) -> sp.Expr:
        """Parse mathematical expression using SymPy"""
        try:
            # Clean expression
            clean_expr = re.sub(r'\s+', ' ', expression.strip())
            
            # Parse with SymPy
            parsed = sp.sympify(clean_expr)
            
            # Simplify
            simplified = simplify(parsed)
            
            return simplified
            
        except Exception as e:
            logger.warning(f"Expression parsing failed for '{expression}': {e}")
            # Return a simple symbolic expression
            x = symbols('x')
            return x
    
    def _expression_to_matrix(self, expr: sp.Expr) -> np.ndarray:
        """Convert symbolic expression to matrix representation"""
        try:
            # Extract coefficients and terms
            if expr.is_polynomial():
                # For polynomials, use coefficient vector
                coeffs = sp.Poly(expr).all_coeffs()
                max_degree = len(coeffs) - 1
                
                # Pad or truncate to desired dimension
                matrix_size = min(self.config.max_dimension, max_degree + 1)
                matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
                
                # Fill diagonal with coefficients
                for i, coeff in enumerate(coeffs):
                    if i < matrix_size:
                        matrix[i, i] = float(coeff)
                        
            else:
                # For non-polynomials, create a matrix based on expression structure
                matrix_size = min(self.config.max_dimension, 64)
                matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
                
                # Hash the expression to create deterministic values
                expr_str = str(expr)
                hash_val = hashlib.md5(expr_str.encode()).digest()
                
                # Fill matrix based on hash
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        idx = (i * matrix_size + j) % len(hash_val)
                        matrix[i, j] = (hash_val[idx] - 128) / 128.0
            
            return matrix
            
        except Exception as e:
            logger.warning(f"Matrix conversion failed: {e}")
            # Return identity matrix as fallback
            size = min(self.config.max_dimension, 32)
            return np.eye(size, dtype=np.float32)
    
    async def _optimize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Optimize matrix using LIMPS"""
        try:
            # Check cache first
            matrix_hash = hashlib.md5(matrix.tobytes()).hexdigest()
            if matrix_hash in self.matrix_cache:
                return self.matrix_cache[matrix_hash]
            
            # Send to LIMPS for optimization
            response = await self.client.post(
                f"{self.config.limps_url}/optimize",
                json={
                    "matrix": matrix.tolist(),
                    "method": "gradient",
                    "params": {
                        "max_iterations": 100,
                        "tolerance": 1e-6
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                optimized_matrix = np.array(result["optimized"], dtype=np.float32)
                
                # Cache result
                if self.config.cache_matrices:
                    self.matrix_cache[matrix_hash] = optimized_matrix
                
                return optimized_matrix
            else:
                logger.warning(f"LIMPS optimization failed: {response.status_code}")
                return matrix
                
        except Exception as e:
            logger.warning(f"Matrix optimization failed: {e}")
            return matrix
    
    def _matrix_to_embedding(self, matrix: np.ndarray) -> np.ndarray:
        """Convert matrix to embedding vector"""
        try:
            # Flatten matrix
            flat = matrix.flatten()
            
            # Pad or truncate to desired dimension
            if len(flat) > self.config.max_dimension:
                embedding = flat[:self.config.max_dimension]
            else:
                embedding = np.zeros(self.config.max_dimension, dtype=np.float32)
                embedding[:len(flat)] = flat
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Matrix to embedding conversion failed: {e}")
            return np.zeros(self.config.max_dimension, dtype=np.float32)
    
    def _generate_fallback_embedding(self, expression: str) -> np.ndarray:
        """Generate fallback embedding when mathematical processing fails"""
        # Hash-based fallback
        hash_val = hashlib.sha256(expression.encode()).digest()
        embedding = np.zeros(self.config.max_dimension, dtype=np.float32)
        
        for i in range(0, min(len(hash_val), self.config.max_dimension // 4)):
            seed = int.from_bytes(hash_val[i:i+4], 'big')
            np.random.seed(seed)
            
            for j in range(4):
                if i * 4 + j < self.config.max_dimension:
                    embedding[i * 4 + j] = np.random.normal(0, 0.1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def embed_code_ast(self, code: str) -> np.ndarray:
        """
        Embed code by analyzing its abstract syntax tree
        
        Args:
            code: Source code string
            
        Returns:
            Code embedding vector
        """
        try:
            import ast
            
            # Parse code to AST
            tree = ast.parse(code)
            
            # Extract features from AST
            features = self._extract_ast_features(tree)
            
            # Convert features to embedding
            embedding = self._features_to_embedding(features)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Code AST embedding failed: {e}")
            return self._generate_fallback_embedding(code)
    
    def _extract_ast_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract features from AST"""
        features = {
            "node_types": [],
            "operators": [],
            "functions": [],
            "variables": [],
            "depth": 0,
            "complexity": 0
        }
        
        def visit_node(node, depth=0):
            features["node_types"].append(type(node).__name__)
            features["depth"] = max(features["depth"], depth)
            
            if isinstance(node, ast.BinOp):
                features["operators"].append(type(node.op).__name__)
            elif isinstance(node, ast.Call):
                if hasattr(node.func, 'id'):
                    features["functions"].append(node.func.id)
            elif isinstance(node, ast.Name):
                features["variables"].append(node.id)
            
            # Recursively visit children
            for child in ast.iter_child_nodes(node):
                visit_node(child, depth + 1)
        
        visit_node(tree)
        
        # Calculate complexity
        features["complexity"] = len(features["node_types"]) + len(features["operators"])
        
        return features
    
    def _features_to_embedding(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert AST features to embedding vector"""
        embedding = np.zeros(self.config.max_dimension, dtype=np.float32)
        
        # Hash-based feature encoding
        feature_str = json.dumps(features, sort_keys=True)
        hash_val = hashlib.sha256(feature_str.encode()).digest()
        
        # Fill embedding based on hash
        for i in range(0, min(len(hash_val), self.config.max_dimension // 4)):
            seed = int.from_bytes(hash_val[i:i+4], 'big')
            np.random.seed(seed)
            
            for j in range(4):
                if i * 4 + j < self.config.max_dimension:
                    embedding[i * 4 + j] = np.random.normal(0, 0.1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def embed_system_of_equations(self, equations: List[str]) -> np.ndarray:
        """
        Embed a system of mathematical equations
        
        Args:
            equations: List of equation strings
            
        Returns:
            System embedding vector
        """
        try:
            # Parse equations
            parsed_eqs = [self._parse_expression(eq) for eq in equations]
            
            # Create system matrix
            system_matrix = self._equations_to_system_matrix(parsed_eqs)
            
            # Optimize system
            if self.config.use_matrix_optimization:
                optimized_system = await self._optimize_matrix(system_matrix)
            else:
                optimized_system = system_matrix
            
            # Convert to embedding
            embedding = self._matrix_to_embedding(optimized_system)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"System of equations embedding failed: {e}")
            return self._generate_fallback_embedding(str(equations))
    
    def _equations_to_system_matrix(self, equations: List[sp.Expr]) -> np.ndarray:
        """Convert system of equations to matrix representation"""
        try:
            # Extract all symbols
            all_symbols = set()
            for eq in equations:
                all_symbols.update(eq.free_symbols)
            
            symbols_list = sorted(list(all_symbols), key=str)
            
            # Create coefficient matrix
            n_eqs = len(equations)
            n_vars = len(symbols_list)
            
            matrix_size = max(n_eqs, n_vars, 32)
            matrix = np.zeros((matrix_size, matrix_size), dtype=np.float32)
            
            # Fill coefficient matrix
            for i, eq in enumerate(equations):
                if i < matrix_size:
                    for j, symbol in enumerate(symbols_list):
                        if j < matrix_size:
                            coeff = eq.coeff(symbol)
                            if coeff != 0:
                                matrix[i, j] = float(coeff)
            
            return matrix
            
        except Exception as e:
            logger.warning(f"System matrix creation failed: {e}")
            # Return identity matrix as fallback
            size = min(self.config.max_dimension, 32)
            return np.eye(size, dtype=np.float32)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'client'):
            asyncio.create_task(self.close())
