"""Semantic token suggestions for the QGI suggester: the real kgirl implementation, re-exported.

This used to be a stub whose available() returned False and whose suggestions were placeholder
"_A/_B/_C" expansions. kgirl.core.matrix_processor ranks the state vocabulary by character-trigram
similarity with stable hashing (deterministic across processes).
"""

from kgirl.core.matrix_processor import MatrixProcessor, matrix_processor

__all__ = ["MatrixProcessor", "matrix_processor"]
