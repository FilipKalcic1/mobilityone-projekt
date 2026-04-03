"""
Scoring Utilities Module

Pure mathematical functions for tool scoring, extracted from tool_registry.py.
These are stateless functions that can be easily tested and reused.
"""
import numpy as np
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (0.0 to 1.0)
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))
