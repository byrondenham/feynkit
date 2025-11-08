"""
Normal forms for convex polytopes and pairing matrices.
"""

from .pairing_matrix import PairingMatrixResult, is_canonical, maximal_pairing_matrix

__all__ = ["maximal_pairing_matrix", "is_canonical", "PairingMatrixResult"]
