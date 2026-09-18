"""
Normal forms and polytope-equivalence utilities.

The two equivalence verbs are :func:`is_unimodular_equivalent` (Liu–Cai,
the natural relation for lattice polytopes / GKZ systems) and
:func:`is_affinely_equivalent` (the broader relation over the rationals,
backed by a brute-force search).

The pairing-matrix canonicalisation (Grinis–Kasprzyk-style) is exposed as
:func:`maximal_pairing_matrix` together with :func:`is_canonical`.
"""

from .affine_equivalence import (
    is_affinely_equivalent,
    is_point_config_equivalent,
    is_unimodular_equivalent,
)
from .pairing_matrix import (
    PairingMatrixResult,
    is_canonical,
    matrix_lexicographic_compare,
    maximal_pairing_matrix,
    symbolic_compare,
)
from .polytope_automorphisms import (
    coefficient_preserving_indices,
    compute_graph_automorphisms,
    compute_polytope_automorphisms,
)

__all__ = [
    "is_unimodular_equivalent",
    "is_affinely_equivalent",
    "is_point_config_equivalent",
    "maximal_pairing_matrix",
    "is_canonical",
    "PairingMatrixResult",
    "symbolic_compare",
    "matrix_lexicographic_compare",
    "compute_polytope_automorphisms",
    "compute_graph_automorphisms",
    "coefficient_preserving_indices",
]
