"""
Feynkit: A unified toolkit for symbolic Feynman integral computations.

The primary interface is :class:`FeynmanIntegral`, an immutable object that
exposes every representation of an integral, graph topology, Symanzik /
Lee-Pomeransky polynomials, parametric representations, GKZ system, Newton
polytope, and toric ideal, as a ``cached_property`` derived lazily from
one underlying graph and its kinematic data.

Quick Start
-----------
>>> import sympy as sp
>>> from feynkit import Edge, Graph, FeynmanIntegral
>>>
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>>
>>> integral = FeynmanIntegral(graph)
>>> integral.symanzik.u                     # Symanzik U
>>> integral.gkz.a_matrix                   # GKZ A-matrix
>>> integral.toric_ideal.generators         # IBP relations
>>> integral.is_affinely_equivalent_to(other)
>>> integral.to_latex()                     # full analysis document
>>>
>>> # Derive a related integral with a different dimension; cache is fresh.
>>> integral.with_(dimension=4)

Submodules
----------
core
    Edge, Graph, validation utilities, error types.
polynomials
    Polynomial transforms (inversion, rescaling, projective forms).
parametrisations
    Schwinger / Feynman / Lee-Pomeransky representation classes.
systems
    GKZ A-matrix, monomial support, Euler operators.
algebra
    Toric ideal, syzygy, monomial-change utilities.
normal_forms
    Pairing-matrix canonicalisation and polytope equivalence.
kinematics
    Momentum products and Mandelstam variables.
visualisation
    TikZ generation and Newton polytope rendering.
io
    LaTeX and text formatters.

References
----------
.. [1] Weinzierl, S. (2022). "Feynman Integrals: A Comprehensive Treatment
        for Students and Researchers." Springer.
.. [2] de la Cruz, L. (2019). "Feynman integrals as A-hypergeometric functions."
        JHEP 12, 123.
.. [3] Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1994).
        "Discriminants, Resultants and Multidimensional Determinants."
        Birkhäuser.
.. [4] Sturmfels, B. (1996). "Gröbner Bases and Convex Polytopes."
        American Mathematical Society.
"""

from feynkit.a_configuration import (
    AConfiguration,
    FiniteIndexResult,
    IntrinsicModel,
    SymmetryPair,
    finite_index_map,
    intrinsic_lattice_model,
    symmetry_pairs,
)
from feynkit.artifacts.conformal import (
    bms_simplex_a_config,
    complete_graph_a_config,
    conformal_companion_a_config,
    massless_polygon_a_config,
)
from feynkit.core import (
    Edge,
    FeynkitError,
    Graph,
    ValidationError,
    __version__,
)
from feynkit.database import FeynkitDatabase
from feynkit.integral import FeynmanIntegral
from feynkit.landau import (
    EdgeDiscriminant,
    LandauAnalysis,
    landau_analysis,
    landau_analysis_from_polynomial,
)
from feynkit.normal_forms import PairingMatrixResult
from feynkit.parametrisations import ParametrisationResult
from feynkit.systems import GKZSystem
from feynkit.types import (
    NewtonPolytope,
    PolytopeAutomorphisms,
    PolytopeEquivalence,
    SymanzikPolynomials,
    ToricIdeal,
)

__all__ = [
    "__version__",
    # Errors
    "FeynkitError",
    "ValidationError",
    # Inputs
    "Edge",
    "Graph",
    # Unified facade
    "FeynmanIntegral",
    # Conformal artifacts
    "massless_polygon_a_config",
    "bms_simplex_a_config",
    "complete_graph_a_config",
    "conformal_companion_a_config",
    # A-configurations (arbitrary GKZ inputs)
    "AConfiguration",
    "FiniteIndexResult",
    "IntrinsicModel",
    "SymmetryPair",
    "finite_index_map",
    "intrinsic_lattice_model",
    "symmetry_pairs",
    # Database
    "FeynkitDatabase",
    # Value types
    "SymanzikPolynomials",
    "NewtonPolytope",
    "ToricIdeal",
    "PolytopeEquivalence",
    "PolytopeAutomorphisms",
    "ParametrisationResult",
    "GKZSystem",
    "PairingMatrixResult",
]
