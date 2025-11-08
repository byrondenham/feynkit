"""
Feynkit: A comprehensive toolkit for symbolic Feynman integral computations.

Feynkit provides a modern, well-documented API for working with Feynman integrals
using parametric representations, algebraic geometry methods, and differential
equation techniques.

Quick Start
-----------
>>> import sympy as sp
>>> from feynkit import (
...     Edge, Graph,
...     create_momentum_products,
...     create_parametrisations,
...     create_gkz_system
... )
>>> from feynkit.algebra import compute_toric_ideal_generators
>>> from feynkit.visualisation import visualise_newton_polytope
>>> from feynkit.io import create_analysis_document
>>>
>>> # Create a bubble diagram
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>>
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>>
>>> # Full analysis with export
>>> D = sp.Symbol('D', positive=True)
>>> nu1, nu2 = sp.symbols('nu1 nu2', positive=True)
>>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
>>> all_param = create_parametrisations(graph, D, 1, {1: nu1, 2: nu2}, p_dot)
>>>
>>> # Export to LaTeX
>>> latex_doc = create_analysis_document(
...     graph=graph,
...     u_polynomial=all_param.u_polynomial,
...     f_polynomial=all_param.f_polynomial,
...     title="My Analysis"
... )

Modules
-------
core
    Core data structures: Edge, Graph, and validation utilities.
kinematics
    Momentum products and Mandelstam variables.
polynomials
    Symanzik polynomial computation and manipulation.
parametrisations
    Schwinger, Feynman, and Lee-Pomeransky parametric representations.
systems
    GKZ hypergeometric systems and Euler operators.
algebra
    Toric ideals and Gröbner basis computations.
visualisation
    TikZ generation and polytope visualisation.
io
    LaTeX and text export utilities.

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

# Import commonly used functions for convenience
from feynkit.algebra import compute_toric_ideal_generators
from feynkit.core import (
    Edge,
    FeynkitError,
    Graph,
    ValidationError,
    __version__,
)
from feynkit.io import create_analysis_document, create_analysis_report
from feynkit.kinematics import create_momentum_products
from feynkit.parametrisations import create_parametrisations
from feynkit.polynomials import calculate_symanzik_polynomials
from feynkit.systems import create_gkz_system
from feynkit.visualisation import visualise_newton_polytope

__all__ = [
    "__version__",
    # Core
    "Edge",
    "Graph",
    "FeynkitError",
    "ValidationError",
    # Convenience imports
    "create_momentum_products",
    "calculate_symanzik_polynomials",
    "create_parametrisations",
    "create_gkz_system",
    "compute_toric_ideal_generators",
    "visualise_newton_polytope",
    "create_analysis_document",
    "create_analysis_report",
]
