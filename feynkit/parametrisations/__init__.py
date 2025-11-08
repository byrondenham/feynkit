"""
Parametric representations module for Feynman integrals.

Provides three standard parametric representations:
- Schwinger parametrisation (alpha parameters, each alpha in [0, infty))
- Feynman parametrisation (a parameters on simplex, sum_i a_u = 1)
- Lee-Pomeransky parametrisation (u parameters with G = U + F)

Classes
-------
Parametrisation
    Abstract base class for all parametrisations.
ParametrisationResult
    Container for parametrisation computation results.
SchwingerParametrisation
    Schwinger parametrisation implementation.
FeynmanParametrisation
    Feynman parametrisation implementation.
LeePomeranskyParametrisation
    Lee-Pomeransky parametrisation implementation.
AllParametrisations
    Container holding all three parametrisations.

Functions
---------
create_parametrisations
    Factory function to create all three parametrisations at once.

Examples
--------
>>> from feynkit.parametrisations import create_parametrisations
>>> from feynkit import Edge, Graph, create_momentum_products
>>> import sympy as sp
>>>
>>> # Setup graph
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>>
>>> # Create parametrisations
>>> D = sp.Symbol('D', positive=True)
>>> nu1, nu2 = sp.symbols('nu1 nu2', positive=True)
>>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
>>> all_param = create_parametrisations(graph, D, 1, {1: nu1, 2: nu2}, p_dot)
>>>
>>> # Compute Schwinger parametrisation
>>> schwinger = all_param.schwinger.compute()
>>> print(schwinger)
"""

from .base import Parametrisation, ParametrisationResult
from .factory import AllParametrisations, create_parametrisations
from .feynman import FeynmanParametrisation
from .lee_pomeransky import LeePomeranskyParametrisation
from .schwinger import SchwingerParametrisation

__all__ = [
    # Base classes
    "Parametrisation",
    "ParametrisationResult",
    # Implementations
    "SchwingerParametrisation",
    "FeynmanParametrisation",
    "LeePomeranskyParametrisation",
    # Factory
    "create_parametrisations",
    "AllParametrisations",
]
