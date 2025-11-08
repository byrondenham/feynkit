"""
Core data structures and utilities for feynkit.

This module provides the fundamental building blocks for representing Feynman
graphs and performing basic operations on them.

Classes
-------
Edge
    Represents an edge (propagator or external leg) in a Feynman graph.
Graph
    Represents a complete Feynman graph with methods for computing Laplacians
    and related polynomials.

Exceptions
----------
FeynkitError
    Base exception class for all feynkit errors.
ValidationError
    Raised when input validation fails.
GraphTopologyError
    Raised when graph topology is invalid.
VertexIndexError
    Raised when vertex indices are out of valid range.
EdgeConnectivityError
    Raised when edge connectivity violates constraints.
ParameterError
    Raised when parameters are missing or invalid.
ComputationError
    Raised when a computation fails.

Examples
--------
>>> import sympy as sp
>>> from feynkit.core import Edge, Graph
>>>
>>> # Create a bubble diagram
>>> m1, m2 = sp.symbols('m1 m2', nonnegative=True)
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>>
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>> print(graph)
>>> L = graph.calculate_laplacian()
>>> W = graph.calculate_w_polynomial()
"""

from .constants import __version__
from .edge import Edge
from .exceptions import (
    ComputationError,
    EdgeConnectivityError,
    FeynkitError,
    GraphTopologyError,
    MatrixError,
    ParameterError,
    PolynomialError,
    ValidationError,
    VertexIndexError,
)
from .graph import Graph

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Edge",
    "Graph",
    # Exceptions
    "FeynkitError",
    "ValidationError",
    "GraphTopologyError",
    "VertexIndexError",
    "EdgeConnectivityError",
    "ParameterError",
    "ComputationError",
    "PolynomialError",
    "MatrixError",
]
