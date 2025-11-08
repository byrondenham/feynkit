"""
Graph representation for Feynman diagrams.

This module provides the Graph class representing complete Feynman diagrams
with methods to compute Laplacians and related polynomials.
"""

from __future__ import annotations

import sympy as sp

from .constants import (
    DEFAULT_ENERGY_SCALE,
    EXTERNAL_PARAM_PREFIX,
    SCALE_ASSUMPTIONS,
    SCHWINGER_PARAM_PREFIX,
)
from .edge import Edge
from .exceptions import GraphTopologyError
from .validation import (
    validate_edge_connectivity,
    validate_nonnegative_integer,
    validate_positive_integer,
)


class Graph:
    """
    A Feynman graph with internal and external edges.

    Represents a Feynman diagram as a graph structure with internal vertices
    and external legs. Provides methods to compute the graph Laplacian and
    related polynomials used in parametric integration.

    Attributes
    ----------
    internal_vertices : int
        Number of internal vertices in the graph (r_int).
    external_legs : int
        Number of external legs (external momenta, n_ext).
    edges : List[Edge]
        Complete list of all edges (both internal and external).
    energy_scale : sp.Symbol
        Energy scale parameter for dimensional analysis (typically μ).
    schwinger_parameters : Dict[int, sp.Symbol]
        Schwinger/Feynman parameters for internal edges, indexed by edge idx.
        Parameters are named 'a_{idx}' by default.
    external_parameters : Dict[int, sp.Symbol]
        Parameters for external legs, indexed 1 to n_ext.
        Parameters are named 'b_{j}' by default.

    Methods
    -------
    calculate_laplacian(include_external=True)
        Compute the graph Laplacian matrix.
    calculate_w_polynomial()
        Compute the W polynomial (determinant of the modified Laplacian).
    expand_w_by_external_parameters()
        Expand W polynomial by external leg parameters.
    get_internal_edges()
        Get list of internal edges.
    get_external_edges()
        Get list of external edges.
    get_loop_count()
        Calculate number of independent loops: L = E - V + 1.

    Examples
    --------
    >>> import sympy as sp
    >>> from feynman.core import Edge, Graph
    >>>
    >>> # Create a simple bubble diagram (2 vertices, 2 internal edges, 2 external legs)
    >>> m1, m2 = sp.symbols('m1 m2', nonnegative=True)
    >>> nu1, nu2, = sp.symbols('nu1 nu2', positive=True)
    >>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
    >>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
    >>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    >>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    >>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    >>>
    >>> # Compute Laplacian
    >>> W = graph.calculate_w_polynomial()
    >>> print(W)
    >>>
    >>> # Get loop count
    >>> loops = graph.get_loop_count()
    >>> print(f"Number of loops: {loops}")

    Notes
    -----
    The vertex indexing convention follows:
    - Internal vertices: 1 to r_int
    - External vertices: (r_int + 1) to (r_int + n_ext)

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    """

    def __init__(
        self,
        internal_vertices: int,
        external_legs: int,
        edges: list[Edge],
        energy_scale: sp.Symbol | None = None,
    ):
        """
        Initialise a Feynman graph.

        Parameters
        ----------
        internal_vertices : int
            Number of internal vertices (must be positive).
        external_legs : int
            Number of external legs (can be zero).
        edges : List[Edge]
            List of all edges in the graph (internal propagators and external legs).
        energy_scale : Optional[sp.Symbol], default None
            Energy scale parameter. If None, creates symbol 'mu' with positive=True.

        Raises
        ------
        ValidationError
            If internal_vertices is not positive or external_legs is negative.
        EdgeConnectivityError
            If any edge violates the vertex indexing constraints.
        GraphTopologyError
            If graph topology is inconsistent.
        """
        # Validate basic parameters
        self.internal_vertices = validate_positive_integer(internal_vertices, "internal_vertices")
        self.external_legs = validate_nonnegative_integer(external_legs, "external_legs")
        self.edges = list(edges)

        # Set or create energy scale
        if energy_scale is not None:
            self.energy_scale = energy_scale
        else:
            self.energy_scale = sp.Symbol(DEFAULT_ENERGY_SCALE, **SCALE_ASSUMPTIONS)

        # Validate edge connectivity
        self._validate_edges()

        # Separate and sort internal and external edges by index
        self._internal_edges = sorted([e for e in self.edges if e.is_internal], key=lambda e: e.idx)
        self._external_edges = sorted(
            [e for e in self.edges if not e.is_internal], key=lambda e: e.idx
        )

        # Validate we have the expected number of external legs
        if len(self._external_edges) != self.external_legs:
            raise GraphTopologyError(
                f"Expected {self.external_legs} external edges, "
                f"but found {len(self._external_edges)}"
            )

        # Create Schwinger/Feynman parameters for internal edges
        self.schwinger_parameters: dict[int, sp.Symbol] = {
            e.idx: sp.Symbol(f"{SCHWINGER_PARAM_PREFIX}_{e.idx}", nonnegative=True)
            for e in self._internal_edges
        }

        # Create parameters for external legs
        self.external_parameters: dict[int, sp.Symbol] = {
            j: sp.Symbol(f"{EXTERNAL_PARAM_PREFIX}_{j}", nonnegative=True)
            for j in range(1, self.external_legs + 1)
        }

    def _validate_edges(self) -> None:
        """
        Validate all edges in the graph.

        Raises
        ------
        EdgeConnectivityError
            If any edge violates connectivity constraints.
        GraphTopologyError
            If edge indices are not unique.
        """
        # Check for duplicate edge indices
        edge_indices = [e.idx for e in self.edges]
        if len(edge_indices) != len(set(edge_indices)):
            duplicates = [idx for idx in edge_indices if edge_indices.count(idx) > 1]
            raise GraphTopologyError(f"Duplicate edge indices found: {set(duplicates)}")

        # Validate each edge's connectivity
        for edge in self.edges:
            validate_edge_connectivity(
                edge_v1=edge.v1,
                edge_v2=edge.v2,
                is_internal=edge.is_internal,
                internal_vertex_count=self.internal_vertices,
                external_leg_count=self.external_legs,
                edge_idx=edge.idx,
            )

    def get_internal_edges(self) -> list[Edge]:
        """
        Get list of internal edges (propagators).

        Returns
        -------
        List[Edge]
            List of internal edges, sorted by index.
        """
        return self._internal_edges.copy()

    def get_external_edges(self) -> list[Edge]:
        """
        Get list of external edges (external legs).

        Returns
        -------
        List[Edge]
            List of external edges, sorted by index.
        """
        return self._external_edges.copy()

    def get_loop_count(self) -> int:
        """
        Calculate the number of independent loops in the graph.

        The loop count L is given by the formula:
            L = E - V + 1
        where E is the number of internal edges and V is the number of internal vertices.

        Returns
        -------
        int
            Number of independent loops.

        Examples
        --------
        >>> # Bubble diagram: 2 edges, 2 vertices -> L = 2 - 2 + 1 = 1
        >>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        >>> graph.get_loop_count()
        1
        """
        return len(self._internal_edges) - self.internal_vertices + 1

    def calculate_laplacian(self, include_external: bool = True) -> sp.Matrix:
        """
        Compute the graph Laplacian matrix.

        The Laplacian matrix encodes the connectivity structure of the graph. Each edge
        contributes to the Laplacian based on its Schwinger parameter (a for internal
        edges, b for external legs).

        For an edge connecting vertices i and j with parameter p:
        - L[i,i] += p (diagonal: sum of parameters at vertex i)
        - L[j,j] += p (diagonal: sum of parameters at vertex j)
        - L[i,j] -= p (off-diagonal: negative parameter)
        - L[j,i] -= p (symmetric)

        Parameters
        ----------
        include_external : bool, default True
            If True, include external legs with parameters b_j in the Laplacian.
            If False, only internal edges with parameters a_i are included.

        Returns
        -------
        sp.Matrix
            The Laplacian matrix of size (r_int + n_ext) × (r_int + n_ext) if
            include_external is True, or r_int × r_int if False.

        Notes
        -----
        Self-loops (tadpoles, where i == j) are skipped as they don't contribute
        to the Laplacian in the standard formulation.

        Examples
        --------
        >>> L = graph.calculate_laplacian()
        >>> print(L)
        >>> L_internal = graph.calculate_laplacian(include_external=False)
        >>> print(L_internal)
        """
        # Determine matrix size based on whether external vertices are included
        size = self.internal_vertices + (self.external_legs if include_external else 0)
        L = sp.zeros(size, size)

        # Add contributions from internal edges with Schwinger parameters a_i
        for edge in self._internal_edges:
            param = self.schwinger_parameters[edge.idx]
            i, j = edge.v1, edge.v2

            # Skip self-loops (tadpoles)
            if i == j:
                continue

            # Standard Laplacian construction (vertex indices are 1-indexed, matrix is 0-indexed)
            L[i - 1, i - 1] += param  # Diagonal element at vertex i
            L[j - 1, j - 1] += param  # Diagonal element at vertex j
            L[i - 1, j - 1] -= param  # Off-diagonal coupling
            L[j - 1, i - 1] -= param  # Symmetric off-diagonal

        # Add contributions from external legs with parameters b_j
        if include_external and self.external_legs > 0:
            for edge in self._external_edges:
                # Map external vertex to leg number
                leg_number = edge.v2 - self.internal_vertices
                param = self.external_parameters[leg_number]
                i, ext = edge.v1, edge.v2

                # External legs connect internal vertex i to external vertex ext
                L[i - 1, i - 1] += param
                L[ext - 1, ext - 1] += param
                L[i - 1, ext - 1] -= param
                L[ext - 1, i - 1] -= param

        return sp.Matrix(L)

    def calculate_w_polynomial(self) -> sp.Expr:
        """
        Compute the W polynomial from the graph Laplacian.

        The W polynomial is obtained by taking the determinant of the Laplacian
        restricted to internal vertices (after removing external vertex rows/columns).
        This polynomial appears in the Symanzik parametrisation of Feynman integrals.

        Returns
        -------
        sp.Expr
            The W polynomial, factored for simplified form. This is a polynomial
            in the Schwinger parameters {a_i} and external leg parameters {b_j}.

        Notes
        -----
        For graphs with external legs (n_ext > 0), we compute det(L_internal) where
        L_internal is the Laplacian with external vertex rows/columns removed.
        For graphs with no external legs, we compute det(L) directly.

        The W polynomial encodes the spanning trees of the graph and is related
        to the Symanzik U polynomial through a variable transformation.

        Examples
        --------
        >>> W = graph.calculate_w_polynomial()
        >>> print(W)
        """
        # Get full Laplacian including external vertices
        L = self.calculate_laplacian(include_external=True)

        if self.external_legs > 0:
            # Remove rows and columns corresponding to external vertices
            # External vertices are indexed r_int to r_int + n_ext - 1 (0-indexed)
            rows_to_keep = list(range(self.internal_vertices))
            M = L.extract(rows_to_keep, rows_to_keep)
        else:
            M = L

        # Return factored determinant
        return sp.factor(M.det())

    def expand_w_by_external_parameters(self) -> dict[tuple[int, ...], sp.Expr]:
        """
        Expand the W polynomial as a polynomial in external parameters b_j.

        Expands W as a sum over monomials in the b parameters and returns a dictionary
        mapping each monomial signature to its coefficient (which depends on a parameters).

        Returns
        -------
        Dict[tuple[int, ...], sp.Expr]
            Dictionary mapping tuples of existing leg indices to polynomial coefficients.
            Each key is a sorted tuple representing a monomial in b variable.
            For example:
            - (1,) represents b_1 with coefficient in a's
            - (1, 2) represents b_1 * b_2 with coefficient
            - () represents the constant term (no b's)

        Notes
        -----
        This expansion is used to extract the Symanzik U and F polynomials.
        The coefficient of a degree-1 term gives U (after variable transformation).
        The coefficients of degree-2 terms give contributions to F related to
        external momentum invariants.

        Examples
        --------
        >>> coeffs = graph.expand_w_by_external_parameters()
        >>> for monomial, coeff in coeffs.items():
        ...     print(f"b_{monomial}: {coeff}")
        """
        W = sp.expand(self.calculate_w_polynomial())

        if self.external_legs == 0:
            return {(): W}

        b_symbols = [self.external_parameters[j] for j in range(1, self.external_legs + 1)]

        # Convert to polynomial in b variables
        P = sp.Poly(W, *b_symbols)
        coefficients = {}

        # Extract each monomial and its coefficient
        for monomial, coeff in P.terms():
            # Convert monomial exponents to list of leg indices
            # monomial is a tuple of exponents (e_1, e_2, ..., e_n_ext)
            indices = []
            for j, exp in enumerate(monomial, start=1):
                if exp > 0:
                    # Add leg index j, repeated exp times (for products like b_1^2)
                    indices.extend([j] * int(exp))

            # Store coefficient with sorted tuple key
            coefficients[tuple(sorted(indices))] = coeff

        return coefficients

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the graph.

        Returns
        -------
        str
            String representation showing graph structure.
        """
        return (
            f"Graph(internal_vertices={self.internal_vertices}, "
            f"external_legs={self.external_legs}, "
            f"edges={len(self.edges)}, "
            f"loops={self.get_loop_count()})"
        )

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the graph.

        Returns
        -------
        str
            Concise description of the graph.
        """
        return (
            f"Feynman Graph: {self.internal_vertices} internal vertices, "
            f"{self.external_legs} external legs, "
            f"{len(self._internal_edges)} propagators, "
            f"{self.get_loop_count()} loops"
        )
