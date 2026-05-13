"""
Factory functions for creating parametric representations.

Provides convenient high-level functions to create all three parametrisations
from a Feynman graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..core.exceptions import ValidationError
from ..core.graph import Graph
from ..core.validation import validate_dimension_parameter, validate_propagator_exponents
from ..polynomials.symanzik import _calculate_symanzik_polynomials
from .feynman import FeynmanParametrisation
from .lee_pomeransky import LeePomeranskyParametrisation
from .schwinger import SchwingerParametrisation


@dataclass
class AllParametrisations:
    """
    Container for all three parametric representations.

    Attributes
    ----------
    schwinger : SchwingerParametrisation
        Schwinger parametrisation instance.
    feynman : FeynmanParametrisation
        Feynman parametrisation instance.
    lee_pomeransky : LeePomeranskyParametrisation
        Lee-Pomeransky parametrisation instance.
    u_polynomial : sp.Expr
        First Symanzik polynomial U.
    f_polynomial : sp.Expr
        Second Symanzik polynomial F.
    """

    schwinger: SchwingerParametrisation
    feynman: FeynmanParametrisation
    lee_pomeransky: LeePomeranskyParametrisation
    u_polynomial: sp.Expr
    f_polynomial: sp.Expr

    def compute_all(self) -> tuple[object, object, object]:
        """
        Compute all three parametrisations.

        Returns
        -------
        tuple
            (schwinger_result, feynman_result, lee_pomeransky_result)
        """
        return (
            self.schwinger.compute(),
            self.feynman.compute(),
            self.lee_pomeransky.compute(),
        )


def _create_parametrisations(
    graph: Graph,
    dimension: sp.Expr,
    loop_count: int,
    propagator_exponents: dict[int, sp.Expr],
    momentum_products: dict[tuple[int, int], sp.Expr],
) -> AllParametrisations:
    """
    Create all three parametric representations for a Feynman graph.

    This is the main factory function that computes Symanzik polynomials
    and creates all parametrisation objects.

    Parameters
    ----------
    graph : Graph
        The Feynman graph to parametrise.
    dimension : sp.Expr
        Spacetime dimension D (typically symbolic, e.g., D = 4 - 2 epsilon).
    loop_count : int
        Number of independent loops in the graph (L = E - V + 1).
    propagator_exponents : Dict[int, sp.Expr]
        Dictionary mapping edge index to its propagator exponent nu_i.
        For standard propagators, nu_i = 1.
    momentum_products : Dict[Tuple[int, int], sp.Expr]
        Momentum dot products for external legs, from create_momentum_products().

    Returns
    -------
    AllParametrisations
        Object containing all three parametrisation instances and Symanzik polynomials.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    ComputationError
        If parametrisation computation fails.

    Examples
    --------
    >>> import sympy as sp
    >>> from feynkit import Edge, Graph, create_momentum_products
    >>> from feynkit.parametrisations import create_parametrisations
    >>>
    >>> # Create bubble diagram
    >>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    >>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    >>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    >>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    >>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    >>>
    >>> # Setup parameters
    >>> D = sp.Symbol('D', positive=True)
    >>> nu1, nu2 = sp.symbols('nu1 nu2', positive=True)
    >>> nus = {1: nu1, 2: nu2}
    >>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
    >>>
    >>> # Create all parametrisations
    >>> all_param = create_parametrisations(graph, D, 1, nus, p_dot)
    >>>
    >>> # Compute specific parametrisation
    >>> schwinger_result = all_param.schwinger.compute()
    >>> print(schwinger_result)
    >>>
    >>> # Or compute all at once
    >>> schw, feyn, lp = all_param.compute_all()

    Notes
    -----
    This function automatically:
    1. Validates all inputs
    2. Computes Symanzik U and F polynomials
    3. Creates all three parametrisation objects
    4. Stores everything in a convenient container

    The returned object allows you to:
    - Access individual parametrisations: `all_param.schwinger.compute()`
    - Access Symanzik polynomials: `all_param.u_polynomial`
    - Compute all at once: `all_param.compute_all()`

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals: A Comprehensive Treatment
            for Students and Researchers." Springer.
    """
    # Validate inputs
    validate_dimension_parameter(dimension)

    internal_edges = graph.get_internal_edges()
    internal_edge_indices = [e.idx for e in internal_edges]
    validate_propagator_exponents(propagator_exponents, internal_edge_indices)

    # Validate loop count
    expected_loops = graph.get_loop_count()
    if loop_count != expected_loops:
        raise ValidationError(
            f"Provided loop_count={loop_count} does not match graph topology "
            f"(expected {expected_loops} loops from L = E - V + 1)"
        )

    # Compute Symanzik polynomials
    u_polynomial, f_polynomial = _calculate_symanzik_polynomials(graph, momentum_products)

    # Create parametrisation instances
    schwinger = SchwingerParametrisation(
        graph=graph,
        dimension=dimension,
        loop_count=loop_count,
        propagator_exponents=propagator_exponents,
        u_polynomial=u_polynomial,
        f_polynomial=f_polynomial,
    )

    feynman = FeynmanParametrisation(
        graph=graph,
        dimension=dimension,
        loop_count=loop_count,
        propagator_exponents=propagator_exponents,
        u_polynomial=u_polynomial,
        f_polynomial=f_polynomial,
    )

    lee_pomeransky = LeePomeranskyParametrisation(
        graph=graph,
        dimension=dimension,
        loop_count=loop_count,
        propagator_exponents=propagator_exponents,
        u_polynomial=u_polynomial,
        f_polynomial=f_polynomial,
    )

    return AllParametrisations(
        schwinger=schwinger,
        feynman=feynman,
        lee_pomeransky=lee_pomeransky,
        u_polynomial=u_polynomial,
        f_polynomial=f_polynomial,
    )
