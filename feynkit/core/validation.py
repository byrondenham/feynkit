"""
Input validation utilities for feynkit.

Provides functions to validate graph structures, parameters, and other inputs
to ensure consistency and catch errors early.
"""

from typing import Any

import sympy as sp

from .constants import MIN_INTERNAL_VERTEX
from .exceptions import EdgeConnectivityError, ParameterError, ValidationError, VertexIndexError


def validate_positive_integer(value: Any, name: str) -> int:
    """
    Validate a value that is a positive integer.

    Parameters
    ----------
    value : Any
        Value to validate.
    name : str
        Name of the parameter for error messages.

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    ValidationError
        If value is not a positive integer.

    Examples
    --------
    >>> validate_positive_integer(5, "loop_count")
    5
    >>> validate_positive_integer(-1, "loop_count")
    Traceback (most recent call last):
        ...
    ValidationError: loop_count must be a positive integer, got -1
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from ValueError

    if int_value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {int_value}")

    return int_value


def validate_nonnegative_integer(value: Any, name: str) -> int:
    """
    Validate that a value is a non-negative integer.

    Parameters
    ----------
    value : Any
        Value to validate.
    name : str
        Name of the parameter for error messages.

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    ValidationError
        If value is not a non-negative integer.
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from TypeError

    if int_value < 0:
        raise ValidationError(f"{name} must be non-negative, got {int_value}")

    return int_value


def validate_vertex_index(
    vertex: int,
    min_val: int,
    max_val: int,
    vertex_type: str = "vertex",
) -> None:
    """
    Validate that a vertex index is in the valid range.

    Parameters
    ----------
    vertex : int
        Vertex index to validate.
    min_val : int
        Minimum valid value (inclusive).
    max_val : int
        Maximum valid value (inclusive).
    vertex_type : str, default "vertex"
        Description of vertex type for error messages.

    Raises
    ------
    VertexIndexError
        If vertex is outside of the valid range.

    Examples
    --------
    >>> validate_vertex_index(2, 1, 5, "internal vertex")
    >>> validate_vertex_index(6, 1, 5, "internal vertex")
    Traceback (most recent call last):
        ...
    VertexIndexError: Invalid internal vertex index 6: must be in range [1, 5]
    """
    if not (min_val <= vertex <= max_val):
        raise VertexIndexError(vertex, min_val, max_val, vertex_type)


def validate_edge_connectivity(
    edge_v1: int,
    edge_v2: int,
    is_internal: bool,
    internal_vertex_count: int,
    external_leg_count: int,
    edge_idx: int,
) -> None:
    """
    Validate that an edge's connectivity follows graph conventions.

    Parameters
    ----------
    edge_v1 : int
        Starting vertex of the edge.
    edge_v2 : int
        Ending vertex of the edge.
    is_internal : bool
        Whether the edge is an internal propagator.
    internal_vertex_count : int
        Number of internal vertices in the graph.
    external_leg_count : int
        Number of external legs in the graph.
    edge_idx : int
        Index of the edge (for error messages).

    Raises
    ------
    EdgeConnectivityError
        If edge connectivity violates graph conventions.

    Notes
    -----
    Conventions:
    - All edges must start from an internal vertex (v1 in [1, r_int])
    - Internal edges must end at an internal vertex (v2 in [1, r_int])
    - External edges must end at an external vertex (v2 in [r_int + 1, r_int + n_ext])
    """
    # v1 must always be an internal vertex
    if not (MIN_INTERNAL_VERTEX <= edge_v1 <= internal_vertex_count):
        raise EdgeConnectivityError(
            f"Edge {edge_idx}: starting vertex v1={edge_v1} must be an internal vertex "
            f"(in range [1, {internal_vertex_count}])"
        )

    if is_internal:
        # Internal edge: v2 must be an internal vertex
        if not (MIN_INTERNAL_VERTEX <= edge_v2 <= internal_vertex_count):
            raise EdgeConnectivityError(
                f"Edge {edge_idx}: internal edge must connect to internal vertex, "
                f"but v2={edge_v2} is outside range [1, {internal_vertex_count}]"
            )
    else:
        # External edge: v2 must be an external vertex
        min_external = internal_vertex_count + 1
        max_external = internal_vertex_count + external_leg_count
        if not (min_external <= edge_v2 <= max_external):
            raise EdgeConnectivityError(
                f"Edge {edge_idx}: external edge must connect external vertex, "
                f"but v2={edge_v2} is outside range [{min_external}, {max_external}]"
            )


def validate_propagator_exponents(exponents: dict[int, sp.Expr], edge_indices: list[int]) -> None:
    """
    Validate that propagator exponents are provided for all internal edges.

    Parameters
    ----------
    exponents : Dict[int, sp.Expr]
        Dictionary mapping edge indices to their exponents.
    edge_indices : List[int]
        List of all internal edge indices that require exponents.

    Raises
    ------
    ParameterError
        If any required exponent is missing.

    Examples
    --------
    >>> import sympy as sp
    >>> nu1, nu2, = sp.symbols('nu1 nu2', positive=True)
    >>> exponents = {1: nu1, 2: nu2}
    >>> validate_propagator_exponents(exponents, [1, 2])
    >>> validate_propagator_exponents(exponents, [1, 2, 3])
    Traceback (most recent call last):
        ...
    ParameterError: Missing propagator exponents for edges: [3]
    """
    missing = [idx for idx in edge_indices if idx not in exponents]
    if missing:
        raise ParameterError(f"Missing parameter exponents for edges: {missing}")


def validate_momentum_products(
    momentum_products: dict[tuple[int, int], sp.Expr],
    external_leg_count: int,
) -> None:
    """
    Validate that momentum dot products are provided for all external leg pairs.

    Parameters
    ----------
    momentum_products : Dict[tuple[int, int], sp.Expr]
        Dictionary mapping (i, j) pairs to momentum dot products p_i · p_j.
    external_leg_count : int
        Number of external legs in the graph.

    Raises
    ------
    ParameterError
        If any required momentum product is missing.

    Notes
    -----
    For n external legs, we expect momentum products for all pairs (i, j) with i < j.
    The dictionary can have either (i, j) of (j, i) as keys (order doesn't matter).
    """
    if external_leg_count <= 1:
        return  # No momentum producs needed

    missing_pairs = []
    for i in range(1, external_leg_count + 1):
        for j in range(i + 1, external_leg_count + 1):
            # Check both orderings
            if (i, j) not in momentum_products and (j, i) not in momentum_products:
                missing_pairs.append((i, j))

    if missing_pairs:
        raise ParameterError(f"Missing momentum products for external leg pairs: {missing_pairs}")


def validate_dimension_parameter(dimension: sp.Expr) -> None:
    """
    Validate that the spacetime dimension parameter is appropriate.

    Parameters
    ----------
    dimensions : sp.Expr
        Spacetime dimension (typically symbolic like D or 4-2ε).

    Raises
    ------
    ValidationError
        If dimension is not a valid SymPy expression.

    Notes
    -----
    This is a basic validation. More sophisticated checks could verify that
    the dimension is positive, real, etc., but we keep it simple for flexibility.
    """
    if not isinstance(dimension, sp.Basic):
        raise ValidationError(
            f"Dimension must be a SymPy expression, got {type(dimension).__name__}"
        )
