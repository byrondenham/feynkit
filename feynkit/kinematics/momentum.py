"""
Momentum product construction for Feynman integrals.

This module provides functions to create momentum dot product dictionaries
for external momenta in various representations (generic or Mandelstam variables).
"""

import sympy as sp

from ..core.exceptions import ValidationError
from ..core.validation import validate_positive_integer


def create_momentum_products(
    n_external: int,
    use_mandelstam: bool = False,
) -> dict[tuple[int, int], sp.Expr]:
    """
    Create a dictionary of momentum dot products for external legs.

    Constructs symbolic representations of dot products between external momenta,
    either in terms of Mandelstam variables s_{ij} or generic momentum products p_i * p_j.

    Parameters
    ----------
    n_external : int
        Number of external legs (external momenta). Must be positive.
    use_mandelstam : bool, default False
        If True, use Mandelstam variables s_{ij} = (p_i + p_j)^2.
        If False, ise generic dot products p_i * p_j.

    Returns
    -------
    Dict[Tuple[int, int], sp.Expr]
        Dictionary mapping pairs (i, j) with i < j to symbolic expressions.
        - If use_mandelstam=True: s_{ij}/2 for n_external > 2, or s/2 for n_external = 2
        - If use_mandelstam=False: symbol 'p{i}p{j}' representing p_i * p_j

    Raises
    ------
    ValidationError
        If n_external is not a positive integer.

    Notes
    -----
    Mandelstam Variables:

    For n_external = 2, there is only one independent kinematic invariant,
    so we use a single variable 's'.

    The factor of 1/2 in Mandelstam variables comes from the relation:
        s_{ij} - (p_i + p_j)^2 = p_i^2 + 2(p_i * p_j) + p_j^2

    For massless external particles (p_i^2 = 0):
        s_{ij} = 2(p_i * p_j) => p_i * p_j = s_{ij}/2

    Generic Momentum Products:

    When use_mandelstam=False, we create generic symbolic dot products
    p_i * p_j without assuming any specific kinematic relations.

    Examples
    --------
    >>> # Generic momentum products for 3 external legs
    >>> p_dot = create_momentum_products(n_external=3, use_mandelstam=False)
    >>> print(p_dot)
    {(1, 2): p1p2, (1, 3): p1p3, (2, 3): p2p3}

    >>> # Mandelstam variables for 2 external legs
    >>> p_dot = create_momentum_products(n_external=2, use_mandelstam=True)
    >>> print(p_dot)
    {(1, 2): s/2}

    >>> # Mandelstam variables for 3 external legs
    >>> p_dot = create_momentum_products(n_external=3, use_mandelstam=True)
    >>> print(p_dot)
    {(1, 2): s12/2, (1, 3): s13/2, (2, 3): s23/2}

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    """
    # Validate input
    n_external = validate_positive_integer(n_external, "n_external")

    momentum_products: dict[tuple[int, int], sp.Expr] = {}

    if use_mandelstam:
        if n_external == 2:
            # Special case: only one kinematic invariant for 2 external legs
            s = sp.Symbol("s", real=True)
            momentum_products[(1, 2)] = s / 2
        else:
            # General case: Mandelstam variables s_{ij} for each pair
            for i in range(1, n_external + 1):
                for j in range(i + 1, n_external + 1):
                    s_ij = sp.Symbol(f"s{i}{j}", real=True)
                    momentum_products[(i, j)] = s_ij / 2
    else:
        # Generic momentum dot products p_i * p_j
        for i in range(1, n_external + 1):
            for j in range(i + 1, n_external + 1):
                p_ij = sp.Symbol(f"p{i}p{j}", real=True)
                momentum_products[(i, j)] = p_ij

    return momentum_products


def get_momentum_product(
    momentum_products: dict[tuple[int, int], sp.Expr],
    i: int,
    j: int,
) -> sp.Expr:
    """
    Retrieve a momentum product from the dictionary, handling both orderings.

    Since momentum products are symmetric (p_i * p_j = p_j * p_i), this function
    checks both (i, j) and (j, i) orderings in the dictionary.

    Parameters
    ----------
    momentum_products : Dict[Tuple[int, int], sp.Expr]
        Dictionary of momentum products.
    i : int
        First external leg index.
    j : int
        Second external leg index.

    Returns
    -------
    sp.Expr
        The momentum product p_i * p_j.

    Raises
    ------
    ValidationError
        If the momentum product for legs (i, j) is not found.

    Examples
    --------
    >>> p_dot = create_momentum_products(n_external=3, use_mandelstam=False)
    >>> prod = get_momentum_product(p_dot, 1, 2)
    >>> print(prod)
    p1p2
    >>> prod_reversed = get_momentum_product(p_dot, 2, 1)
    >>> print(prod_reversed)
    p1p2
    """
    # Try both orderings
    if (i, j) in momentum_products:
        return momentum_products[(i, j)]
    elif (j, i) in momentum_products:
        return momentum_products[(j, i)]
    else:
        raise ValidationError(
            f"Momentum product for external legs ({i}, {j}) not found in dictionary. "
            f"Available pairs: {list(momentum_products.keys())}"
        )


def validate_momentum_products_complete(
    momentum_products: dict[tuple[int, int], sp.Expr], n_external: int
) -> None:
    """
    Validate that momentum products are provided for all required external leg pairs.

    Parameters
    ----------
    momentum_products : Dict[Tuple[int, int], sp.Expr]
        Dictionary of momentum products to validate.
    n_external : int
        Number of external legs.

    Raises
    ------
    ValidationError
        If any required momentum product is missing.

    Examples
    --------
    >>> p_dot = create_momentum_products(n_external=3, use_mandelstam=False)
    >>> validate_momentum_products_complete(p_dot, n_external=3)
    >>> # No error raised - all products present
    """
    if n_external <= 1:
        return  # No momentum products needed

    missing_pairs = []
    for i in range(1, n_external + 1):
        for j in range(i + 1, n_external + 1):
            # Check both orderings
            if (i, j) not in momentum_products and (j, i) not in momentum_products:
                missing_pairs.append((i, j))

    if missing_pairs:
        raise ValidationError(f"Missing momentum products for external leg pairs: {missing_pairs}")
