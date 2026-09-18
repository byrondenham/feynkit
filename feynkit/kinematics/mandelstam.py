"""
Mandelstam variable utilities for Feynman integrals.

Provides functions to work with Mandelstam variables and kinematic constraints.
"""

import sympy as sp


def create_mandelstam_variables(n_external: int) -> dict[tuple[int, int], sp.Symbol]:
    """
    Create Mandelsam variable symbols for all external leg pairs.

    Parameters
    ----------
    n_external : int
        Number of external legs.

    Returns
    -------
    Dict[Tuple[int, int], sp.Symbol]
        Dictionary mapping leg pairs (i, j) to Mandelstam variables s_{ij}.

    Examples
    --------
    >>> s_vars = create_mandelstam_variables(n_external=3)
    >>> print(s_vars)
    {(1, 2): s12, (1, 3): s13, (2, 3): s23}
    """
    if n_external == 2:
        # Single variable for 2-point function
        return {(1, 2): sp.Symbol("s", real=True)}

    mandelstam_vars = {}
    for i in range(1, n_external + 1):
        for j in range(i + 1, n_external + 1):
            mandelstam_vars[(i, j)] = sp.Symbol(f"s{i}{j}", real=True)

    return mandelstam_vars


def mandelstam_constraints(
    n_external: int,
    mandelstam_vars: dict[tuple[int, int], sp.Symbol],
    massless: bool = True,
) -> list[sp.Expr]:
    """
    Generate kinematic constraints for Mandelstam variables.

    For massless external particles with momentum conservation,
    the Mandelstam variables satisfy:
        sum_i p_i = 0 => sum_{i<j} s_{ij} = 0

    Parameters
    ----------
    n_external : int
        Number of external legs.
    mandelstam_vars : Dict[Tuple[int, int], sp.Symbol]
        Dictionary of Mandelstam variables.
    massless : bool, default True
        Whether external particles are massless.

    Returns
    -------
    List[sp.Expr]
        List of constraint equations (expressions that should equal zero).

    Notes
    -----
    For n external massless particles, there are n(n-1)/2 Mandelstam variables
    but only n(n-3)/2 independent ones due to momentum conservation.

    Examples
    --------
    >>> s_vars = create_mandelstam_variables(n_external=3)
    >>> constraints = mandelstam_constraints(3, s_vars, massless=True)
    >>> # For 3 massless particles: s12 + s13 + s23 = 0
    """
    if n_external <= 2:
        return []  # No constraints for 2-point functions

    constraints = []

    if massless:
        # For massless particles: sum of all Mandelstam variables = 0
        total = sum(mandelstam_vars.values())
        constraints.append(total)

    return constraints
