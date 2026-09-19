"""
Mandelstam variable utilities for Feynman integrals.

Provides functions to work with Mandelstam variables and kinematic constraints.
"""

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class KinematicInvariants:
    """Standard independent invariants for n external legs.

    Attributes
    ----------
    n_external : int
        Number of external legs.
    external_masses : list[sp.Symbol]
        The squared external momenta p_i^2, one per leg (for two legs both
        entries are the single invariant s = p^2).
    mandelstam : dict[tuple[int, int], sp.Symbol]
        Planar multi-particle invariants s_{i...j-1} = (p_i + ... + p_{j-1})^2,
        keyed by (i, j) with j - i >= 2 and (i, j) != (1, n). Together with the
        external masses these are the n(n-1)/2 independent invariants.
    momentum_products : dict[tuple[int, int], sp.Expr]
        Every dot product p_i . p_j (i < j) written in the invariants.
    """

    n_external: int
    external_masses: list[sp.Symbol]
    mandelstam: dict[tuple[int, int], sp.Symbol] = field(default_factory=dict)
    momentum_products: dict[tuple[int, int], sp.Expr] = field(default_factory=dict)

    @property
    def symbols(self) -> list[sp.Symbol]:
        """All independent invariants, masses first."""
        return list(self.external_masses) + list(self.mandelstam.values())


def standard_invariants(n_external: int) -> KinematicInvariants:
    """Express the external dot products in the standard planar invariants.

    The independent invariants are the external masses p_i^2 and the planar
    multi-particle invariants s_{i...j-1} = (p_i + ... + p_{j-1})^2 for
    j - i >= 2, (i, j) != (1, n). There are n(n-1)/2 of them, the number of
    independent dot products of n momenta subject to conservation. For n = 4
    this is the familiar set {s, t, p_1^2, ..., p_4^2} with u eliminated.

    For two legs the only invariant is s = p_1^2 = p_2^2 and the product is
    p_1 . p_2 = -s, so that F = -s a_1 a_2 / mu^2 + ... and the normal
    threshold of the massive bubble sits at s = (m_1 + m_2)^2.

    Conventions follow Weinzierl (2022), section 2: mostly-minus metric,
    F = -sum over 2-forests of s_F prod a_e + U sum m_e^2 a_e.
    """
    n = n_external
    if n < 2:
        raise ValueError("At least two external legs are needed")
    if n == 2:
        s = sp.Symbol("s", real=True)
        return KinematicInvariants(2, [s, s], {}, {(1, 2): -s})

    masses = [sp.Symbol(f"p{i}^2", real=True) for i in range(1, n + 1)]
    mandelstam: dict[tuple[int, int], sp.Symbol] = {}
    for i in range(1, n + 1):
        for j in range(i + 2, n + 1):
            if (i, j) == (1, n):
                continue
            mandelstam[(i, j)] = sp.Symbol("s" + "".join(str(k) for k in range(i, j)), real=True)

    # Unknowns: dot products of the n-1 independent momenta p_1 .. p_{n-1}.
    unknowns = {(a, b): sp.Symbol(f"_x{a}_{b}") for a in range(1, n) for b in range(a, n)}

    def dot(a: int, b: int) -> sp.Expr:
        """p_a . p_b with p_n = -(p_1 + ... + p_{n-1})."""
        if a > b:
            a, b = b, a
        if b < n:
            return unknowns[(a, b)]
        if a < n:
            return -sum(dot(a, k) for k in range(1, n))
        return sum(dot(k, m) for k in range(1, n) for m in range(1, n))

    def square(legs: list[int]) -> sp.Expr:
        return sum(dot(a, b) for a in legs for b in legs)

    equations = [sp.Eq(square([i]), masses[i - 1]) for i in range(1, n + 1)]
    equations += [sp.Eq(square(list(range(i, j))), sym) for (i, j), sym in mandelstam.items()]
    solution = sp.solve(equations, list(unknowns.values()), dict=True)
    if len(solution) != 1:
        raise RuntimeError("Invariant system is not uniquely solvable")  # pragma: no cover
    sol = solution[0]

    products: dict[tuple[int, int], sp.Expr] = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            products[(i, j)] = sp.expand(dot(i, j).subs(sol))
    return KinematicInvariants(n, masses, mandelstam, products)
