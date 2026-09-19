"""
Frozen value types returned by :class:`feynkit.FeynmanIntegral` and the
polytope-equivalence verbs.

These types are kept in their own module to avoid circular imports between
:mod:`feynkit.integral` (which uses them) and downstream modules (which
also produce them, e.g. the polytope-equivalence verbs in
:mod:`feynkit.normal_forms.affine_equivalence`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp


@dataclass(frozen=True)
class SymanzikPolynomials:
    """
    The Symanzik U and F polynomials together with the Lee-Pomeransky G = U + F.

    All four polynomials are presented in two complete parameter sets so the
    user can pick whichever is convenient: the Schwinger parameters
    ``a_i`` (the Schwinger form, used internally by the U/F construction) and
    the Lee-Pomeransky parameters ``u_i`` (used by the GKZ system and the
    Newton polytope).

    Attributes
    ----------
    u, f
        First and second Symanzik polynomials in Schwinger parameters.
    u_lp, f_lp
        The same polynomials after substituting ``a_i -> u_i``.
    g
        Lee-Pomeransky polynomial ``G(u) = U(u) + F(u)``.
    schwinger_parameters
        The ``a_i`` symbols, in edge-index order.
    lp_parameters
        The ``u_i`` symbols, in edge-index order.
    """

    u: sp.Expr
    f: sp.Expr
    u_lp: sp.Expr
    f_lp: sp.Expr
    g: sp.Expr
    schwinger_parameters: list[sp.Symbol]
    lp_parameters: list[sp.Symbol]


@dataclass(frozen=True)
class NewtonPolytope:
    """
    Newton polytope of the Lee-Pomeransky G polynomial.

    Wraps the monomial support together with the homogenised GKZ A-matrix
    (whose columns are the support exponent vectors with a leading row of
    ones).

    Attributes
    ----------
    support
        Pairs ``(exponent_vector, coefficient)`` for every monomial of G.
    a_matrix
        The GKZ A-matrix; columns are homogenised exponent vectors.
    parameters
        The Lee-Pomeransky parameters labelling the rows of the
        un-homogenised exponent vectors.
    """

    support: list[tuple[tuple[int, ...], sp.Expr]]
    a_matrix: sp.Matrix
    parameters: list[sp.Symbol]

    @property
    def points(self) -> list[tuple[int, ...]]:
        """All exponent vectors appearing in the support."""
        return [exp_vec for exp_vec, _ in self.support]

    @property
    def coefficients(self) -> list[sp.Expr]:
        """Coefficients in the same order as :attr:`points`."""
        return [coeff for _, coeff in self.support]


@dataclass(frozen=True)
class ToricIdeal:
    """
    Toric ideal associated with the GKZ A-matrix.

    Attributes
    ----------
    generators
        Polynomial generators of the toric ideal in the variables ``z_j``.
    a_matrix
        The GKZ A-matrix the ideal is associated with.
    z_variables
        The differential / monomial variables ``[z_1, ..., z_m]``.
    """

    generators: list[sp.Expr]
    a_matrix: sp.Matrix
    z_variables: list[sp.Symbol]


@dataclass(frozen=True)
class PolytopeAutomorphisms:
    """
    Unimodular automorphism group of a convex lattice polytope.

    Aut(P) = { (U, t) : U in GL_n(Z), |det U| = 1, t in Z ^n, {Up + t : p in P} = P }.

    Attributes
    ----------
    maps
        All automorphisms as ``(U, t)`` pairs where ``U`` is an
        ``ImmutableMatrix`` in ``GL_n(Z)`` and ``t`` is an integer column
        vector (also an ``ImmutableMatrix``).
    order
        ``|Aut(P)|``, equal to ``len(maps)``.
    vertex_permutations
        For each automorphism, the induced permutation on the (hull) vertex
        list as a list of ints: ``vertex_permutations[k][i] = j`` means the
        k-th automorphism sends vertex i to vertex j.
    vertex_orbits
        Partition of vertex indices into orbits under the full group action.
    """

    maps: list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]]
    order: int
    vertex_permutations: list[list[int]]
    vertex_orbits: list[list[int]]


@dataclass(frozen=True)
class PolytopeEquivalence:
    """
    Result of an equivalence test between two convex (lattice) polytopes.

    Attributes
    ----------
    equivalent
        Whether the two polytopes were found equivalent under the named
        relation.
    relation
        ``"unimodular"``, integer map with det +/-1 (Liu-Cai).
        ``"affine_polytope"``, rational/integer affine map, hull vertices only.
        ``"affine_point_config"``, rational/integer affine map, all A-columns.
    witness_map
        Linear part ``M`` of the witness affine map ``v -> M*v + t``.
        For ``"unimodular"`` results this is in ``GL_n(Z)``; for affine
        results it may be rational.  ``None`` on failure.
    translation
        Translation vector ``t`` of the witness map (column vector).
        ``None`` on failure or when not extracted.
    determinant
        ``det(M)`` of the witness linear part.  ``None`` on failure.
    vertex_correspondence
        For positive results, a list ``[j_0, j_1, ...]`` such that input
        point ``i`` of source is mapped to point ``j_i`` of target.
    """

    equivalent: bool
    relation: Literal["unimodular", "affine_polytope", "affine_point_config"]
    witness_map: sp.Matrix | None = None
    translation: sp.Matrix | None = None
    determinant: sp.Expr | None = None
    vertex_correspondence: list[int] | None = None
