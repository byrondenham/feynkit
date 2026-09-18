"""
Principal A-determinant (edge part) for Landau singularity analysis.

For a Feynman integral with Lee–Pomeransky polynomial G, this module computes
the 1-face contribution to the principal A-determinant,

    E_A^(1)(G) = ∏_{τ edge of New(G)}  Δ_{A_τ}(G|_τ)

where Δ_{A_τ}(G|_τ) is the discriminant of G restricted to the edge τ,
viewed as a univariate polynomial in the edge direction.  The zero locus of
E_A^(1) in kinematic space gives the leading Landau singularity surfaces
(normal thresholds and IR singularities).

References
----------
Gelfand–Kapranov–Zelevinsky (1994) §10.1, Theorem 10.1.4.
Klausen (2020), §3, "Counting master integrals".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from .systems.monomial import extract_monomial_support

if TYPE_CHECKING:
    from .integral import FeynmanIntegral

__all__ = [
    "EdgeDiscriminant",
    "LandauAnalysis",
    "landau_analysis",
    "landau_analysis_from_polynomial",
]


# ─── data types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EdgeDiscriminant:
    """Discriminant of G restricted to one edge of its Newton polytope.

    Attributes
    ----------
    edge_exponents
        Exponent vectors of the monomials on this edge.
    edge_coefficients
        Corresponding kinematic coefficients (same order).
    discriminant
        Polynomial in kinematic variables whose zeros are the Landau
        surfaces contributed by this edge.
    """

    edge_exponents: tuple[tuple[int, ...], ...]
    edge_coefficients: tuple[sp.Expr, ...]
    discriminant: sp.Expr


@dataclass(frozen=True)
class LandauAnalysis:
    """Edge-part Landau analysis of a Feynman integral.

    Attributes
    ----------
    edge_discriminants
        One entry per edge of New(G) whose discriminant depends on kinematic
        variables.
    landau_polynomial
        Product of all edge discriminants (a polynomial in kinematics).
    landau_surfaces
        Irreducible kinematic factors of ``landau_polynomial``; each one
        defines a Landau surface.
    """

    edge_discriminants: tuple[EdgeDiscriminant, ...]
    landau_polynomial: sp.Expr
    landau_surfaces: tuple[sp.Expr, ...]


# ─── internal helpers ────────────────────────────────────────────────────────


def _primitive_direction(pts: np.ndarray) -> np.ndarray:
    """Primitive integer direction vector for a collinear set of lattice points."""
    diffs = pts - pts[0]
    direction: np.ndarray | None = None
    for diff in diffs[1:]:
        if np.any(diff != 0):
            direction = diff.copy().astype(int)
            break
    if direction is None:
        return np.zeros(pts.shape[1], dtype=int)
    g = 0
    for v in direction:
        g = int(np.gcd(g, abs(int(v))))
    if g == 0:
        return np.asarray(direction, dtype=int)
    return np.asarray(direction // g, dtype=int)


def _points_on_segment(all_pts: np.ndarray, i: int, j: int) -> list[int]:
    """Indices of all integer lattice points on the closed segment [pts[i], pts[j]]."""
    pi = all_pts[i].astype(float)
    pj = all_pts[j].astype(float)
    direction = pj - pi
    on_seg = [i, j]
    for k in range(len(all_pts)):
        if k in (i, j):
            continue
        diff = all_pts[k].astype(float) - pi
        # Find parameter t such that diff = t * direction
        t_vals: list[float] = []
        for dim in range(len(direction)):
            if abs(direction[dim]) > 1e-9:
                t_vals.append(diff[dim] / direction[dim])
        if not t_vals:
            continue
        t = t_vals[0]
        if not (1e-9 < t < 1.0 - 1e-9):
            continue
        if not all(abs(tv - t) < 1e-9 for tv in t_vals):
            continue
        if all(abs(diff[dim] - t * direction[dim]) < 1e-9 for dim in range(len(direction))):
            on_seg.append(k)
    return sorted(on_seg)


def _hull_edges(pts: np.ndarray) -> list[list[int]]:
    """Enumerate all 1-faces (edges) of the convex hull of pts.

    In d dimensions a pair of hull vertices forms an edge iff they appear
    together in at least d−1 facets.  Interior lattice points on the
    segment are included in each returned list.
    """
    n, d = pts.shape

    if d == 1:
        imin = int(np.argmin(pts[:, 0]))
        imax = int(np.argmax(pts[:, 0]))
        return [_points_on_segment(pts, imin, imax)]

    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(pts.astype(float))
    except QhullError:
        # Points span a lower-dimensional affine subspace — project and recurse.
        basis = (pts - pts[0]).astype(float)
        _, sv, vt = np.linalg.svd(basis, full_matrices=False)
        rank = int(np.sum(sv > 1e-6))
        if rank == 0:
            return []
        proj = basis @ vt[:rank].T  # shape (n, rank)
        return _hull_edges(proj)  # indices are preserved

    hull_verts = hull.vertices.tolist()
    facet_sets = [frozenset(row.tolist()) for row in hull.simplices]

    edges: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for ii in range(len(hull_verts)):
        for jj in range(ii + 1, len(hull_verts)):
            vi, vj = hull_verts[ii], hull_verts[jj]
            pair = (vi, vj)
            if pair in seen:
                continue
            seen.add(pair)
            shared = sum(1 for fs in facet_sets if vi in fs and vj in fs)
            if shared >= d - 1:
                seg = _points_on_segment(pts, vi, vj)
                edges.append(seg)

    return edges


def _univariate_discriminant(coeffs: list[sp.Expr], t_exps: list[int]) -> sp.Expr:
    """Discriminant of P(t) = Σ coeffs[k] * t^{t_exps[k]}.

    Returns Res(P, P') / lc(P)^{deg P - 1}, the standard polynomial
    discriminant.  Returns Integer(1) for linear or constant P.
    """
    _t = sp.Symbol("_t_landau_internal_")
    P = sp.Integer(0)
    for c, e in zip(coeffs, t_exps):
        P = P + c * _t**e
    poly = sp.Poly(P, _t)
    deg = poly.degree()
    if deg <= 1:
        return sp.Integer(1)
    dP = sp.diff(P, _t)
    dpoly = sp.Poly(dP, _t)
    res = poly.resultant(dpoly)
    lc = poly.nth(deg)
    disc = sp.cancel(res / lc ** (deg - 1))
    return sp.factor(disc)


def _factor_list(expr: sp.Expr, kinematic_syms: set[sp.Symbol]) -> list[sp.Expr]:
    """Irreducible factors of the numerator of expr involving kinematic symbols."""
    num, _den = sp.fraction(sp.together(expr))
    coeff, factors = sp.factor_list(num)
    result: list[sp.Expr] = []
    for base, _exp in factors:
        if base.free_symbols & kinematic_syms:
            result.append(base)
    return result


# ─── public API ──────────────────────────────────────────────────────────────


def landau_analysis_from_polynomial(
    g_poly: sp.Expr,
    lp_parameters: list[sp.Symbol],
) -> LandauAnalysis:
    """Edge-part Landau analysis for a Lee–Pomeransky polynomial.

    Parameters
    ----------
    g_poly
        Lee–Pomeransky polynomial G(u; kinematics).
    lp_parameters
        Schwinger parameters u_i (the integration variables of g_poly).

    Returns
    -------
    LandauAnalysis
        Edge discriminants, their product, and the irreducible Landau surfaces.
    """
    if not lp_parameters or g_poly == sp.Integer(0):
        return LandauAnalysis((), sp.Integer(1), ())
    support = extract_monomial_support(g_poly, lp_parameters)
    if not support:
        return LandauAnalysis((), sp.Integer(1), ())

    kinematic_syms: set[sp.Symbol] = g_poly.free_symbols - set(lp_parameters)
    all_exps = np.array([list(e) for e, _ in support], dtype=int)

    edge_index_lists = _hull_edges(all_exps)

    non_trivial: list[EdgeDiscriminant] = []
    for idx_list in edge_index_lists:
        if len(idx_list) < 2:
            continue

        edge_pts = all_exps[idx_list]
        direction = _primitive_direction(edge_pts)
        raw_t = [int(np.dot(direction, pt)) for pt in edge_pts]
        t_min = min(raw_t)
        t_exps = [v - t_min for v in raw_t]

        edge_coeffs = [support[k][1] for k in idx_list]
        disc = _univariate_discriminant(edge_coeffs, t_exps)

        if disc.free_symbols & kinematic_syms:
            edge_exps = tuple(support[k][0] for k in idx_list)
            non_trivial.append(
                EdgeDiscriminant(
                    edge_exponents=edge_exps,
                    edge_coefficients=tuple(edge_coeffs),
                    discriminant=disc,
                )
            )

    if not non_trivial:
        return LandauAnalysis((), sp.Integer(1), ())

    raw: sp.Expr = sp.Integer(1)
    for ed in non_trivial:
        raw = sp.expand(raw * ed.discriminant)
    # Keep only the numerator: the denominator is a scale factor with no zeros
    landau_poly, _den = sp.fraction(sp.together(raw))
    landau_poly = sp.factor(landau_poly)

    surfaces = tuple(_factor_list(landau_poly, kinematic_syms))

    return LandauAnalysis(
        edge_discriminants=tuple(non_trivial),
        landau_polynomial=landau_poly,
        landau_surfaces=surfaces,
    )


def landau_analysis(integral: FeynmanIntegral) -> LandauAnalysis:
    """Edge-part Landau analysis of a :class:`~feynkit.FeynmanIntegral`.

    Computes the edge-part principal A-determinant E_A^(1)(G) of the
    Lee–Pomeransky polynomial G.  Its zero locus in kinematic space is the
    leading Landau variety of the integral.

    Parameters
    ----------
    integral
        A :class:`~feynkit.FeynmanIntegral` instance.

    Returns
    -------
    LandauAnalysis
    """
    s = integral.symanzik
    return landau_analysis_from_polynomial(s.g, s.lp_parameters)
