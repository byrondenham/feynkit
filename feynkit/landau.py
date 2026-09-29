"""
Principal A-determinant and Landau surfaces of a Feynman integral.

For the Lee-Pomeransky polynomial G = U + F with Newton polytope P, the
principal A-determinant is the product over all faces tau of P of the
A-discriminant of G restricted to tau (Gelfand, Kapranov and Zelevinsky
1994, chapter 10, theorem 1.2). Its zero locus in kinematic space is the
singular locus of the GKZ system, and it contains the Landau variety
(Klausen 2023, section 5.1). This module computes the reduced form: every
irreducible kinematic factor once, without the multiplicities.

Faces contribute as follows.

- A vertex contributes its coefficient (Fevola, Mizera and Telen 2023,
  example 2.1).
- A face whose lattice points form a simplex has discriminant 1.
- An edge contributes the discriminant of the univariate polynomial in
  the lattice coordinate along the edge.
- Any other face contributes the elimination ideal of {f_tau = 0,
  t_i d f_tau / d t_i = 0} in the torus, computed with a Gröbner basis
  at mu = 1 when a sufficient condition shows that to be exact (see
  ``landau_analysis_from_polynomial``) and, when the distinct
  coefficients of the face that are not constant are linearly
  independent linear forms in the kinematic symbols or in the squares of
  those that occur only squared, in fresh
  symbols standing for them, a change of coordinates that keeps the
  Gröbner basis small. It contributes the factors of the generator or,
  when there are several, of their greatest common divisor, the
  codimension-one part of their zero set. Faces with more points than
  ``max_face_points`` are skipped and listed in
  ``LandauAnalysis.skipped_faces``.

The factors are candidate codimension-one singular loci on all sheets of the
integral. Membership is necessary for a singularity, not sufficient, and
the list is neither complete for physical sheets nor guaranteed exhaustive
(Fevola, Mizera and Telen 2023, section 2).

For one-loop graphs, :func:`one_loop_principal_a_determinant` gives the
closed form of Dlapa, Helmer, Papathanasiou and Tellander (2023, eq.
1LoopEA): the product of the principal minors of the modified Cayley
matrix of the cycle and, for a graph with bridges, the poles of the
bridges' propagators. It is used as an independent check of the face
computation.
"""

from __future__ import annotations

import numbers
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
import sympy.core.random as sympy_random
from sympy.polys.matrices import DomainMatrix

from ._exact import affine_rank
from .core.exceptions import ComputationError, ValidationError
from .polytope import faces as _faces
from .polytope import lattice_coordinates as _lattice_coordinates
from .systems.monomial import extract_monomial_support

if TYPE_CHECKING:
    from .core.edge import Edge
    from .integral import FeynmanIntegral

__all__ = [
    "FaceDiscriminant",
    "LandauAnalysis",
    "landau_analysis",
    "landau_analysis_from_polynomial",
    "one_loop_bridge_poles",
    "one_loop_landau_surfaces",
    "one_loop_landau_surfaces_by_type",
    "one_loop_principal_a_determinant",
]


# --- data types --------------------------------------------------------------


@dataclass(frozen=True)
class FaceDiscriminant:
    """A-discriminant of G restricted to one face of its Newton polytope.

    Attributes
    ----------
    dimension
        Dimension of the face.
    exponents
        Exponent vectors of the monomials of G on the face.
    coefficients
        Their kinematic coefficients, in the same order.
    discriminant
        Polynomial in the kinematic variables; 1 when the face is a simplex
        or its discriminant carries no kinematics.
    is_simplex
        True when the face's lattice points are affinely independent.
    principal
        True when the discriminant was obtained as a single generator. For
        faces of dimension two or more it is False when the elimination
        ideal needed more than one generator; the discriminant is then the
        product of the kinematic factors of their greatest common divisor,
        since an irreducible polynomial defines a codimension-one component
        of their common zero set exactly when it divides every generator.
    """

    dimension: int
    exponents: tuple[tuple[int, ...], ...]
    coefficients: tuple[sp.Expr, ...]
    discriminant: sp.Expr
    is_simplex: bool
    principal: bool = True


@dataclass(frozen=True)
class LandauAnalysis:
    """Reduced principal A-determinant of a Feynman integral.

    Attributes
    ----------
    face_discriminants
        One entry per face of the Newton polytope of G, all dimensions,
        with the exact discriminant computed for that face.
    principal_a_determinant
        Product of ``landau_surfaces`` (the reduced principal A-determinant).
    landau_surfaces
        The distinct irreducible factors of the face discriminants that
        carry kinematics, one per candidate singular surface. A factor
        built from the energy scale alone is not a singular surface and is
        excluded, though the face discriminants that have it keep it. It
        can come from a face eliminated with the scale as a variable (see
        ``landau_analysis_from_polynomial``'s ``scale`` argument), or from a
        vertex or an edge whose coefficients carry the scale in a
        numerator.
    skipped_faces
        Exponent sets of faces that were too large to eliminate.
    """

    face_discriminants: tuple[FaceDiscriminant, ...]
    principal_a_determinant: sp.Expr
    landau_surfaces: tuple[sp.Expr, ...]
    skipped_faces: tuple[tuple[tuple[int, ...], ...], ...] = ()


# --- discriminants -----------------------------------------------------------


def _univariate_discriminant(coeffs: list[sp.Expr], t_exps: list[int]) -> sp.Expr:
    """Discriminant of P(t) = sum coeffs[k] * t^{t_exps[k]}.

    Returns Res(P, P') / lc(P)^{deg P - 1}, the standard polynomial
    discriminant.  Returns Integer(1) for linear or constant P.
    """
    _t = sp.Symbol("_t_landau_internal_")
    P = sp.Integer(0)
    for c, e in zip(coeffs, t_exps, strict=True):
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


def _normalised(irreducible: sp.Expr) -> sp.Expr:
    """An irreducible polynomial as :func:`_factor_list` writes it, without factoring it.

    sp.factor_list makes a factor primitive over the integers, with a positive
    leading coefficient in the order sp.Poly gives the symbols.
    """
    _, poly = sp.Poly(irreducible).clear_denoms(convert=True)
    _, poly = poly.primitive()
    return (-poly if poly.LC() < 0 else poly).as_expr()


# --- public API --------------------------------------------------------------


def _singular_binary() -> str | None:
    """Path to the Singular binary, or None if it is not installed."""
    import shutil

    return shutil.which("Singular")


# subprocess.run waits through a C int of milliseconds, which holds about 2.1e6 s.
_MAX_TIMEOUT = 2_000_000


def _check_timeout(timeout: object, *, optional: bool = True) -> None:
    """Raise ValidationError unless ``timeout`` is a number of seconds greater
    than 0 and at most 2,000,000, or None when ``optional``."""
    if timeout is None and optional:
        return
    if (
        isinstance(timeout, numbers.Real)
        and not isinstance(timeout, bool)
        and not timeout <= 0
        and timeout <= _MAX_TIMEOUT  # false for NaN
    ):
        return
    what = "None or a number" if optional else "a number"
    raise ValidationError(
        f"timeout must be {what} of seconds greater than 0 and at most {_MAX_TIMEOUT:,}; "
        f"got {timeout!r:.60}"
    )


def _eliminate_sympy(
    system: list[sp.Expr], to_eliminate: list[sp.Symbol], kin: list[sp.Symbol]
) -> list[sp.Poly]:
    basis = sp.groebner(system, *to_eliminate, *kin, order="lex")
    return [sp.Poly(g, *kin) for g in basis.exprs if not (g.free_symbols & set(to_eliminate))]


# A term of Singular's output, and a factor of a term: a coefficient or a power of a variable.
_SINGULAR_TERM = re.compile(r"([+-]?)([^+-]+)")
_SINGULAR_FACTOR = re.compile(r"([0-9]+)(?:/([0-9]+))?|v([0-9]+)(?:\^([0-9]+))?")


def _read_singular_polynomial(
    line: str, first: int, n_vars: int
) -> dict[tuple[int, ...], Fraction] | None:
    """The terms of a polynomial Singular printed in v_first, ..., v_{first + n_vars - 1}.

    Returns None when the line is anything else, such as an error message,
    a term in another variable or a coefficient with denominator 0.
    """
    terms: dict[tuple[int, ...], Fraction] = {}
    end = 0
    for match in _SINGULAR_TERM.finditer(line):
        if match.start() != end:
            return None
        end = match.end()
        sign, body = match.groups()
        coefficient = Fraction(-1 if sign == "-" else 1)
        exponents = [0] * n_vars
        for factor in body.split("*"):
            parsed = _SINGULAR_FACTOR.fullmatch(factor)
            if parsed is None:
                return None
            numerator, denominator, index, power = parsed.groups()
            if numerator is not None:
                if denominator is not None and not int(denominator):
                    return None
                coefficient *= Fraction(int(numerator), int(denominator or 1))
                continue
            k = int(index) - first
            if not 0 <= k < n_vars:
                return None
            exponents[k] += int(power or 1)
        key = tuple(exponents)
        terms[key] = terms.get(key, Fraction(0)) + coefficient
    if end != len(line):
        return None
    return terms


def _eliminate_singular(
    system: list[sp.Expr],
    to_eliminate: list[sp.Symbol],
    kin: list[sp.Symbol],
    binary: str,
    *,
    points: int,
    timeout: float | None = None,
) -> list[sp.Poly]:
    """Elimination ideal via Singular's ``eliminate`` (Decker et al., Singular 4).

    Singular prints each generator expanded on one line, which can hold tens
    of thousands of terms; the line is read term by term, since SymPy's
    parser recurses once per term. ``points`` is the number of points of the
    face, for the error messages.

    Raises
    ------
    ComputationError
        If Singular fails, runs past ``timeout`` seconds, or prints anything
        but polynomials in ``kin``.
    """
    names = {sym: f"v{i}" for i, sym in enumerate(to_eliminate + kin)}

    def render(expr: sp.Expr) -> str:
        return str(sp.expand(expr).subs(names, simultaneous=True)).replace("**", "^")

    ring_vars = ",".join(names[v] for v in to_eliminate + kin)
    ideal = ",".join(render(g) for g in system)
    product = "*".join(names[v] for v in to_eliminate)
    script = (
        f"ring r = 0, ({ring_vars}), dp;\n"
        f"ideal I = {ideal};\n"
        f"ideal E = eliminate(I, {product});\n"
        "int k; for (k = 1; k <= size(E); k++) { print(string(E[k])); }\n"
        "quit;\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "elim.sing"
        path.write_text(script)
        try:
            result = subprocess.run(
                [binary, "-q", "--no-warn", str(path)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ComputationError(
                f"Singular did not eliminate a face with {points} points within "
                f"timeout={timeout} s"
            ) from None
    size = f"{len(result.stdout)} characters of output"
    if result.returncode != 0:
        reason = " ".join(result.stderr.split())[:100] or f"exit status {result.returncode}"
        raise ComputationError(
            f"Singular failed on a face with {points} points, {size}: {reason}"
        ) from None
    out: list[sp.Poly] = []
    for text in result.stdout.splitlines():
        line = text.replace(" ", "")
        if not line or line == "0":
            continue
        terms = _read_singular_polynomial(line, len(to_eliminate), len(kin))
        if terms is None:
            raise ComputationError(
                f"cannot read Singular's elimination ideal of a face with {points} points "
                f"from {size}, at {repr(text.strip())[:60]}"
            ) from None
        polynomial = {key: sp.Rational(c.numerator, c.denominator) for key, c in terms.items() if c}
        if polynomial:
            out.append(sp.Poly.from_dict(polynomial, *kin))
    return out


def _renaming(coeffs: list[sp.Expr]) -> tuple[dict[sp.Expr, sp.Symbol], bool]:
    """A fresh symbol for each distinct coefficient that is not constant, and whether an
    atom is squared; or ({}, False).

    The renaming is returned only when those coefficients are linearly
    independent linear forms, with rational coefficients, in atoms: symbols
    that occur only to the first power, or only squared, as masses do.
    Completed to a basis of the linear forms in the atoms, they are then new
    coordinates on the polynomial ring of the atoms, and the other basis
    elements are variables the face does not involve.
    """
    distinct = list(dict.fromkeys(c for c in coeffs if c.free_symbols))
    if not distinct:
        return {}, False
    symbols = sorted(set().union(*(c.free_symbols for c in distinct)), key=str)
    powers: dict[int, int] = {}
    rows: list[list[sp.Expr]] = []
    for c in distinct:
        try:
            poly = sp.Poly(c, *symbols)
        except sp.PolynomialError:
            return {}, False
        if poly.domain not in (sp.ZZ, sp.QQ):
            return {}, False
        row: list[sp.Expr] = [sp.Integer(0)] * len(symbols)
        for monomial, value in poly.terms():
            used = [(k, e) for k, e in enumerate(monomial) if e]
            if len(used) != 1:
                return {}, False
            ((k, e),) = used
            if e > 2 or powers.setdefault(k, e) != e:
                return {}, False
            row[k] = value
        rows.append(row)
    if sp.Matrix(rows).rank() < len(distinct):
        return {}, False
    renaming = {c: sp.Dummy(f"c{i}") for i, c in enumerate(distinct, start=1)}
    return renaming, 2 in powers.values()


def _unit_scale_is_exact(support: list[tuple[tuple[int, ...], sp.Expr]], scale: sp.Symbol) -> bool:
    """Whether the faces can be eliminated at scale = 1.

    They can when every coefficient is a power of the scale times an
    expression free of it, with the power an affine function of the
    exponents, as for G = U + F, where it is 0 on U and -2 on F, unless the
    kinematics contain the scale. Rescaling the scale then multiplies the
    coefficients by a character of the torus, which maps the singular locus
    of every face to itself, so that away from scale = 0 the locus is
    determined by its slice at scale = 1.
    """
    lifted: list[tuple[int, ...]] = []
    for exponents, c in support:
        power = 0
        for part, sign in zip(sp.fraction(sp.together(c)), (1, -1), strict=True):
            dependent = part.as_independent(scale, as_Add=False)[1]
            if dependent == 1:
                continue
            base, k = dependent.as_base_exp()
            if base != scale or not k.is_Integer:
                return False
            power += sign * int(k)
        lifted.append((*exponents, power))
    return affine_rank(lifted) == affine_rank([exponents for exponents, _ in support])


def _elimination_discriminant(
    coeffs: list[sp.Expr],
    exps: list[tuple[int, ...]],
    kinematic_syms: set[sp.Symbol],
    backend: str = "auto",
    *,
    scale: sp.Symbol | None = None,
    timeout: float | None = None,
) -> tuple[list[sp.Expr], bool]:
    """Kinematic locus where f = sum c_k t^{e_k} has a singular point in the torus.

    Returns (distinct kinematic factors, principal) where principal is True
    when the elimination ideal had at most one generator. The factors are
    those of the generator or, when there are several, of their greatest
    common divisor: an irreducible polynomial defines a codimension-one
    component of their common zeros exactly when it divides every one of
    them. Uses Singular when installed and ``backend`` is "auto" or
    "singular", else SymPy; ``timeout`` limits each Singular run, in
    seconds.

    The coefficients are taken at ``scale`` = 1, which the caller passes
    only when :func:`_unit_scale_is_exact`: rescaling the scale is then a
    torus action on the coefficients, which maps the locus to itself, so
    this loses only the factor mu itself and the component at mu = 0, which
    can need generators of its own. When :func:`_renaming` applies, the
    elimination and the factorisation run in the fresh symbols, and each
    factor is substituted back. It stays irreducible, being the image of an
    irreducible polynomial under a change of coordinates, unless an atom is
    squared: a factor in squared masses can split, as m_1^2 - m_2^2 does,
    and is factored again.
    """
    if scale is not None:
        coeffs = [sp.sympify(c).subs(scale, 1) for c in coeffs]
    renaming, squared = _renaming(coeffs)
    back = {fresh: c for c, fresh in renaming.items()}
    dim = len(exps[0])
    t = list(sp.symbols(f"_t1:{dim + 1}"))
    w = sp.Symbol("_w")
    f = sum(
        renaming.get(c, c) * sp.prod(ti**e for ti, e in zip(t, ek, strict=True))
        for c, ek in zip(coeffs, exps, strict=True)
    )
    num, _den = sp.fraction(sp.together(f))
    num = sp.expand(num)
    system = [num] + [sp.expand(ti * sp.diff(num, ti)) for ti in t] + [1 - w * sp.prod(t)]
    kin = sorted(num.free_symbols - set(t), key=str)
    to_eliminate = [w, *t]

    binary = _singular_binary() if backend in ("auto", "singular") else None
    if backend == "singular" and binary is None:
        raise RuntimeError("Singular backend requested but the 'Singular' binary was not found")
    if not kin:
        return [], True
    if binary is not None:
        eliminated = _eliminate_singular(
            system, to_eliminate, kin, binary, points=len(coeffs), timeout=timeout
        )
    else:
        eliminated = _eliminate_sympy(system, to_eliminate, kin)
    if not eliminated:
        return [], True

    # sp.factor_list makes each factor primitive with a positive leading coefficient in the
    # order sp.Poly gives the symbols; factoring in that order normalises these factors as
    # _factor_list normalises those of the other faces.
    order = sp.Poly(sp.Add(*kin)).gens
    common = eliminated[0]
    for g in eliminated[1:]:
        common = common.gcd(g)
    factors: dict[sp.Expr, None] = {}
    for fac, _exp in common.reorder(*order).factor_list()[1]:
        if back:
            image = sp.expand(fac.as_expr().xreplace(back))
            pieces = _factor_list(image, kinematic_syms) if squared else [_normalised(image)]
            for piece in pieces:
                factors.setdefault(piece, None)
        elif fac.free_symbols & kinematic_syms:
            factors.setdefault(fac.as_expr(), None)
    return list(factors), len(eliminated) == 1


def _face_discriminant(
    coeffs: list[sp.Expr],
    exps_ambient: list[tuple[int, ...]],
    dimension: int,
    kinematic_syms: set[sp.Symbol],
    max_face_points: int,
    scale: sp.Symbol | None = None,
    timeout: float | None = None,
) -> tuple[sp.Expr, list[sp.Expr], bool, bool] | None:
    """Return (discriminant, its kinematic factors, is_simplex, principal), or None if the
    face is skipped."""
    n_pts = len(coeffs)
    is_simplex = n_pts == dimension + 1
    if dimension == 0:
        disc = sp.together(coeffs[0])
        return disc, _factor_list(disc, kinematic_syms), True, True
    if is_simplex:
        return sp.Integer(1), [], True, True
    if n_pts > max_face_points:
        return None
    lattice = _lattice_coordinates(np.array(exps_ambient, dtype=int))
    if dimension == 1:
        disc = _univariate_discriminant(coeffs, [c[0] for c in lattice])
        return disc, _factor_list(disc, kinematic_syms), False, True
    factors, principal = _elimination_discriminant(
        coeffs, lattice, kinematic_syms, scale=scale, timeout=timeout
    )
    return sp.Mul(*factors), factors, False, principal


def _reduced(
    factors_by_face: list[list[sp.Expr]], surface_syms: set[sp.Symbol]
) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    """The distinct factors, over all faces, that involve surface_syms, and their product."""
    seen: dict[sp.Expr, None] = {}
    for factors in factors_by_face:
        for fac in factors:
            if fac.free_symbols & surface_syms:
                seen.setdefault(fac, None)
    surfaces = tuple(seen)
    return (sp.Mul(*surfaces) if surfaces else sp.Integer(1)), surfaces


# --- public API --------------------------------------------------------------


def landau_analysis_from_polynomial(
    g_poly: sp.Expr,
    lp_parameters: list[sp.Symbol],
    *,
    max_face_points: int = 12,
    scale: sp.Symbol | None = None,
    timeout: float | None = None,
) -> LandauAnalysis:
    """Reduced principal A-determinant of a polynomial in the given variables.

    Parameters
    ----------
    g_poly
        The polynomial, typically G = U + F.
    lp_parameters
        Its variables; every other symbol is treated as kinematic.
    max_face_points
        Faces of dimension two or more with more monomials than this are
        not eliminated and are reported in ``skipped_faces``.
    scale
        The energy scale mu, if ``g_poly`` carries one. When every
        coefficient of ``g_poly`` is a power of mu times an expression free
        of it, with the power an affine function of the exponents, as for
        G = U + F whose kinematics do not contain mu, the faces of
        dimension two or more that are not simplices are eliminated at
        mu = 1. That finds the same factors but mu itself and those that
        came only from mu = 0, where a face could need a second generator,
        so these discriminants carry no mu. The test is made once, on all
        of ``g_poly``, and is sufficient, not necessary; when it fails, as
        when a momentum product is set to mu^2, and without ``scale``, mu
        is eliminated as a variable. A factor that is mu alone is not a
        kinematic singularity and is left out of ``landau_surfaces`` and
        ``principal_a_determinant``. Vertex and edge discriminants keep mu
        where it occurs, for instance a vertex coefficient m_1^2 / mu^2 is
        still that coefficient.
    timeout
        The most seconds to give Singular for each face it eliminates, at
        most 2,000,000; None, the default, sets no limit. The SymPy
        fallback is not limited.

    Raises
    ------
    ValidationError
        If ``timeout`` is neither None nor a number greater than 0 and at
        most 2,000,000.
    ComputationError
        If Singular fails on a face, runs past ``timeout``, or prints
        output that is not an elimination ideal.
    """
    _check_timeout(timeout)
    g_poly = sp.expand(g_poly)
    if g_poly == 0:
        return LandauAnalysis((), sp.Integer(1), ())
    support = extract_monomial_support(g_poly, lp_parameters)
    if not support:
        return LandauAnalysis((), sp.Integer(1), ())
    kinematic_syms: set[sp.Symbol] = g_poly.free_symbols - set(lp_parameters)
    surface_syms = kinematic_syms - ({scale} if scale is not None else set())
    unit_scale = scale if scale is not None and _unit_scale_is_exact(support, scale) else None
    exps = np.array([list(e) for e, _ in support], dtype=int)

    faces: list[FaceDiscriminant] = []
    factors_by_face: list[list[sp.Expr]] = []
    skipped: list[tuple[tuple[int, ...], ...]] = []
    for dimension, idx in _faces(exps):
        face_exps = [tuple(int(x) for x in exps[i]) for i in idx]
        face_coeffs = [support[i][1] for i in idx]
        result = _face_discriminant(
            face_coeffs, face_exps, dimension, kinematic_syms, max_face_points, unit_scale, timeout
        )
        if result is None:
            skipped.append(tuple(face_exps))
            continue
        disc, factors, is_simplex, principal = result
        if not (disc.free_symbols & kinematic_syms):
            disc = sp.Integer(1)
        factors_by_face.append(factors)
        faces.append(
            FaceDiscriminant(
                dimension=dimension,
                exponents=tuple(face_exps),
                coefficients=tuple(face_coeffs),
                discriminant=disc,
                is_simplex=is_simplex,
                principal=principal,
            )
        )

    e_a, surfaces = _reduced(factors_by_face, surface_syms)
    return LandauAnalysis(tuple(faces), e_a, surfaces, tuple(skipped))


def landau_analysis(
    integral: FeynmanIntegral, *, max_face_points: int = 12, timeout: float | None = None
) -> LandauAnalysis:
    """Reduced principal A-determinant of G = U + F for a Feynman integral.

    See the module docstring for what the faces contribute and for the
    caveats on interpreting the factors as Landau singularities, and
    :func:`landau_analysis_from_polynomial` for the arguments and errors.
    """
    sym = integral.symanzik
    return landau_analysis_from_polynomial(
        sym.g,
        list(sym.lp_parameters),
        max_face_points=max_face_points,
        scale=integral.graph.energy_scale,
        timeout=timeout,
    )


# --- one-loop closed form ----------------------------------------------------


def _one_loop_cycle(
    integral: FeynmanIntegral,
) -> tuple[list[Edge], list[list[int]], list[tuple[Edge, list[int]]]]:
    """The cycle of a one-loop graph and the trees attached to it.

    Returns the internal edges of the cycle in cycle order; for each vertex
    of the cycle, the legs at it or on the tree attached to it there; and
    each bridge, an internal edge on no cycle, with the legs on its side away
    from the cycle, in internal-edge order. Removing a vertex on a single
    internal edge, a leaf, and carrying its legs to its neighbour until no
    leaf is left leaves the cycle; the edges removed are the bridges. A
    self-loop counts twice at its vertex, which is therefore never a leaf.

    Raises
    ------
    ValueError
        If the graph is not connected, or the edges left are not one cycle.
    """
    graph = integral.graph
    internal = graph.get_internal_edges()
    n_int = graph.internal_vertices
    legs: dict[int, list[int]] = {v: [] for v in range(1, n_int + 1)}
    for ext in graph.get_external_edges():
        legs[ext.v1].append(ext.v2 - n_int)
    incident: dict[int, list[int]] = {v: [] for v in legs}
    for k, e in enumerate(internal):
        incident[e.v1].append(k)
        incident[e.v2].append(k)

    def other_end(k: int, v: int) -> int:
        return internal[k].v2 if internal[k].v1 == v else internal[k].v1

    # Removing leaves needs a connected graph: a separate tree would end in a vertex without edges.
    reached, frontier = {1}, [1]
    while frontier:
        v = frontier.pop()
        for w in (other_end(k, v) for k in incident[v]):
            if w not in reached:
                reached.add(w)
                frontier.append(w)
    if len(reached) != n_int:
        raise ValueError("The closed form applies to connected one-loop graphs only")

    bridges: list[tuple[int, list[int]]] = []
    leaves = [v for v, ks in incident.items() if len(ks) == 1]
    while leaves:
        v = leaves.pop()
        (k,) = incident.pop(v)
        w = other_end(k, v)
        bridges.append((k, list(legs[v])))
        legs[w].extend(legs.pop(v))
        incident[w].remove(k)
        if len(incident[w]) == 1:
            leaves.append(w)
    cycle = [k for k, e in enumerate(internal) if e.v1 in incident and e.v2 in incident]
    start = internal[cycle[0]].v1
    order_edges: list[int] = []
    order_vertices = [start]
    v, previous = start, -1
    while True:
        k = next(k for k in incident[v] if k != previous)
        order_edges.append(k)
        previous = k
        v = other_end(k, v)
        if v == start:
            break
        order_vertices.append(v)
    if len(order_edges) != len(cycle):
        raise ValueError("The closed form applies to connected one-loop graphs only")
    return (
        [internal[k] for k in order_edges],
        [legs[v] for v in order_vertices],
        [(internal[k], far) for k, far in sorted(bridges)],
    )


def _momentum_squared(integral: FeynmanIntegral) -> Callable[[list[int]], sp.Expr]:
    """The square of the total momentum of a set of legs, from the momentum products.

    The momenta sum to zero, so the momentum q_S of the legs in S is minus
    that of the others, and q_S^2 = -sum over a in S and c not in S of
    p_a . p_c. F is written in the same products, so the two agree when the
    products have been changed, as by setting p_3^2 = 0 in them.
    """
    products = integral.momentum_products
    n_legs = integral.graph.external_legs

    def q_squared(legs: list[int]) -> sp.Expr:
        inside = set(legs)
        return sp.expand(
            -sum(
                products.get((min(a, c), max(a, c)), sp.Integer(0))
                for a in legs
                for c in range(1, n_legs + 1)
                if c not in inside
            )
        )

    return q_squared


def one_loop_principal_a_determinant(integral: FeynmanIntegral) -> sp.Expr:
    """Reduced principal A-determinant of a one-loop integral in closed form.

    The product of :func:`one_loop_landau_surfaces`; see there for the
    construction and references.
    """
    surfaces = one_loop_landau_surfaces(integral)
    return sp.Mul(*surfaces) if surfaces else sp.Integer(1)


def _factor_list_reproducibly(expr: sp.Expr, kinematic_syms: set[sp.Symbol]) -> list[sp.Expr]:
    """:func:`_factor_list`, with SymPy's random choices drawn from a fixed seed.

    SymPy factors a multivariate polynomial by Wang's algorithm, which draws
    evaluation points from a generator the whole process shares, so the time
    it takes depends on what ran before, though the factors do not. After some
    histories a Gram minor of the massless hexagon, 130 terms that factor in a
    fifth of a second, had not factored after seven minutes. The generator is
    seeded for each call and its state restored afterwards.
    """
    state = sympy_random.rng.getstate()
    sympy_random.rng.seed(0)
    try:
        return _factor_list(expr, kinematic_syms)
    finally:
        sympy_random.rng.setstate(state)


def _modified_cayley_matrix(integral: FeynmanIntegral) -> sp.Matrix:
    """The modified Cayley matrix of the cycle of a one-loop integral.

    Y_00 = 0, Y_0i = 1, Y_ii = 2 m_i^2 and Y_ij = m_i^2 + m_j^2 - q_ij^2, with
    q_ij the momentum between propagators i and j of the cycle, in cycle order.
    """
    edges, legs_at, _bridges = _one_loop_cycle(integral)
    n = len(edges)
    q_squared = _momentum_squared(integral)
    masses = [e.get_mass() ** 2 for e in edges]
    y = sp.zeros(n + 1, n + 1)
    for i in range(1, n + 1):
        y[0, i] = y[i, 0] = 1
        y[i, i] = 2 * masses[i - 1]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # Cutting edges i and j isolates the cycle vertices strictly after edge i up to edge j.
            legs = [leg for k in range(i, j) for leg in legs_at[k]]
            y[i, j] = y[j, i] = masses[i - 1] + masses[j - 1] - q_squared(legs)
    return y


def one_loop_landau_surfaces_by_type(
    integral: FeynmanIntegral,
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """Irreducible factors of the one-loop closed form, split by type.

    Dlapa, Helmer, Papathanasiou and Tellander (2023, eq. 1LoopEA): with the
    modified Cayley matrix Y of size (n+1), Y_00 = 0, Y_0i = 1,
    Y_ii = 2 m_i^2 and Y_ij = m_i^2 + m_j^2 - q_ij^2 where q_ij is the
    momentum flowing between propagators i and j, the reduced principal
    A-determinant is the product of the principal minors of Y that are not
    identically zero. Minors containing index 0 are Gram determinants
    (second-type singularities); minors not containing index 0 are Cayley
    determinants (first type).

    For a graph with bridges the matrix is that of the cycle, with the legs
    of each tree attached to the cycle at the vertex where the tree meets
    it; the poles of the bridges are :func:`one_loop_bridge_poles`.

    Returns
    -------
    tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]
        ``(first_type, second_type)``, each in first-encountered order over
        the principal minors, smallest subsets first. A factor that arises
        from minors of both kinds is listed in both tuples;
        :func:`one_loop_landau_surfaces` merges them and lists it under
        first type only.

    Raises
    ------
    ValueError
        If the integral has no loop or more than one, or its graph is not connected.
    """
    if integral.loop_count != 1:
        raise ValueError("The closed form applies to one-loop integrals only")
    y = _modified_cayley_matrix(integral)
    n = y.rows - 1
    kinematic_syms = y.free_symbols
    # The minors are taken with a symbol for each distinct entry that is not constant and
    # factored before the entries are substituted back. When the entries are linearly
    # independent forms in symbols alone (see _renaming), each factor stays irreducible.
    entries = [sp.expand(y[i, j]) for i in range(n + 1) for j in range(i, n + 1)]
    names, squared = _renaming(entries)
    exact = bool(names) and not squared
    if not names:
        distinct = dict.fromkeys(entry for entry in entries if entry.free_symbols)
        names = {entry: sp.Dummy(f"y{k}") for k, entry in enumerate(distinct, start=1)}
    back = {symbol: entry for entry, symbol in names.items()}
    # Minors in the polynomial ring of the symbols, much faster than Matrix.det.
    symbolic = DomainMatrix.from_Matrix(
        y.applyfunc(lambda entry: names.get(sp.expand(entry), entry))
    )
    first: dict[sp.Expr, None] = {}
    second: dict[sp.Expr, None] = {}
    for size in range(1, n + 2):
        for subset in combinations(range(n + 1), size):
            minor = symbolic.domain.to_sympy(symbolic.extract(list(subset), list(subset)).det())
            if minor == 0:
                continue
            factors = _factor_list_reproducibly(minor, set(back))
            images = [sp.expand(f.xreplace(back)) for f in factors]
            if any(image == 0 for image in images):
                continue
            target = second if 0 in subset else first
            for image in images:
                if exact:
                    pieces = [_normalised(image)]
                else:
                    pieces = _factor_list_reproducibly(image, kinematic_syms)
                for fac in pieces:
                    if fac.free_symbols & kinematic_syms:
                        target.setdefault(fac, None)
    return tuple(first), tuple(second)


def one_loop_bridge_poles(integral: FeynmanIntegral) -> tuple[sp.Expr, ...]:
    """Irreducible factors of the poles of the bridges of a one-loop graph.

    A bridge b, an internal edge on no cycle, carries the momentum q_b of the
    legs on its side away from the cycle, so the integral is that of the
    cycle, with the legs of each tree attached to the cycle moved to the
    vertex where the tree meets it, times the propagator
    1/(m_b^2 - q_b^2)^nu_b of each bridge. Returns the distinct factors of the
    m_b^2 - q_b^2, bridges in internal-edge order, which the faces of the
    Newton polytope give as well; () for a graph without bridges.

    Raises
    ------
    ValueError
        If the integral has no loop or more than one, or its graph is not connected.
    """
    if integral.loop_count != 1:
        raise ValueError("The closed form applies to one-loop integrals only")
    _edges, _legs_at, bridges = _one_loop_cycle(integral)
    q_squared = _momentum_squared(integral)
    masses = [sp.sympify(e.get_mass()) for e in integral.graph.get_internal_edges()]
    kinematic_syms = set().union(*(m.free_symbols for m in masses)) | set().union(
        *(sp.sympify(v).free_symbols for v in integral.momentum_products.values())
    )
    poles: dict[sp.Expr, None] = {}
    for edge, legs in bridges:
        pole = sp.expand(edge.get_mass() ** 2 - q_squared(legs))
        for fac in _factor_list(pole, kinematic_syms):
            poles.setdefault(fac, None)
    return tuple(poles)


def one_loop_landau_surfaces(integral: FeynmanIntegral) -> tuple[sp.Expr, ...]:
    """Irreducible factors of the one-loop principal A-determinant in closed form.

    The union of the first and second-type factors of
    :func:`one_loop_landau_surfaces_by_type`, first type first and with
    second-type factors already listed under first type dropped, followed by
    the bridge poles of :func:`one_loop_bridge_poles` not already listed.

    Raises
    ------
    ValueError
        If the integral has no loop or more than one, or its graph is not connected.
    """
    first, second = one_loop_landau_surfaces_by_type(integral)
    surfaces = first + tuple(s for s in second if s not in first)
    return surfaces + tuple(p for p in one_loop_bridge_poles(integral) if p not in surfaces)
