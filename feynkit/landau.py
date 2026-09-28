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
  t_i d f_tau / d t_i = 0} in the torus, computed with a Gröbner basis.
  Faces with more points than ``max_face_points`` are skipped and listed
  in ``LandauAnalysis.skipped_faces``.

The factors are candidate codimension-one singular loci on all sheets of the
integral. Membership is necessary for a singularity, not sufficient, and
the list is neither complete for physical sheets nor guaranteed exhaustive
(Fevola, Mizera and Telen 2023, section 2).

For one-loop graphs, :func:`one_loop_principal_a_determinant` gives the
closed form of Dlapa, Helmer, Papathanasiou and Tellander (2023, eq.
1LoopEA): the product of the principal minors of the modified Cayley
matrix. It is used as an independent check of the face computation.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

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
        ideal needed more than one generator; the product of their
        kinematic factors is then reported.
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
        built from the energy scale alone is not a singular surface (see
        ``landau_analysis_from_polynomial``'s ``scale`` argument) and is
        excluded, even though the face discriminants themselves keep it.
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


# --- public API --------------------------------------------------------------


def _singular_binary() -> str | None:
    """Path to the Singular binary, or None if it is not installed."""
    import shutil

    return shutil.which("Singular")


def _eliminate_sympy(
    system: list[sp.Expr], to_eliminate: list[sp.Symbol], kin: list[sp.Symbol]
) -> list[sp.Poly]:
    basis = sp.groebner(system, *to_eliminate, *kin, order="lex")
    return [sp.Poly(g, *kin) for g in basis.exprs if not (g.free_symbols & set(to_eliminate))]


# A term of Singular's output, and a factor of a term: a coefficient or a power of a variable.
_SINGULAR_TERM = re.compile(r"([+-]?)([^+-]+)")
_SINGULAR_FACTOR = re.compile(r"(\d+)(?:/(\d+))?|v(\d+)(?:\^(\d+))?")


def _read_singular_polynomial(
    line: str, first: int, n_vars: int
) -> dict[tuple[int, ...], Fraction] | None:
    """The terms of a polynomial Singular printed in v_first, ..., v_{first + n_vars - 1}.

    Returns None when the line is anything else, such as an error message
    or a term in another variable.
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
        reason = result.stderr.strip()[:200] or f"exit status {result.returncode}"
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
                f"from {size}, at {text.strip()[:60]!r}"
            ) from None
        polynomial = {key: sp.Rational(c.numerator, c.denominator) for key, c in terms.items() if c}
        if polynomial:
            out.append(sp.Poly.from_dict(polynomial, *kin))
    return out


def _elimination_discriminant(
    coeffs: list[sp.Expr],
    exps: list[tuple[int, ...]],
    kinematic_syms: set[sp.Symbol],
    backend: str = "auto",
    *,
    timeout: float | None = None,
) -> tuple[sp.Expr, bool]:
    """Kinematic locus where f = sum c_k t^{e_k} has a singular point in the torus.

    Returns (product of distinct kinematic factors, principal) where principal
    is True when the elimination ideal had at most one generator. Uses
    Singular when installed and ``backend`` is "auto" or "singular", else
    SymPy; ``timeout`` limits each Singular run, in seconds.
    """
    dim = len(exps[0])
    t = list(sp.symbols(f"_t1:{dim + 1}"))
    w = sp.Symbol("_w")
    f = sum(
        c * sp.prod(ti**e for ti, e in zip(t, ek, strict=True))
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
        return sp.Integer(1), True
    if binary is not None:
        eliminated = _eliminate_singular(
            system, to_eliminate, kin, binary, points=len(coeffs), timeout=timeout
        )
    else:
        eliminated = _eliminate_sympy(system, to_eliminate, kin)
    if not eliminated:
        return sp.Integer(1), True

    # sp.factor_list makes each factor primitive with a positive leading coefficient in the
    # order sp.Poly gives the symbols; factoring in that order normalises these factors as
    # _factor_list normalises those of the other faces.
    order = sp.Poly(sp.Add(*kin)).gens
    factors: dict[sp.Expr, None] = {}
    for g in eliminated:
        for fac, _exp in g.reorder(*order).factor_list()[1]:
            if fac.free_symbols & kinematic_syms:
                factors.setdefault(fac.as_expr(), None)
    return (sp.Mul(*factors) if factors else sp.Integer(1)), len(eliminated) == 1


def _face_discriminant(
    coeffs: list[sp.Expr],
    exps_ambient: list[tuple[int, ...]],
    dimension: int,
    kinematic_syms: set[sp.Symbol],
    max_face_points: int,
    timeout: float | None = None,
) -> tuple[sp.Expr, bool, bool] | None:
    """Return (discriminant, is_simplex, principal), or None if the face is skipped."""
    n_pts = len(coeffs)
    is_simplex = n_pts == dimension + 1
    if dimension == 0:
        return sp.together(coeffs[0]), True, True
    if is_simplex:
        return sp.Integer(1), True, True
    if n_pts > max_face_points:
        return None
    lattice = _lattice_coordinates(np.array(exps_ambient, dtype=int))
    if dimension == 1:
        t_exps = [c[0] for c in lattice]
        return _univariate_discriminant(coeffs, t_exps), False, True
    disc, principal = _elimination_discriminant(coeffs, lattice, kinematic_syms, timeout=timeout)
    return disc, False, principal


def _reduced(
    factors_by_face: list[sp.Expr], kinematic_syms: set[sp.Symbol]
) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    seen: dict[sp.Expr, None] = {}
    for disc in factors_by_face:
        for fac in _factor_list(disc, kinematic_syms):
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
        The energy scale mu, if ``g_poly`` carries one. Every coefficient
        of F is homogeneous of the same negative degree in mu, so it can
        survive factorisation as a standalone factor of a face
        discriminant; that factor is not a kinematic singularity and is
        left out of ``landau_surfaces`` and ``principal_a_determinant``.
        Face discriminants are unaffected and keep mu where it occurs, for
        instance a vertex coefficient m_1^2 / mu^2 is still that
        coefficient.
    timeout
        The most seconds to give Singular for each face it eliminates; None,
        the default, sets no limit. The SymPy fallback is not limited.

    Raises
    ------
    ValidationError
        If ``timeout`` is not None and not positive.
    ComputationError
        If Singular fails on a face, runs past ``timeout``, or prints
        output that is not an elimination ideal.
    """
    if timeout is not None and not timeout > 0:
        raise ValidationError(f"timeout must be positive; got {timeout}")
    g_poly = sp.expand(g_poly)
    if g_poly == 0:
        return LandauAnalysis((), sp.Integer(1), ())
    support = extract_monomial_support(g_poly, lp_parameters)
    if not support:
        return LandauAnalysis((), sp.Integer(1), ())
    kinematic_syms: set[sp.Symbol] = g_poly.free_symbols - set(lp_parameters)
    surface_syms = kinematic_syms - ({scale} if scale is not None else set())
    exps = np.array([list(e) for e, _ in support], dtype=int)

    faces: list[FaceDiscriminant] = []
    skipped: list[tuple[tuple[int, ...], ...]] = []
    for dimension, idx in _faces(exps):
        face_exps = [tuple(int(x) for x in exps[i]) for i in idx]
        face_coeffs = [support[i][1] for i in idx]
        result = _face_discriminant(
            face_coeffs, face_exps, dimension, kinematic_syms, max_face_points, timeout
        )
        if result is None:
            skipped.append(tuple(face_exps))
            continue
        disc, is_simplex, principal = result
        if not (disc.free_symbols & kinematic_syms):
            disc = sp.Integer(1)
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

    e_a, surfaces = _reduced([f.discriminant for f in faces], surface_syms)
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


def _one_loop_cycle(integral: FeynmanIntegral) -> tuple[list[Edge], list[list[int]]]:
    """Internal edges in cycle order and, for each cycle vertex, the legs attached to it."""
    graph = integral.graph
    internal = graph.get_internal_edges()
    n_int = graph.internal_vertices
    adjacency: dict[int, list[tuple[int, Edge]]] = {v: [] for v in range(1, n_int + 1)}
    for e in internal:
        adjacency[e.v1].append((e.v2, e))
        adjacency[e.v2].append((e.v1, e))
    if len(internal) == 1:  # tadpole: a single self-loop
        return internal, [[ext.v2 - n_int for ext in graph.get_external_edges()]]
    start = internal[0].v1
    order_edges: list[Edge] = []
    order_vertices: list[int] = [start]
    prev_edge: Edge | None = None
    v = start
    while True:
        nxt = next((w, e) for (w, e) in adjacency[v] if e is not prev_edge)
        w, e = nxt
        order_edges.append(e)
        prev_edge = e
        v = w
        if v == start:
            break
        order_vertices.append(v)
    legs_at: dict[int, list[int]] = {v: [] for v in order_vertices}
    for ext in graph.get_external_edges():
        legs_at[ext.v1].append(ext.v2 - n_int)
    return order_edges, [legs_at[v] for v in order_vertices]


def one_loop_principal_a_determinant(integral: FeynmanIntegral) -> sp.Expr:
    """Reduced principal A-determinant of a one-loop integral in closed form.

    The product of :func:`one_loop_landau_surfaces`; see there for the
    construction and references.
    """
    surfaces = one_loop_landau_surfaces(integral)
    return sp.Mul(*surfaces) if surfaces else sp.Integer(1)


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
        If the integral has more than one loop.
    """
    if integral.loop_count != 1:
        raise ValueError("The closed form applies to one-loop integrals only")
    edges, legs_at = _one_loop_cycle(integral)
    n = len(edges)
    products = integral.momentum_products
    n_legs = integral.graph.external_legs
    inv = None
    if n_legs >= 2:
        from .kinematics.mandelstam import standard_invariants

        inv = standard_invariants(n_legs)

    def dot(a: int, b: int) -> sp.Expr:
        if a == b:
            if inv is not None and integral.use_mandelstam:
                return inv.external_masses[a - 1]
            return -sum(
                products.get((min(a, c), max(a, c)), sp.Integer(0))
                for c in range(1, n_legs + 1)
                if c != a
            )
        return products.get((min(a, b), max(a, b)), sp.Integer(0))

    def q_squared(legs: list[int]) -> sp.Expr:
        return sp.expand(sum(dot(a, b) for a in legs for b in legs))

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

    kinematic_syms = set().union(*(sp.sympify(x).free_symbols for x in masses)) | set().union(
        *(v.free_symbols for v in products.values())
    )
    first: dict[sp.Expr, None] = {}
    second: dict[sp.Expr, None] = {}
    for size in range(1, n + 2):
        for subset in combinations(range(n + 1), size):
            minor = sp.expand(y.extract(list(subset), list(subset)).det())
            if minor == 0:
                continue
            target = second if 0 in subset else first
            for fac in _factor_list(minor, kinematic_syms):
                target.setdefault(fac, None)
    return tuple(first), tuple(second)


def one_loop_landau_surfaces(integral: FeynmanIntegral) -> tuple[sp.Expr, ...]:
    """Irreducible factors of the one-loop principal A-determinant in closed form.

    The union of the first and second-type factors of
    :func:`one_loop_landau_surfaces_by_type`, first type first and with
    second-type factors already listed under first type dropped.

    Raises
    ------
    ValueError
        If the integral has more than one loop.
    """
    first, second = one_loop_landau_surfaces_by_type(integral)
    return first + tuple(s for s in second if s not in first)
