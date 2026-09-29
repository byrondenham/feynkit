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
  codimension-one part of their zero set. A component that arises only as
  a limit under special kinematics can then be missing: at p_1^2 = 0 the
  massive triangle loses p_2^2 - p_3^2, since the locus of its top face is
  p_2^2 = p_3^2 = 0, of codimension two, while the one-loop closed form
  keeps it. Faces with more points than ``max_face_points`` are skipped
  and listed in ``LandauAnalysis.skipped_faces``.

The factors are candidate codimension-one singular loci on all sheets of the
integral. Membership is necessary for a singularity, not sufficient, and
the list is neither complete for physical sheets nor guaranteed exhaustive
(Fevola, Mizera and Telen 2023, section 2).

For one-loop graphs, :func:`one_loop_principal_a_determinant` gives the
closed form of Dlapa, Helmer, Papathanasiou and Tellander (2023, eq.
1LoopEA): the product of the principal minors of the modified Cayley
matrix of the cycle and, for a graph with bridges, the poles of the
bridges' propagators. It is used as an independent check of the face
computation for generic kinematics.
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
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.rings import PolyElement, PolyRing

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


def _normalised_poly(irreducible: PolyElement) -> sp.Expr:
    """:func:`_normalised` of an element of a ring whose symbols are in SymPy's order.

    The leading term in all the ring's symbols is the leading term in those
    that occur, so the sign is the one :func:`_normalised` gives.
    """
    _, poly = irreducible.clear_denoms()
    _, poly = poly.primitive()
    return (-poly if poly.LC < 0 else poly).as_expr()


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


def _factor_order(item: tuple[sp.Poly, int]) -> tuple[object, ...]:
    """The order of sp.factor_list: by the length of the dense representation, the
    number of generators, the multiplicity, the domain and the representation."""
    poly, multiplicity = item
    rep = poly.rep.to_list()
    return (len(rep), len(poly.gens), multiplicity, str(poly.domain), rep)


def _factor_bases(expr: sp.Expr) -> list[tuple[sp.Poly, int]] | None:
    """The polynomials sp.factor_list factors one at a time in :func:`_factor_list`, with
    their exponents: the factors of the numerator that are not numbers, as SymPy's
    ``together`` writes it. None when one is not a polynomial in symbols over the
    integers or the rationals."""
    num, _den = sp.fraction(sp.together(expr))
    numerator, _denominator = sp.together(num).as_numer_denom()
    bases: list[tuple[sp.Poly, int]] = []
    for arg in sp.Mul.make_args(numerator):
        if arg.is_Number:
            continue
        base, exp = arg.args if arg.is_Pow else (arg, sp.Integer(1))
        if hasattr(arg, "_eval_factor") or base.is_Number or not (exp.is_Integer and exp > 0):
            return None
        try:
            poly = sp.Poly(base)
        except BasePolynomialError:
            return None
        if poly.domain not in (sp.ZZ, sp.QQ) or not all(g.is_Symbol for g in poly.gens):
            return None
        bases.append((poly, int(exp)))
    return bases


def _ring_bases(poly: PolyElement) -> list[tuple[sp.Poly, int]]:
    """:func:`_factor_bases` of poly.as_expr(), read off the polynomial.

    For a polynomial, ``together`` pulls out the numbers and the gcd of the
    monomials, so the bases are the symbols of that gcd, in the order Mul
    gives them, and then the rest, a sum, over the integers. The rest may
    differ from SymPy's by a constant, which does not change its factors.
    Reading them off spares ``together`` on polynomials of thousands of
    terms.
    """
    ring = poly.ring
    monomials = list(poly.keys())
    if not monomials:
        return []
    gcd = [min(m[k] for m in monomials) for k in range(ring.ngens)]
    powers = sp.Mul(*(g**e for g, e in zip(ring.symbols, gcd, strict=True) if e))
    bases = [
        (sp.Poly(base), int(exp))
        for base, exp in (arg.as_base_exp() for arg in sp.Mul.make_args(powers))
        if not base.is_Number
    ]
    rest = {tuple(a - b for a, b in zip(m, gcd, strict=True)): c for m, c in poly.items()}
    if len(rest) > 1:
        used = [k for k in range(ring.ngens) if any(m[k] for m in rest)]
        terms = {tuple(m[k] for k in used): ring.domain.to_sympy(c) for m, c in rest.items()}
        gens = [ring.symbols[k] for k in used]
        _, integral = sp.Poly.from_dict(terms, *gens, domain=sp.QQ).clear_denoms(convert=True)
        bases.append((integral, 1))
    return bases


def _factorize_singular(
    polys: list[sp.Poly], binary: str
) -> dict[sp.Poly, list[tuple[sp.Poly, int]]] | None:
    """Poly.factor_list without the constant, for each polynomial, by Singular's
    ``factorize`` in one run.

    Each factor is written in the polynomial's generators and domain,
    primitive over the integers with a positive leading coefficient, and
    the factors are in SymPy's order, so the result is the one SymPy gives.
    None if Singular fails, prints anything else, or gives factors whose
    product, with the multiplicities, is not the polynomial up to a constant.
    """
    gens = list(dict.fromkeys(g for poly in polys for g in poly.gens))
    index = {g: k for k, g in enumerate(gens)}

    def render(poly: sp.Poly) -> str:
        _, integral = poly.clear_denoms(convert=True)
        return "+".join(
            "*".join(
                [str(c)] + [f"v{index[g]}^{e}" for g, e in zip(poly.gens, powers, strict=True) if e]
            )
            for powers, c in integral.terms()
        )

    script = "\n".join(
        [
            f"ring r = 0, ({','.join(f'v{k}' for k in range(len(gens)))}), dp;",
            "proc show(poly p) { list L = factorize(p); int i;",
            '  print("@" + string(ncols(L[1])));',
            "  for (i = 1; i <= ncols(L[1]); i++) {",
            '    print(string(L[2][i]) + ":" + string(L[1][i]));',
            "  }",
            "}",
            *(f"show({render(poly)});" for poly in polys),
            "quit;\n",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "factorize.sing"
        path.write_text(script)
        try:
            result = subprocess.run(
                [binary, "-q", "--no-warn", str(path)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    if result.returncode != 0:
        return None
    lines = [line.replace(" ", "") for line in result.stdout.splitlines() if line.strip()]
    out: dict[sp.Poly, list[tuple[sp.Poly, int]]] = {}
    position = 0
    for poly in polys:
        head = lines[position] if position < len(lines) else ""
        if not (head.startswith("@") and head[1:].isdigit()):
            return None
        count = int(head[1:])
        body = lines[position + 1 : position + 1 + count]
        position += 1 + count
        if len(body) != count:
            return None
        places = [index[g] for g in poly.gens]
        factors: dict[sp.Poly, int] = {}
        for line in body:
            multiplicity, colon, text = line.partition(":")
            terms = _read_singular_polynomial(text, 0, len(gens)) if colon else None
            if terms is None or not multiplicity.isdigit():
                return None
            local = {}
            for exponents, c in terms.items():
                if sum(exponents) != sum(exponents[k] for k in places):
                    return None
                if c:
                    local[tuple(exponents[k] for k in places)] = sp.Rational(
                        c.numerator, c.denominator
                    )
            if not any(sum(powers) for powers in local):
                continue
            factor = sp.Poly.from_dict(local, *poly.gens, domain=sp.QQ)
            _, factor = factor.clear_denoms(convert=True)
            _, factor = factor.primitive()
            factor = (-factor if factor.LC() < 0 else factor).set_domain(poly.domain)
            factors[factor] = factors.get(factor, 0) + int(multiplicity)
        product = sp.Poly(1, *poly.gens, domain=sp.QQ)
        for factor, k in factors.items():
            product *= factor.set_domain(sp.QQ) ** k
        if product.monic() != poly.set_domain(sp.QQ).monic():
            return None
        out[poly] = sorted(factors.items(), key=_factor_order)
    if position != len(lines):
        return None
    return out


def _factor_lists(exprs: list[sp.Expr], kinematic_syms: set[sp.Symbol]) -> list[list[sp.Expr]]:
    """:func:`_factor_list` of each expression, factored by Singular in one run when it is
    installed.

    SymPy factors a multivariate polynomial by Wang's algorithm, which draws
    evaluation points from a generator the whole process shares. Its time
    depends on what ran before, though its factors do not, and some points
    make it run for over ten minutes on a polynomial it otherwise factors in
    a fraction of a second, as on a Cayley minor of the massive box without
    Mandelstam variables. Singular's ``factorize`` does not depend on that
    state. Factorisation over the rationals is unique up to units, and the
    factors are written and ordered as sp.factor_list writes and orders
    them, so the result is the same either way. An expression that is not a
    polynomial over the integers or the rationals goes to SymPy, and so does
    every expression when Singular fails. An element of a polynomial ring
    over the rationals, whose symbols are in SymPy's order, stands for its
    expression.
    """
    binary = _singular_binary()
    plans = [
        _ring_bases(expr) if isinstance(expr, PolyElement) else _factor_bases(expr)
        for expr in (exprs if binary is not None else [])
    ]
    distinct = list(dict.fromkeys(poly for plan in plans if plan for poly, _ in plan))
    factored = _factorize_singular(distinct, binary) if binary is not None and distinct else {}
    if factored is None:
        plans, factored = [], {}
    result: list[list[sp.Expr]] = []
    for k, expr in enumerate(exprs):
        plan = plans[k] if plans else None
        if plan is None:
            if isinstance(expr, PolyElement):
                expr = expr.as_expr()
            result.append(_factor_list(expr, kinematic_syms))
            continue
        merged: dict[sp.Poly, int] = {}
        for base, exp in plan:
            for factor, multiplicity in factored[base]:
                merged[factor] = merged.get(factor, 0) + multiplicity * exp
        ordered = [factor.as_expr() for factor, _ in sorted(merged.items(), key=_factor_order)]
        result.append([factor for factor in ordered if factor.free_symbols & kinematic_syms])
    return result


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
    them. A component that arises only as a limit under special kinematics
    can then be missing, as p_2^2 - p_3^2 is for the massive triangle at
    p_1^2 = 0, whose top face has the locus p_2^2 = p_3^2 = 0. Eliminates
    and factors with Singular when it is installed and ``backend`` is
    "auto" or "singular", else with SymPy; ``timeout`` limits each
    elimination by Singular, in seconds.

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
    common = common.reorder(*order)
    factored = None
    if binary is not None and not common.is_ground:
        factored = _factorize_singular([common], binary)
    pairs = factored[common] if factored is not None else common.factor_list()[1]
    factors: dict[sp.Expr, None] = {}
    if back:
        images = [sp.expand(fac.as_expr().xreplace(back)) for fac, _exp in pairs]
        if not squared:
            pieces_of = [[_normalised(image)] for image in images]
        elif binary is not None:
            pieces_of = _factor_lists(images, kinematic_syms)
        else:
            pieces_of = [_factor_list(image, kinematic_syms) for image in images]
        for pieces in pieces_of:
            for piece in pieces:
                factors.setdefault(piece, None)
    else:
        for fac, _exp in pairs:
            if fac.free_symbols & kinematic_syms:
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
) -> tuple[sp.Expr, list[sp.Expr] | None, bool, bool] | None:
    """Return (discriminant, its kinematic factors, is_simplex, principal), or None if the
    face is skipped. The factors of a vertex or an edge are None: they are those of the
    discriminant, which the caller factors with the others."""
    n_pts = len(coeffs)
    is_simplex = n_pts == dimension + 1
    if dimension == 0:
        return sp.together(coeffs[0]), None, True, True
    if is_simplex:
        return sp.Integer(1), [], True, True
    if n_pts > max_face_points:
        return None
    lattice = _lattice_coordinates(np.array(exps_ambient, dtype=int))
    if dimension == 1:
        return _univariate_discriminant(coeffs, [c[0] for c in lattice]), None, False, True
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
    # The discriminants of the vertices and edges, factored together at the end.
    pending: list[tuple[int, sp.Expr]] = []
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
        if factors is None:
            pending.append((len(factors_by_face), disc))
        if not (disc.free_symbols & kinematic_syms):
            disc = sp.Integer(1)
        factors_by_face.append(factors or [])
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

    discriminants = [disc for _, disc in pending]
    for (k, _), factors in zip(pending, _factor_lists(discriminants, kinematic_syms), strict=True):
        factors_by_face[k] = factors
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
    p_a . p_c. Each product is read as F reads it, under (a, c) or (c, a),
    so the two agree when the products have been changed, as by setting
    p_3^2 = 0 in them.
    """
    products = integral.momentum_products
    n_legs = integral.graph.external_legs

    def dot(a: int, c: int) -> sp.Expr:
        low, high = min(a, c), max(a, c)
        return products.get((low, high), products.get((high, low), sp.Integer(0)))

    def q_squared(legs: list[int]) -> sp.Expr:
        inside = set(legs)
        return sp.expand(
            -sum(dot(a, c) for a in legs for c in range(1, n_legs + 1) if c not in inside)
        )

    return q_squared


def one_loop_principal_a_determinant(integral: FeynmanIntegral) -> sp.Expr:
    """Reduced principal A-determinant of a one-loop integral in closed form.

    The product of :func:`one_loop_landau_surfaces`; see there for the
    construction and references.
    """
    surfaces = one_loop_landau_surfaces(integral)
    return sp.Mul(*surfaces) if surfaces else sp.Integer(1)


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


def _images(
    factors: dict[sp.Expr, None], back: dict[sp.Symbol, sp.Expr]
) -> dict[sp.Expr, PolyElement]:
    """Each factor with the symbols of ``back`` replaced by their values.

    The images are computed in the ring of the values' symbols over the
    rationals, in SymPy's order, with the powers of each value kept, much
    faster than substituting into expressions and expanding them.
    """
    if not factors:
        return {}
    symbols = set().union(*(sp.sympify(value).free_symbols for value in back.values()))
    ring = PolyRing(sp.Poly(sp.Add(*symbols)).gens, sp.QQ)
    names = list(back)
    values = [ring.from_expr(back[name]) for name in names]
    powers: dict[tuple[int, int], PolyElement] = {}

    def power(k: int, e: int) -> PolyElement:
        if (k, e) not in powers:
            powers[k, e] = values[k] if e == 1 else power(k, e - 1) * values[k]
        return powers[k, e]

    images: dict[sp.Expr, PolyElement] = {}
    for factor in factors:
        image = ring.zero
        for monomial, c in sp.Poly(factor, *names).terms():
            term = ring(c)
            for k, e in enumerate(monomial):
                if e:
                    term *= power(k, e)
            image += term
        images[factor] = image
    return images


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
    minors: list[tuple[bool, sp.Expr]] = []
    for size in range(1, n + 2):
        for subset in combinations(range(n + 1), size):
            minor = symbolic.domain.to_sympy(symbolic.extract(list(subset), list(subset)).det())
            if minor != 0:
                minors.append((0 in subset, minor))
    # The minors are factored together, and then the images that need it.
    factor_lists = _factor_lists([minor for _, minor in minors], set(back))
    image_of = _images(dict.fromkeys(f for factors in factor_lists for f in factors), back)
    kept: list[tuple[bool, list[PolyElement]]] = []
    for (gram, _minor), factors in zip(minors, factor_lists, strict=True):
        images = [image_of[f] for f in factors]
        if all(images):
            kept.append((gram, images))
    unique = list(dict.fromkeys(image for _, images in kept for image in images))
    if exact:
        pieces = {image: [_normalised_poly(image)] for image in unique}
    else:
        pieces = dict(zip(unique, _factor_lists(unique, kinematic_syms), strict=True))
    first: dict[sp.Expr, None] = {}
    second: dict[sp.Expr, None] = {}
    for gram, images in kept:
        target = second if gram else first
        for image in images:
            for fac in pieces[image]:
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
