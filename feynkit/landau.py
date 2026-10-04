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
  codimension-one part of their zero set. When the ideal is zero, a
  component projects onto all of kinematic space; it is left out, and the
  minimal primes give the hypersurfaces the other components project onto.
  The result is, by definition, the principal Landau determinant of
  Fevola, Mizera and Telen (2024, definition 3.5): the hypersurfaces onto
  which components of the faces' incidence varieties in the torus
  project. The faces are eliminated smallest first, by number of points
  and then by dimension. A face whose elimination runs past its time
  limit, ``timeout`` or, for a face of more than ``LARGE_FACE_POINTS``
  points, ``large_face_timeout``, a face with more points than
  ``max_face_points`` when that is given, and the faces left when
  ``total_timeout`` runs out are skipped and listed in
  ``LandauAnalysis.skipped_faces``; the factors of those faces are then
  missing from the result, at generic kinematics too.

The principal Landau determinant is conjectured to lie in the Euler
discriminant, the locus where |chi| of the complement of {G = 0} in the
torus drops (Fevola, Mizera and Telen 2024, definition 3.2 and conjecture
3.6), and can be strictly smaller (their example 3.10). At special
kinematics a singular point of a face can leave the torus, and no face
sees its limit: at p_1^2 = 0 and p_2^2 = p_3^2 the massive triangle's top
face has no singular point in the torus, since it has moved onto the facet
u_3 = 0, so p_2^2 - p_3^2 is not in the principal Landau determinant,
though |chi| drops from 6 to 5 there. For a Feynman integral whose
momentum products are not the generic ones, :func:`landau_analysis` also
restricts the surfaces of the parent family, the same graph and masses
with generic external kinematics, when asked to with ``limits=True``, and
tests each irreducible factor of
the restrictions that is not in the principal Landau determinant: it is a
limit surface when the number of critical points, |chi| for generic
exponents, drops at two random rational points of it, and a candidate
otherwise (see :class:`LimitSurface`). This is evidence, not proof. It is
not known whether every such factor lies in the Euler discriminant, nor,
beyond one loop, whether the two lists give all of it; in general they do
not, since at generic kinematics the parachute has a component of the
Euler discriminant outside the principal Landau determinant (Fevola,
Mizera and Telen 2024, eq. (3.18)). On the one-loop bubbles, triangles and
boxes with massless legs checked, the principal Landau determinant and the
limit surfaces together give the one-loop closed form.

The factors are candidate codimension-one singular loci on all sheets of the
integral. Membership is necessary for a singularity, not sufficient, and
the list is neither complete for physical sheets nor guaranteed exhaustive
(Fevola, Mizera and Telen 2023, section 2).

For one-loop graphs, :func:`one_loop_principal_a_determinant` gives the
closed form of Dlapa, Helmer, Papathanasiou and Tellander (2023, eq.
1LoopEA): the product of the principal minors of the modified Cayley
matrix of the cycle and, for a graph with bridges, the poles of the
bridges' propagators. It is used as an independent check of the face
computation for generic kinematics, and of the limit surfaces at special
kinematics.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import numbers
import random
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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
    "DEFAULT_FACE_TIMEOUT",
    "DEFAULT_LARGE_FACE_TIMEOUT",
    "DEFAULT_LIMITS",
    "LARGE_FACE_POINTS",
    "FaceDiscriminant",
    "LandauAnalysis",
    "LimitSurface",
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
    dominant
        True when the elimination ideal of a face of dimension two or more
        is zero: a component of the face's incidence variety projects onto
        a dense subset of the kinematic space. Such components are left out
        (Fevola, Mizera and Telen 2024, definition 3.5), and the
        discriminant holds the factors of the others that project onto
        hypersurfaces, found through the minimal primes of the incidence
        ideal. Without Singular they are not sought, and the discriminant
        of such a face is 1.
    """

    dimension: int
    exponents: tuple[tuple[int, ...], ...]
    coefficients: tuple[sp.Expr, ...]
    discriminant: sp.Expr
    is_simplex: bool
    principal: bool = True
    dominant: bool = False


@dataclass(frozen=True)
class LimitSurface:
    """A factor of a restricted surface of the parent family, outside the principal Landau
    determinant.

    The parent family has the same graph and masses with generic external
    kinematics; its surfaces, restricted to the kinematics analysed, factor
    into irreducible polynomials, and this is one of them that the faces of
    the family itself do not give. It is tested for a drop of the Euler
    characteristic: the number of critical points of
    sum_e nu_e log u_e - (D/2) log G on the complement of {G = 0} in the
    torus, which is |chi| of that complement for generic exponents, is
    counted with :func:`~feynkit.point_count.critical_point_count` at a
    random rational point of the family and at random rational points of the
    surface. A drop at every point is evidence that the surface lies in the
    Euler discriminant (Fevola, Mizera and Telen 2024, definition 3.2), not a
    proof: the points are random, and the counts are taken modulo two primes.

    Attributes
    ----------
    surface
        The irreducible polynomial, in the kinematic symbols of the family.
    parent_surfaces
        The Landau surfaces of the parent family whose restrictions it
        divides.
    generic_count
        The count at a random point of the family, or None if it was not
        made or failed.
    counts
        The counts at the points of the surface, in the order drawn; the
        test stops at the first that fails.
    points
        Those points, each as (symbol, value) pairs sorted by name, with the
        energy scale left out when the analysis took it to be 1.
    reason
        None when the count dropped at every point, otherwise why the
        surface is only a candidate.
    """

    surface: sp.Expr
    parent_surfaces: tuple[sp.Expr, ...]
    generic_count: int | None = None
    counts: tuple[int, ...] = ()
    points: tuple[tuple[tuple[sp.Symbol, sp.Rational], ...], ...] = ()
    reason: str | None = None

    @property
    def confirmed(self) -> bool:
        """Whether the count dropped at every point of the surface tested."""
        return self.reason is None


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
        Exponent sets of the faces whose discriminants are left out, in the
        order of the faces of the polytope: those whose elimination ran past
        its time limit, those with more points than ``max_face_points``,
        and those left when ``total_timeout`` ran out. ``landau_surfaces``
        and ``principal_a_determinant`` are then incomplete, whatever the
        kinematics: the massless pentagon's polytope, of 15 points, runs
        past its time limit, and its factor is missing.
    limit_surfaces
        The factors of the parent family's surfaces, restricted to these
        kinematics, that are not in ``landau_surfaces`` and at whose random
        points the count of critical points dropped; see
        :class:`LimitSurface`. Empty without a parent.
    limit_candidates
        The other such factors: the count did not drop at a point, no
        rational point was found on the factor, a count failed or ran past
        its time, or the test was not asked for.
    parent
        The analysis of the parent family, the same graph and masses with
        generic external kinematics, or None when there is no parent. When
        it skipped faces the limit surfaces may be incomplete.
    timed_out_faces
        The faces of ``skipped_faces`` whose elimination ran past its time
        limit, ``timeout`` or ``large_face_timeout``, in the same order.
    unattempted_faces
        The faces of ``skipped_faces`` left when ``total_timeout`` ran out,
        the one it cut short included, in the same order.
    """

    face_discriminants: tuple[FaceDiscriminant, ...]
    principal_a_determinant: sp.Expr
    landau_surfaces: tuple[sp.Expr, ...]
    skipped_faces: tuple[tuple[tuple[int, ...], ...], ...] = ()
    limit_surfaces: tuple[LimitSurface, ...] = ()
    limit_candidates: tuple[LimitSurface, ...] = ()
    parent: LandauAnalysis | None = None
    timed_out_faces: tuple[tuple[tuple[int, ...], ...], ...] = ()
    unattempted_faces: tuple[tuple[tuple[int, ...], ...], ...] = ()


# --- discriminants -----------------------------------------------------------


def _univariate_discriminant(coeffs: list[sp.Expr], t_exps: list[int]) -> sp.Expr:
    """Discriminant of P(t) = sum coeffs[k] * t^{t_exps[k]}.

    For n = deg P this is (-1)^(n(n-1)/2) Res(P, P') / lc(P), the standard
    polynomial discriminant, a polynomial in the coefficients.  Returns
    Integer(1) for linear or constant P.
    """
    _t = sp.Symbol("_t_landau_internal_")
    P = sp.Integer(0)
    for c, e in zip(coeffs, t_exps, strict=True):
        P = P + c * _t**e
    poly = sp.Poly(P, _t)
    deg = poly.degree()
    if deg <= 1:
        return sp.Integer(1)
    dpoly = sp.Poly(sp.diff(P, _t), _t)
    res = poly.resultant(dpoly)
    lc = poly.nth(deg)
    disc = sp.cancel((-1) ** (deg * (deg - 1) // 2) * res / lc)
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

# Faces with more points than this are large: they get large_face_timeout instead of timeout,
# and without Singular, whose SymPy fallback cannot be stopped, they are skipped when
# max_face_points is None. Every one-loop box has at most 14 points.
LARGE_FACE_POINTS = 14

# The seconds Singular gets for each face of at most LARGE_FACE_POINTS points by default, several
# times the longest such a face has taken on the graphs the tests and the report exercise, about
# 9 s; a face past it is skipped.
DEFAULT_FACE_TIMEOUT = 60

# The seconds Singular gets for each larger face by default: a probe. On the graphs measured, the
# large faces that gave a component took under a second, those that took longer added nothing,
# and many never finish.
DEFAULT_LARGE_FACE_TIMEOUT = 5

# Whether landau_analysis looks for limit surfaces when not told: True, False, or "one-loop"
# for one-loop integrals only.
DEFAULT_LIMITS: bool | str = False


def _check_timeout(timeout: object, *, optional: bool = True, name: str = "timeout") -> None:
    """Raise ValidationError unless ``timeout`` is a number of seconds greater
    than 0 and at most 2,000,000, or None when ``optional``; ``name`` is the
    argument's, for the message."""
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
        f"{name} must be {what} of seconds greater than 0 and at most {_MAX_TIMEOUT:,}; "
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


class _EliminationTimeout(ComputationError):
    """Singular ran past the time limit on a face."""


def _singular_ideal(
    system: list[sp.Expr], to_eliminate: list[sp.Symbol], kin: list[sp.Symbol]
) -> tuple[str, str]:
    """Singular's declarations of the ring and of the ideal of ``system``, in the variables
    v0, v1, ..., those to eliminate first, and the product of those to eliminate."""
    names = {sym: f"v{i}" for i, sym in enumerate(to_eliminate + kin)}

    def render(expr: sp.Expr) -> str:
        return str(sp.expand(expr).subs(names, simultaneous=True)).replace("**", "^")

    ring_vars = ",".join(names[v] for v in to_eliminate + kin)
    ideal = ",".join(render(g) for g in system)
    declarations = f"ring r = 0, ({ring_vars}), dp;\nideal I = {ideal};\n"
    return declarations, "*".join(names[v] for v in to_eliminate)


def _run_singular(script: str, binary: str, *, points: int, timeout: float | None) -> str:
    """What Singular prints on stdout for a script eliminating a face with ``points`` points.

    Raises
    ------
    ComputationError
        If Singular fails, or runs past ``timeout`` seconds, as the subclass
        _EliminationTimeout.
    """
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
            raise _EliminationTimeout(
                f"Singular did not eliminate a face with {points} points within "
                f"timeout={timeout} s"
            ) from None
    if result.returncode != 0:
        reason = " ".join(result.stderr.split())[:100] or f"exit status {result.returncode}"
        raise ComputationError(
            f"Singular failed on a face with {points} points, "
            f"{len(result.stdout)} characters of output: {reason}"
        ) from None
    return result.stdout


def _read_generator(
    text: str, n_eliminated: int, kin: list[sp.Symbol], *, points: int, size: int
) -> sp.Poly | None:
    """A generator of an elimination ideal that Singular printed on one line, or None for 0.

    Raises
    ------
    ComputationError
        If the line is not a polynomial in ``kin``.
    """
    line = text.replace(" ", "")
    if not line or line == "0":
        return None
    terms = _read_singular_polynomial(line, n_eliminated, len(kin))
    if terms is None:
        raise ComputationError(
            f"cannot read Singular's elimination ideal of a face with {points} points "
            f"from {size} characters of output, at {repr(text.strip())[:60]}"
        ) from None
    polynomial = {key: sp.Rational(c.numerator, c.denominator) for key, c in terms.items() if c}
    return sp.Poly.from_dict(polynomial, *kin) if polynomial else None


class _SingularElimination(NamedTuple):
    """The generators of an elimination ideal and, when it is zero and a decomposition was
    asked for, the generators of the elimination ideal of each minimal prime."""

    generators: list[sp.Poly]
    components: list[list[sp.Poly]] | None


def _eliminate_singular(
    system: list[sp.Expr],
    to_eliminate: list[sp.Symbol],
    kin: list[sp.Symbol],
    binary: str,
    *,
    points: int,
    timeout: float | None = None,
    decompose: bool = False,
) -> _SingularElimination:
    """Elimination ideal via Singular's ``eliminate`` (Decker et al., Singular 4).

    Singular prints each generator expanded on one line, which can hold tens
    of thousands of terms; the line is read term by term, since SymPy's
    parser recurses once per term. ``points`` is the number of points of the
    face, for the error messages.

    With ``decompose``, a zero elimination ideal is followed in the same run
    by the minimal associated primes of the ideal of ``system``, the ideals
    of the irreducible components of its variety, from ``minAssChar`` in
    Singular's primdec.lib, which works with characteristic sets, each
    eliminated in turn. A component projects onto a dense subset of the
    kinematic space exactly when its elimination ideal is zero, an empty
    list here. Singular prints ``@`` before each component's generators.

    Raises
    ------
    ComputationError
        If Singular fails, runs past ``timeout`` seconds, as the subclass
        _EliminationTimeout, or prints anything but polynomials in ``kin``,
        with the components after the generators.
    """
    declarations, product = _singular_ideal(system, to_eliminate, kin)
    script = (
        declarations
        + f"ideal E = eliminate(I, {product});\n"
        + "int k; for (k = 1; k <= size(E); k++) { print(string(E[k])); }\n"
    )
    if decompose:
        script += (
            "if (size(E) == 0) {\n"
            '  LIB "primdec.lib";\n'
            "  list L = minAssChar(I); int i; ideal F;\n"
            "  for (i = 1; i <= size(L); i++) {\n"
            f'    F = eliminate(L[i], {product}); print("@");\n'
            "    for (k = 1; k <= ncols(F); k++) { print(string(F[k])); }\n"
            "  }\n"
            "}\n"
        )
    stdout = _run_singular(script + "quit;\n", binary, points=points, timeout=timeout)
    generators: list[sp.Poly] = []
    components: list[list[sp.Poly]] | None = None
    for text in stdout.splitlines():
        if decompose and text.strip() == "@":
            if generators:
                raise ComputationError(
                    f"cannot read Singular's elimination ideal of a face with {points} points: "
                    "components follow a non-zero ideal"
                )
            components = [*(components or []), []]
            continue
        generator = _read_generator(text, len(to_eliminate), kin, points=points, size=len(stdout))
        if generator is not None:
            (generators if components is None else components[-1]).append(generator)
    if decompose and not generators and components is None:
        components = []
    return _SingularElimination(generators, components)


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


def _flint_available() -> bool:
    """Whether python-flint can be imported; the analogue of :func:`_singular_binary`."""
    return importlib.util.find_spec("flint") is not None


def _factorize_flint(polys: list[sp.Poly]) -> dict[sp.Poly, list[tuple[sp.Poly, int]]] | None:
    """Poly.factor_list without the constant, for each polynomial, by python-flint.

    Each factor is written as :func:`_factorize_singular` writes it: in the polynomial's
    generators and domain, primitive with a positive leading coefficient, and in SymPy's
    order. The polynomials must be over the integers or the rationals. A polynomial whose
    factors do not multiply back to it up to a constant is left out of the result, and so
    is every polynomial when python-flint cannot be used; None if it cannot.
    """
    try:
        from flint import fmpz_mpoly, fmpz_mpoly_ctx
    except ImportError:
        return None
    out: dict[sp.Poly, list[tuple[sp.Poly, int]]] = {}
    for poly in polys:
        if poly.domain not in (sp.ZZ, sp.QQ):
            continue
        zeros = (0,) * len(poly.gens)
        try:
            ctx = fmpz_mpoly_ctx.get(tuple(f"x{k}" for k in range(len(poly.gens))), "lex")
            _, integral = poly.clear_denoms(convert=True)
            _, integral_factors = fmpz_mpoly(
                {powers: int(c) for powers, c in integral.terms()}, ctx
            ).factor()
        except (ArithmeticError, TypeError, ValueError):
            continue
        factors: dict[sp.Poly, int] = {}
        for factor, multiplicity in integral_factors:
            terms = factor.to_dict()
            if list(terms) == [zeros]:
                continue
            piece = sp.Poly.from_dict(
                {powers: int(c) for powers, c in terms.items()}, *poly.gens, domain=sp.ZZ
            )
            _, piece = piece.primitive()
            piece = (-piece if piece.LC() < 0 else piece).set_domain(poly.domain)
            factors[piece] = factors.get(piece, 0) + int(multiplicity)
        product = sp.Poly(1, *poly.gens, domain=sp.QQ)
        for piece, k in factors.items():
            product *= piece.set_domain(sp.QQ) ** k
        if product.monic() != poly.set_domain(sp.QQ).monic():
            continue
        out[poly] = sorted(factors.items(), key=_factor_order)
    return out


def _factorize(
    polys: list[sp.Poly], binary: str | None
) -> dict[sp.Poly, list[tuple[sp.Poly, int]]]:
    """The factorisation of as many of the polynomials as Singular, else python-flint, gives;
    the rest are left to SymPy."""
    factored = _factorize_singular(polys, binary) if binary is not None and polys else None
    if factored is None and polys and _flint_available():
        factored = _factorize_flint(polys)
    return factored or {}


def _factor_lists(exprs: list[sp.Expr], kinematic_syms: set[sp.Symbol]) -> list[list[sp.Expr]]:
    """:func:`_factor_list` of each expression, factored by Singular in one run when it is
    installed, else by python-flint when that is.

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
    every expression that neither Singular nor python-flint factors. An element of a polynomial ring
    over the rationals, whose symbols are in SymPy's order, stands for its
    expression.
    """
    binary = _singular_binary()
    plans = [
        _ring_bases(expr) if isinstance(expr, PolyElement) else _factor_bases(expr)
        for expr in (exprs if binary is not None or _flint_available() else [])
    ]
    distinct = list(dict.fromkeys(poly for plan in plans if plan for poly, _ in plan))
    factored = _factorize(distinct, binary)
    result: list[list[sp.Expr]] = []
    for k, expr in enumerate(exprs):
        plan = plans[k] if plans else None
        if plan is None or any(base not in factored for base, _ in plan):
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


class _Elimination(NamedTuple):
    """What :func:`_elimination_discriminant` finds for a face."""

    factors: list[sp.Expr]
    principal: bool
    dominant: bool


def _gcd(polys: list[sp.Poly]) -> sp.Poly:
    """The greatest common divisor of one or more polynomials."""
    common = polys[0]
    for g in polys[1:]:
        common = common.gcd(g)
    return common


def _elimination_discriminant(
    coeffs: list[sp.Expr],
    exps: list[tuple[int, ...]],
    kinematic_syms: set[sp.Symbol],
    backend: str = "auto",
    *,
    scale: sp.Symbol | None = None,
    timeout: float | None = None,
) -> _Elimination:
    """Kinematic locus where f = sum c_k t^{e_k} has a singular point in the torus.

    Returns the distinct kinematic factors, whether the elimination ideal
    had at most one generator (principal) and whether it was zero
    (dominant). The factors are those of the generator or, when there are
    several, of their greatest common divisor: an irreducible polynomial
    defines a codimension-one component of their common zeros exactly when
    it divides every one of them. A zero ideal means that a component of the
    incidence variety projects onto a dense subset of the kinematic space.
    With Singular, the ideal of each minimal prime is then eliminated in
    turn, the dominant components, whose ideals are zero, are left out, and
    the factors are those of the greatest common divisor of each of the
    others, which is 1 unless the component projects onto a hypersurface
    (Fevola, Mizera and Telen 2024, definition 3.5 and example 3.9). The
    SymPy fallback has no decomposition and gives no factor for such a
    face. A component that arises only as the limit of a singular point
    leaving the torus is in no face's locus, as p_2^2 - p_3^2 is in none
    for the massive triangle at p_1^2 = 0, whose top face has the locus
    p_2^2 = p_3^2 = 0; see :class:`LimitSurface`. Eliminates
    and factors with Singular when it is installed and ``backend`` is
    "auto" or "singular", else with SymPy; ``timeout`` limits the face's
    elimination and decomposition by Singular together, in seconds, and
    running past it raises _EliminationTimeout.

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
        return _Elimination([], True, False)
    components: list[list[sp.Poly]] | None = None
    if binary is not None:
        eliminated, components = _eliminate_singular(
            system,
            to_eliminate,
            kin,
            binary,
            points=len(coeffs),
            timeout=timeout,
            decompose=True,
        )
    else:
        eliminated = _eliminate_sympy(system, to_eliminate, kin)
    principal, dominant = len(eliminated) <= 1, not eliminated
    if dominant:
        if components is None:
            return _Elimination([], principal, dominant)
        # The gcd of the generators of a prime of height two or more is 1.
        eliminated = [_gcd(generators) for generators in components if generators]
        eliminated = [g for g in eliminated if not g.is_ground]
        if not eliminated:
            return _Elimination([], principal, dominant)
        eliminated = [sp.prod(eliminated[1:], start=eliminated[0])]

    # sp.factor_list makes each factor primitive with a positive leading coefficient in the
    # order sp.Poly gives the symbols; factoring in that order normalises these factors as
    # _factor_list normalises those of the other faces.
    order = sp.Poly(sp.Add(*kin)).gens
    common = _gcd(eliminated).reorder(*order)
    factored = _factorize([common], binary) if not common.is_ground else {}
    pairs = factored[common] if common in factored else common.factor_list()[1]
    factors: dict[sp.Expr, None] = {}
    if back:
        images = [sp.expand(fac.as_expr().xreplace(back)) for fac, _exp in pairs]
        if not squared:
            pieces_of = [[_normalised(image)] for image in images]
        else:
            pieces_of = _factor_lists(images, kinematic_syms)
        for pieces in pieces_of:
            for piece in pieces:
                factors.setdefault(piece, None)
    else:
        for fac, _exp in pairs:
            if fac.free_symbols & kinematic_syms:
                factors.setdefault(fac.as_expr(), None)
    return _Elimination(list(factors), principal, dominant)


def _face_discriminant(
    coeffs: list[sp.Expr],
    exps_ambient: list[tuple[int, ...]],
    dimension: int,
    kinematic_syms: set[sp.Symbol],
    max_face_points: int | None,
    scale: sp.Symbol | None = None,
    timeout: float | None = None,
) -> tuple[sp.Expr, list[sp.Expr] | None, bool, bool, bool] | None:
    """Return (discriminant, its kinematic factors, is_simplex, principal, dominant), or None
    if the face has more than ``max_face_points`` points. The factors of a vertex or an edge
    are None: they are those of the discriminant, which the caller factors with the others."""
    n_pts = len(coeffs)
    is_simplex = n_pts == dimension + 1
    if dimension == 0:
        return sp.together(coeffs[0]), None, True, True, False
    if is_simplex:
        return sp.Integer(1), [], True, True, False
    if max_face_points is not None and n_pts > max_face_points:
        return None
    lattice = _lattice_coordinates(np.array(exps_ambient, dtype=int))
    if dimension == 1:
        disc = _univariate_discriminant(coeffs, [c[0] for c in lattice])
        return disc, None, False, True, False
    factors, principal, dominant = _elimination_discriminant(
        coeffs, lattice, kinematic_syms, scale=scale, timeout=timeout
    )
    return sp.Mul(*factors), factors, False, principal, dominant


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


# --- limit surfaces ----------------------------------------------------------

# Kinematic values are drawn as a / b with a in [1, 97] and b in [1, 11].
_NUMERATORS = (1, 97)
_DENOMINATORS = (1, 11)
# Draws tried for a point, of the family or of a surface, before giving up.
_POINT_TRIES = 50
# Points of each surface at which the count must drop.
_POINTS_PER_SURFACE = 2


def _rational_root(poly: sp.Poly) -> sp.Rational | None:
    """A rational root of a univariate polynomial of degree 1 or 2, or None."""
    if poly.degree() == 1:
        a, b = poly.all_coeffs()
        return sp.Rational(-b / a)
    if poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        root = sp.sqrt(b * b - 4 * a * c)
        if root.is_Rational:
            return sp.Rational((-b + root) / (2 * a))
    return None


class _Family:
    """The polynomials G(u; z) of a family, for counting critical points at points z.

    The kinematic symbols in ``fixed`` keep their values there, the energy
    scale 1 when the analysis eliminated at scale 1; the others are drawn.
    """

    def __init__(
        self,
        g: sp.Expr,
        variables: list[sp.Symbol],
        fixed: dict[sp.Symbol, sp.Expr],
        *,
        seed: int,
    ) -> None:
        self.g = g
        self.variables = variables
        self.fixed = fixed
        self.symbols = sorted(g.free_symbols - set(variables) - set(fixed), key=str)
        self.coefficients = [c for _, c in extract_monomial_support(g, variables)]
        self.seed = seed
        self.rng = random.Random(seed)

    def _draw(self) -> dict[sp.Symbol, sp.Rational]:
        return {
            x: sp.Rational(self.rng.randint(*_NUMERATORS), self.rng.randint(*_DENOMINATORS))
            for x in self.symbols
        }

    def point(
        self, avoid: list[sp.Expr], on: sp.Expr | None = None
    ) -> dict[sp.Symbol, sp.Rational] | None:
        """A random rational point off every polynomial of ``avoid``: on {on = 0} when
        ``on`` is given, and otherwise off every coefficient of G as well. None if
        _POINT_TRIES draws found none.

        On a surface, the other symbols are drawn and a symbol in which ``on`` then
        has degree 1, or degree 2 with a rational root, is solved for.
        """
        checks = avoid if on is not None else avoid + self.coefficients
        for _ in range(_POINT_TRIES):
            values = self._draw()
            if on is not None:
                unknowns = [x for x in self.symbols if x in on.free_symbols]
                self.rng.shuffle(unknowns)
                for x in unknowns:
                    rest = {y: v for y, v in values.items() if y != x}
                    root = _rational_root(sp.Poly(on.xreplace({**rest, **self.fixed}), x))
                    if root is not None:
                        values[x] = root
                        break
                else:
                    continue
            at = {**values, **self.fixed}
            if not any(sp.sympify(h).xreplace(at) == 0 for h in checks):
                return values
        return None

    def count(self, values: dict[sp.Symbol, sp.Rational], timeout: float) -> int:
        """The number of critical points of the likelihood function at the point."""
        from . import point_count

        g = sp.expand(self.g.xreplace({**values, **self.fixed}))
        return point_count.critical_point_count(
            g, self.variables, {}, seed=self.seed, timeout=timeout
        )


def _restricted_factors(
    parent: LandauAnalysis,
    restriction: Mapping[sp.Symbol, sp.Expr],
    surfaces: tuple[sp.Expr, ...],
    kinematic_syms: set[sp.Symbol],
    surface_syms: set[sp.Symbol],
) -> dict[sp.Expr, list[sp.Expr]]:
    """The irreducible factors of the parent's surfaces, restricted, that are not in
    ``surfaces``, each with the parent surfaces whose restrictions it divides.
    Restrictions that vanish identically are left out."""
    images = [(h, sp.expand(h.xreplace(dict(restriction)))) for h in parent.landau_surfaces]
    images = [(h, image) for h, image in images if image != 0]
    known = {_normalised(h) for h in surfaces}
    found: dict[sp.Expr, list[sp.Expr]] = {}
    factor_lists = _factor_lists([image for _, image in images], kinematic_syms)
    for (h, _), factors in zip(images, factor_lists, strict=True):
        for factor in factors:
            if factor.free_symbols & surface_syms and _normalised(factor) not in known:
                found.setdefault(factor, []).append(h)
    return found


def _confirmed(
    family: _Family,
    candidates: dict[sp.Expr, list[sp.Expr]],
    surfaces: tuple[sp.Expr, ...],
    *,
    confirm: bool,
    confirm_timeout: float,
) -> list[LimitSurface]:
    """Each candidate with the counts that confirm it, or the reason they do not."""
    generic: int | None = None
    shared: str | None = "not tested, since confirm is False" if not confirm else None
    others = list(surfaces) + list(candidates)
    if shared is None:
        point = family.point(others)
        if point is None:
            shared = "no random point of the family was found off every surface"
        else:
            try:
                generic = family.count(point, confirm_timeout)
            except (ComputationError, RuntimeError, ValidationError) as exc:
                shared = f"the count at a random point of the family failed: {exc}"
    records: list[LimitSurface] = []
    for h, parents in candidates.items():
        counts: list[int] = []
        points: list[dict[sp.Symbol, sp.Rational]] = []
        reason = shared
        deadline = time.monotonic() + confirm_timeout
        while reason is None and len(counts) < _POINTS_PER_SURFACE:
            point = family.point([x for x in others if x != h], on=h)
            left = deadline - time.monotonic()
            if point is None:
                reason = "no rational point found on it"
            elif left <= 0:
                reason = f"its counts ran past confirm_timeout={confirm_timeout} s"
            else:
                try:
                    counts.append(family.count(point, left))
                except (ComputationError, RuntimeError, ValidationError) as exc:
                    reason = f"the count failed: {exc}"
                    break
                points.append(point)
                if generic is not None and counts[-1] >= generic:
                    reason = f"the count, {counts[-1]}, did not drop below {generic}"
        records.append(
            LimitSurface(
                surface=h,
                parent_surfaces=tuple(parents),
                generic_count=generic,
                counts=tuple(counts),
                points=tuple(
                    tuple(sorted(point.items(), key=lambda item: str(item[0]))) for point in points
                ),
                reason=reason,
            )
        )
    return records


def _generic_parent(
    integral: FeynmanIntegral,
) -> tuple[FeynmanIntegral, dict[sp.Symbol, sp.Expr]] | None:
    """The integral with the same graph and masses and generic external kinematics, and
    the restriction taking its kinematic symbols to the integral's; None when the
    integral's momentum products are already those.

    The generic products are linear forms in the invariants, as many as the
    products and independent, so equating them with the integral's products
    gives each invariant uniquely.
    """
    parent = integral.with_(momentum_products=None)
    generic = {key: sp.sympify(v) for key, v in parent.momentum_products.items()}
    given = {key: sp.sympify(v) for key, v in integral.momentum_products.items()}
    given = {(min(key), max(key)): v for key, v in given.items()}
    invariants = sorted(set().union(*(v.free_symbols for v in generic.values())), key=str)
    if not invariants or set(given) != {(min(k), max(k)) for k in generic}:
        return None
    fresh = {x: sp.Dummy(x.name) for x in invariants}
    equations = [v.xreplace(fresh) - given[(min(key), max(key))] for key, v in generic.items()]
    matrix, rhs = sp.linear_eq_to_matrix(equations, [fresh[x] for x in invariants])
    if matrix.rows != matrix.cols or matrix.det() == 0:
        return None
    solution = matrix.LUsolve(rhs)
    restriction = {x: sp.expand(solution[k]) for k, x in enumerate(invariants)}
    if _renames(restriction, parent):
        return None
    return parent, restriction


def _renames(restriction: dict[sp.Symbol, sp.Expr], parent: FeynmanIntegral) -> bool:
    """Whether the restriction only renames the invariants: its values are linearly
    independent linear forms, without constant terms, in symbols that are neither masses
    nor the energy scale. The family is then the parent family in new coordinates."""
    used = sorted(set().union(*(v.free_symbols for v in restriction.values())), key=str)
    own = parent.symanzik.g.free_symbols - set(parent.symanzik.lp_parameters) - set(restriction)
    if not used or set(used) & own:
        return False
    try:
        matrix, constants = sp.linear_eq_to_matrix(list(restriction.values()), used)
    except (ValueError, sp.PolynomialError):
        return False
    return all(c == 0 for c in constants) and matrix.rank() == len(restriction)


# --- public API --------------------------------------------------------------


def landau_analysis_from_polynomial(
    g_poly: sp.Expr,
    lp_parameters: list[sp.Symbol],
    *,
    max_face_points: int | None = None,
    scale: sp.Symbol | None = None,
    timeout: float | None = DEFAULT_FACE_TIMEOUT,
    large_face_timeout: float | None = DEFAULT_LARGE_FACE_TIMEOUT,
    total_timeout: float | None = None,
    parent: sp.Expr | None = None,
    restriction: Mapping[sp.Symbol, sp.Expr] | None = None,
    confirm: bool = True,
    confirm_timeout: float = 60,
    seed: int = 0,
) -> LandauAnalysis:
    """Reduced principal A-determinant of a polynomial in the given variables.

    Parameters
    ----------
    g_poly
        The polynomial, typically G = U + F.
    lp_parameters
        Its variables; every other symbol is treated as kinematic.
    max_face_points
        Faces with more monomials than this are not eliminated and are
        reported in ``skipped_faces``; the surfaces and their product then
        lack those faces' factors. None, the default, sets no limit, and
        the time limits decide instead. Without Singular, None stands for
        :data:`LARGE_FACE_POINTS`, 14, which covers every one-loop box,
        since SymPy's elimination cannot be stopped.
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
        The most seconds to give Singular for each face it eliminates of at
        most :data:`LARGE_FACE_POINTS` (14) points, at most 2,000,000,
        :data:`DEFAULT_FACE_TIMEOUT` (60) by default; None sets no limit. A
        face that runs past it is skipped and listed in ``skipped_faces``
        and ``timed_out_faces``: one of 14 points of the generic massive
        parachute had not finished after 400 s. It limits each face, its
        decomposition into minimal primes included; the other steps are not
        limited, namely the SymPy fallback, the discriminants of edges and
        the factorisations, which SymPy can take minutes over.
    large_face_timeout
        The same for each larger face, :data:`DEFAULT_LARGE_FACE_TIMEOUT`
        (5) by default: a short probe. The number of points is a poor
        measure of the cost. Faces of 20 and 28 points of a two-loop box
        give a component in under a second, and the polytope of the
        generic massive parachute, 19 points, gives one in 0.2 s, while on
        the graphs measured the large faces that took longer added nothing,
        and many never finish.
    total_timeout
        The most seconds to give the faces of more than
        :data:`LARGE_FACE_POINTS` points together, at most 2,000,000; None,
        the default, sets no limit. The faces are attempted smallest first,
        by number of points and then by dimension, so these come last, each
        within its time limit or the time left, whichever is shorter. When
        it runs out, the face under way and those not yet attempted are
        skipped and listed in ``skipped_faces`` and ``unattempted_faces``.
        It never cuts short a face of up to :data:`LARGE_FACE_POINTS`
        points, which is treated as without it, nor a vertex, a simplex or
        an edge, which need no elimination.
    parent, restriction
        A polynomial in the same variables of which ``g_poly`` is a
        restriction, and the map from its kinematic symbols to expressions
        in those of ``g_poly`` that restricts it: substituted, ``parent``
        must be ``g_poly``. The restriction defaults to no substitution.
        The parent is analysed with the same ``max_face_points``, ``scale``
        and time limits, ``total_timeout`` a total of its own, and the
        irreducible factors of its surfaces,
        restricted, that are not among ``landau_surfaces`` and do not
        vanish identically, are tested and listed in ``limit_surfaces`` or
        ``limit_candidates``. Without a parent both are empty.
    confirm
        Whether to test each factor for a drop of the number of critical
        points (see :class:`LimitSurface`); when False every factor is a
        candidate.
    confirm_timeout
        The most seconds to give the counts of each factor, and the count at
        a random point of the family, at most 2,000,000. Its point search
        is not limited.
    seed
        Seed of the random points and of the random exponents of the counts.

    Raises
    ------
    ValidationError
        If ``timeout``, ``large_face_timeout`` or ``total_timeout`` is
        neither None nor a number greater than 0 and at most 2,000,000, or
        ``confirm_timeout`` is not such a number; if
        ``restriction`` is given without ``parent``; or if ``parent`` does
        not restrict to ``g_poly``.
    ComputationError
        If Singular fails on a face or prints output that is not an
        elimination ideal.
    """
    _check_timeout(timeout)
    _check_timeout(large_face_timeout, name="large_face_timeout")
    _check_timeout(total_timeout, name="total_timeout")
    _check_timeout(confirm_timeout, optional=False, name="confirm_timeout")
    if parent is None and restriction is not None:
        raise ValidationError("a restriction needs a parent polynomial")
    parent_g: sp.Expr | None = None
    if parent is not None:
        restriction = dict(restriction or {})
        parent_g = sp.expand(parent)
        if sp.expand(parent_g.xreplace(restriction) - sp.expand(g_poly)) != 0:
            raise ValidationError(
                "the parent polynomial does not restrict to the polynomial under the restriction"
            )
    limits = (timeout, large_face_timeout, total_timeout)
    analysis = _analysis(g_poly, lp_parameters, max_face_points, scale, *limits)
    if parent_g is None or restriction is None:
        return analysis
    parent_analysis = _parent_analysis(
        parent_g, tuple(lp_parameters), max_face_points, scale, *limits
    )
    g_poly = sp.expand(g_poly)
    kinematic_syms = g_poly.free_symbols - set(lp_parameters)
    surface_syms = kinematic_syms - ({scale} if scale is not None else set())
    candidates = _restricted_factors(
        parent_analysis, restriction, analysis.landau_surfaces, kinematic_syms, surface_syms
    )
    fixed: dict[sp.Symbol, sp.Expr] = {}
    if scale is not None and scale in g_poly.free_symbols:
        support = extract_monomial_support(g_poly, lp_parameters)
        if _unit_scale_is_exact(support, scale):
            fixed[scale] = sp.Integer(1)
    family = _Family(g_poly, list(lp_parameters), fixed, seed=seed)
    records = _confirmed(
        family,
        candidates,
        analysis.landau_surfaces,
        confirm=confirm,
        confirm_timeout=confirm_timeout,
    )
    return dataclasses.replace(
        analysis,
        limit_surfaces=tuple(r for r in records if r.confirmed),
        limit_candidates=tuple(r for r in records if not r.confirmed),
        parent=parent_analysis,
    )


@functools.lru_cache(maxsize=16)
def _parent_analysis(
    parent_g: sp.Expr,
    lp_parameters: tuple[sp.Symbol, ...],
    max_face_points: int | None,
    scale: sp.Symbol | None,
    timeout: float | None,
    large_face_timeout: float | None,
    total_timeout: float | None,
) -> LandauAnalysis:
    """The principal Landau determinant of a parent family, kept for the next restriction
    of the same family: analysing the kinematics of one graph in turn, each with every
    set of massless legs, analyses its generic family once. The analysis is immutable."""
    limits = (timeout, large_face_timeout, total_timeout)
    return _analysis(parent_g, list(lp_parameters), max_face_points, scale, *limits)


def _analysis(
    g_poly: sp.Expr,
    lp_parameters: list[sp.Symbol],
    max_face_points: int | None,
    scale: sp.Symbol | None,
    timeout: float | None,
    large_face_timeout: float | None,
    total_timeout: float | None,
) -> LandauAnalysis:
    """The principal Landau determinant of :func:`landau_analysis_from_polynomial`.

    The faces are computed smallest first, so that the faces left when
    ``total_timeout`` runs out are the largest, and listed in the order of
    :func:`feynkit.polytope.faces`.
    """
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
    if max_face_points is None and _singular_binary() is None:
        max_face_points = LARGE_FACE_POINTS
    # The seconds the faces of more than LARGE_FACE_POINTS points have taken, which alone count
    # against total_timeout.
    spent = 0.0

    listed = [
        (dimension, [tuple(int(x) for x in exps[i]) for i in idx], [support[i][1] for i in idx])
        for dimension, idx in _faces(exps)
    ]
    results: dict[int, tuple[sp.Expr, list[sp.Expr] | None, bool, bool, bool] | None] = {}
    timed_out: set[int] = set()
    unattempted: set[int] = set()
    for k in sorted(range(len(listed)), key=lambda k: (len(listed[k][1]), listed[k][0])):
        dimension, face_exps, face_coeffs = listed[k]
        points = len(face_exps)
        limit, cut = (timeout if points <= LARGE_FACE_POINTS else large_face_timeout), False
        eliminated = dimension >= 2 and points > dimension + 1
        if max_face_points is not None and points > max_face_points:
            eliminated = False
        charged = eliminated and points > LARGE_FACE_POINTS
        if charged and total_timeout is not None:
            left = total_timeout - spent
            if left <= 0:
                results[k] = None
                unattempted.add(k)
                continue
            if limit is None or left < limit:
                limit, cut = left, True
        began = time.monotonic()
        try:
            results[k] = _face_discriminant(
                face_coeffs,
                face_exps,
                dimension,
                kinematic_syms,
                max_face_points,
                unit_scale,
                limit,
            )
        except _EliminationTimeout:
            results[k] = None
            (unattempted if cut else timed_out).add(k)
        if charged:
            spent += time.monotonic() - began

    faces: list[FaceDiscriminant] = []
    factors_by_face: list[list[sp.Expr]] = []
    skipped: list[tuple[tuple[int, ...], ...]] = []
    # The discriminants of the vertices and edges, factored together at the end.
    pending: list[tuple[int, sp.Expr]] = []
    for k, (dimension, face_exps, face_coeffs) in enumerate(listed):
        result = results[k]
        if result is None:
            skipped.append(tuple(face_exps))
            continue
        disc, factors, is_simplex, principal, dominant = result
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
                dominant=dominant,
            )
        )

    discriminants = [disc for _, disc in pending]
    for (k, _), factors in zip(pending, _factor_lists(discriminants, kinematic_syms), strict=True):
        factors_by_face[k] = factors
    e_a, surfaces = _reduced(factors_by_face, surface_syms)
    return LandauAnalysis(
        tuple(faces),
        e_a,
        surfaces,
        tuple(skipped),
        timed_out_faces=tuple(tuple(listed[k][1]) for k in sorted(timed_out)),
        unattempted_faces=tuple(tuple(listed[k][1]) for k in sorted(unattempted)),
    )


def landau_analysis(
    integral: FeynmanIntegral,
    *,
    max_face_points: int | None = None,
    timeout: float | None = DEFAULT_FACE_TIMEOUT,
    large_face_timeout: float | None = DEFAULT_LARGE_FACE_TIMEOUT,
    total_timeout: float | None = None,
    limits: bool | str | None = None,
    confirm: bool = True,
    confirm_timeout: float = 60,
    seed: int = 0,
) -> LandauAnalysis:
    """Reduced principal A-determinant of G = U + F for a Feynman integral.

    See the module docstring for what the faces contribute and for the
    caveats on interpreting the factors as Landau singularities, and
    :func:`landau_analysis_from_polynomial` for the arguments and errors.

    The parent family is the integral with the same graph and masses and the
    generic momentum products, those the integral would have been given by
    default, when its own differ. Its invariants are found from the
    integral's products, which they determine uniquely, and give the
    restriction. When the products are the generic ones, or the generic
    ones in invariants renamed by an invertible linear map, there is no
    parent, and ``limit_surfaces`` and ``limit_candidates`` are empty, and
    there is none unless ``limits`` asks for it. The parent is analysed
    with the same ``max_face_points`` and time limits, and a
    ``total_timeout`` of its own; when it skips faces, listed in
    ``parent.skipped_faces``, the limit surfaces may be incomplete.

    ``limits`` is True to look for limit surfaces, False not to, and
    "one-loop" to look for them only at one loop; None, the default, takes
    :data:`DEFAULT_LIMITS`, which is False.
    """
    if limits is None:
        limits = DEFAULT_LIMITS
    if not (isinstance(limits, bool) or limits == "one-loop"):
        raise ValidationError(f"limits must be True, False or 'one-loop'; got {limits!r:.60}")
    wanted = limits is True or (limits == "one-loop" and integral.loop_count == 1)
    sym = integral.symanzik
    found = _generic_parent(integral) if wanted else None
    return landau_analysis_from_polynomial(
        sym.g,
        list(sym.lp_parameters),
        max_face_points=max_face_points,
        scale=integral.graph.energy_scale,
        timeout=timeout,
        large_face_timeout=large_face_timeout,
        total_timeout=total_timeout,
        parent=found[0].symanzik.g if found is not None else None,
        restriction=found[1] if found is not None else None,
        confirm=confirm,
        confirm_timeout=confirm_timeout,
        seed=seed,
    )


# --- one-loop closed form ----------------------------------------------------


def _require_one_loop(integral: FeynmanIntegral) -> None:
    """Raise ValueError unless the graph is connected and has exactly one loop.

    Connectivity is checked first, so a disconnected graph is refused for that
    and not for its loop count, which sums over its components.
    """
    if len(set(integral.graph._components())) > 1:
        raise ValueError("The closed form applies to connected one-loop graphs only")
    if integral.loop_count != 1:
        raise ValueError("The closed form applies to one-loop integrals only")


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
) -> dict[sp.Expr, PolyElement | sp.Expr]:
    """Each factor with the symbols of ``back`` replaced by their values.

    The images are computed in the ring of the values' symbols over the
    rationals, in SymPy's order, with the powers of each value kept, much
    faster than substituting into expressions and expanding them. A value
    that is not a polynomial over the rationals, such as m_1 / mu, a
    Landau variable's s = -m^2 (1 - x)^2 / x or one holding sqrt(2), and one
    holding a Float, which the ring would turn into a fraction, leave the
    images expanded expressions.
    """
    if not factors:
        return {}
    names = list(back)
    symbols = set().union(*(sp.sympify(value).free_symbols for value in back.values()))
    values = None
    if symbols and not any(sp.sympify(value).has(sp.Float) for value in back.values()):
        ring = PolyRing(sp.Poly(sp.Add(*symbols)).gens, sp.QQ)
        try:
            values = [ring.from_expr(back[name]) for name in names]
        except ValueError:
            values = None
    if values is None:
        return {factor: sp.expand(factor.xreplace(back)) for factor in factors}
    powers: dict[tuple[int, int], PolyElement] = {}

    def power(k: int, e: int) -> PolyElement:
        if (k, e) not in powers:
            powers[k, e] = values[k] if e == 1 else power(k, e - 1) * values[k]
        return powers[k, e]

    images: dict[sp.Expr, PolyElement | sp.Expr] = {}
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
    _require_one_loop(integral)
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
    kept: list[tuple[bool, list[PolyElement | sp.Expr]]] = []
    for (gram, _minor), factors in zip(minors, factor_lists, strict=True):
        images = [image_of[f] for f in factors]
        if not any(image == 0 for image in images):
            kept.append((gram, images))
    unique = list(dict.fromkeys(image for _, images in kept for image in images))
    if exact:
        pieces = {
            image: [
                _normalised_poly(image) if isinstance(image, PolyElement) else _normalised(image)
            ]
            for image in unique
        }
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
    _require_one_loop(integral)
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
