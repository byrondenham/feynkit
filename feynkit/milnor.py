"""
The Euler characteristic of the Milnor fibre of a hypersurface germ, through Lê numbers.

Let f be a polynomial in n + 1 variables with rational coefficients that
vanishes at a rational point p, and F its Milnor fibre at p. This module
computes the reduced Euler characteristic chi~(F) = chi(F) - 1 and the Lê
numbers of f at p, following the lectures of D. B. Massey, "Non-isolated
hypersurface singularities and Lê cycles", arXiv:1410.3312 (page numbers
below are that PDF's):

- If df(p) != 0, F is contractible and chi~(F) = 0.
- If p is an isolated critical point, chi~(F) = (-1)^n mu, mu the Milnor
  number, the length at p of the ring modulo the Jacobian ideal (Ex. 3.11,
  p. 21, with Thm 4.1, p. 22).
- Otherwise let s >= 1 be the dimension of the critical locus at p and
  z_0, ..., z_n generic linear coordinates centred at p. By Lê's attaching
  theorem (D. T. Lê, Ann. Inst. Fourier 23 (1973) 261-270; Massey, Thm 2.23,
  p. 14), F is obtained up to homotopy from the Milnor fibre of f restricted
  to the hyperplane V(z_0) by attaching (Gamma^1 . V(f))_p cells of dimension
  n, where Gamma^1 is the relative polar curve, the closure of
  V(df/dz_1, ..., df/dz_n) less the critical locus. So
  chi~(F) = chi~(F_1) + (-1)^n (Gamma^1 . V(f))_p, with f_1 = f|V(z_0), whose
  critical locus has dimension s - 1; after s steps the restriction has an
  isolated critical point.

Each step splits (Gamma^1 . V(f))_p into gamma^1 + lambda^0, the intersection
numbers of Gamma^1 with V(z_0) and V(df/dz_0) (the Teissier trick, Massey,
Ex. 3.3, p. 19). The first is the polar number of the step, the second its
Lê number lambda^0, and Massey's Ex. 4.6(1), p. 24, gives the Lê numbers of f
from those along the flag: lambda^0_f = lambda^0_{f_0} and
lambda^k_f = lambda^0_{f_k} - gamma^1_{f_{k-1}} for 1 <= k <= s, with f_k the
restriction to V(z_0, ..., z_{k-1}) and lambda^0_{f_s} its Milnor number. By
Massey's Thm 4.1, p. 22, and the formula after it on p. 23,
chi~(F) = sum_k (-1)^(n-k) lambda^k_f, which the attaching sum equals
once every Teissier split holds.

The coordinates are drawn from a seed as a unit upper triangular change of
the variables less p, with random integer coefficients below 2^28, applied
one shear per step so that the polynomial stays sparse. Generic coordinates
satisfy the conditions under which Lê and polar numbers exist (Massey,
Def. 3.8, p. 20, and Remark 3.9); each step checks those that concern it:
its polar curve has dimension at most 1 at p and meets V(z_k), V(df_k/dz_k)
and V(f_k) there in finitely many points, the critical locus loses one
dimension (Ex. 4.6(1)), and the Teissier split holds. If a check fails the
computation draws new coordinates once, then gives up with the reason, never
a guess.

Singular computes over F_p, at the largest primes below 2^29 that divide no
numerator or denominator of a coefficient of f centred at p. The numbers
over F_p equal those over Q for all but finitely many primes, so the results
are probabilistic, like those of
:func:`~feynkit.point_count.critical_point_count`. The polar curve is
saturated in the polynomial ring by repeated ideal quotients until it stops
growing; local orderings serve only for lengths of ideals that are finite at
p, where they are fast. Both public functions compute two independent flags,
one at each of the two largest such primes, which must give the same Lê and
polar numbers, and check the first against the Iomdine-Lê formula (Massey,
Ex. 4.6(2), p. 24), which :func:`milnor_fibre_euler_characteristic` can
skip. Generic coordinates give the same Lê numbers (p. 22), so
flags that disagree leave the germ undecided, even when their Euler
characteristics agree.

Every ComputationError raised for a germ the computation cannot decide has a
message that starts with "undecided: " and goes on with the reason.

The second half of the module, :func:`torus_cut_milnor_number`, takes the Milnor fibre
of a Laurent polynomial at a point of a toric variety and cuts it by the torus; see
its docstring.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import numbers
import random
import re
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import sympy as sp

from . import _exact
from .core.exceptions import ComputationError, ValidationError
from .landau import _check_timeout, _singular_binary
from .polytope import PolytopeData, polytope_data
from .toric import (
    NormalCone,
    OrbitChart,
    _completion,
    _face_key,
    _minimal_set,
    _saturated_basis,
    normal_cone,
    orbit_chart,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "LeNumbers",
    "NumberField",
    "TorusCutMilnorNumber",
    "le_numbers",
    "milnor_fibre_euler_characteristic",
    "torus_cut_milnor_number",
]

# The seconds each Singular run gets by default.
DEFAULT_TIMEOUT = 120

# Primes stay below 2^29, where Singular's prime fields work in every ring.
_PRIME_LIMIT = 2**29

# The coefficients of the random coordinates and hyperplanes lie in [1, _COEFFICIENT_LIMIT),
# below every prime used, so none vanishes modulo the prime.
_COEFFICIENT_LIMIT = 2**28

# How many sets of coordinates a flag may try before the computation gives up.
_ATTEMPTS = 2

Method = Literal["submersion", "isolated", "attaching"]


@dataclass(frozen=True)
class LeNumbers:
    """The Lê numbers of a hypersurface germ with respect to a flag of coordinates.

    Massey, "Non-isolated hypersurface singularities and Lê cycles",
    arXiv:1410.3312, Def. 3.8, p. 20, defines them; the computation follows
    Ex. 4.6(1), p. 24 (see the module docstring).

    Attributes
    ----------
    numbers
        lambda^0, ..., lambda^s at the point; (mu,) at an isolated critical
        point and () where f is a submersion.
    polar_numbers
        gamma^1 of f_0, ..., f_{s-1}, f_k the restriction of f to
        V(z_0, ..., z_{k-1}): the intersection number at the point of its
        relative polar curve with V(z_k). gamma^1 of f_0 is Massey's polar
        number gamma^1_{f,z}(0). Empty unless s >= 1.
    critical_dimension
        s, the dimension of the critical locus of f at the point; 0 at an
        isolated critical point and -1 where f is a submersion.
    flag
        The coordinates z_0, ..., z_n of the first flag, linear forms in the
        variables less their values at the point, as rows of integer
        coefficients in the order of the variables. The rows form a unit upper
        triangular matrix, and only z_0, ..., z_{s-1} differ from the
        variables. Over F_p the forms are reduced modulo ``prime``.
    prime
        The prime of the first flag; a second flag, at the next prime below
        it that divides no coefficient, gave the same numbers. None for a
        submersion, decided exactly.
    method
        "submersion" (df != 0 at the point), "isolated" (a Milnor number by a
        local standard basis) or "attaching" (Lê's attaching theorem along the
        flag).
    """

    numbers: tuple[int, ...]
    polar_numbers: tuple[int, ...]
    critical_dimension: int
    flag: tuple[tuple[int, ...], ...]
    prime: int | None
    method: Method

    @property
    def reduced_euler_characteristic(self) -> int:
        """chi~(F) = chi(F) - 1 = sum_k (-1)^(n-k) lambda^k, F the Milnor fibre and n + 1 the
        number of variables (Massey, arXiv:1410.3312, Thm 4.1, p. 22, and the formula after it
        on p. 23)."""
        n = len(self.flag) - 1
        return sum((-1) ** (n - k) * value for k, value in enumerate(self.numbers))


# --- the germ ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberField:
    """A number field Q(a), given by the minimal polynomial of a over Q.

    Points with coordinates in Q(a) are given as polynomials in ``generator`` with rational
    coefficients, together with a NumberField as the argument ``field`` of the functions of
    this module. The computation then runs over F_p(a) with a minimal polynomial. At each
    prime p it uses an irreducible factor of degree 2 or more of the reduction of the minimal
    polynomial, the whole polynomial when that stays irreducible: the residue field at one
    prime of Q(a) over p. Only primes where the reduction is squarefree are used, so p is
    unramified, and no coefficient has p in a denominator. A prime where the polynomial splits
    into factors is fine: the factor is a reduction of the same point at one prime over p, and
    the two primes must agree as for rational points. This also covers fields such as
    Q(sqrt(2), sqrt(3)), where no prime keeps a minimal polynomial of degree 4 irreducible.

    Attributes
    ----------
    minimal_polynomial
        An irreducible polynomial of degree 2 or more in ``generator`` with rational
        coefficients.
    generator
        The symbol that stands for a.

    Raises
    ------
    ValidationError
        On construction, if the generator is not a symbol, or the polynomial is not
        irreducible over Q of degree 2 or more in it with rational coefficients.
    """

    minimal_polynomial: sp.Expr
    generator: sp.Symbol

    def __post_init__(self) -> None:
        if not isinstance(self.generator, sp.Symbol):
            raise ValidationError("the generator of a number field must be a symbol")
        try:
            poly = sp.Poly(sp.sympify(self.minimal_polynomial), self.generator, domain="QQ")
        except (sp.SympifyError, sp.PolynomialError, sp.CoercionFailed, TypeError) as exc:
            raise ValidationError(
                "the minimal polynomial must be a polynomial in the generator with rational "
                "coefficients"
            ) from exc
        if poly.degree() < 2:
            raise ValidationError(
                "the minimal polynomial must have degree 2 or more; use rational coordinates "
                "for a rational point"
            )
        if not poly.is_irreducible:
            raise ValidationError("the minimal polynomial must be irreducible over Q")


@dataclass(frozen=True)
class _Field:
    """The monic minimal polynomial of a, lowest coefficient first, and a's symbol."""

    coefficients: tuple[Fraction, ...]
    symbol: sp.Symbol

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def expr(self) -> sp.Expr:
        return sum(
            (
                sp.Rational(c.numerator, c.denominator) * self.symbol**i
                for i, c in enumerate(self.coefficients)
            ),
            start=sp.Integer(0),
        )

    def reduce(self, element: sp.Expr) -> sp.Expr:
        """The remainder of a polynomial in a modulo the minimal polynomial."""
        return sp.rem(sp.expand(element), self.expr, self.symbol)

    def vector(self, element: sp.Expr) -> tuple[Fraction, ...]:
        """The coordinates of the element in the basis 1, a, a^2, ... (no trailing zeros)."""
        poly = sp.Poly(self.reduce(element), self.symbol, domain="QQ")
        coefficients = [Fraction(int(c.p), int(c.q)) for c in reversed(poly.all_coeffs())]
        while coefficients and coefficients[-1] == 0:
            coefficients.pop()
        return tuple(coefficients)

    def power(self, element: sp.Expr, exponent: int) -> sp.Expr:
        """element^exponent in the field, reduced; the element must not be zero if exponent < 0."""
        base = self.reduce(element)
        if exponent < 0:
            base, exponent = sp.invert(base, self.expr, self.symbol), -exponent
        result: sp.Expr = sp.Integer(1)
        for _ in range(exponent):
            result = self.reduce(result * base)
        return result


def _field(field: NumberField | None) -> _Field | None:
    """The data of a NumberField, or None; the argument must be one or None."""
    if field is None:
        return None
    if not isinstance(field, NumberField):
        raise ValidationError("field must be a NumberField or None")
    poly = sp.Poly(sp.sympify(field.minimal_polynomial), field.generator, domain="QQ")
    lead = poly.LC()
    coefficients = tuple(
        Fraction(int((c / lead).p), int((c / lead).q)) for c in reversed(poly.all_coeffs())
    )
    return _Field(coefficients, field.generator)


# A coefficient in a field is the vector of its coordinates in 1, a, a^2, ...
Coefficient = Fraction | tuple[Fraction, ...]


@dataclass(frozen=True)
class _Germ:
    """f centred at the point: f(at + x) as exponent vectors and coefficients.

    The coefficients are rational, or vectors of rationals (coordinates in 1, a, a^2, ...)
    when ``field`` is set; terms with a zero coefficient are dropped.
    """

    terms: tuple[tuple[tuple[int, ...], Coefficient], ...]
    variables: int
    submersion: bool
    field: _Field | None = None


def _rational(value: object) -> Fraction:
    """A coordinate of the point: an int, a Fraction or a SymPy rational, which registers
    itself as a numbers.Rational."""
    if isinstance(value, numbers.Rational) and not isinstance(value, bool):
        return Fraction(int(value.numerator), int(value.denominator))
    raise ValidationError(f"a coordinate of the point must be rational, not {value!r:.60}")


def _element(value: object, field: _Field) -> sp.Expr:
    """A coordinate in the field: a rational or a polynomial in a with rational coefficients."""
    try:
        expr = sp.sympify(value)
        if expr.free_symbols - {field.symbol}:
            raise sp.PolynomialError
        sp.Poly(expr, field.symbol, domain="QQ")
    except (sp.SympifyError, sp.PolynomialError, sp.CoercionFailed, TypeError) as exc:
        raise ValidationError(
            f"a coordinate of the point must be a polynomial in {field.symbol} with rational "
            f"coefficients, not {value!r:.60}"
        ) from exc
    return field.reduce(expr)


def _field_terms(
    expr: sp.Expr, variables: Sequence[sp.Symbol], field: _Field
) -> dict[tuple[int, ...], tuple[Fraction, ...]]:
    """The non-zero terms of a polynomial in the variables and a, with coefficients reduced
    modulo the minimal polynomial."""
    poly = sp.Poly(sp.expand(expr), *variables, field.symbol, domain="QQ")
    grouped: dict[tuple[int, ...], sp.Expr] = {}
    for monomial, c in poly.terms():
        key = tuple(int(e) for e in monomial[:-1])
        grouped[key] = grouped.get(key, sp.Integer(0)) + c * field.symbol ** int(monomial[-1])
    out = {}
    for key, value in grouped.items():
        vector = field.vector(value)
        if vector:
            out[key] = vector
    return out


def _germ(
    polynomial: object,
    variables: Sequence[sp.Symbol],
    at: Sequence[object],
    field: _Field | None = None,
) -> _Germ:
    """f(at + x) after validating the input.

    Raises
    ------
    ValidationError
        If the variables are not one or more distinct symbols, f is not a non-zero polynomial
        with rational coefficients (or, over a field, coefficients polynomial in a) in them,
        ``at`` is not one rational number (or element of the field) per variable, or f does
        not vanish at ``at``.
    """
    variables = tuple(variables)
    if (
        not variables
        or not all(isinstance(x, sp.Symbol) for x in variables)
        or len(set(variables)) != len(variables)
        or (field is not None and field.symbol in variables)
    ):
        raise ValidationError("variables must be one or more distinct symbols")
    if field is not None:
        return _field_germ(polynomial, variables, at, field)
    try:
        f = sp.sympify(polynomial)
        if f.free_symbols - set(variables):
            raise sp.PolynomialError
        poly = sp.Poly(f, *variables, domain="QQ")
    except (sp.SympifyError, sp.PolynomialError, sp.CoercionFailed, TypeError) as exc:
        raise ValidationError(
            "f must be a polynomial with rational coefficients in the variables"
        ) from exc
    if poly.is_zero:
        raise ValidationError("f must not be zero")
    if isinstance(at, (str, bytes)) or not isinstance(at, Sequence) or len(at) != len(variables):
        raise ValidationError(
            f"at must give one rational number for each of the {len(variables)} variables"
        )
    point = [_rational(value) for value in at]
    shift = {
        x: x + sp.Rational(q.numerator, q.denominator)
        for x, q in zip(variables, point, strict=True)
    }
    centred = sp.Poly(poly.as_expr().subs(shift, simultaneous=True), *variables, domain="QQ")
    terms = tuple(
        (tuple(int(e) for e in monomial), Fraction(int(c.p), int(c.q)))
        for monomial, c in centred.terms()
    )
    if any(sum(monomial) == 0 for monomial, _ in terms):
        raise ValidationError("f must vanish at the point")
    return _Germ(
        terms=terms,
        variables=len(variables),
        submersion=any(sum(monomial) == 1 for monomial, _ in terms),
    )


def _field_germ(
    polynomial: object, variables: tuple[sp.Symbol, ...], at: Sequence[object], field: _Field
) -> _Germ:
    try:
        f = sp.sympify(polynomial)
        if f.free_symbols - {*variables, field.symbol}:
            raise sp.PolynomialError
        sp.Poly(f, *variables, field.symbol, domain="QQ")
    except (sp.SympifyError, sp.PolynomialError, sp.CoercionFailed, TypeError) as exc:
        raise ValidationError(
            f"f must be a polynomial in the variables and {field.symbol} with rational coefficients"
        ) from exc
    if isinstance(at, (str, bytes)) or not isinstance(at, Sequence) or len(at) != len(variables):
        raise ValidationError(
            f"at must give one element of the field for each of the {len(variables)} variables"
        )
    point = [_element(value, field) for value in at]
    shift = {x: x + q for x, q in zip(variables, point, strict=True)}
    terms = _field_terms(f.subs(shift, simultaneous=True), variables, field)
    if not terms:
        raise ValidationError("f must not be zero")
    if any(sum(monomial) == 0 for monomial in terms):
        raise ValidationError("f must vanish at the point")
    return _Germ(
        terms=tuple(terms.items()),
        variables=len(variables),
        submersion=any(sum(monomial) == 1 for monomial in terms),
        field=field,
    )


@functools.lru_cache(maxsize=256)
def _factor_modulo(coefficients: tuple[Fraction, ...], p: int) -> tuple[int, ...] | None:
    """The irreducible factor of the minimal polynomial modulo p that the computation uses,
    as its coefficients from the lowest, monic; None if p does not suit it.

    p suits when it is no denominator of a coefficient, the reduction is squarefree, and
    some irreducible factor has degree 2 or more. The factor is the one of the greatest
    degree, the first by coefficients among those.
    """
    if any(c.denominator % p == 0 for c in coefficients):
        return None
    t = sp.Symbol("t")
    poly = sp.Poly([_residue(c, p) for c in reversed(coefficients)], t, modulus=p)
    _, factors = poly.factor_list()
    if any(m != 1 for _, m in factors):
        return None
    found = []
    for factor, _ in factors:
        if factor.degree() >= 2:
            lead = int(factor.LC()) % p
            monic = [int(c) % p * pow(lead, -1, p) % p for c in factor.all_coeffs()]
            found.append((-factor.degree(), tuple(reversed(monic))))
    if not found:
        return None
    return min(found)[1]


def _primes(germ: _Germ, count: int) -> list[int]:
    """The ``count`` largest primes below 2^29 that divide no numerator or denominator of a
    coefficient of f centred at the point and, over a field, suit its minimal polynomial."""
    values = [
        abs(n)
        for _, c in germ.terms
        for part in (c if isinstance(c, tuple) else (c,))
        for n in (part.numerator, part.denominator)
        if n
    ]
    found: list[int] = []
    p = _PRIME_LIMIT
    while len(found) < count:
        p = int(sp.prevprime(p))
        if all(v % p for v in values) and (
            germ.field is None or _factor_modulo(germ.field.coefficients, p) is not None
        ):
            found.append(p)
    return found


# --- coordinates ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Draw:
    """Random coordinates and hyperplanes for one attempt.

    shears[k] holds the coefficients c_{k,i}, i > k, of z_k = x_k + sum_i c_{k,i} x_i;
    hyperplanes are generic linear forms through the point, for dimension tests.
    """

    shears: tuple[tuple[int, ...], ...]
    hyperplanes: tuple[tuple[int, ...], ...]


def _draw(seed: int, flag: int, attempt: int, m: int) -> _Draw:
    rng = random.Random(f"milnor:{seed}:{flag}:{attempt}")
    shears = tuple(
        tuple(rng.randrange(1, _COEFFICIENT_LIMIT) for _ in range(m - 1 - k)) for k in range(m - 1)
    )
    hyperplanes = tuple(
        tuple(rng.randrange(1, _COEFFICIENT_LIMIT) for _ in range(m)) for _ in range(m)
    )
    return _Draw(shears, hyperplanes)


def _rows(draw: _Draw, m: int, used: int) -> tuple[tuple[int, ...], ...]:
    """The coordinates z_0, ..., z_{m-1}: the first ``used`` from the shears, the rest the
    variables."""
    rows = []
    for k in range(m):
        row = [0] * m
        row[k] = 1
        if k < used:
            row[k + 1 :] = draw.shears[k]
        rows.append(tuple(row))
    return tuple(rows)


# --- Singular ---------------------------------------------------------------------------------

# satq: I : h^infinity by repeated quotients in the polynomial ring, which finishes where
# elim.lib's sat and saturation in a local ordering stall. isol: whether V(I) has dimension at
# most 0 at the origin, that is, the origin is not in the closure of V(I) less the origin.
# ldim: the dimension of V(I) at the origin, the least number of generic hyperplanes through
# the origin that leave it isolated.
_PROCS = """\
proc satq(ideal I, ideal h)
{
  ideal Q = std(I);
  ideal N = std(quotient(Q, h));
  while (size(reduce(N, Q)) != 0) { Q = N; N = std(quotient(Q, h)); }
  return(Q);
}
proc isol(ideal I)
{
  ideal K = satq(I, maxideal(1));
  int k;
  for (k = 1; k <= ncols(K); k++) { if (jet(K[k], 0) != 0) { return(1); } }
  return(0);
}
proc ldim(ideal I, ideal L)
{
  if (isol(I)) { return(0); }
  ideal C = I;
  int j;
  for (j = 1; j <= ncols(L); j++) { C = C + L[j]; if (isol(C)) { return(j); } }
  return(ncols(L) + 1);
}
"""


def _residue(c: Fraction, p: int) -> int:
    return c.numerator * pow(c.denominator, -1, p) % p


def _linear(coefficients: Sequence[int], first: int) -> str:
    """sum_i c_i x(first + i) for Singular."""
    return "+".join(f"{c}*x({first + i})" for i, c in enumerate(coefficients)) or "0"


def _coefficient(c: Coefficient, p: int) -> str:
    """A coefficient modulo p as Singular reads it: an integer, or a polynomial in a."""
    if not isinstance(c, tuple):
        return str(_residue(c, p))
    pieces = [f"{_residue(x, p)}*a^{i}" if i else str(_residue(x, p)) for i, x in enumerate(c)]
    return "(" + "+".join(pieces) + ")"


def _ring(germ: _Germ, p: int, name: str, order: str) -> list[str]:
    """The declaration of a ring in x(1), ..., x(m) over F_p, or over F_p(a) with the minimal
    polynomial the prime uses."""
    m = germ.variables
    if germ.field is None:
        return [f"ring {name} = {p}, (x(1..{m})), {order};"]
    factor = _factor_modulo(germ.field.coefficients, p)
    if factor is None:
        raise ComputationError(f"undecided: the prime {p} does not suit the minimal polynomial")
    polynomial = "+".join(f"{c}*a^{i}" for i, c in enumerate(factor))
    return [f"ring {name} = ({p},a), (x(1..{m})), {order};", f"minpoly = {polynomial};"]


def _prelude(germ: _Germ, p: int, draw: _Draw) -> list[str]:
    """Rings S (global) and Lc (local) in x(1), ..., x(m), the procedures, f centred at the
    origin as F in S, the shears SH and the hyperplanes L."""
    terms = []
    for monomial, c in germ.terms:
        powers = [
            f"x({i + 1})^{e}" if e > 1 else f"x({i + 1})" for i, e in enumerate(monomial) if e
        ]
        terms.append("*".join([_coefficient(c, p), *powers]))
    shears = [_linear(row, k + 2) for k, row in enumerate(draw.shears)] or ["0"]
    return [
        *_ring(germ, p, "Lc", "ds"),
        "poly Fl; ideal Gl; ideal Jl;",
        *_ring(germ, p, "S", "dp"),
        _PROCS,
        f"poly F = {'+'.join(terms)};",
        f"ideal SH = {', '.join(shears)};",
        f"ideal L = {', '.join(_linear(row, 1) for row in draw.hyperplanes)};",
        "ideal dead = 0;",
        "int m = nvars(S); int k; int i; int s; int sk; int dg; int gv; int g1; int l0; int mu;",
        "ideal J; ideal P; ideal G; poly H;",
    ]


# One level of the flag: the critical dimension sk of f_k (printed as D), then either its
# Milnor number (I) or, after the shear that makes x(k+1) the coordinate z_k, the polar curve G
# and its intersection numbers with V(f_k), V(z_k) and V(df_k/dz_k) (P). A level that breaks a
# condition stops the run; Python reads and checks every number again.
_LEVELS = """\
for (k = 0; k < m; k++) {
  J = dead;
  for (i = k + 1; i <= m; i++) { J = J + diff(F, var(i)); }
  sk = ldim(J, L);
  print("D " + string(k) + " " + string(sk));
  if (k == 0) { s = sk; }
  if (sk != s - k) { break; }
  if (sk == 0) {
    setring Lc; Fl = imap(S, F); Jl = imap(S, dead);
    for (i = k + 1; i <= m; i++) { Jl = Jl + diff(Fl, var(i)); }
    mu = vdim(std(Jl));
    setring S;
    print("I " + string(k) + " " + string(mu));
    break;
  }
  if (k == m - 1) { break; }
  F = subst(F, var(k + 1), var(k + 1) - SH[k + 1]);
  P = dead;
  for (i = k + 2; i <= m; i++) { P = P + diff(F, var(i)); }
  G = satq(P, ideal(diff(F, var(k + 1))));
  dg = 2;
  if (isol(G)) { dg = 0; } else { if (isol(G + L[1])) { dg = 1; } }
  if (dg > 1) { print("P " + string(k) + " 2 -1 -1 -1"); break; }
  setring Lc; Fl = imap(S, F); Gl = imap(S, G);
  gv = vdim(std(Gl + Fl));
  g1 = vdim(std(Gl + var(k + 1)));
  l0 = vdim(std(Gl + diff(Fl, var(k + 1))));
  setring S;
  print("P " + string(k) + " " + string(dg) + " " + string(gv) + " " + string(g1) + " " + string(l0));
  if (gv < 0 || g1 < 0 || l0 < 0) { break; }
  F = subst(F, var(k + 1), 0);
  dead = dead + var(k + 1);
}
"""


def _flag_script(germ: _Germ, p: int, draw: _Draw) -> str:
    return "\n".join([*_prelude(germ, p, draw), _LEVELS, 'print("END");', "quit;"]) + "\n"


def _check_script(
    germ: _Germ, p: int, draw: _Draw, level: int, js: Sequence[int], last: bool
) -> str:
    """The Iomdine-Lê check at ``level``: for each j, lambda^0 of f_level + z_level^j in the
    rotated flag (z_{level+1}, ..., z_n, z_level), printed as J. At the last level of the
    flag it is a Milnor number."""
    lines = _prelude(germ, p, draw)
    for k in range(level):
        lines += [
            f"F = subst(F, var({k + 1}), var({k + 1}) - SH[{k + 1}]);",
            f"F = subst(F, var({k + 1}), 0); dead = dead + var({k + 1});",
        ]
    k = level + 1
    lines.append(f"F = subst(F, var({k}), var({k}) - SH[{k}]);")
    for j in js:
        lines.append(f"H = F + var({k})^{j};")
        if last:
            lines += [
                "setring Lc; Fl = imap(S, H); Jl = imap(S, dead);",
                f"for (i = {k}; i <= m; i++) {{ Jl = Jl + diff(Fl, var(i)); }}",
                f'mu = vdim(std(Jl)); setring S; print("J {j} " + string(mu));',
            ]
            continue
        lines += [
            f"H = subst(H, var({k + 1}), var({k + 1}) - SH[{k + 1}]);",
            f"P = dead + diff(H, var({k}));",
            f"for (i = {k + 2}; i <= m; i++) {{ P = P + diff(H, var(i)); }}",
            f"G = satq(P, ideal(diff(H, var({k + 1}))));",
            "dg = 2;",
            "if (isol(G)) { dg = 0; } else { if (isol(G + L[1])) { dg = 1; } }",
            "l0 = -1;",
            "if (dg <= 1) {",
            "  setring Lc; Fl = imap(S, H); Gl = imap(S, G);",
            f"  l0 = vdim(std(Gl + diff(Fl, var({k + 1})))); setring S;",
            "}",
            f'print("J {j} " + string(l0));',
        ]
    lines += ['print("END");', "quit;"]
    return "\n".join(lines) + "\n"


def _run(script: str, binary: str, timeout: float) -> tuple[str, bool]:
    """What Singular prints for a script, and whether it ran past ``timeout`` seconds.

    Raises
    ------
    ComputationError
        If Singular exits with an error status or reports an error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "milnor.sing"
        path.write_text(script)
        try:
            result = subprocess.run(
                [binary, "-q", "--no-warn", str(path)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or b""
            return out.decode(errors="replace") if isinstance(out, bytes) else out, True
    # Singular reports an error in the script on stdout, after a question mark.
    error = next(
        (line for line in result.stdout.splitlines() if line.lstrip().startswith("?")), None
    )
    if result.returncode != 0 or error is not None:
        reason = (
            " ".join((error or result.stderr).split())[:100] or f"exit status {result.returncode}"
        )
        raise ComputationError(f"undecided: Singular failed computing Lê numbers: {reason}")
    return result.stdout, False


_LINE = re.compile(r"([DIPJ]) (-?\d+(?: -?\d+)*)")


def _read(text: str) -> dict[str, dict[int, tuple[int, ...]]]:
    """The printed numbers by kind and level (or exponent j for J)."""
    found: dict[str, dict[int, tuple[int, ...]]] = {kind: {} for kind in "DIPJ"}
    for line in text.splitlines():
        match = _LINE.fullmatch(line.strip())
        if match:
            key, *values = (int(v) for v in match.group(2).split())
            found[match.group(1)][key] = tuple(values)
    return found


# --- one flag ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Flag:
    """The outcome of one flag: its Lê numbers, the attaching sum and lambda^0 of each f_k."""

    le: LeNumbers
    attaching: int
    lambda0: tuple[int, ...]
    draw: _Draw


@dataclass(frozen=True)
class _Levels:
    """What a run found: the critical dimension s, (gv, g1, l0) for each level k < s, the
    intersection numbers of the polar curve of f_k with V(f_k), V(z_k) and V(df_k/dz_k), and
    the Milnor number of f_s."""

    s: int
    polar: tuple[tuple[int, int, int], ...]
    mu: int


def _judge(found: dict[str, dict[int, tuple[int, ...]]], n: int) -> _Levels | str:
    """The levels of a finished run, or the condition that failed."""
    dims, polar, iso = found["D"], found["P"], found["I"]
    if 0 not in dims:
        return "Singular printed no critical dimension"
    s = dims[0][0]
    if not 0 <= s <= n:
        return f"the critical locus has dimension {s} at the point"
    levels = []
    for k in range(s + 1):
        if k not in dims:
            return f"level {k}: no critical dimension"
        if dims[k][0] != s - k:
            return (
                f"level {k}: the critical locus of the restriction has dimension {dims[k][0]}, "
                f"not {s - k}"
            )
        if k == s:
            break
        if k not in polar:
            return f"level {k}: no polar curve"
        dg, gv, g1, l0 = polar[k]
        if dg > 1:
            return f"level {k}: the polar curve has dimension 2 or more at the point"
        if min(gv, g1, l0) < 0:
            return f"level {k}: the polar curve meets V(f), V(z) or V(df/dz) in a curve"
        if gv != g1 + l0:
            return f"level {k}: the Teissier split fails, {gv} != {g1} + {l0}"
        levels.append((gv, g1, l0))
    if s not in iso or iso[s][0] < 0:
        return f"level {s}: no finite Milnor number"
    return _Levels(s, tuple(levels), iso[s][0])


def _flag(germ: _Germ, p: int, seed: int, flag: int, binary: str, timeout: float) -> _Flag:
    """Lê numbers along one flag at prime p, with a second set of coordinates if the first
    fails a condition of Def. 3.8.

    Raises
    ------
    ComputationError
        If Singular fails or runs past ``timeout``, or every set of coordinates fails.
    """
    n = germ.variables - 1
    reasons = []
    for attempt in range(_ATTEMPTS):
        draw = _draw(seed, flag, attempt, germ.variables)
        text, timed_out = _run(_flag_script(germ, p, draw), binary, timeout)
        found = _read(text)
        if timed_out:
            where = max(found["D"], default=None)
            at = "" if where is None else f" at level {where} of the flag"
            raise ComputationError(f"undecided: Singular ran past timeout={timeout} s{at}")
        result = _judge(found, n)
        if isinstance(result, str):
            reasons.append(result)
            continue
        s, levels, mu = result.s, result.polar, result.mu
        lambda0 = [l0 for _, _, l0 in levels] + [mu]
        polar = [g1 for _, g1, _ in levels]
        numbers_ = [lambda0[0]] + [lambda0[k] - polar[k - 1] for k in range(1, s + 1)]
        if min(numbers_) < 0:
            reasons.append(f"a negative Lê number {tuple(numbers_)}")
            continue
        # Lê's attaching theorem level by level. With gv = g1 + l0 at every level it equals
        # Thm 4.1's sum_k (-1)^(n-k) lambda^k term by term.
        attaching = sum((-1) ** (n - k) * gv for k, (gv, _, _) in enumerate(levels))
        attaching += (-1) ** (n - s) * mu
        le = LeNumbers(
            numbers=tuple(numbers_),
            polar_numbers=tuple(polar),
            critical_dimension=s,
            flag=_rows(draw, germ.variables, s),
            prime=p,
            method="isolated" if s == 0 else "attaching",
        )
        return _Flag(le, attaching, tuple(lambda0), draw)
    raise ComputationError(
        f"undecided: {_ATTEMPTS} flags of coordinates failed the conditions for Lê numbers ("
        + "; ".join(reasons)
        + ")"
    )


def _submersion(germ: _Germ) -> LeNumbers:
    m = germ.variables
    return LeNumbers(
        numbers=(),
        polar_numbers=(),
        critical_dimension=-1,
        flag=tuple(tuple(int(i == k) for i in range(m)) for k in range(m)),
        prime=None,
        method="submersion",
    )


def _check(germ: _Germ, p: int, found: _Flag, binary: str, timeout: float) -> None:
    """The Iomdine-Lê formula (Massey, Ex. 4.6(2), p. 24) on the flag's own coordinates.

    For f_k, with critical dimension s - k, and j > 1 + lambda^0/gamma^1 (any j >= 2 when
    gamma^1 = 0), lambda^0 of f_k + z_k^j in the rotated flag is
    lambda^0_{f_k} + (j - 1) lambda^1_{f_k}, where lambda^1_{f_k} = lambda^{k+1}_f by
    Ex. 4.6(1). At the last level, k = s - 1, the left side is a Milnor number, a cheap
    check that runs for every s >= 1. When s = 2, level 0 is checked as well if it finishes
    within ``timeout``.

    Raises
    ------
    ComputationError
        If the formula fails, or the last level runs past ``timeout``.
    """
    le = found.le
    s = le.critical_dimension
    for level in [s - 1] + ([0] if s == 2 else []):
        gamma, lam0 = le.polar_numbers[level], found.lambda0[level]
        first = 2 if gamma == 0 else lam0 // gamma + 2
        js = (first, first + 1)
        last = level == s - 1
        text, timed_out = _run(_check_script(germ, p, found.draw, level, js, last), binary, timeout)
        if timed_out:
            if last:
                raise ComputationError(
                    f"undecided: the Iomdine-Lê check at level {level} ran past timeout={timeout} s"
                )
            continue
        got = _read(text)["J"]
        for j in js:
            want = lam0 + (j - 1) * le.numbers[level + 1]
            value = got.get(j, (None,))[0]
            if value != want:
                raise ComputationError(
                    f"undecided: the Iomdine-Lê check fails at level {level}, j = {j}: lambda^0 is "
                    f"{value}, "
                    f"the Lê numbers {le.numbers} give {want}"
                )


# --- public -----------------------------------------------------------------------------------


def _validate(seed: object, timeout: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(f"seed must be an integer, not {seed!r:.60}")
    _check_timeout(timeout, optional=False)


def _binary() -> str:
    binary = _singular_binary()
    if binary is None:
        raise RuntimeError("Lê numbers at a critical point need Singular, which was not found")
    return binary


def _decide(germ: _Germ, seed: int, check: bool, timeout: float) -> _Flag:
    """The first of two flags at the two primes of the germ, which must give the same Lê and
    polar numbers, checked against the Iomdine-Lê formula if ``check``.

    Raises
    ------
    RuntimeError
        If Singular is not installed.
    ComputationError
        If either flag is undecided, the flags disagree, or the check fails.
    """
    binary = _binary()
    primes = _primes(germ, 2)
    first = _flag(germ, primes[0], seed, 0, binary, timeout)
    second = _flag(germ, primes[1], seed, 1, binary, timeout)
    a, b = first.le, second.le
    if (a.numbers, a.polar_numbers) != (b.numbers, b.polar_numbers):
        raise ComputationError(
            f"undecided: two flags give the Lê numbers {a.numbers} and {b.numbers}, polar "
            f"numbers {a.polar_numbers} and {b.polar_numbers}, modulo {primes[0]} and {primes[1]}"
        )
    if check and a.critical_dimension >= 1:
        _check(germ, primes[0], first, binary, timeout)
    return first


def le_numbers(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    at: Sequence[int | Fraction | sp.Expr],
    *,
    seed: int = 0,
    timeout: float = DEFAULT_TIMEOUT,
    field: NumberField | None = None,
) -> LeNumbers:
    """The Lê numbers of f at a rational point, with respect to random coordinates.

    For generic coordinates the Lê numbers do not depend on the coordinates
    (Massey, "Non-isolated hypersurface singularities and Lê cycles",
    arXiv:1410.3312, p. 22). Two flags, drawn from ``seed``, are computed at
    the two largest primes below 2^29 that divide no numerator or denominator
    of a coefficient of f centred at the point, one at each. They must give
    the same Lê and polar numbers: flags that disagree leave the germ
    undecided, even when their Euler characteristics agree. The first flag is
    then checked against the Iomdine-Lê formula, as
    :func:`milnor_fibre_euler_characteristic` checks it. The results are
    those of computations over F_p, exact for all but finitely many primes,
    not a certificate. The module docstring describes the method.

    Parameters
    ----------
    polynomial
        f, a polynomial with rational coefficients in ``variables``.
    variables
        The n + 1 variables of f.
    at
        The point, one rational number per variable, where f must vanish; with ``field``,
        each coordinate is a polynomial in the field's generator with rational coefficients.
    seed
        Seed of the random coordinates.
    timeout
        The most seconds each Singular run gets, at most 2,000,000.
    field
        A :class:`NumberField` Q(a) if the coordinates of the point are in it, None for a
        rational point (the default, which takes the code path and gives the numbers of a
        rational point). Then f may have coefficients that are polynomials in a, and the
        computation runs over F_p(a); see :class:`NumberField`.

    Raises
    ------
    ValidationError
        If ``variables`` are not one or more distinct symbols, f is not a
        non-zero polynomial with rational coefficients in them, ``at`` does
        not give one rational number per variable, f does not vanish there,
        ``seed`` is not an integer, or ``timeout`` is not a number of seconds
        greater than 0 and at most 2,000,000.
    RuntimeError
        If the point is critical and Singular is not installed.
    ComputationError
        If Singular fails or runs past ``timeout``, a flag fails the
        conditions under which Lê numbers exist at two sets of coordinates,
        the two flags disagree, or the Iomdine-Lê check fails; the message
        starts with "undecided: " and gives the reason.
    """
    _validate(seed, timeout)
    germ = _germ(polynomial, variables, at, _field(field))
    if germ.submersion:
        return _submersion(germ)
    return _decide(germ, seed, True, timeout).le


def milnor_fibre_euler_characteristic(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    at: Sequence[int | Fraction | sp.Expr],
    *,
    seed: int = 0,
    check: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
    field: NumberField | None = None,
) -> int:
    """chi~(F) = chi(F) - 1, F the Milnor fibre of f at a rational point where f vanishes.

    0 where df != 0; (-1)^n mu at an isolated critical point, n + 1 the
    number of variables; otherwise the sum of Lê's attaching theorem along a
    flag of random coordinates (see the module docstring). The flags are
    those of :func:`le_numbers`, computed and compared the same way. Since
    each level checks the Teissier split, the attaching sum equals
    sum_k (-1)^(n-k) lambda^k (Massey, "Non-isolated hypersurface
    singularities and Lê cycles", arXiv:1410.3312, Thm 4.1, p. 22, and
    p. 23). The result is that of computations over F_p,
    exact for all but finitely many primes, not a certificate.

    Parameters
    ----------
    polynomial
        f, a polynomial with rational coefficients in ``variables``.
    variables
        The n + 1 variables of f.
    at
        The point, one rational number per variable, where f must vanish; with ``field``,
        each coordinate is a polynomial in the field's generator with rational coefficients.
    seed
        Seed of the random coordinates; :func:`le_numbers` with the same seed
        computes the same flags.
    check
        Whether to check the first flag against the Iomdine-Lê formula
        (Massey, Ex. 4.6(2), p. 24): at the last level of the flag, where it
        is a Milnor number, whenever the critical locus has dimension s >= 1,
        and when s = 2 also at the first level, unless that run passes
        ``timeout``.
    timeout
        The most seconds each Singular run gets, at most 2,000,000.
    field
        As for :func:`le_numbers`.

    Raises
    ------
    ValidationError
        As :func:`le_numbers`.
    RuntimeError
        If the point is critical and Singular is not installed.
    ComputationError
        If Singular fails or runs past ``timeout``, a flag fails the
        conditions under which Lê numbers exist at two sets of coordinates,
        the two flags disagree, or the Iomdine-Lê check fails; the message
        starts with "undecided: " and gives the reason.
    """
    _validate(seed, timeout)
    germ = _germ(polynomial, variables, at, _field(field))
    if germ.submersion:
        return 0
    return _decide(germ, seed, check, timeout).attaching


# --- the torus cut ------------------------------------------------------------------------------

TorusMethod = Literal["smooth chart", "newton"]

_UNDECIDED = "undecided: "


@dataclass(frozen=True)
class TorusCutMilnorNumber:
    """The Milnor fibre of a Laurent polynomial at a point of a toric variety, cut by the torus.

    Let G be a Laurent polynomial whose Newton polytope P has dimension N, X_P the
    projective toric variety of P, T its open torus and x a point of the orbit O_F of a
    face F. Let g be a local equation of G at x (in a chart, g = x^(-v) G, see
    :class:`~feynkit.toric.OrbitChart`) and F_x its Milnor fibre at x, in the sense of
    Dimca, "Sheaves in Topology", Springer 2004 (Dim04), Prop. 4.2.2, p. 103. Then

        beta(x) = chi(F_x intersected with T) - 1_T(x),   mu^T(x) = (-1)^(N-1) beta(x),

    where 1_T(x) is 1 if x lies in T and 0 otherwise. At a point of T that is an isolated
    critical point of g, beta = chi~(F_x) = (-1)^(N-1) mu, so mu^T = mu there. The
    definition is for x in the closure of the zero set, where g(x) = 0. Off the closure beta
    is 0 by convention, although the displayed formula would give -1 at a point of T, where
    F_x is empty; the functions here do not take such points.

    Attributes
    ----------
    value
        mu^T(x), or None if undecided.
    beta
        beta(x), or None if undecided.
    method
        "smooth chart": the normal cone of F is smooth, so the chart is C^r x (C^*)^(d-r)
        with coordinates y, t, and beta is the signed sum over subsets I of the y
        coordinates of chi~ of the Milnor fibre of g restricted to y_I = 0 (see
        :func:`torus_cut_milnor_number`), each by Lê's attaching theorem
        (:func:`milnor_fibre_euler_characteristic`). "newton": the sum of Euler
        characteristics of orbit pieces from the Newton polyhedron of g (Matsui and
        Takeuchi, Cor. 3.6 with Rem. 3.7).
    terms
        Pairs (subset, number), sorted by subset. For "smooth chart" the subset is I,
        a tuple of indices of the y coordinates counted from 0, and the number is
        chi~(F(g|y_I = 0)), which is -1 where g restricted to y_I = 0 is zero. For
        "newton" the subset is J, the indices of the t coordinates, counted from 0, that
        are non-zero in the orbit piece, all y coordinates being non-zero, and the number
        is chi(T_J intersected with F_x), the Euler characteristic of that piece of the
        Milnor fibre. Empty if the result is undecided.
    prime
        For "smooth chart", the smallest prime of the first flag over the terms that
        needed Lê numbers (:class:`LeNumbers`); the other terms are exact. None if no term
        needed one, and for "newton", which computes over Q.
    reason
        Why the result is undecided, or None. When it is not None, value and beta are
        None.
    chart
        The chart in which the point was given and g computed (:class:`~feynkit.toric.
        OrbitChart`); None if the cone is not smooth and no chart was passed.
    """

    value: int | None
    beta: int | None
    method: TorusMethod
    terms: tuple[tuple[tuple[int, ...], int], ...]
    prime: int | None
    reason: str | None
    chart: OrbitChart | None = None


def _undecided(method: TorusMethod, reason: str) -> TorusCutMilnorNumber:
    return TorusCutMilnorNumber(None, None, method, (), None, reason)


def _laurent_terms(polynomial: object, variables: Sequence[sp.Symbol]) -> dict:
    """G as a dictionary from exponent vectors to rational coefficients.

    Raises
    ------
    ValidationError
        If the variables are not one or more distinct symbols, or G is not a non-zero Laurent
        polynomial with rational coefficients in them.
    """
    variables = tuple(variables)
    if (
        not variables
        or not all(isinstance(x, sp.Symbol) for x in variables)
        or len(set(variables)) != len(variables)
    ):
        raise ValidationError("variables must be one or more distinct symbols")
    message = (
        "the polynomial must be a Laurent polynomial with rational coefficients in the variables"
    )
    try:
        f = sp.sympify(polynomial)
        if f.free_symbols - set(variables):
            raise ValidationError(message)
        pieces = sp.Add.make_args(sp.expand(f))
    except (sp.SympifyError, TypeError) as exc:
        raise ValidationError(message) from exc
    index = {x: i for i, x in enumerate(variables)}
    out: dict[tuple[int, ...], Fraction] = {}
    for piece in pieces:
        coefficient, monomial = piece.as_independent(*variables, as_Add=False)
        if not coefficient.is_Rational:
            raise ValidationError(message)
        exponent = [0] * len(variables)
        for base, power in monomial.as_powers_dict().items():
            if base == 1:
                continue
            if base not in index or not sp.sympify(power).is_Integer:
                raise ValidationError(message)
            exponent[index[base]] += int(power)
        key = tuple(exponent)
        out[key] = out.get(key, Fraction(0)) + Fraction(int(coefficient.p), int(coefficient.q))
    out = {key: c for key, c in out.items() if c != 0}
    if not out:
        raise ValidationError("the polynomial must not be zero")
    return out


def _shifted(
    local: dict,
    k: int,
    kept: Sequence[int],
    point: Sequence[Fraction | sp.Expr],
    shift: Sequence[int],
    s: Sequence[sp.Symbol],
    y: Sequence[sp.Symbol],
) -> sp.Expr:
    """The polynomial in s and the y_j, j in kept, of g restricted to the other y_i = 0, with t
    replaced by point + s and multiplied by the unit prod_a t_a^shift_a, which clears the
    negative exponents of t."""
    total: sp.Expr = sp.Integer(0)
    for exponent, c in local.items():
        if any(e for j, e in enumerate(exponent[k:]) if j not in kept):
            continue
        term: sp.Expr = sp.Rational(c.numerator, c.denominator)
        for a in range(k):
            centre = point[a]
            if isinstance(centre, Fraction):
                centre = sp.Rational(centre.numerator, centre.denominator)
            term *= (centre + s[a]) ** (exponent[a] + shift[a])
        for j in kept:
            term *= y[j] ** exponent[k + j]
        total += term
    return sp.expand(total)


def _reduced_chi(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
    seed: int,
    timeout: float,
    field: _Field | None = None,
) -> tuple[int, int | None]:
    """chi~ of the Milnor fibre at the origin of a polynomial that vanishes there, and the
    prime of the first flag (None if no Lê numbers were needed).

    Raises
    ------
    ComputationError
        As :func:`milnor_fibre_euler_characteristic`.
    """
    if expr == 0 or (field is not None and not _field_terms(expr, variables, field)):
        # The fibre {0 = epsilon} is empty, so chi = 0 and chi~ = -1.
        return -1, None
    germ = _germ(expr, variables, [0] * len(variables), field)
    if germ.submersion:
        return 0, None
    flag = _decide(germ, seed, True, timeout)
    return flag.attaching, flag.le.prime


def _smooth_chart(
    local: dict,
    k: int,
    r: int,
    point: Sequence[Fraction | sp.Expr],
    shift: Sequence[int],
    seed: int,
    timeout: float,
    sign: int,
    field: _Field | None = None,
) -> TorusCutMilnorNumber:
    s = [sp.Symbol(f"s{a + 1}") for a in range(k)]
    y = [sp.Symbol(f"y{j + 1}") for j in range(r)]
    terms = []
    primes = []
    beta = 0
    for size in range(r + 1):
        for subset in itertools.combinations(range(r), size):
            kept = [j for j in range(r) if j not in subset]
            expr = _shifted(local, k, kept, point, shift, s, y)
            try:
                chi, prime = _reduced_chi(expr, [*s, *(y[j] for j in kept)], seed, timeout, field)
            except ComputationError as exc:
                message = str(exc)
                if not message.startswith(_UNDECIDED):
                    raise
                return _undecided(
                    "smooth chart",
                    f"restriction to y_i = 0 for i in {list(subset)}: {message[len(_UNDECIDED) :]}",
                )
            terms.append((subset, chi))
            beta += (-1) ** size * chi
            if prime is not None:
                primes.append(prime)
    return TorusCutMilnorNumber(
        sign * beta, beta, "smooth chart", tuple(terms), min(primes, default=None), None
    )


def _compact_faces(
    support: Sequence[tuple[int, ...]], deadline: float | None = None
) -> list[tuple[int, tuple[int, ...]]]:
    """The compact faces of the Newton polyhedron of a polynomial supported in the non-negative
    orthant, as (dimension, indices into support of the points on the face).

    A face of the polyhedron is compact exactly when a strictly positive weight is least on
    it. Add to the support the points p + e_i, e_i the unit vectors. The faces of the convex
    hull of the enlarged set that contain no added point are the compact faces: the points
    of a face contain p + e_i whenever they contain p and the weight vanishes on e_i,
    and contain no p when the weight is negative on e_i, while a positive weight is least on
    a point of the support alone. So the exposing weight of such a face is strictly positive,
    and conversely a strictly positive weight puts no p + e_i on its face. This is exact
    integer arithmetic.
    """
    d = len(support[0])
    added = [tuple(a + (i == j) for j, a in enumerate(p)) for p in support for i in range(d)]
    hull = polytope_data([*support, *dict.fromkeys(q for q in added if q not in set(support))])
    count = len(support)
    found = []
    for dim, idx in hull.faces:
        if deadline is not None and time.monotonic() > deadline:
            raise ComputationError("undecided: computing the compact faces ran past the timeout")
        if all(i < count for i in idx):
            found.append((dim, idx))
    return found


_NONDEGENERATE = re.compile(r"N (\d+) ([01])")


def _singular_faces(
    coefficients: dict, support: Sequence[tuple[int, ...]], faces: list, timeout: float
) -> list[int]:
    """The positions in faces whose polynomial has a singular point in the torus, so that the
    face is degenerate, decided over Q by a Gröbner basis.

    The face polynomial f_gamma is quasi-homogeneous for a positive weight, so by Euler's
    relation every common zero of its partial derivatives in the torus is a zero of f_gamma,
    and f_gamma = 0 is smooth and reduced there exactly if the partial derivatives and
    1 - z x_1 ... x_d generate the unit ideal.

    Raises
    ------
    RuntimeError
        If Singular is not installed.
    ComputationError
        If Singular fails or runs past ``timeout``.
    """
    d = len(support[0])
    binary = _binary()
    product = "*".join(f"x({i + 1})" for i in range(d))
    lines = [f"ring R = 0, (x(1..{d}), z), dp;", "poly F; ideal J;"]
    for position, (_, idx) in enumerate(faces):
        if len(idx) < 2:
            continue
        pieces = []
        for i in idx:
            c = coefficients[support[i]]
            powers = [f"x({a + 1})^{e}" for a, e in enumerate(support[i]) if e]
            pieces.append("*".join([f"({c.numerator}/{c.denominator})", *powers]))
        partials = ", ".join(f"diff(F, x({a + 1}))" for a in range(d))
        lines += [
            f"F = {'+'.join(pieces)};",
            f"J = ideal({partials}, 1 - z*{product});",
            "J = std(J);",
            f'print("N {position} " + string(reduce(1, J) == 0));',
        ]
    lines += ['print("END");', "quit;"]
    text, timed_out = _run("\n".join(lines) + "\n", binary, timeout)
    if timed_out:
        raise ComputationError(
            f"undecided: Singular ran past timeout={timeout} s checking nondegeneracy"
        )
    checked = {
        int(m.group(1)): m.group(2) == "1"
        for m in (_NONDEGENERATE.fullmatch(line.strip()) for line in text.splitlines())
        if m
    }
    needed = [p for p, (_, idx) in enumerate(faces) if len(idx) >= 2]
    if set(checked) != set(needed):
        raise ComputationError("undecided: Singular did not check every compact face")
    return [p for p in needed if not checked[p]]


def _cone_volume(points: Sequence[Sequence[int]]) -> int:
    """The normalised volume, with respect to Z^m, of the convex hull of the points and the
    origin, for points in Z^m."""
    hull = polytope_data([*points, [0] * len(points[0])])
    if hull.dimension != len(points[0]):
        raise ComputationError("a compact facet lies in a hyperplane through the origin")
    return hull.normalized_volume * math.prod(hull.smith_invariants)


def _newton(
    local: dict,
    k: int,
    r: int,
    point: Sequence[Fraction],
    shift: Sequence[int],
    timeout: float,
    sign: int,
) -> TorusCutMilnorNumber:
    s = [sp.Symbol(f"s{a + 1}") for a in range(k)]
    y = [sp.Symbol(f"y{j + 1}") for j in range(r)]
    expr = _shifted(local, k, range(r), point, shift, s, y)
    poly = sp.Poly(expr, *s, *y, domain="QQ")
    coefficients = {
        tuple(int(e) for e in monomial): Fraction(int(c.p), int(c.q))
        for monomial, c in poly.terms()
    }
    support = sorted(coefficients)
    try:
        faces = _compact_faces(support, time.monotonic() + timeout)
    except ComputationError as exc:
        message = str(exc)
        if not message.startswith(_UNDECIDED):
            raise
        return _undecided("newton", message[len(_UNDECIDED) :])
    try:
        degenerate = _singular_faces(coefficients, support, faces, timeout)
    except ComputationError as exc:
        message = str(exc)
        if not message.startswith(_UNDECIDED):
            raise
        return _undecided("newton", f"nondegeneracy not decided: {message[len(_UNDECIDED) :]}")
    if degenerate:
        shown = [support[i] for i in faces[degenerate[0]][1]]
        return _undecided(
            "newton",
            f"nondegeneracy not verified: the compact face with exponents {shown} of the Newton "
            f"polyhedron has a singular point in the torus ({len(degenerate)} such faces)",
        )
    terms = []
    total = 0
    for size in range(k + 1):
        for subset in itertools.combinations(range(k), size):
            active = [*subset, *range(k, k + r)]
            m = len(active)
            chi = 0
            if m:
                outside = [c for c in range(k + r) if c not in active]
                volume = 0
                for dim, idx in faces:
                    if dim == m - 1 and all(support[i][c] == 0 for i in idx for c in outside):
                        volume += _cone_volume([[support[i][c] for c in active] for i in idx])
                chi = (-1) ** (m - 1) * volume
            terms.append((subset, chi))
            total += chi
    beta = total - (1 if r == 0 else 0)
    return TorusCutMilnorNumber(sign * beta, beta, "newton", tuple(terms), None, None)


def _check_chart(data: PolytopeData, key: tuple[int, ...], base: NormalCone, chart: object) -> None:
    """Whether chart is an OrbitChart of the face for data.

    Raises
    ------
    ValidationError
        If it is not.
    """
    if not isinstance(chart, OrbitChart) or chart.face != key:
        raise ValidationError("chart must be an OrbitChart of the face")
    d = base.lattice_dimension
    torus = _completion(_saturated_basis(base.rays, d), d)
    vertex = tuple(data.chart.coordinates[min(set(key) & set(data.vertex_indices))])
    if chart.torus_basis != tuple(tuple(w) for w in torus) or chart.vertex != vertex:
        raise ValidationError("chart is not a chart of this polytope and face")
    rows = [list(w) for w in chart.torus_basis] + [list(u) for u in chart.rays]
    if len(rows) != d or abs(_exact.determinant(rows)) != 1:
        raise ValidationError("the rays and torus basis of chart do not form a basis of Z^d")
    if any(not set(key) <= _minimal_set(data, u) for u in chart.rays):
        raise ValidationError("a ray of chart is not in the normal cone of the face")


def torus_cut_milnor_number(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    face: Sequence[Sequence[int]],
    coordinates: Sequence[int | Fraction | sp.Rational],
    *,
    seed: int = 0,
    timeout: float = DEFAULT_TIMEOUT,
    method: TorusMethod = "smooth chart",
    chart: OrbitChart | None = None,
    field: NumberField | None = None,
) -> TorusCutMilnorNumber:
    """beta and mu^T at a point of the orbit of a face, in a smooth chart.

    See :class:`TorusCutMilnorNumber` for the definitions and the sign: with N the
    dimension of the Newton polytope P of G, beta(x) = chi(F_x intersected with T) - 1_T(x)
    and mu^T(x) = (-1)^(N-1) beta(x).

    The normal cone of the face must be smooth (:func:`~feynkit.toric.normal_cone`), or a
    chart of a smooth piece of its subdivision must be passed as ``chart``. Then
    :func:`~feynkit.toric.orbit_chart` gives a chart C^r_y x (C^*)^(d-r)_t of X_P around the
    orbit, d = N, and g(t, y) = x^(-v) G. The point x is (t, y) = (``coordinates``, 0).
    With a cone that is not smooth and no ``chart``, the result is undecided.

    "smooth chart" (the default). T is the set where no y_i vanishes, and near x the
    coordinates t do not vanish. The stalk at x of the nearby cycles of g on the constant
    sheaf of a closed analytic set Z is the cohomology of F_x intersected with Z (Dim04,
    Def. 4.2.1 and Prop. 4.2.2, p. 103), and Matsui and Takeuchi, arXiv:0809.3148
    (MT11), (3.7)-(3.8), p. 10, write chi(F_x intersected with a locally closed orbit) as
    the Euler characteristic of that stalk. Inclusion and exclusion over the closed sets
    {y_I = 0}, I a subset of the r coordinates y, writes the indicator function of T as
    a signed sum of their indicator functions, and Euler characteristics add, so

        beta(x) = sum over I of (-1)^|I| chi~(F_x(g|y_I = 0)),

    with chi~ = chi - 1 the reduced Euler characteristic. Each term is
    :func:`milnor_fibre_euler_characteristic` of g restricted to y_I = 0 as a germ in the
    remaining t and y; where the restriction is the zero polynomial the fibre is empty and
    chi~ = -1. Where every restriction has an isolated critical point at x, mu^T(x) is the
    sum of their Milnor numbers.

    "newton". The same numbers from the Newton polyhedron, by MT11 Cor. 3.6 with Rem. 3.7
    (arXiv pp. 9-11). Hypotheses (MT11 Section 3, p. 8, Def. 3.2, p. 9): S is a finitely
    generated subsemigroup of Z^n containing 0 whose cone K(S) is strongly convex of
    dimension n, here S = N^d; f in C[S] is non-zero and vanishes at the fixed point 0,
    that is 0 is not in its support; and f is non-degenerate, which means that for every
    compact face gamma of the Newton polyhedron Gamma_+(f), the hypersurface
    {L(f_gamma) = 0} of the torus (C^*)^n is smooth and reduced. Rem. 3.7 moves the point
    to the fixed point: g is multiplied by a unit monomial in t, and t is translated by
    ``coordinates`` so that f(s, y) is a polynomial that vanishes at 0, and its toric
    structure is that of C^d with coordinates (s, y). For each face Delta of the orthant,
    that is a subset of the d coordinates that are not 0, Cor. 3.6 gives

        chi(T_Delta intersected with F_x) = (-1)^(dim Delta - 1) sum_i Vol_Z(Gamma_i),

    summed over the compact faces gamma_i of Gamma_+(f) inside Delta of dimension
    dim Delta - 1, Gamma_i the convex hull of gamma_i and 0, Vol_Z the volume normalised for
    Z^Delta (so (dim Delta)! times the Euclidean volume); the sum is 0 if Gamma_+(f) misses Delta. The y_i are
    non-zero exactly on the part of the fibre in T, so beta is the sum of these over the
    subsets of t coordinates, less 1_T(x). Non-degeneracy is verified exactly for every
    compact face with at least two points, over Q: its partial derivatives and
    1 - z x_1 ... x_d must generate the unit ideal (the face polynomial is
    quasi-homogeneous for a positive weight, so by Euler's relation a common zero of the
    partial derivatives in the torus is a zero of the polynomial). The compact faces are
    those faces of the convex hull of the support that a strictly positive weight selects,
    found in exact integer arithmetic. If a face is degenerate, the result is
    undecided: Cor. 3.6 does not apply, though the germ may still have a value.

    Parameters
    ----------
    polynomial
        G, a Laurent polynomial with rational coefficients in ``variables``, non-zero.
    variables
        The n variables of G.
    face
        The exponent vectors, in the order of ``variables``, of all the terms of G on a
        face of its Newton polytope P (the face itself, so every term that lies on it);
        the whole of P for the top face.
    coordinates
        The point x of the orbit, as the d - r non-zero rational values of the torus
        coordinates t_1, ..., t_(d-r) of the chart; none for a vertex. The convention is
        that of :class:`~feynkit.toric.OrbitChart`, which this function uses as it is
        returned by ``orbit_chart(data, face)`` for ``data = polytope_data(points)``,
        ``points`` the sorted exponent vectors of the terms of G. Let c(alpha) be the
        coordinates of an exponent alpha in the lattice chart of ``data`` (``data.chart``),
        v the chart's vertex and w_1, ..., w_(d-r) its ``torus_basis``. A term with
        exponent alpha contributes t_1^<w_1, c(alpha) - v> ... t_(d-r)^<w_(d-r), c(alpha) - v>
        times the y monomial given by the rays. So t_a is the character of the lattice
        spanned by the exponents whose chart coordinates are the a-th vector m_a of the
        basis dual to (w_1, ..., w_(d-r), u_1, ..., u_r); it is a monomial x^e in the
        variables only up to the chart basis ``data.chart.basis`` (e = sum_j (m_a)_j b_j),
        and is x_a itself only when that basis is the identity. Another chart of the same
        orbit, for instance one from ``orbit_chart(data, face, cone=piece)`` or the same
        chart written out, can be passed as ``chart``.
    chart
        An :class:`~feynkit.toric.OrbitChart` of the face for ``polytope_data(points)`` with
        a smooth cone, in which ``coordinates`` are given; its rays, torus basis and
        vertex must be those of a chart of this polytope and face, and the torus basis that
        of ``orbit_chart(data, face)``. If None, ``orbit_chart(data, face)``, which needs the
        normal cone of the face to be smooth. The chart used is returned in the result.
    seed
        Seed of the random coordinates of the Lê computations.
    timeout
        The most seconds each Singular run gets, at most 2,000,000.
    method
        "smooth chart" or "newton".
    field
        A :class:`NumberField` Q(a) if ``coordinates`` are in it, each a polynomial in the
        field's generator with rational coefficients; None, the default, for rational
        coordinates, which take the code path and give the numbers they always did. The
        Lê computations then run over F_p(a) (see :class:`NumberField`) and the result holds
        for the point whose coordinates are those polynomials in a root of the minimal
        polynomial. Only the "smooth chart" method takes a field; "newton" is undecided for it.

    Returns
    -------
    TorusCutMilnorNumber
        With ``reason`` set and value None if a Lê computation or the nondegeneracy check is
        undecided, ran past ``timeout``, or the cone is not smooth. The two-flag agreement of
        :func:`le_numbers` is part of every Lê computation. A check against the Iomdine-Lê
        formula that :func:`milnor_fibre_euler_characteristic` skips on a timeout at level 0
        is skipped here as well, and not reported.

    Raises
    ------
    ValidationError
        If ``polynomial`` is not a non-zero Laurent polynomial with rational coefficients
        in distinct ``variables``, ``face`` is not the set of terms on a face of P,
        ``coordinates`` is not d - r non-zero rationals, the face polynomial does not vanish
        at them (the point is not in the closure of V(G), and the numbers are not defined
        there: by convention beta is 0 off the closure, which is not computed here, while
        the displayed formula would give -1 at a point of T), ``chart`` is not a chart of
        the face, ``seed`` is not an integer,
        ``timeout`` is not a number of seconds greater than 0 and at most 2,000,000, or
        ``method`` is unknown.
    RuntimeError
        If Singular is needed and not installed.
    ComputationError
        Only for a failure that is not a condition on the germ, which would be a bug.
    """
    _validate(seed, timeout)
    if method not in ("smooth chart", "newton"):
        raise ValidationError('method must be "smooth chart" or "newton"')
    fld = _field(field)
    if fld is not None and (
        fld.symbol in tuple(variables) or re.fullmatch(r"[sy]\d+", fld.symbol.name)
    ):
        raise ValidationError("the generator of the field must not be one of the variables")
    terms = _laurent_terms(polynomial, variables)
    points = sorted(terms)
    data = polytope_data(points)
    position = {p: i for i, p in enumerate(data.points)}
    if isinstance(face, (str, bytes)) or not isinstance(face, Sequence):
        raise ValidationError("face must be a sequence of exponent vectors of terms of G")
    try:
        indices = [position[tuple(_exact._as_int(c, "an exponent") for c in p)] for p in face]
    except (KeyError, TypeError, ValidationError) as exc:
        raise ValidationError("face must be a sequence of exponent vectors of terms of G") from exc
    key = _face_key(data, indices)
    base = normal_cone(data, key)
    if chart is None:
        if not base.smooth:
            return _undecided(
                method,
                f"the normal cone of the face {list(key)} is not smooth: the chart is not "
                "C^r x (C^*)^(d-r) and a smooth subdivision of the cone is needed",
            )
        chart = orbit_chart(data, key)
    else:
        _check_chart(data, key, base, chart)
    k, r = chart.torus_dimension, chart.normal_dimension
    if (
        isinstance(coordinates, (str, bytes))
        or not isinstance(coordinates, Sequence)
        or len(coordinates) != k
    ):
        raise ValidationError(f"coordinates must give {k} rational numbers, one for each t")
    point: list[Fraction] | list[sp.Expr]
    if fld is None:
        point = [_rational(c) for c in coordinates]
        zero = [c == 0 for c in point]
    else:
        point = [_element(c, fld) for c in coordinates]
        zero = [not fld.vector(c) for c in point]
    if any(zero):
        raise ValidationError("the coordinates t must not be zero")
    local = chart.local_equation(
        {tuple(data.chart.coordinates[i]): terms[data.points[i]] for i in range(len(points))}
    )
    sign = (-1) ** (data.dimension - 1)
    if fld is None:
        value_at_x = sum(
            (
                c
                * math.prod(
                    (t**e for t, e in zip(point, exponent[:k], strict=True)), start=Fraction(1)
                )
                for exponent, c in local.items()
                if not any(exponent[k:])
            ),
            start=Fraction(0),
        )
        vanishes = value_at_x == 0
    else:
        total: sp.Expr = sp.Integer(0)
        for exponent, c in local.items():
            if any(exponent[k:]):
                continue
            product: sp.Expr = sp.Rational(c.numerator, c.denominator)
            for t, e in zip(point, exponent[:k], strict=True):
                product = fld.reduce(product * fld.power(t, e))
            total += product
        vanishes = not fld.vector(total)
    if not vanishes:
        raise ValidationError(
            "the face polynomial does not vanish at coordinates: the point is not in the "
            "closure of V(G)"
        )
    shift = [max(0, -min(e[a] for e in local)) for a in range(k)]
    if method == "newton":
        if fld is not None:
            return _undecided(
                method, "the newton method takes rational points only, not a point over a field"
            )
        result = _newton(local, k, r, point, shift, timeout, sign)
    else:
        result = _smooth_chart(local, k, r, point, shift, seed, timeout, sign, fld)
    return dataclasses.replace(result, chart=chart)
