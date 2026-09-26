"""
Finite-field point counts of V = {G = 0} in the torus.

Let X be the complement of V = {G = 0} in the torus (C^*)^N. The number of
master integrals, subsectors included, symmetries unused and D symbolic, is
C = (-1)^N chi(X) (Bitoun, Bogner, Klausen and Panzer, arXiv:1712.09215,
Corollary 37). For generic exponents |chi(X)| is also the number of critical
points of sum_e nu_e log u_e - (D/2) log G on X, all of them regular (Fevola,
Mizera and Telen, Comput. Phys. Commun. 303 (2024) 109278, proof of
Theorem 3.1, after Huh 2013), and |chi(X)| <= N! Vol(Newt G) (Bitoun et al.,
Theorem 44, crediting Kouchnirenko).

If #V(F_q) = P(q) for every finite field F_q whose characteristic avoids a
finite set, then chi(V) = P(1) (Katz, appendix to Hausel and
Rodriguez-Villegas, Invent. Math. 174 (2008) 555, Theorem 6.1.2(3)), and
chi(X) = -P(1), since chi is additive and vanishes on the torus.
count_torus_points counts #V(F_p) for finitely many primes p at one rational
kinematic point, fits a polynomial exactly and checks it at further primes.
Its result is a candidate, never a proof: Katz's theorem needs every finite
field of all but finitely many characteristics, which no finite sample
establishes.

G has degree at most 2 in some variable x (for a Feynman graph, in every
u_e), so each count is a sum over the other variables of the number of roots
of a x^2 + b x + c in F_p^*, read off a table of Legendre symbols.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import sympy as sp

from .core.exceptions import ValidationError
from .landau import LandauAnalysis, _factor_list, landau_analysis_from_polynomial
from .polytope import polytope_data

__all__ = ["TorusCount", "count_torus_points"]

# A polynomial as (exponent vector, coefficient) pairs with non-zero coefficients.
_Terms = tuple[tuple[tuple[int, ...], Fraction], ...]
# The coefficients a, b and c of x^2, x and 1, each keyed by the exponents of the other variables.
_Parts = tuple[
    dict[tuple[int, ...], Fraction],
    dict[tuple[int, ...], Fraction],
    dict[tuple[int, ...], Fraction],
]
# The numpy backend evaluates at most this many points at once, unless one grid of the
# last two other variables, (p - 1)^2 points, is larger, as for p > 2049: then each chunk
# holds one grid.
_CHUNK = 2**22
# Invariants are drawn from the non-zero integers of [-20, 20], in increasing order.
_INVARIANTS = tuple(k for k in range(-20, 21) if k != 0)
# The square test looks at this many admissible draws.
_ADMISSIBLE_DRAWS = 200
# Draws tried before giving up on finding an admissible one.
_MAX_DRAWS = 10_000
_BACKENDS = ("numpy",)


@dataclass(frozen=True)
class TorusCount:
    """Point counts of V = {G = 0} in the torus over finite fields, and what they suggest.

    Every ``candidate_*`` field is a candidate, not a result: the counts fit a
    polynomial on finitely many primes, while Katz's theorem needs every finite
    field of all but finitely many characteristics.

    Attributes
    ----------
    variables
        The variables u_1, ..., u_N of G.
    eliminated
        The variable solved for, the first of lowest degree in G.
    point
        The kinematic point as (quantity, value) pairs in the order of the symbol
        names. A symbol that occurs in G only to even powers, such as a mass m_e,
        is given by its square m_e**2.
    on_shell
        The substitutions made in the momentum products before counting, as
        (symbol, value) pairs; empty for :func:`count_torus_points`.
    seed
        The seed the point was drawn with, or None when the caller gave it.
    on_landau_surface
        Whether a coefficient of G or a face discriminant vanishes at the point,
        which only a given point with ``allow_singular`` allows; C may be smaller
        there than for generic kinematics.
    excluded_primes
        2 and the primes up to ``max_prime`` that divide the numerator or the
        denominator of a coefficient of G, of a non-zero face discriminant or of
        a non-zero irreducible factor of one, at the point.
    fit_primes, verification_primes
        The primes the fit used and the primes it was checked at, in order.
    counts
        (p, #V(F_p)) for every prime counted, in order.
    candidate_polynomial
        The coefficients of P(q), constant term first, N of them; None when the
        counts are not polynomial on the tested primes.
    candidate_euler_characteristic
        chi(X) = -P(1), or None.
    candidate_master_count
        C = (-1)^N chi(X), or None.
    reason
        Why the counts are not polynomial on the tested primes; None when there
        is a candidate.
    skipped_faces
        How many faces the Landau analysis skipped; their discriminants are
        missing from the admissibility test and from the excluded primes.
    backend
        The counting backend, "numpy" or "flint".
    """

    variables: tuple[sp.Symbol, ...]
    eliminated: sp.Symbol
    point: tuple[tuple[sp.Expr, Fraction], ...]
    on_shell: tuple[tuple[sp.Symbol, sp.Expr], ...]
    seed: int | None
    on_landau_surface: bool
    excluded_primes: tuple[int, ...]
    fit_primes: tuple[int, ...]
    verification_primes: tuple[int, ...]
    counts: tuple[tuple[int, int], ...]
    candidate_polynomial: tuple[int, ...] | None
    candidate_euler_characteristic: int | None
    candidate_master_count: int | None
    reason: str | None
    skipped_faces: int
    backend: str


def _legendre(d: Fraction, p: int) -> int:
    """The Legendre symbol (d/p) of a rational d; 0 when p divides the numerator of d.

    p must be an odd prime that does not divide the reduced denominator of d;
    when it does, (d/p) is undefined and the result is 0.
    """
    residue = d.numerator * d.denominator % p
    if residue == 0:
        return 0
    return 1 if pow(residue, (p - 1) // 2, p) == 1 else -1


def _split(terms: _Terms, k: int) -> _Parts:
    """The coefficients a, b, c of x^2, x and 1, x being variable k of degree at most 2.

    A term of higher degree in x raises ValueError.
    """
    a: dict[tuple[int, ...], Fraction] = {}
    b: dict[tuple[int, ...], Fraction] = {}
    c: dict[tuple[int, ...], Fraction] = {}
    for monomial, coefficient in terms:
        if monomial[k] > 2:
            raise ValueError(f"the term {monomial} has degree {monomial[k]} in variable {k}")
        (a, b, c)[2 - monomial[k]][monomial[:k] + monomial[k + 1 :]] = coefficient
    return a, b, c


def _modular(part: dict[tuple[int, ...], Fraction], p: int) -> dict[tuple[int, ...], int]:
    """The non-zero residues mod p of the coefficients; p must not divide a denominator."""
    residues = {}
    for monomial, coefficient in part.items():
        residue = coefficient.numerator * pow(coefficient.denominator, -1, p) % p
        if residue:
            residues[monomial] = residue
    return residues


def _roots(a: int, b: int, c: int, p: int) -> int:
    """The number of roots in F_p^* of a x^2 + b x + c, the coefficients residues mod p."""
    if a:
        return 1 + _legendre(Fraction(b * b - 4 * a * c), p) - (c == 0)
    if b:
        return int(c != 0)
    return (p - 1) * (c == 0)


def _count_numpy(parts: _Parts, others: int, p: int) -> int:
    """#V(F_p) in the torus: the roots in F_p^* of a x^2 + b x + c, summed over the others.

    p must be an odd prime that divides no denominator of the coefficients.
    ``others`` is the number of other variables. The last two, or the only
    one, form a grid evaluated by broadcasting, with a, b and c grouped by
    their exponents there. The variables before them run in chunks of at most
    _CHUNK points, or of one grid when that is larger, (p - 1)^2 points for
    p > 2049. int64 arithmetic is exact up to p of about 10^9 when G has
    degree at most 2 in the grid variables; the comment in the loop gives the
    bound.
    """
    a, b, c = (_modular(part, p) for part in parts)
    if others == 0:
        return _roots(a.get((), 0), b.get((), 0), c.get((), 0), p)
    legendre = -np.ones(p, dtype=np.int64)
    legendre[0] = 0
    legendre[np.arange(1, p, dtype=np.int64) ** 2 % p] = 1
    grid = min(others, 2)
    outer = others - grid
    top = max([e for part in (a, b, c) for monomial in part for e in monomial] + [0])
    base = np.arange(1, p, dtype=np.int64)
    powers = [np.ones(p - 1, dtype=np.int64)]
    for _ in range(top):
        powers.append(powers[-1] * base % p)

    def grid_monomial(exponents: tuple[int, ...]) -> np.ndarray:
        if grid == 1:
            return powers[exponents[0]]
        return powers[exponents[0]][:, None] * powers[exponents[1]][None, :] % p

    grouped: list[list[tuple[np.ndarray, dict[tuple[int, ...], int]]]] = []
    for part in (a, b, c):
        groups: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
        for monomial, coefficient in part.items():
            groups.setdefault(monomial[outer:], {})[monomial[:outer]] = coefficient
        grouped.append([(grid_monomial(g), terms) for g, terms in groups.items()])

    size = max(1, _CHUNK // (p - 1) ** grid)
    total_outer = (p - 1) ** outer
    total = 0
    for start in range(0, total_outer, size):
        index = np.arange(start, min(total_outer, start + size), dtype=np.int64)
        digits = [index // (p - 1) ** t % (p - 1) for t in range(outer)]
        # Reduction mod p is deferred: weight is reduced once per group and value once per
        # part. Every term is below p, so weight stays below T p for the T outer terms of a
        # group, and value below E (p - 1)^2 for the E grid exponents of a part, where
        # E <= (d + 1)^2 for degree d in the grid variables and E <= 9 for a Feynman graph.
        # int64 holds both exactly while T p and E (p - 1)^2 stay below 2^63: for E = 9, up
        # to p of about 10^9, far above the default max_prime of 1000.
        values = []
        for groups_of_part in grouped:
            value = np.zeros((len(index),) + (p - 1,) * grid, dtype=np.int64)
            for grid_values, terms in groups_of_part:
                weight = np.zeros(len(index), dtype=np.int64)
                for exponents, coefficient in terms.items():
                    term = np.full(len(index), coefficient, dtype=np.int64)
                    for digit, e in zip(digits, exponents, strict=True):
                        if e:
                            term = term * powers[e][digit] % p
                    weight += term
                value += (weight % p).reshape((-1,) + (1,) * grid) * grid_values
            values.append(value % p)
        av, bv, cv = values
        zero_c = cv == 0
        delta = (bv * bv - 4 * av * cv) % p
        roots = np.where(
            av != 0,
            1 + legendre[delta] - zero_c,
            np.where(bv != 0, ~zero_c, (p - 1) * zero_c),
        )
        total += int(roots.sum())
    return total


def _interpolate(xs: Sequence[int], ys: Sequence[int]) -> list[Fraction]:
    """The coefficients, constant term first, of the polynomial of degree below len(xs) that
    takes the value ys[i] at xs[i]; exact Lagrange interpolation.

    xs and ys must have the same length and the xs must be distinct; otherwise
    ValueError is raised.
    """
    n = len(xs)
    if len(ys) != n:
        raise ValueError(f"xs and ys differ in length: {n} and {len(ys)}")
    if len(set(xs)) != n:
        raise ValueError(f"the xs {list(xs)} are not distinct")
    coefficients = [Fraction(0)] * n
    for i in range(n):
        basis = [Fraction(1)]
        denominator = 1
        for j in range(n):
            if j == i:
                continue
            basis = [Fraction(0), *basis]
            for k in range(len(basis) - 1):
                basis[k] -= xs[j] * basis[k + 1]
            denominator *= xs[i] - xs[j]
        for k in range(n):
            coefficients[k] += ys[i] * basis[k] / denominator
    return coefficients


# --- the kinematic point -------------------------------------------------------


def _terms(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> _Terms:
    """expr as a polynomial in symbols with rational coefficients."""
    expr = sp.expand(expr)
    if not symbols:
        if not expr.is_Rational:
            raise ValidationError(f"{expr} is not a rational number")
        return (((), Fraction(int(expr.p), int(expr.q))),) if expr != 0 else ()
    try:
        poly = sp.Poly(expr, *symbols, domain="QQ")
    except (sp.PolynomialError, sp.CoercionFailed) as exc:
        names = ", ".join(map(str, symbols))
        raise ValidationError(f"{expr} is not a polynomial in {names}") from exc
    return tuple(
        (tuple(int(e) for e in monomial), Fraction(int(c.p), int(c.q)))
        for monomial, c in poly.terms()
        if c != 0
    )


def _evaluate(terms: _Terms, values: Sequence[Fraction]) -> Fraction:
    total = Fraction(0)
    for monomial, coefficient in terms:
        term = coefficient
        for value, exponent in zip(values, monomial, strict=True):
            if exponent:
                term *= value**exponent
        total += term
    return total


def _is_square(value: Fraction) -> bool:
    """Whether value is the square of a non-zero rational."""
    if value <= 0:
        return False
    return all(math.isqrt(n) ** 2 == n for n in (value.numerator, value.denominator))


@dataclass(frozen=True)
class _Kinematics:
    """G, its face discriminants and their factors as polynomials in the drawn quantities.

    ``symbols`` are the kinematic symbols in name order. A symbol in ``even``
    occurs in G only to even powers and is drawn through its square, so the
    polynomials here are written in those squares. ``discriminants`` holds the
    numerator and denominator of every face discriminant that is not constant,
    ``factors`` their distinct irreducible factors, and ``square_factors`` the
    factors from faces of dimension at least 1 with principal discriminants,
    which the square test asks to be squares.
    """

    symbols: tuple[sp.Symbol, ...]
    even: frozenset[sp.Symbol]
    coefficients: tuple[tuple[tuple[int, ...], _Terms], ...]
    discriminants: tuple[tuple[_Terms, _Terms], ...]
    factors: tuple[_Terms, ...]
    square_factors: tuple[_Terms, ...]


def _degrees(expr: sp.Expr, symbol: sp.Symbol) -> list[int]:
    try:
        return [d for (d,) in sp.Poly(expr, symbol).monoms()]
    except sp.PolynomialError as exc:
        raise ValidationError(
            f"G is not a polynomial in {symbol}; pass the energy scale as scale"
        ) from exc


def _kinematics(
    g: sp.Expr,
    variables: Sequence[sp.Symbol],
    landau: LandauAnalysis,
    unit: Mapping[sp.Symbol, int],
) -> _Kinematics:
    symbols = tuple(sorted(g.free_symbols - set(variables), key=lambda x: x.name))
    even = frozenset(x for x in symbols if all(d % 2 == 0 for d in _degrees(g, x)))
    squares = {x: sp.Dummy(f"{x.name}_squared") for x in even}
    drawn = [squares.get(x, x) for x in symbols]
    roots = {x: sp.sqrt(square) for x, square in squares.items()}

    def rewrite(expr: sp.Expr) -> sp.Expr:
        return sp.expand(sp.sympify(expr).subs(unit).subs(roots))

    coefficients = tuple(
        (tuple(int(e) for e in monomial), _terms(rewrite(c), drawn))
        for monomial, c in sp.Poly(g, *variables).terms()
    )
    discriminants: list[tuple[_Terms, _Terms]] = []
    factors: dict[_Terms, None] = {}
    square_factors: dict[_Terms, None] = {}
    for face in landau.face_discriminants:
        numerator, denominator = sp.fraction(sp.together(sp.sympify(face.discriminant).subs(unit)))
        numerator, denominator = rewrite(numerator), rewrite(denominator)
        if not numerator.free_symbols:
            continue
        discriminants.append((_terms(numerator, drawn), _terms(denominator, drawn)))
        for base in _factor_list(numerator, set(drawn)):
            factor = _terms(base, drawn)
            factors.setdefault(factor, None)
            if face.dimension >= 1 and face.principal:
                square_factors.setdefault(factor, None)
    return _Kinematics(
        symbols, even, coefficients, tuple(discriminants), tuple(factors), tuple(square_factors)
    )


def _key(x: sp.Symbol, even: frozenset[sp.Symbol]) -> sp.Expr:
    return x**2 if x in even else x


def _rational(raw: object, key: sp.Expr) -> Fraction:
    value = sp.sympify(raw)
    if not value.is_Rational:
        raise ValidationError(f"point gives {key} the value {raw!r}, which is not rational")
    return Fraction(int(value.p), int(value.q))


def _given_point(
    point: Mapping[sp.Expr, int | Fraction], kinematics: _Kinematics
) -> tuple[Fraction, ...]:
    """The drawn quantities at a point the caller gave: a symbol's value, or its square's."""
    even = kinematics.even
    values: dict[sp.Symbol, Fraction] = {}
    for raw_key, raw in point.items():
        key = sp.sympify(raw_key)
        if key in kinematics.symbols:
            value = _rational(raw, key)
            values[key] = value**2 if key in even else value
        elif isinstance(key, sp.Pow) and key.exp == 2 and key.base in even:
            values[key.base] = _rational(raw, key)
            if values[key.base] < 0:
                raise ValidationError(f"point gives {key} the negative value {raw}")
        else:
            expected = ", ".join(str(_key(x, even)) for x in kinematics.symbols) or "none"
            raise ValidationError(f"point has the key {key}; the keys are {expected}")
    missing = [str(_key(x, even)) for x in kinematics.symbols if x not in values]
    if missing:
        raise ValidationError(f"point gives no value for {', '.join(missing)}")
    return tuple(values[x] for x in kinematics.symbols)


def _admissible(kinematics: _Kinematics, values: Sequence[Fraction]) -> bool:
    """Whether every coefficient of G and every face discriminant is non-zero at values."""
    if any(_evaluate(terms, values) == 0 for _, terms in kinematics.coefficients):
        return False
    return all(
        _evaluate(numerator, values) != 0 and _evaluate(denominator, values) != 0
        for numerator, denominator in kinematics.discriminants
    )


def _draw(kinematics: _Kinematics, seed: int) -> tuple[Fraction, ...]:
    """The first of 200 admissible draws whose square-test factors are all non-zero rational
    squares, else the first admissible draw.

    Each draw takes the symbols in name order: a symbol occurring only to even
    powers gets its square from randint(1, 20), any other symbol a choice
    among the non-zero integers of [-20, 20].
    """
    rng = random.Random(seed)
    first: tuple[Fraction, ...] | None = None
    admissible = 0
    for _ in range(_MAX_DRAWS):
        values = tuple(
            Fraction(rng.randint(1, 20) if x in kinematics.even else rng.choice(_INVARIANTS))
            for x in kinematics.symbols
        )
        if not _admissible(kinematics, values):
            continue
        if all(_is_square(_evaluate(f, values)) for f in kinematics.square_factors):
            return values
        if first is None:
            first = values
        admissible += 1
        if admissible == _ADMISSIBLE_DRAWS:
            break
    if first is None:
        raise ValidationError(f"no admissible kinematic point in {_MAX_DRAWS} draws")
    return first


# --- primes and the fit -----------------------------------------------------------


def _excluded(
    kinematics: _Kinematics, values: Sequence[Fraction], primes: Sequence[int]
) -> tuple[int, ...]:
    """2 and the primes dividing a coefficient of G, a non-zero face discriminant or a non-zero
    factor of one at values; at a point on a Landau surface a vanishing discriminant thus
    still contributes the primes of its other factors."""
    quantities = [_evaluate(terms, values) for _, terms in kinematics.coefficients]
    for numerator, denominator in kinematics.discriminants:
        top, bottom = _evaluate(numerator, values), _evaluate(denominator, values)
        if top != 0 and bottom != 0:
            quantities.append(top / bottom)
    quantities += [_evaluate(factor, values) for factor in kinematics.factors]
    numbers = [n for q in quantities if q != 0 for n in (q.numerator, q.denominator)]
    return (2, *(p for p in primes if any(n % p == 0 for n in numbers)))


def _volume_bound(terms: _Terms) -> int:
    """N! Vol of the Newton polytope: the normalised volume times the lattice index, or 0
    when the polytope is not full-dimensional."""
    data = polytope_data([monomial for monomial, _ in terms])
    if not data.is_full_dimensional:
        return 0
    return data.normalized_volume * data.sublattice_index


def count_torus_points(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    *,
    scale: sp.Symbol | None = None,
    seed: int = 0,
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    allow_singular: bool = False,
    landau: LandauAnalysis | None = None,
    volume_bound: int | None = None,
    verification: int = 4,
    max_prime: int = 1000,
    max_evaluations: int = 2 * 10**9,
    backend: str = "numpy",
) -> TorusCount:
    """Count the points of {polynomial = 0} in (F_p^*)^N and fit a candidate polynomial.

    Parameters
    ----------
    polynomial
        G, polynomial in ``variables`` with coefficients polynomial in the
        kinematic symbols, once ``scale`` is set to 1.
    variables
        The N variables of G.
    scale
        The energy scale mu, if G carries one. The Landau analysis runs with mu
        symbolic, then mu is set to 1.
    seed
        Seed of the kinematic point, drawn with random.Random(seed): in the
        order of the symbol names, each invariant from the non-zero integers of
        [-20, 20], and the square of each symbol occurring only to even powers,
        such as a mass, from 1 to 20. The first of 200 admissible draws, those
        where every coefficient of G and every face discriminant is non-zero,
        whose square-test factors are all non-zero rational squares is used,
        else the first admissible one. The square-test factors are the
        irreducible factors of the principal discriminants of the faces of
        dimension at least 1.
    point
        The kinematic point instead of a draw, keyed by symbol, or by the
        square x**2 of a symbol x occurring only to even powers.
    allow_singular
        Count at a given point on a Landau surface instead of raising.
    landau
        The Landau analysis of ``polynomial`` with mu symbolic, if already
        computed.
    volume_bound
        The bound N! Vol(Newt G) on C; computed from the Newton polytope of G at
        the point by default.
    verification
        The number of primes after the N + 1 fit primes to check the fit at. More
        are added until every quadratic character the counts could depend on,
        of -1 and of each square-test factor that is not a square at the point,
        has taken both signs.
    max_prime
        The largest prime to count at.
    max_evaluations
        The most evaluations of G to spend; a prime p costs (p - 1)^(N - 1).
    backend
        "numpy", which evaluates G on grids of points at once.

    Returns
    -------
    TorusCount
        The counts and, when they are polynomial on the tested primes, the
        candidate polynomial, Euler characteristic and master count.

    Raises
    ------
    ValidationError
        If the arguments are malformed, the point lies on a Landau surface and
        ``allow_singular`` is false, no admissible point is drawn, G has degree
        above 2 in every variable, there are too few primes up to
        ``max_prime``, or counting needs more than ``max_evaluations``.
    """
    if backend not in _BACKENDS:
        raise ValidationError(f"backend must be one of {', '.join(_BACKENDS)}; got {backend!r}")
    if verification < 1:
        raise ValidationError(f"verification must be at least 1; got {verification}")
    variables = tuple(variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValidationError("variables must be one or more distinct symbols")
    g = sp.expand(sp.sympify(polynomial))
    if g == 0:
        raise ValidationError("the polynomial is zero")
    n = len(variables)
    if landau is None:
        landau = landau_analysis_from_polynomial(g, list(variables), scale=scale)
    unit = {scale: 1} if scale is not None else {}
    kinematics = _kinematics(sp.expand(g.subs(unit)), variables, landau, unit)

    values = _draw(kinematics, seed) if point is None else _given_point(point, kinematics)
    on_surface = not _admissible(kinematics, values)
    if on_surface and not allow_singular:
        raise ValidationError(
            "the point lies on a Landau surface, where a coefficient of G or a face "
            "discriminant vanishes; pass allow_singular=True to count there"
        )
    specialised = tuple(
        (monomial, value)
        for monomial, terms in kinematics.coefficients
        if (value := _evaluate(terms, values)) != 0
    )
    if not specialised:
        raise ValidationError("G vanishes identically at the point")
    degrees = [max(monomial[k] for monomial, _ in specialised) for k in range(n)]
    k = degrees.index(min(degrees))
    if degrees[k] > 2:
        raise ValidationError(
            f"G has degree at least {degrees[k]} in every variable; the count solves a quadratic"
        )
    parts = _split(specialised, k)
    bound = _volume_bound(specialised) if volume_bound is None else volume_bound

    primes = list(sp.primerange(3, max_prime + 1))
    excluded = _excluded(kinematics, values, primes)
    usable = [p for p in primes if p not in excluded]
    if len(usable) < n + 1 + verification:
        raise ValidationError(
            f"{len(usable)} primes up to max_prime={max_prime} are not excluded, and the fit "
            f"and its check need {n + 1 + verification}"
        )
    needed = sum((p - 1) ** (n - 1) for p in usable[: n + 1 + verification])
    if needed > max_evaluations:
        raise ValidationError(
            f"counting needs at least {needed} evaluations of G, above "
            f"max_evaluations={max_evaluations}"
        )

    fit = usable[: n + 1]
    counts = [(p, _count_numpy(parts, n - 1, p)) for p in fit]
    verified: list[int] = []

    def outcome(polynomial_: tuple[int, ...] | None, reason: str | None) -> TorusCount:
        chi = None if polynomial_ is None else -sum(polynomial_)
        return TorusCount(
            variables=variables,
            eliminated=variables[k],
            point=tuple(
                (_key(x, kinematics.even), v)
                for x, v in zip(kinematics.symbols, values, strict=True)
            ),
            on_shell=(),
            seed=seed if point is None else None,
            on_landau_surface=on_surface,
            excluded_primes=excluded,
            fit_primes=tuple(fit),
            verification_primes=tuple(verified),
            counts=tuple(counts),
            candidate_polynomial=polynomial_,
            candidate_euler_characteristic=chi,
            candidate_master_count=None if chi is None else (-1) ** n * chi,
            reason=reason,
            skipped_faces=len(landau.skipped_faces),
            backend=backend,
        )

    coefficients = _interpolate(fit, [c for _, c in counts])
    if any(c.denominator != 1 for c in coefficients):
        return outcome(None, "the polynomial through the fit counts has non-integer coefficients")
    if coefficients[n] != 0:
        return outcome(None, f"the polynomial through the fit counts has a q^{n} term")
    fitted = tuple(int(c) for c in coefficients[:n])
    master = (-1) ** (n + 1) * sum(fitted)
    if not 0 <= master <= bound:
        return outcome(None, f"the candidate master count {master} is not in [0, {bound}]")

    at_point = [_evaluate(f, values) for f in kinematics.square_factors]
    characters = [Fraction(-1)] + [v for v in at_point if v != 0 and not _is_square(v)]
    signs: dict[Fraction, set[int]] = {d: {_legendre(d, p) for p in fit} for d in characters}
    spent = sum((p - 1) ** (n - 1) for p in fit)
    remaining = iter(usable[n + 1 :])
    while len(verified) < verification or not all({-1, 1} <= s for s in signs.values()):
        p = next(remaining, None)
        if p is None:
            raise ValidationError(
                f"the primes up to max_prime={max_prime} do not give every quadratic character "
                "the check needs both signs"
            )
        spent += (p - 1) ** (n - 1)
        if spent > max_evaluations:
            raise ValidationError(
                f"checking the fit needs more than max_evaluations={max_evaluations} evaluations"
            )
        observed = _count_numpy(parts, n - 1, p)
        counts.append((p, observed))
        verified.append(p)
        for d in characters:
            signs[d].add(_legendre(d, p))
        predicted = sum(c * p**i for i, c in enumerate(fitted))
        if observed != predicted:
            return outcome(
                None, f"the count at p = {p} is {observed}, where the fit predicts {predicted}"
            )
    return outcome(fitted, None)
