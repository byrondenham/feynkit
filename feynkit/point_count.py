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

import importlib.util
import itertools
import math
import random
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from ._exact import smith_invariants
from .core.exceptions import ComputationError, ValidationError
from .landau import (
    LandauAnalysis,
    _factor_list,
    _singular_binary,
    landau_analysis_from_polynomial,
)
from .polytope import PolytopeData, polytope_data

__all__ = ["TorusCount", "count_torus_points", "critical_point_count"]

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
_BACKENDS = ("numpy", "flint")


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
        denominator of a coefficient of G, of a non-zero face discriminant, of
        a non-zero irreducible factor of one or of the discriminant of G on an
        edge, at the point.
    max_prime
        The largest prime the count could use, and the limit of
        ``excluded_primes``.
    fit_primes, verification_primes
        The primes the fit used and the primes it was checked at, in order.
    counts
        (p, #V(F_p)) for every prime counted, in order.
    candidate_polynomial
        The coefficients of P(q), constant term first, N of them; None when
        there is no candidate (see ``reason``).
    candidate_euler_characteristic
        chi(X) = -P(1), or None.
    candidate_master_count
        C = (-1)^N chi(X), or None.
    reason
        Why there is no candidate, naming the test that failed: the polynomial
        through the fit counts has non-integer coefficients or a q^N term; the
        master count the fit gives is outside [0, N! Vol(Newt G)], Newt G being
        the Newton polytope of G at the point and Vol its Euclidean volume; an
        edge of lattice length at least 3 or a face whose lattice quotient has
        exponent at least 3 lets characters of order above 2, which the check
        does not cover, into the counts; or the fit disagrees with the count at
        a check prime. None when there is a candidate.
    skipped_faces
        How many faces the Landau analysis skipped; their discriminants are
        missing from the admissibility test, the excluded primes and the
        characters the check covers.
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
    max_prime: int
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
        # E <= (d + 1)^2 for degree d in the grid variables and E <= 9 for a Feynman graph;
        # delta below takes 4 (p - 1)^2. int64 holds all three exactly while they stay below
        # 2^63: for E = 9, up to p of about 10^9 (_exact_limit), far above the default
        # max_prime of 1000.
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


def _exact_limit(parts: _Parts, others: int) -> int | None:
    """The largest prime for which _count_numpy is exact in int64, or None without a limit.

    This is where max(4, E) (p - 1)^2 reaches 2^63, E being the most grid
    exponents of a part (see _count_numpy); one variable is counted with
    Python integers.
    """
    if others == 0:
        return None
    grid = min(others, 2)
    width = max(len({monomial[others - grid :] for monomial in part}) for part in parts)
    return math.isqrt((2**63 - 1) // max(4, width)) + 1


def _flint_available() -> bool:
    """Whether python-flint can be imported; the analogue of _singular_binary."""
    return importlib.util.find_spec("flint") is not None


def _count_flint(parts: _Parts, others: int, p: int) -> int:
    """#V(F_p) in the torus, counting the roots of nmod_poly([c, b, a], p) point by point.

    A cross-check of _count_numpy that needs no Legendre symbol, and far slower.
    """
    import flint

    a, b, c = (_modular(part, p) for part in parts)

    def value(part: dict[tuple[int, ...], int], y: tuple[int, ...]) -> int:
        return (
            sum(
                coefficient * math.prod(pow(v, e, p) for v, e in zip(y, monomial, strict=True))
                for monomial, coefficient in part.items()
            )
            % p
        )

    total = 0
    for y in itertools.product(range(1, p), repeat=others):
        av, bv, cv = value(a, y), value(b, y), value(c, y)
        if av == bv == cv == 0:
            total += p - 1
        elif av or bv:
            roots = flint.nmod_poly([cv, bv, av], p).roots()
            total += sum(1 for root, _ in roots if int(root) != 0)
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
    polynomials here are written in those squares. A face discriminant can
    still be odd in such a symbol m, since the Landau analysis factorises in m
    and drops multiplicities (m^2 becomes m); it is replaced by its product
    with its image under m -> -m, which is even in m and vanishes wherever
    either does.

    ``discriminants`` holds the numerator and denominator of every face
    discriminant that is not constant, ``factors`` their distinct irreducible
    factors, and ``square_factors`` the factors from faces of dimension at
    least 1 with principal discriminants, which the square test asks to be
    squares. ``edges`` holds the coefficient of G at every vertex and the
    discriminant of G on every edge, as a polynomial in the edge's lattice
    coordinate, with the constants and contents the Landau analysis drops and
    with the lattice index of edges that are simplices.
    """

    symbols: tuple[sp.Symbol, ...]
    even: frozenset[sp.Symbol]
    coefficients: tuple[tuple[tuple[int, ...], _Terms], ...]
    discriminants: tuple[tuple[_Terms, _Terms], ...]
    factors: tuple[_Terms, ...]
    square_factors: tuple[_Terms, ...]
    edges: tuple[_Terms, ...]


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

    def norm(expr: sp.Expr) -> sp.Expr:
        expr = sp.expand(expr)
        for x in sorted(even, key=lambda x: x.name):
            image = sp.expand(expr.subs(x, -x))
            if image != expr:
                expr = sp.expand(expr * image)
        return expr

    by_monomial = {
        tuple(int(e) for e in monomial): rewrite(c)
        for monomial, c in sp.Poly(g, *variables).terms()
    }
    coefficients = tuple((monomial, _terms(c, drawn)) for monomial, c in by_monomial.items())
    discriminants: list[tuple[_Terms, _Terms]] = []
    factors: dict[_Terms, None] = {}
    square_factors: dict[_Terms, None] = {}
    edges: dict[_Terms, None] = {}
    for face in landau.face_discriminants:
        if face.dimension <= 1:
            local = [(e, by_monomial.get(e, sp.Integer(0))) for e in face.exponents]
            edges.setdefault(_terms(_edge_discriminant(local), drawn), None)
        numerator, denominator = sp.fraction(sp.together(sp.sympify(face.discriminant).subs(unit)))
        numerator, denominator = rewrite(norm(numerator)), rewrite(norm(denominator))
        if not numerator.free_symbols:
            continue
        discriminants.append((_terms(numerator, drawn), _terms(denominator, drawn)))
        for base in _factor_list(numerator, set(drawn)):
            factor = _terms(base, drawn)
            factors.setdefault(factor, None)
            if face.dimension >= 1 and face.principal:
                square_factors.setdefault(factor, None)
    return _Kinematics(
        symbols,
        even,
        coefficients,
        tuple(discriminants),
        tuple(factors),
        tuple(square_factors),
        tuple(edges),
    )


def _edge_discriminant(terms: Sequence[tuple[tuple[int, ...], sp.Expr]]) -> sp.Expr:
    """The coefficient of a vertex, or the discriminant of G on an edge as a polynomial in the
    edge's lattice coordinate t, where G restricts to sum_i c_i t^(k_i)."""
    if len(terms) == 1:
        return terms[0][1]
    start = terms[0][0]
    step = next(tuple(x - y for x, y in zip(e, start, strict=True)) for e, _ in terms if e != start)
    size = math.gcd(*step)
    step = tuple(x // size for x in step)
    axis = next(j for j, x in enumerate(step) if x)
    positions = [(e[axis] - start[axis]) // step[axis] for e, _ in terms]
    lowest = min(positions)
    if max(positions) - lowest < 2:
        return sp.Integer(1)
    t = sp.Dummy("t")
    polynomial = sum(
        (c * t ** (k - lowest) for (_, c), k in zip(terms, positions, strict=True)), sp.Integer(0)
    )
    return sp.expand(sp.discriminant(polynomial, t))


def _key(x: sp.Symbol, even: frozenset[sp.Symbol]) -> sp.Expr:
    return x**2 if x in even else x


def _rational(raw: object, key: sp.Expr) -> Fraction:
    value = sp.sympify(raw)
    if not value.is_Rational:
        raise ValidationError(f"point gives {key} the value {raw!r}, which is not rational")
    return Fraction(int(value.p), int(value.q))


def _given_point(
    point: Mapping[sp.Expr, int | Fraction],
    symbols: Sequence[sp.Symbol],
    even: frozenset[sp.Symbol],
) -> tuple[Fraction, ...]:
    """The value of each of symbols at a point the caller gave, or of its square for a symbol
    in even, which occurs only to even powers and may be keyed by itself or by its square."""
    values: dict[sp.Symbol, Fraction] = {}
    for raw_key, raw in point.items():
        key = sp.sympify(raw_key)
        if key in symbols:
            symbol, value = key, _rational(raw, key)
            if key in even:
                value = value**2
        elif isinstance(key, sp.Pow) and key.exp == 2 and key.base in even:
            symbol, value = key.base, _rational(raw, key)
            if value < 0:
                raise ValidationError(f"point gives {key} the negative value {raw}")
        else:
            expected = ", ".join(str(_key(x, even)) for x in symbols) or "none"
            raise ValidationError(f"point has the key {key}; the keys are {expected}")
        if symbol in values:
            raise ValidationError(f"point gives more than one value for {_key(symbol, even)}")
        values[symbol] = value
    missing = [str(_key(x, even)) for x in symbols if x not in values]
    if missing:
        raise ValidationError(f"point gives no value for {', '.join(missing)}")
    return tuple(values[x] for x in symbols)


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
    """2 and the primes dividing a coefficient of G, a non-zero face discriminant, a non-zero
    factor of one or a non-zero edge discriminant at values; at a point on a Landau surface a
    vanishing discriminant thus still contributes the primes of its other factors. Every
    quadratic character the check covers is thus defined at every prime not excluded."""
    quantities = [_evaluate(terms, values) for _, terms in kinematics.coefficients]
    for numerator, denominator in kinematics.discriminants:
        top, bottom = _evaluate(numerator, values), _evaluate(denominator, values)
        if top != 0 and bottom != 0:
            quantities.append(top / bottom)
    quantities += [_evaluate(terms, values) for terms in kinematics.factors + kinematics.edges]
    numbers = [n for q in quantities if q != 0 for n in (q.numerator, q.denominator)]
    return (2, *(p for p in primes if any(n % p == 0 for n in numbers)))


def _coprime_base(numbers: Sequence[int]) -> list[int]:
    """Pairwise coprime integers above 1 of whose powers every non-zero number is, up to sign,
    a product; found with gcds alone, without factorising."""
    base: list[int] = []
    pending = [abs(n) for n in numbers]
    while pending:
        x = pending.pop()
        if x <= 1:
            continue
        for i, b in enumerate(base):
            g = math.gcd(x, b)
            if g > 1:
                del base[i]
                pending += [g, x // g, b // g]
                break
        else:
            base.append(x)
    return base


def _square_classes(values: Sequence[Fraction]) -> list[int]:
    """The class of each non-zero value in Q^* / (Q^*)^2 as a vector over F_2, a bit mask.

    Bit 0 is the sign. The others are the parities of the exponents of the
    elements of a coprime base of the numerators and denominators that are
    not squares: their square-free parts are above 1 with disjoint prime
    supports, so the classes are those of the prime supports of the
    square-free parts of the values, with the sign.
    """
    base = _coprime_base([n for v in values for n in (v.numerator, v.denominator)])
    base = [b for b in base if math.isqrt(b) ** 2 != b]
    classes = []
    for value in values:
        vector = int(value < 0)
        for i, b in enumerate(base, start=1):
            for n in (abs(value.numerator), value.denominator):
                while n % b == 0:
                    n //= b
                    vector ^= 1 << i
        classes.append(vector)
    return classes


def _insert(pivots: dict[int, int], vector: int) -> bool:
    """Add vector to a row-echelon basis over F_2 keyed by leading bit; whether it was
    independent of the vectors already there."""
    while vector:
        top = vector.bit_length() - 1
        if top not in pivots:
            pivots[top] = vector
            return True
        vector ^= pivots[top]
    return False


def _characters(kinematics: _Kinematics, values: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """Independent generators d_1, ..., d_k of the quadratic characters (d/p) that the check
    covers: d = -1 and the non-zero values at the point of every factor of a face
    discriminant, vertex coefficient and edge discriminant, up to squares."""
    candidates = [Fraction(-1)] + [
        value
        for terms in kinematics.factors + kinematics.edges
        if (value := _evaluate(terms, values)) != 0
    ]
    pivots: dict[int, int] = {}
    return tuple(
        d
        for d, vector in zip(candidates, _square_classes(candidates), strict=True)
        if _insert(pivots, vector)
    )


def _signs(characters: Sequence[Fraction], p: int) -> int:
    """The vector over F_2 of the characters at p: bit i is set when (d_i/p) = -1."""
    return sum(1 << i for i, d in enumerate(characters) if _legendre(d, p) == -1)


def _volume_bound(data: PolytopeData) -> int:
    """N! Vol of the Newton polytope: the normalised volume times the lattice index, or 0
    when the polytope is not full-dimensional."""
    if not data.is_full_dimensional:
        return 0
    return data.normalized_volume * data.sublattice_index


def _higher_order(data: PolytopeData) -> str | None:
    """Why the counts may depend on characters of order above 2, or None.

    The check covers quadratic characters only. On an edge of lattice length
    k, G is a polynomial of degree k in the edge's lattice coordinate, whose
    roots mod p follow its Galois group: for k >= 3 they can need a character
    of order 3 or more even when every lattice point of the edge is present,
    as for u^3 - u^2 - 2u + 1, which for p != 7 has three roots mod p when
    p = +-1 mod 7 and none otherwise. On a face of dimension 2 or more whose
    points span a lattice L in the lattice L_sat of the integer points of its
    affine hull, G is, up to a monomial, pulled back from the torus of L along
    an isogeny whose non-empty fibres over F_p are torsors under
    Hom(L_sat/L, F_p^*). The counts can then depend on p through power-residue
    characters whose orders divide the exponent of L_sat/L, its largest Smith
    invariant, which may be smaller than its index. When G has degree at most
    2 in every variable, as for a Feynman graph, every edge has lattice length
    at most 2.
    """
    for dimension, indices in sorted(data.faces):
        if dimension == 0:
            continue
        if dimension == 1:
            start, end = (data.points[i] for i in indices if i in data.vertex_indices)
            size = math.gcd(*(x - y for x, y in zip(end, start, strict=True)))
            found = f"an edge of lattice length {size}"
        else:
            points = [data.points[i] for i in indices]
            differences = [[x - y for x, y in zip(q, points[0], strict=True)] for q in points[1:]]
            size = max(smith_invariants(differences, data.ambient_dimension), default=1)
            found = f"a face of dimension {dimension} whose lattice quotient has exponent {size}"
        if size >= 3:
            return (
                f"the Newton polytope has {found}, so the counts may depend on characters of "
                "order above 2, which the check does not cover"
            )
    return None


def _analyses(landau: LandauAnalysis, g: sp.Expr, variables: Sequence[sp.Symbol]) -> bool:
    """Whether landau is an analysis of g: its faces carry the coefficients of g and, with the
    faces it skipped, cover the support of g."""
    terms = {tuple(int(e) for e in m): c for m, c in sp.Poly(g, *variables).terms()}
    seen: dict[tuple[int, ...], sp.Expr] = {}
    for face in landau.face_discriminants:
        seen.update(zip(face.exponents, face.coefficients, strict=True))
    if set(seen).union(*landau.skipped_faces) != set(terms):
        return False
    return all(sp.cancel(c - terms[e]) == 0 for e, c in seen.items())


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
        Seed of the kinematic point, an integer, drawn with
        random.Random(seed): in the order of the symbol names, each invariant
        from the non-zero integers of [-20, 20], and the square of each symbol
        occurring only to even powers, such as a mass, from 1 to 20; a
        discriminant odd in such a symbol m enters through its product with
        its image under m -> -m. The first of 200 admissible draws, those
        where every coefficient of G and every face discriminant is non-zero,
        whose square-test factors are all non-zero rational squares is used,
        else the first admissible one. The square-test factors are the
        irreducible factors of the principal discriminants of the faces of
        dimension at least 1.
    point
        The kinematic point instead of a draw, keyed by symbol, or by the
        square x**2 of a symbol x occurring only to even powers.
    allow_singular
        Count at a given point where a coefficient of G or a face discriminant
        vanishes instead of raising.
    landau
        The Landau analysis of ``polynomial`` with mu symbolic, if already
        computed; its faces must carry the terms of ``polynomial``.
    volume_bound
        The bound N! Vol(Newt G) on C; computed from the Newton polytope of G at
        the point by default.
    verification
        The least number of primes after the N + 1 fit primes to check the fit
        at. More are added until every non-trivial product of the quadratic
        characters (d/p) the counts could depend on takes both signs on the fit
        and check primes together, so that no such character is constant on
        the sample. The d are -1 and the values at the point of every
        irreducible factor of every face discriminant, principal or not, of
        every vertex coefficient of G and of the discriminant of G on every
        edge, in its lattice coordinate.
    max_prime
        The largest prime to count at; at most about 10^9, above which the
        int64 counts would not be exact.
    max_evaluations
        The most evaluations of G to spend; a prime p costs (p - 1)^(N - 1).
    backend
        "numpy", which evaluates G on grids of points at once, or "flint", which
        counts the roots of each quadratic with python-flint point by point: a
        slower cross-check that needs python-flint installed.

    Returns
    -------
    TorusCount
        The counts and, when the fit passes its tests, the candidate
        polynomial, Euler characteristic and master count. When an
        edge of the Newton polytope of G at the point has lattice length at
        least 3, or a face of dimension 2 or more has a lattice quotient of
        exponent at least 3 (its largest Smith invariant, not its index), the
        counts may depend on characters of order above 2, which the check does
        not cover: the fit primes are counted, but a fit is refused without a
        check.

    Raises
    ------
    ValidationError
        If the arguments are malformed, ``landau`` analyses another
        polynomial, a coefficient of G or a face discriminant vanishes at the
        point and ``allow_singular`` is false, no admissible point is drawn, G
        has degree above 2 in every variable, ``max_prime`` is above the int64
        limit, there are too few primes up to ``max_prime`` for the fit and its
        check, or counting needs more than ``max_evaluations``.
    RuntimeError
        If ``backend`` is "flint" and python-flint is not installed.
    """
    if backend not in _BACKENDS:
        raise ValidationError(f"backend must be one of {', '.join(_BACKENDS)}; got {backend!r}")
    if backend == "flint" and not _flint_available():
        raise RuntimeError("the flint backend needs python-flint, which is not installed")
    if verification < 1:
        raise ValidationError(f"verification must be at least 1; got {verification}")
    variables = tuple(variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValidationError("variables must be one or more distinct symbols")
    g = sp.expand(sp.sympify(polynomial))
    if g == 0:
        raise ValidationError("the polynomial is zero")
    if point is None and not isinstance(seed, int):
        raise ValidationError(f"seed must be an integer; got {seed!r}")
    n = len(variables)
    if landau is None:
        landau = landau_analysis_from_polynomial(g, list(variables), scale=scale)
    elif not _analyses(landau, g, variables):
        raise ValidationError(
            "landau is not a Landau analysis of the polynomial; from "
            "FeynmanIntegral.torus_count it must analyse the integral with on_shell applied"
        )
    unit = {scale: 1} if scale is not None else {}
    kinematics = _kinematics(sp.expand(g.subs(unit)), variables, landau, unit)

    values = (
        _draw(kinematics, seed)
        if point is None
        else _given_point(point, kinematics.symbols, kinematics.even)
    )
    on_surface = not _admissible(kinematics, values)
    if on_surface and not allow_singular:
        raise ValidationError(
            "a coefficient of G or a face discriminant vanishes at the point; pass "
            "allow_singular=True to count there"
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
    limit = _exact_limit(parts, n - 1)
    if limit is not None and max_prime > limit:
        raise ValidationError(
            f"max_prime must be at most {limit}, where the int64 counts are exact; "
            f"got {max_prime}"
        )
    polytope = polytope_data([monomial for monomial, _ in specialised])
    bound = _volume_bound(polytope) if volume_bound is None else volume_bound
    # A fit the check cannot vouch for is refused before the check, so only the fit primes are
    # counted; their counts are still reported, and a fit that fails outright says so instead.
    refusal = _higher_order(polytope)
    checks = verification if refusal is None else 0

    primes = list(sp.primerange(3, max_prime + 1))
    excluded = _excluded(kinematics, values, primes)
    usable = [p for p in primes if p not in excluded]
    if len(usable) < n + 1 + checks:
        raise ValidationError(
            f"{len(usable)} primes up to max_prime={max_prime} are not excluded, and the fit "
            f"and its check need {n + 1 + checks}"
        )
    needed = sum((p - 1) ** (n - 1) for p in usable[: n + 1 + checks])
    if needed > max_evaluations:
        raise ValidationError(
            f"counting needs at least {needed} evaluations of G, above "
            f"max_evaluations={max_evaluations}"
        )

    count = _count_flint if backend == "flint" else _count_numpy
    fit = usable[: n + 1]
    counts = [(p, count(parts, n - 1, p)) for p in fit]
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
            max_prime=max_prime,
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
        return outcome(
            None,
            f"the master count {master} given by the fit is not in [0, N! Vol(Newt G)] = "
            f"[0, {bound}]",
        )
    if refusal is not None:
        return outcome(None, refusal)

    # Every non-trivial product of the characters must take both signs on the fit and check
    # primes together: their vectors over F_2 must affinely span F_2^k, that is their
    # differences from the first must have rank k.
    characters = _characters(kinematics, values)
    first = _signs(characters, fit[0])
    spanned: dict[int, int] = {}
    for p in fit[1:]:
        _insert(spanned, _signs(characters, p) ^ first)
    spent = sum((p - 1) ** (n - 1) for p in fit)
    remaining = iter(usable[n + 1 :])
    while len(verified) < verification or len(spanned) < len(characters):
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
        observed = count(parts, n - 1, p)
        counts.append((p, observed))
        verified.append(p)
        _insert(spanned, _signs(characters, p) ^ first)
        predicted = sum(c * p**i for i, c in enumerate(fitted))
        if observed != predicted:
            return outcome(
                None, f"the count at p = {p} is {observed}, where the fit predicts {predicted}"
            )
    return outcome(fitted, None)


def _singular_polynomial(poly: sp.Poly, p: int) -> str:
    """poly, with integer coefficients in the ring variables v0, v1, ..., reduced mod p for
    Singular."""
    terms = []
    for monomial, coefficient in poly.terms():
        residue = int(coefficient) % p
        if residue:
            powers = [f"v{i}^{e}" if e > 1 else f"v{i}" for i, e in enumerate(monomial) if e]
            terms.append("*".join([str(residue), *powers]))
    return "+".join(terms) or "0"


def critical_point_count(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction],
    *,
    seed: int = 0,
    timeout: float = 300,
) -> int:
    """The number of critical points of sum_e nu_e log u_e - (D/2) log G on X, modulo primes.

    X is the complement of {G = 0} in the torus. For generic exponents nu_e and
    D the critical points are all regular and number |chi(X)| (Fevola, Mizera
    and Telen, Comput. Phys. Commun. 303 (2024) 109278, proof of Theorem 3.1,
    eq. (3.2), after Huh 2013). The exponents are random rationals drawn with
    random.Random(seed). The critical points are the zeros of the ideal I of
    nu_e G - (D/2) u_e dG/du_e, e = 1, ..., N, and 1 - t u_1 ... u_N G, and
    their number is its vector-space dimension vdim(std(I)).

    Singular computes it over F_p rather than Q, where the Groebner basis can
    take hours, at the two largest primes below 2^31 that divide no numerator
    or denominator of a coefficient of the generators or of an exponent, and
    the two results must agree. The dimension over F_p equals the one over Q
    for all but finitely many p, so this is a cross-check, not a certificate.

    Parameters
    ----------
    polynomial
        G, with the energy scale already set to 1.
    variables
        The N variables of G.
    point
        A value for every other symbol of G, keyed by the symbol or, for a
        symbol occurring only to even powers, by its square, as
        :attr:`TorusCount.point` gives them.
    seed
        Seed of the random exponents.
    timeout
        The most seconds to give Singular.

    Raises
    ------
    RuntimeError
        If Singular is not installed, fails, or prints anything but the two
        counts.
    ValidationError
        If ``variables`` are not one or more distinct symbols, ``timeout`` is
        not positive, G is not a polynomial with rational coefficients in its
        symbols, or ``point`` misses a symbol of G, has any other key, gives a
        value that is not rational or a negative value for a square, keys by
        its square a symbol occurring to odd powers, or G vanishes
        identically at the point.
    ComputationError
        If Singular runs out of time, the counts modulo the two primes differ,
        or the critical points do not form a finite set at the exponents drawn.
    """
    variables = tuple(variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValidationError("variables must be one or more distinct symbols")
    if not timeout > 0:
        raise ValidationError(f"timeout must be positive; got {timeout}")
    g = sp.expand(sp.sympify(polynomial))
    symbols = tuple(sorted(g.free_symbols - set(variables), key=lambda x: x.name))
    try:
        sp.Poly(g, *variables, *symbols, domain="QQ")
    except (sp.PolynomialError, sp.CoercionFailed) as exc:
        raise ValidationError(
            "G must be a polynomial with rational coefficients in its symbols; set the energy "
            "scale to 1"
        ) from exc
    even = frozenset(x for x in symbols if all(d % 2 == 0 for d in _degrees(g, x)))
    values = _given_point(point, symbols, even)
    # A symbol in even occurs only to even powers, and its value is that of its square.
    rationals = [sp.Rational(v.numerator, v.denominator) for v in values]
    substitution = {
        x: sp.sqrt(v) if x in even else v for x, v in zip(symbols, rationals, strict=True)
    }
    g = sp.expand(g.subs(substitution))
    if g == 0:
        raise ValidationError("G vanishes identically at the point")
    binary = _singular_binary()
    if binary is None:
        raise RuntimeError("critical_point_count needs Singular, which was not found")

    rng = random.Random(seed)
    exponents = [
        sp.Rational(rng.randint(1, 10**6), rng.randint(1, 10**6)) for _ in range(len(variables) + 1)
    ]
    half_d, *nu = exponents
    t = sp.Dummy("t")
    generators = [
        nu_e * g - half_d * u * sp.diff(g, u) for nu_e, u in zip(nu, variables, strict=True)
    ]
    generators.append(1 - t * sp.Mul(*variables) * g)
    polys = [sp.Poly(generator, t, *variables, domain="QQ") for generator in generators]
    avoided = [
        abs(int(n))
        for q in [c for poly in polys for c in poly.coeffs()] + exponents
        for n in (q.p, q.q)
    ]
    primes: list[int] = []
    p = 2**31
    while len(primes) < 2:
        p = sp.prevprime(p)
        if all(n % p for n in avoided):
            primes.append(p)
    # No coefficient vanishes modulo the primes, so each generator keeps its support there.
    integral = [poly.clear_denoms(convert=True)[1] for poly in polys]
    ring = ",".join(f"v{i}" for i in range(len(variables) + 1))
    script = ""
    for i, prime in enumerate(primes):
        ideal = ",".join(_singular_polynomial(poly, prime) for poly in integral)
        script += (
            f"ring r{i} = {prime}, ({ring}), dp;\n"
            f"ideal I{i} = {ideal};\n"
            f"print(vdim(std(I{i})));\n"
        )
    script += "quit;\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "critical.sing"
        path.write_text(script)
        try:
            run = subprocess.run(
                [binary, "-q", "--no-warn", str(path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ComputationError(f"Singular did not finish within timeout={timeout} s") from exc
    if run.returncode != 0:
        raise RuntimeError(f"Singular failed: {run.stderr.strip()}")
    # Singular reports an error in the script on stdout and still exits with 0.
    try:
        counts = [int(line) for line in run.stdout.split()]
    except ValueError:
        counts = []
    if len(counts) != 2:
        output = (run.stdout.strip() or run.stderr.strip() or "nothing")[:500]
        raise RuntimeError(f"Singular printed {output!r} instead of two counts")
    first, second = counts
    if first != second:
        raise ComputationError(
            f"the critical points number {first} modulo {primes[0]} and {second} modulo "
            f"{primes[1]}"
        )
    if first < 0:
        raise ComputationError("the critical points do not form a finite set at these exponents")
    return first
