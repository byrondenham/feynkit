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

from collections.abc import Sequence
from fractions import Fraction

import numpy as np

# A polynomial as (exponent vector, coefficient) pairs with non-zero coefficients.
_Terms = tuple[tuple[tuple[int, ...], Fraction], ...]
# The coefficients a, b and c of x^2, x and 1, each keyed by the exponents of the other variables.
_Parts = tuple[
    dict[tuple[int, ...], Fraction],
    dict[tuple[int, ...], Fraction],
    dict[tuple[int, ...], Fraction],
]
# The numpy backend evaluates at most this many points at once.
_CHUNK = 2**22


def _legendre(d: Fraction, p: int) -> int:
    """The Legendre symbol (d/p) of a rational d, for an odd prime p; 0 when p divides d."""
    residue = d.numerator * d.denominator % p
    if residue == 0:
        return 0
    return 1 if pow(residue, (p - 1) // 2, p) == 1 else -1


def _split(terms: _Terms, k: int) -> _Parts:
    """The coefficients a, b, c of x^2, x and 1, x being variable k of degree at most 2."""
    a: dict[tuple[int, ...], Fraction] = {}
    b: dict[tuple[int, ...], Fraction] = {}
    c: dict[tuple[int, ...], Fraction] = {}
    for monomial, coefficient in terms:
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

    ``others`` is the number of other variables. The last two, or the only
    one, form a grid evaluated by broadcasting, with a, b and c grouped by
    their exponents there; the variables before them run in chunks of at most
    _CHUNK points. Residues stay below p, so int64 arithmetic is exact.
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
                    weight = (weight + term) % p
                value = (value + weight.reshape((-1,) + (1,) * grid) * grid_values) % p
            values.append(value)
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
    takes the value ys[i] at xs[i]; exact Lagrange interpolation."""
    n = len(xs)
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
