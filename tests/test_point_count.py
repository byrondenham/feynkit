"""Tests for the finite-field point counts of feynkit.point_count.

Oracles: brute-force enumeration of (F_p^*)^N for small p, and the massive
bubble, whose zero set is an affine conic with p - 3 - (lambda/p) points in
the torus, lambda being the Kallen function of s, m_1^2 and m_2^2.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import pytest

from feynkit import FeynmanIntegral
from feynkit import point_count as pc
from feynkit.polytope import polytope_data

SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23)
# Brute force costs (p - 1)^n evaluations, so more variables get smaller primes.
LARGEST_PRIME = {1: 23, 2: 23, 3: 13, 4: 7}


def brute_force(terms: pc._Terms, n: int, p: int) -> int:
    """#V(F_p) in (F_p^*)^n by evaluating the polynomial at every point."""
    residues = [(monomial, c.numerator * pow(c.denominator, -1, p) % p) for monomial, c in terms]
    total = 0
    for u in itertools.product(range(1, p), repeat=n):
        value = sum(
            c * math.prod(pow(x, e, p) for x, e in zip(u, m, strict=True)) for m, c in residues
        )
        total += value % p == 0
    return total


def random_terms(rng: random.Random, n: int) -> pc._Terms:
    """A random polynomial in n variables of degree at most 2 in each."""
    terms: dict[tuple[int, ...], Fraction] = {}
    for _ in range(rng.randint(1, 6)):
        monomial = tuple(rng.randint(0, 2) for _ in range(n))
        terms[monomial] = Fraction(rng.randint(-5, 5), rng.choice([1, 1, 2, 3]))
    return tuple((m, c) for m, c in terms.items() if c != 0)


def bubble_terms(s: int, m1: int, m2: int) -> pc._Terms:
    """G of the massive bubble at mu = 1, with m1 and m2 the squared masses."""
    return (
        ((1, 0), Fraction(1)),
        ((0, 1), Fraction(1)),
        ((2, 0), Fraction(m1)),
        ((1, 1), Fraction(m1 + m2 - s)),
        ((0, 2), Fraction(m2)),
    )


def kallen(s: int, m1: int, m2: int) -> int:
    return s * s + m1 * m1 + m2 * m2 - 2 * s * m1 - 2 * s * m2 - 2 * m1 * m2


class TestKernel:
    def test_counts_match_brute_force(self) -> None:
        rng = random.Random(1)
        checked = 0
        for _ in range(200):
            n = rng.randint(1, 4)
            terms = random_terms(rng, n)
            if not terms:
                continue
            k = rng.randrange(n)
            for p in SMALL_PRIMES:
                if p > LARGEST_PRIME[n] or any(c.denominator % p == 0 for _, c in terms):
                    continue
                assert pc._count_numpy(pc._split(terms, k), n - 1, p) == brute_force(terms, n, p)
                checked += 1
        assert checked > 1000

    def test_chunks_of_the_outer_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_CHUNK", 7)
        rng = random.Random(2)
        for _ in range(20):
            terms = random_terms(rng, 4)
            if not terms or any(c.denominator % 11 == 0 for _, c in terms):
                continue
            assert pc._count_numpy(pc._split(terms, 0), 3, 11) == brute_force(terms, 4, 11)

    def test_any_variable_of_degree_two_can_be_solved_for(self) -> None:
        terms = bubble_terms(2, 5, 8)
        assert pc._count_numpy(pc._split(terms, 0), 1, 17) == pc._count_numpy(
            pc._split(terms, 1), 1, 17
        )

    @pytest.mark.parametrize(("s", "m1", "m2"), [(2, 5, 8), (1, 1, 1), (1, 2, 6), (3, 7, 5)])
    def test_massive_bubble_has_p_minus_3_minus_the_legendre_symbol(
        self, s: int, m1: int, m2: int
    ) -> None:
        lam = kallen(s, m1, m2)
        terms = bubble_terms(s, m1, m2)
        for p in (7, 17, 19, 23, 29, 31, 37, 41):
            if any(x % p == 0 for x in (lam, s, m1, m2, m1 + m2 - s)):
                continue
            expected = p - 3 - pc._legendre(Fraction(lam), p)
            assert pc._count_numpy(pc._split(terms, 0), 1, p) == expected

    def test_a_bad_prime_breaks_the_pattern(self) -> None:
        # s = 1, m_e^2 = (2, 6): lambda = 1 is a square, so p - 4 points for good p, but the
        # coefficient 6 vanishes mod 3 and there are no points at all, not -1.
        terms = bubble_terms(1, 2, 6)
        assert pc._count_numpy(pc._split(terms, 0), 1, 3) == 0
        assert pc._count_numpy(pc._split(terms, 0), 1, 5) == 5 - 4

    def test_one_variable(self) -> None:
        # x^2 - 2 has two roots mod 7 (3 and 4) and none mod 5; 3x + 1 one; 0 all of F_p^*.
        assert pc._count_numpy(pc._split((((2,), Fraction(1)), ((0,), Fraction(-2))), 0), 0, 7) == 2
        assert pc._count_numpy(pc._split((((2,), Fraction(1)), ((0,), Fraction(-2))), 0), 0, 5) == 0
        assert pc._count_numpy(pc._split((((1,), Fraction(3)), ((0,), Fraction(1))), 0), 0, 7) == 1
        assert pc._count_numpy(pc._split((((1,), Fraction(7)),), 0), 0, 7) == 6

    def test_legendre(self) -> None:
        assert [pc._legendre(Fraction(d), 7) for d in range(7)] == [0, 1, 1, -1, 1, -1, -1]
        assert pc._legendre(Fraction(-39), 41) == 1
        assert pc._legendre(Fraction(3, 4), 11) == pc._legendre(Fraction(3), 11)

    def test_interpolation_is_exact(self) -> None:
        assert pc._interpolate([3, 7, 11], [1, 5, 9]) == [-2, 1, 0]
        assert pc._interpolate([2, 3, 5], [4, 9, 25]) == [0, 0, 1]
        assert pc._interpolate([1, 2], [0, 1]) == [-1, 1]
        assert pc._interpolate([5], [7]) == [7]


@pytest.mark.parametrize(
    "cnickel",
    [
        "11e|e|:nz",
        "11e|e|:nn",
        "12e|2e|e|:nzz",
        "111e|e|:nnn",
        "12e|3e|3e|e|:nnzz",
        "12ee|22e|e|:nnnn",
    ],
)
def test_graph_polynomials_have_degree_at_most_two_in_each_variable(cnickel: str) -> None:
    # U and the kinematic part of F are square-free, and the mass term U sum_e m_e^2 u_e adds
    # one to the degree in u_e: degree 2 for a massive propagator, 1 for a massless one, and
    # the edges of the Newton polytope have lattice length at most 2.
    fi = FeynmanIntegral.from_cnickel(cnickel)
    support = [alpha for alpha, _ in fi.newton_polytope.support]
    for k, edge in enumerate(fi.graph.get_internal_edges()):
        assert max(alpha[k] for alpha in support) == (1 if edge.get_mass() == 0 else 2)
    data = polytope_data(support)
    for dimension, indices in data.faces:
        if dimension == 1:
            ends = [data.points[i] for i in indices if i in data.vertex_indices]
            assert math.gcd(*(x - y for x, y in zip(ends[0], ends[1], strict=True))) <= 2
