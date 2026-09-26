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
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import point_count as pc
from feynkit.core.exceptions import ValidationError
from feynkit.landau import landau_analysis_from_polynomial
from feynkit.point_count import count_torus_points
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

    def test_two_outer_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Five variables leave two outer ones, both decoded from each chunk index.
        monkeypatch.setattr(pc, "_CHUNK", 7)
        rng = random.Random(3)
        checked = 0
        for _ in range(10):
            terms = random_terms(rng, 5)
            if not terms:
                continue
            k = rng.randrange(5)
            for p in (5, 7):
                assert pc._count_numpy(pc._split(terms, k), 4, p) == brute_force(terms, 5, p)
                checked += 1
        assert checked >= 16

    def test_split_rejects_a_degree_above_two(self) -> None:
        with pytest.raises(ValueError, match="degree 3"):
            pc._split((((3, 1), Fraction(1)), ((0, 0), Fraction(-1))), 0)

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
        # 3/2 = 7 mod 11 is not a square, while 3 is: a dropped denominator shows.
        assert pc._legendre(Fraction(3, 2), 11) == -1

    def test_interpolation_is_exact(self) -> None:
        assert pc._interpolate([3, 7, 11], [1, 5, 9]) == [-2, 1, 0]
        assert pc._interpolate([2, 3, 5], [4, 9, 25]) == [0, 0, 1]
        assert pc._interpolate([1, 2], [0, 1]) == [-1, 1]
        assert pc._interpolate([5], [7]) == [7]

    def test_interpolation_rejects_malformed_input(self) -> None:
        with pytest.raises(ValueError, match="length"):
            pc._interpolate([3, 7], [1, 5, 9])
        with pytest.raises(ValueError, match="length"):
            pc._interpolate([3, 7, 11], [1, 5])
        with pytest.raises(ValueError, match="distinct"):
            pc._interpolate([3, 7, 3], [1, 5, 1])


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


MU = sp.Symbol("mu", positive=True)
U = sp.symbols("u_1 u_2")
S = sp.Symbol("s")
M1, M2 = sp.symbols("m_1 m_2", nonnegative=True)


def bubble_g(massive: bool = True) -> sp.Expr:
    """G of the bubble with the energy scale mu, massive or massless."""
    u1, u2 = U
    mass = (M1**2 * u1 + M2**2 * u2) * (u1 + u2) if massive else 0
    return u1 + u2 + (mass - S * u1 * u2) / MU**2


def bubble_count(**kwargs: object) -> pc.TorusCount:
    return count_torus_points(bubble_g(), U, scale=MU, **kwargs)  # type: ignore[arg-type]


class TestCountTorusPoints:
    def test_massless_bubble_is_a_graph_over_one_variable(self) -> None:
        # u_2 = u_1 / (s u_1 - 1): p - 2 points, P(q) = q - 2, chi(X) = 1 and C = 1.
        count = count_torus_points(bubble_g(massive=False), U, scale=MU)
        assert all(n == p - 2 for p, n in count.counts)
        assert count.candidate_polynomial == (-2, 1)
        assert count.candidate_euler_characteristic == 1
        assert count.candidate_master_count == 1
        assert count.reason is None

    def test_two_mass_bubble_with_a_square_kallen_function(self) -> None:
        # lambda(1, 2, 6) = 1: p - 4 points and C = 3. The coefficient 6 excludes 3.
        count = bubble_count(point={S: 1, M1**2: 2, M2**2: 6})
        assert 3 in count.excluded_primes
        assert all(n == p - 4 for p, n in count.counts)
        assert count.candidate_polynomial == (-4, 1)
        assert count.candidate_master_count == 3
        assert count.seed is None
        assert count.point == ((M1**2, 2), (M2**2, 6), (S, 1))

    def test_two_mass_bubble_on_its_threshold(self) -> None:
        # lambda(9, 1, 4) = 0: p - 3 points and C = 2, where generic kinematics give 3.
        with pytest.raises(ValidationError, match="Landau surface"):
            bubble_count(point={S: 9, M1**2: 1, M2**2: 4})
        count = bubble_count(point={S: 9, M1**2: 1, M2**2: 4}, allow_singular=True)
        assert count.on_landau_surface
        assert count.candidate_polynomial == (-3, 1)
        assert count.candidate_master_count == 2

    def test_check_covers_both_signs_of_every_character(self) -> None:
        # lambda = -39 is not a square. (-39/p) = -1 at the fit primes 7, 17, 19 and at 23, 29,
        # 31 and 37, where the counts fit p - 2 and C = 1; 41 gives (-39/41) = 1, 37 points
        # against 39, and rejects the fit.
        count = bubble_count(point={S: 2, M1**2: 5, M2**2: 8})
        assert count.excluded_primes[:5] == (2, 3, 5, 11, 13)
        assert count.fit_primes == (7, 17, 19)
        assert count.verification_primes == (23, 29, 31, 37, 41)
        assert count.candidate_polynomial is None
        assert count.candidate_master_count is None
        assert count.reason == "the count at p = 41 is 37, where the fit predicts 39"

    def test_counts_that_are_not_polynomial(self) -> None:
        # lambda(1, 1, 1) = -3: p - 3 - (-3/p) is not a polynomial in p.
        count = bubble_count(point={S: 1, M1**2: 1, M2**2: 1})
        assert count.candidate_polynomial is None
        assert count.verification_primes == ()
        assert count.reason == "the polynomial through the fit counts has non-integer coefficients"

    def test_draws_follow_the_seed(self) -> None:
        # Every draw is admissible for the massless bubble and nothing must be a square, so
        # the first draw is used: one choice among the non-zero integers of [-20, 20].
        for seed in (0, 7):
            count = count_torus_points(bubble_g(massive=False), U, scale=MU, seed=seed)
            expected = random.Random(seed).choice([k for k in range(-20, 21) if k != 0])
            assert count.point == ((S, expected),)
            assert count.seed == seed

    def test_masses_are_drawn_through_their_squares(self) -> None:
        for seed in (0, 1):
            count = bubble_count(seed=seed)
            assert count == bubble_count(seed=seed)
            assert [key for key, _ in count.point] == [M1**2, M2**2, S]
            m1, m2, s = (value for _, value in count.point)
            assert 1 <= m1 <= 20 and 1 <= m2 <= 20 and -20 <= s <= 20 and s != 0
            assert pc._is_square(Fraction(kallen(int(s), int(m1), int(m2))))
            assert count.candidate_master_count == 3

    def test_landau_analysis_keeps_the_energy_scale_symbolic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # At mu = 1 the 2-face of the massive bubble becomes principal and the square test
        # would also ask s to be a square.
        seen: list[tuple[sp.Expr, object]] = []

        def spy(g: sp.Expr, variables: list[sp.Symbol], **kwargs: object) -> object:
            seen.append((g, kwargs.get("scale")))
            return landau_analysis_from_polynomial(g, variables, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(pc, "landau_analysis_from_polynomial", spy)
        bubble_count()
        assert len(seen) == 1
        assert seen[0][1] == MU
        assert MU in seen[0][0].free_symbols

    def test_a_given_landau_analysis_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        analysis = landau_analysis_from_polynomial(bubble_g(), list(U), scale=MU)
        expected = bubble_count(seed=3)

        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("the Landau analysis was repeated")

        monkeypatch.setattr(pc, "landau_analysis_from_polynomial", fail)
        assert bubble_count(seed=3, landau=analysis) == expected

    def test_a_symbol_occurring_only_squared_can_be_given_directly(self) -> None:
        by_square = bubble_count(point={S: 1, M1**2: 4, M2**2: 25})
        by_symbol = bubble_count(point={S: 1, M1: 2, M2: 5})
        assert by_symbol == by_square

    @pytest.mark.parametrize(
        ("point", "message"),
        [
            ({S: 1, M1**2: 2}, "no value for m_2"),
            ({S: 1, M1**2: 2, M2**2: 6, sp.Symbol("t"): 1}, "the key t"),
            ({S**2: 1, M1**2: 2, M2**2: 6}, "the key s"),
            ({S: sp.sqrt(2), M1**2: 2, M2**2: 6}, "not rational"),
            ({S: 1, M1**2: -2, M2**2: 6}, "negative"),
        ],
        ids=["missing", "unknown", "square-of-invariant", "irrational", "negative-square"],
    )
    def test_malformed_points_raise(self, point: dict[sp.Expr, object], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            bubble_count(point=point)

    def test_energy_scale_must_be_named(self) -> None:
        with pytest.raises(ValidationError, match="energy scale"):
            count_torus_points(bubble_g(), U)

    def test_degree_above_two_in_every_variable_raises(self) -> None:
        x, y = sp.symbols("x y")
        with pytest.raises(ValidationError, match="degree at least 3"):
            count_torus_points(x**3 + y**3 + S * x * y + 1, [x, y])

    def test_polynomial_vanishing_at_the_point_raises(self) -> None:
        x, y = sp.symbols("x y")
        with pytest.raises(ValidationError, match="vanishes identically"):
            count_torus_points(S * x + S * y, [x, y], point={S: 0}, allow_singular=True)

    def test_budget_and_prime_limits(self) -> None:
        with pytest.raises(ValidationError, match="max_evaluations=10"):
            bubble_count(max_evaluations=10)
        with pytest.raises(ValidationError, match="max_prime=13"):
            bubble_count(point={S: 1, M1**2: 2, M2**2: 6}, max_prime=13)

    def test_check_that_runs_out_of_primes_raises(self) -> None:
        # (-39/p) = -1 at every prime from 7 to 37 that is not excluded, so no prime up to 37
        # gives the character its other sign.
        with pytest.raises(ValidationError, match="max_prime=37 do not give every quadratic"):
            bubble_count(point={S: 2, M1**2: 5, M2**2: 8}, max_prime=37)

    def test_check_that_runs_over_the_budget_raises(self) -> None:
        # The fit and the first four checks cost 6 + 16 + 18 + 22 + 28 + 30 + 36 = 156
        # evaluations; the character of -39 then needs 41 as well.
        with pytest.raises(ValidationError, match="checking the fit needs more than"):
            bubble_count(point={S: 2, M1**2: 5, M2**2: 8}, max_evaluations=156)

    def test_rational_points_exclude_their_denominators(self) -> None:
        count = count_torus_points(bubble_g(massive=False), U, scale=MU, point={S: Fraction(3, 2)})
        assert count.point == ((S, Fraction(3, 2)),)
        assert count.excluded_primes == (2, 3)
        assert count.candidate_master_count == 1

    def test_square_test_falls_back_to_the_first_admissible_draw(self) -> None:
        # The discriminant 4 (x^2 + 1) is never a square for a non-zero integer x.
        u, x = sp.symbols("u x")
        count = count_torus_points(u**2 + 2 * x * u - 1, [u], seed=5)
        expected = random.Random(5).choice([k for k in range(-20, 21) if k != 0])
        assert count.point == ((x, expected),)

    def test_candidate_above_the_volume_bound_is_rejected(self) -> None:
        count = bubble_count(point={S: 1, M1**2: 2, M2**2: 6}, volume_bound=2)
        assert count.candidate_master_count is None
        assert count.reason == "the candidate master count 3 is not in [0, 2]"

    def test_argument_checks(self) -> None:
        with pytest.raises(ValidationError, match="backend"):
            bubble_count(backend="gpu")
        with pytest.raises(ValidationError, match="verification"):
            bubble_count(verification=0)
        with pytest.raises(ValidationError, match="distinct"):
            count_torus_points(bubble_g(), [U[0], U[0]], scale=MU)
        with pytest.raises(ValidationError, match="zero"):
            count_torus_points(sp.Integer(0), U)
