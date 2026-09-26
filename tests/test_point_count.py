"""Tests for the finite-field point counts of feynkit.point_count.

Oracles: brute-force enumeration of (F_p^*)^N for small p, and the massive
bubble, whose zero set is an affine conic with p - 3 - (lambda/p) points in
the torus, lambda being the Källén function of s, m_1^2 and m_2^2.
"""

from __future__ import annotations

import itertools
import math
import random
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import landau as landau_module
from feynkit import point_count as pc
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary, landau_analysis, landau_analysis_from_polynomial
from feynkit.point_count import count_torus_points, critical_point_count
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

    @pytest.mark.parametrize("chunk", [7, 180])
    def test_two_outer_variables(self, monkeypatch: pytest.MonkeyPatch, chunk: int) -> None:
        # Five variables leave two outer ones, both decoded from each chunk index. A chunk of
        # 7 points holds one outer point; one of 180 holds 11 at p = 5 and 5 at p = 7, so the
        # chunks do not start at a digit boundary.
        monkeypatch.setattr(pc, "_CHUNK", chunk)
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


@pytest.mark.parametrize(
    "cnickel",
    [
        "11e|e|:zz",
        "11e|e|:nz",
        "11e|e|:nn",
        "12e|2e|e|:zzz",
        "12e|2e|e|:nzz",
        "12e|2e|e|:nnn",
        "12e|3e|3e|e|:zzzz",
        "12e|3e|3e|e|:nnzz",
        "12e|3e|3e|e|:nnnn",
        "12e|23|3|e|:zzzzz",
        "12e|23|3|e|:nzzzz",
        "12e|23|3|e|:nnnnn",
        "12e|22|e|:nzzz",
        "12ee|22e|e|:nnnn",
        "111e|e|:nnn",
    ],
)
def test_graph_polytopes_admit_no_characters_of_higher_order(cnickel: str) -> None:
    # Every edge has lattice length at most 2 and every face lattice index 1, for generic
    # kinematics and with every external p_i^2 = 0, so the check's guard never refuses them.
    fi = FeynmanIntegral.from_cnickel(cnickel)
    g = fi.symanzik.g
    variables = list(fi.symanzik.lp_parameters)
    on_shell = {x: 0 for x in g.free_symbols if x.name.startswith("p") and x.name.endswith("^2")}
    for h in (g, sp.expand(g.subs(on_shell))):
        assert pc._higher_order(polytope_data(sp.Poly(h, *variables).monoms())) is None


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
        with pytest.raises(ValidationError, match="face discriminant vanishes at the point"):
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
            ({S: 1, M1: 2, M1**2: 4, M2**2: 6}, "more than one value for m_1"),
        ],
        ids=[
            "missing",
            "unknown",
            "square-of-invariant",
            "irrational",
            "negative-square",
            "symbol-and-square",
        ],
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
        with pytest.raises(ValidationError, match="max_prime must be at most"):
            bubble_count(max_prime=2 * 10**9)

    def test_budget_covers_the_fit_and_the_first_checks(self) -> None:
        # The fit primes 5, 11 and 13 cost 26 evaluations and the checks at 17, 19, 23 and 29
        # another 84; the whole is refused before anything is counted.
        with pytest.raises(ValidationError, match="counting needs at least 110 evaluations"):
            bubble_count(point={S: 1, M1**2: 2, M2**2: 6}, max_evaluations=50)

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
        # 2 is excluded anyway; the denominator 3 of 5/3 is excluded for its own sake.
        count = count_torus_points(bubble_g(massive=False), U, scale=MU, point={S: Fraction(5, 3)})
        assert count.excluded_primes == (2, 3, 5)

    def test_square_test_falls_back_to_the_first_admissible_draw(self) -> None:
        # The discriminant 4 (x^2 + 1) is never a square for a non-zero integer x.
        u, x = sp.symbols("u x")
        count = count_torus_points(u**2 + 2 * x * u - 1, [u], seed=5)
        expected = random.Random(5).choice([k for k in range(-20, 21) if k != 0])
        assert count.point == ((x, expected),)

    def test_candidate_above_the_volume_bound_is_rejected(self) -> None:
        count = bubble_count(point={S: 1, M1**2: 2, M2**2: 6}, volume_bound=2)
        assert count.candidate_master_count is None
        assert count.reason == (
            "the master count 3 given by the fit is not in [0, N! Vol(Newt G)] = [0, 2]"
        )

    def test_argument_checks(self) -> None:
        with pytest.raises(ValidationError, match="backend"):
            bubble_count(backend="gpu")
        with pytest.raises(ValidationError, match="verification"):
            bubble_count(verification=0)
        with pytest.raises(ValidationError, match="distinct"):
            count_torus_points(bubble_g(), [U[0], U[0]], scale=MU)
        with pytest.raises(ValidationError, match="zero"):
            count_torus_points(sp.Integer(0), U)
        with pytest.raises(ValidationError, match="seed must be an integer"):
            bubble_count(seed=None)
        assert bubble_count(seed=None, point={S: 1, M1**2: 2, M2**2: 6}).seed is None

    def test_a_landau_analysis_of_another_polynomial_raises(self) -> None:
        # Another support, and then the same support with other coefficients.
        for g in (bubble_g(massive=False), bubble_g().subs(S, sp.Symbol("t"))):
            other = landau_analysis_from_polynomial(g, list(U), scale=MU)
            with pytest.raises(ValidationError, match="not a Landau analysis of the polynomial"):
                bubble_count(landau=other)

    def test_a_character_of_a_face_that_is_not_principal_is_covered(self) -> None:
        # The one-mass triangle at m_1^2 = 17, p_i^2 = (-5, -8, 18). 801, the Källén function
        # of the p_i^2, is a factor of the discriminant of the top face, which is not
        # principal. (801/p) = -1 at every prime from 7 to 43 that is not excluded, so the
        # check runs on to 47, where (801/47) = 1, and 47 rejects the fit.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        g = fi.symanzik.g
        names = {x.name: x for x in g.free_symbols}
        point = {names["m_1"] ** 2: 17, names["p1^2"]: -5, names["p2^2"]: -8, names["p3^2"]: 18}
        count = count_torus_points(
            g, list(fi.symanzik.lp_parameters), scale=fi.graph.energy_scale, point=point
        )
        assert count.candidate_master_count is None
        assert count.verification_primes[-1] == 47
        assert count.reason is not None and count.reason.startswith("the count at p = 47 ")

    def test_products_of_characters_are_covered(self) -> None:
        # The edge polynomial u^2 + (x + y) u + (x^2 + x y + y^2)/4 has discriminant x y, so
        # the count is 1 + (x y/p). At x = 2, y = 45 the characters of -1, 2 and 5 each take
        # both signs on 7, 11, 17, ..., 29, while (90/p) = -1 there; 31 rejects the fit.
        u, x, y = sp.symbols("u x y")
        g = u**2 + (x + y) * u + (x**2 + x * y + y**2) / 4
        count = count_torus_points(g, [u], point={x: 2, y: 45})
        assert count.candidate_master_count is None
        assert count.verification_primes[-1] == 31

    def test_a_constant_the_landau_analysis_drops_is_covered(self) -> None:
        # u^2 + 37 has 1 + (-37/p) roots, and (-37/p) = -1 at every odd prime up to 17, where
        # the counts fit 0. The Landau analysis sees no kinematics, but the vertex coefficient
        # 37 and the edge discriminant -148 carry the character, and 19 rejects the fit.
        u = sp.Symbol("u")
        count = count_torus_points(u**2 + 37, [u])
        assert count.candidate_master_count is None
        assert count.verification_primes[-1] == 19

    def test_the_check_primes(self) -> None:
        # u_1 + u_2 + x u_1 u_2 has p - 2 points and C = 1. At x = 1 the only character is
        # (-1/p), which takes both signs on the fit primes 3, 5 and 7, so the check takes
        # exactly `verification` primes after them.
        u1, u2 = U
        x = sp.Symbol("x")
        g = u1 + u2 + x * u1 * u2
        count = count_torus_points(g, U, point={x: 1})
        assert count.fit_primes == (3, 5, 7)
        assert count.verification_primes == (11, 13, 17, 19)
        assert count.candidate_master_count == 1
        # One check prime is enough, since the fit primes count towards the coverage.
        assert count_torus_points(g, U, point={x: 1}, verification=1).verification_primes == (11,)
        # At x = (5 * 13 * 17)^2 the fit primes 3, 7 and 11 are all 3 mod 4, as are 19 and
        # 23, so (-1/p) needs 29.
        count = count_torus_points(g, U, point={x: (5 * 13 * 17) ** 2}, verification=1)
        assert count.fit_primes == (3, 7, 11)
        assert count.verification_primes == (19, 23, 29)
        assert count.candidate_master_count == 1

    def test_a_fit_with_a_q_to_the_n_term_is_rejected(self) -> None:
        # u^2 - 11 has no roots mod 3 and two mod 5: the line through the fit counts is q - 3.
        u, x = sp.symbols("u x")
        count = count_torus_points(u**2 - x, [u], point={x: 11})
        assert count.fit_primes == (3, 5)
        assert count.reason == "the polynomial through the fit counts has a q^1 term"

    def test_volume_bound(self) -> None:
        # 1 - u^2 has two roots, so C = 2, while the segment [0, 2] has normalised volume 1 in
        # the lattice it spans, of index 2. A polytope that is not full-dimensional bounds C
        # by 0.
        u, v = sp.symbols("u v")
        count = count_torus_points(1 - u**2, [u])
        assert count.candidate_master_count == 2
        assert pc._volume_bound(polytope_data([(0,), (2,)])) == 2
        assert pc._volume_bound(polytope_data([(0, 0), (1, 1)])) == 0
        assert count_torus_points(1 + u * v, [u, v]).candidate_master_count == 0

    def test_edge_discriminants_are_excluded(self) -> None:
        # The edge 1 + x u^3 has lattice length 3 and discriminant -27 x^2, which the Landau
        # analysis does not see since the edge is a simplex; mod 3, 1 + u^3 = (1 + u)^3. The
        # counts, p - 1 - gcd(3, p - 1), fail the fit, which the reason prefers to the edge.
        u, v, x = sp.symbols("u v x")
        count = count_torus_points(1 + x * u**3 + v, [u, v], point={x: 1})
        assert count.excluded_primes == (2, 3)
        assert count.reason == "the polynomial through the fit counts has non-integer coefficients"

    def test_a_character_of_higher_order_refuses_the_candidate(self) -> None:
        # v + u^6 + 108 has p - 1 points less the sixth roots of -108 = -4 * 27, which exist
        # only when p = 1 mod 3 and 2 is a cube mod p: at 31, but at no prime up to 29. The
        # counts fit q - 1 and C = 0, where chi(X) = 6. The edge from 108 to u^6 has lattice
        # length 6, so the candidate is refused before the check, and the budget need only
        # cover the fit primes 5, 7 and 11.
        u, v, x = sp.symbols("u v x")
        g = v + u**6 + x
        count = count_torus_points(g, [u, v], point={x: 108})
        assert count.counts == ((5, 4), (7, 6), (11, 10))
        assert count.verification_primes == ()
        assert count.candidate_polynomial is None
        assert count.candidate_euler_characteristic is None
        assert count.candidate_master_count is None
        assert count.reason == (
            "the Newton polytope has an edge of lattice length 6, so the counts may depend on "
            "characters of order above 2, which the check does not cover"
        )
        assert count_torus_points(g, [u, v], point={x: 108}, max_evaluations=20) == count
        terms = (((0, 1), Fraction(1)), ((6, 0), Fraction(1)), ((0, 0), Fraction(108)))
        assert pc._count_numpy(pc._split(terms, 1), 1, 31) == 31 - 7

    def test_edges_and_faces_that_admit_characters_of_higher_order(self) -> None:
        def reason(*monomials: tuple[int, ...]) -> str | None:
            return pc._higher_order(polytope_data(monomials))

        # Lattice length 2 passes; length 3 fails even with every point present.
        assert reason((0,), (2,)) is None
        assert reason((0,), (1,), (2,), (3,)) is not None
        assert "an edge of lattice length 3" in str(reason((0,), (3,)))
        # A face is judged by the exponent of its lattice quotient, the largest Smith invariant.
        # A triangle with quotient Z/2 and the square [0, 2]^2 with (Z/2)^2, of index 4, pass;
        # 1 + u^2 v + u v^2 (Z/3) and 1 + u + u^2 v^4 (Z/4) fail, with edges of length 1 or 2.
        assert reason((0, 0), (1, 1), (2, 0)) is None
        assert reason((0, 0), (2, 0), (0, 2), (2, 2)) is None
        found = "a face of dimension 2 whose lattice quotient has exponent"
        assert f"{found} 3," in str(reason((0, 0), (2, 1), (1, 2)))
        assert f"{found} 4," in str(reason((0, 0), (1, 0), (2, 4)))

    def test_a_face_whose_quotient_has_exponent_two_is_accepted(self) -> None:
        # (1 - u^2)(1 - v^2) has 4p - 8 points, so C = 4: X is (C^* less +-1)^2. The square
        # [0, 2]^2 has lattice quotient (Z/2)^2, of index 4 but exponent 2.
        u, v = sp.symbols("u v")
        count = count_torus_points(sp.expand((1 - u**2) * (1 - v**2)), [u, v])
        assert all(n == 4 * p - 8 for p, n in count.counts)
        assert count.candidate_master_count == 4

    def test_an_edge_of_length_three_refuses_the_candidate(self) -> None:
        # u^3 - 3u + 1 has a cyclic Galois group and discriminant 81, a square, so for p != 3
        # the number of its roots mod p, 0 or 3, follows a cubic character that no quadratic
        # character sees. Mod 3 it is (u + 1)^3, with one root, and 3 is excluded. The counts
        # at 5, 7 and 11 fit q - 1 and C = 0, where C = 3. The edge [0, 3] holds every point,
        # so its index is 1, but its lattice length is 3.
        u, v = sp.symbols("u v")
        count = count_torus_points(v + u**3 - 3 * u + 1, [u, v])
        assert count.counts == ((5, 4), (7, 6), (11, 10))
        assert count.candidate_master_count is None
        assert str(count.reason).startswith("the Newton polytope has an edge of lattice length 3")

    def test_draws_are_admissible(self) -> None:
        # The first draw of seed 63 gives x = y = 9, where the coefficient x - y vanishes.
        u, x, y = sp.symbols("u x y")
        count = count_torus_points((x - y) * u + 1, [u], seed=63)
        (_, first_x), (_, first_y) = count.point
        assert first_x != first_y
        assert not count.on_landau_surface

    def test_the_variable_of_lowest_degree_is_eliminated(self) -> None:
        u1, u2 = U
        g = u1 + u2 + (M1**2 * u1 * (u1 + u2) - S * u1 * u2) / MU**2
        assert count_torus_points(g, U, scale=MU).eliminated == u2
        assert count_torus_points(bubble_g(massive=False), U, scale=MU).eliminated == u1

    @pytest.mark.parametrize(
        ("cnickel", "master"),
        [("12e|23|3|e|:nzzzz", 7), ("12e|22|e|:nzzz", 3), ("12ee|22e|e|:nnnn", None)],
    )
    def test_face_discriminants_odd_in_a_mass(self, cnickel: str, master: int | None) -> None:
        # The Landau analysis factorises in the masses, so a factor m_1^2 of a discriminant
        # becomes m_1; the count multiplies such a discriminant by its image under
        # m_1 -> -m_1, a polynomial in m_1^2. Faces of more than 5 points are skipped to save
        # time; enough odd discriminants remain. The candidates 7 and 3 agree with counts at
        # further primes; the last graph's result is not pinned.
        fi = FeynmanIntegral.from_cnickel(cnickel)
        g, mu = fi.symanzik.g, fi.graph.energy_scale
        variables = list(fi.symanzik.lp_parameters)
        analysis = landau_analysis_from_polynomial(g, variables, scale=mu, max_face_points=5)
        masses = [x for x in g.free_symbols if x.name.startswith("m_")]
        assert any(
            sp.expand(face.discriminant.subs(m, -m) - face.discriminant) != 0
            for face in analysis.face_discriminants
            for m in masses
        )
        count = count_torus_points(g, variables, scale=mu, landau=analysis)
        assert count.skipped_faces == len(analysis.skipped_faces) > 0
        assert count.counts
        if master is not None:
            assert count.candidate_master_count == master


requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")


class TestFlint:
    def test_flint_agrees_with_numpy(self) -> None:
        pytest.importorskip("flint")
        rng = random.Random(5)
        for _ in range(100):
            n = rng.randint(1, 3)
            terms = random_terms(rng, n)
            if not terms:
                continue
            parts = pc._split(terms, rng.randrange(n))
            for p in (3, 5, 7, 11, 13):
                if any(c.denominator % p == 0 for _, c in terms):
                    continue
                assert pc._count_flint(parts, n - 1, p) == pc._count_numpy(parts, n - 1, p)

    def test_flint_backend_gives_the_numpy_result(self) -> None:
        pytest.importorskip("flint")
        numpy = bubble_count(seed=2)
        flint = bubble_count(seed=2, backend="flint")
        assert flint.backend == "flint"
        assert flint.counts == numpy.counts
        assert flint.candidate_master_count == numpy.candidate_master_count

    def test_flint_backend_counts_every_prime_with_flint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both backends give the same counts, so only a spy shows which one counted.
        counted: list[int] = []

        def spy(parts: pc._Parts, others: int, p: int) -> int:
            counted.append(p)
            return pc._count_numpy(parts, others, p)

        monkeypatch.setattr(pc, "_flint_available", lambda: True)
        monkeypatch.setattr(pc, "_count_flint", spy)
        count = bubble_count(seed=2, backend="flint")
        assert counted == [p for p, _ in count.counts]

    def test_flint_backend_needs_python_flint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_flint_available", lambda: False)
        with pytest.raises(RuntimeError, match="python-flint"):
            bubble_count(backend="flint")


class TestCriticalPointCount:
    @requires_singular
    def test_two_mass_bubble_has_three_critical_points(self) -> None:
        g = bubble_g().subs(MU, 1)
        for seed in (0, 1):
            assert critical_point_count(g, U, {S: 1, M1**2: 1, M2**2: 1}, seed=seed) == 3

    @requires_singular
    def test_agrees_with_the_candidates(self) -> None:
        massless = count_torus_points(bubble_g(massive=False), U, scale=MU)
        massive = bubble_count(point={S: 1, M1**2: 2, M2**2: 6})
        for massive_, count in ((False, massless), (True, massive)):
            g = bubble_g(massive=massive_).subs(MU, 1)
            assert critical_point_count(g, U, dict(count.point)) == count.candidate_master_count

    @requires_singular
    def test_masses_are_given_through_their_squares(self) -> None:
        # s = 1 = (m_2 - m_1)^2 is the pseudo-threshold of m_e = (2, 3), where lambda = 0 and 2
        # critical points remain; m_e = (4, 9) would give the generic 3.
        g = bubble_g().subs(MU, 1)
        assert critical_point_count(g, U, {S: 1, M1**2: 4, M2**2: 9}) == 2
        assert critical_point_count(g, U, {S: 1, M1: 2, M2: 3}) == 2

    @requires_singular
    def test_fully_massive_kite(self) -> None:
        # 30 critical points, as for generic masses; modulo two primes this takes a fraction
        # of a second, where the same count over Q ran for more than 15 minutes.
        fi = FeynmanIntegral.from_cnickel("12e|23|3|e|:nnnnn")
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        names = {x.name: x for x in g.free_symbols}
        point = {names["s"]: 13} | {
            names[f"m_{e}"] ** 2: m for e, m in enumerate((2, 3, 5, 7, 11), start=1)
        }
        assert critical_point_count(g, list(fi.symanzik.lp_parameters), point) == 30

    @pytest.mark.parametrize(
        ("point", "message"),
        [
            ({S: 1, M1**2: 1}, "no value for m_2"),
            ({S**2: 4, M1**2: 1, M2**2: 1}, r"the key s\*\*2"),
            ({S: 1, M1**2: 1, M2**2: 1, U[0]: 1}, "the key u_1"),
            ({S: 1, M1**2: 1, M2**2: 1, sp.Symbol("t"): 1}, "the key t"),
            ({S: 1, M1**2: -1, M2**2: 1}, "negative"),
        ],
        ids=["missing", "square-of-invariant", "variable", "unknown", "negative-square"],
    )
    def test_malformed_points_raise(self, point: dict[sp.Expr, object], message: str) -> None:
        # The keys follow count_torus_points: s occurs to the first power, so s**2 = 4 would
        # leave its sign open.
        with pytest.raises(ValidationError, match=message):
            critical_point_count(bubble_g().subs(MU, 1), U, point)  # type: ignore[arg-type]

    def test_malformed_arguments_raise(self) -> None:
        g, point = bubble_g().subs(MU, 1), {S: 1, M1**2: 1, M2**2: 1}
        for variables in ([], [U[0], U[0]]):
            with pytest.raises(ValidationError, match="distinct"):
                critical_point_count(g, variables, point)
        with pytest.raises(ValidationError, match="energy scale"):
            critical_point_count(bubble_g(), U, point)
        with pytest.raises(ValidationError, match="rational coefficients"):
            critical_point_count(sp.sqrt(2) * U[0] + U[1], U, {})
        with pytest.raises(ValidationError, match="timeout"):
            critical_point_count(g, U, point, timeout=0)

    def test_polynomial_vanishing_at_the_point_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Raised before Singular is needed. Unchecked, the zero generators would send the search
        # for primes through every prime below 2^31; hiding Singular makes a regression fail
        # at once instead.
        monkeypatch.setattr(pc, "_singular_binary", lambda: None)
        x, y = sp.symbols("x y")
        with pytest.raises(ValidationError, match="vanishes identically"):
            critical_point_count(S * x + S * y, [x, y], {S: 0})

    def test_needs_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_singular_binary", lambda: None)
        with pytest.raises(RuntimeError, match="Singular"):
            critical_point_count(bubble_g().subs(MU, 1), U, {S: 1, M1**2: 1, M2**2: 1})


def fake_singular(
    monkeypatch: pytest.MonkeyPatch, stdout: str = "", error: Exception | None = None
) -> dict[str, object]:
    """Replace Singular by a stub printing stdout or raising error; returns what it was given."""
    seen: dict[str, object] = {}

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, script=Path(args[-1]).read_text())
        if error is not None:
            raise error
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(pc, "_singular_binary", lambda: "Singular")
    monkeypatch.setattr(pc.subprocess, "run", run)
    return seen


class TestCriticalPointsModuloPrimes:
    G, POINT = bubble_g().subs(MU, 1), {S: 1, M1**2: 1, M2**2: 1}

    def test_counts_modulo_the_two_largest_primes_below_2_to_the_31(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = fake_singular(monkeypatch, "3\n3\n")
        assert critical_point_count(self.G, U, self.POINT, timeout=7) == 3
        assert seen["timeout"] == 7
        script = str(seen["script"])
        assert "ring r0 = 2147483647," in script
        assert "ring r1 = 2147483629," in script
        # The coefficients, cleared of denominators up to about 6 * 10^10, reach Singular mod p.
        ideals = [line for line in script.splitlines() if line.startswith("ideal")]
        assert max(int(n) for line in ideals for n in re.findall(r"\d+", line)) < 2**31

    def test_primes_avoid_the_denominators_of_the_point(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = fake_singular(monkeypatch, "3\n3\n")
        critical_point_count(self.G, U, {S: Fraction(1, 2147483647), M1**2: 1, M2**2: 1})
        assert "ring r0 = 2147483629," in str(seen["script"])
        assert "ring r1 = 2147483587," in str(seen["script"])

    def test_counts_that_differ_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, "3\n4\n")
        with pytest.raises(ComputationError, match="3 modulo 2147483647 and 4 modulo"):
            critical_point_count(self.G, U, self.POINT)

    def test_a_locus_that_is_not_finite_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, "-1\n-1\n")
        with pytest.raises(ComputationError, match="finite"):
            critical_point_count(self.G, U, self.POINT)

    def test_output_that_is_not_two_counts_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Singular exits with 0 after an error in the script and prints the error.
        for stdout in ("   ? error occurred in or before critical.sing line 2\n", "3\n", ""):
            fake_singular(monkeypatch, stdout)
            with pytest.raises(RuntimeError, match="Singular"):
                critical_point_count(self.G, U, self.POINT)

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, error=subprocess.TimeoutExpired(["Singular"], 7))
        with pytest.raises(ComputationError, match="timeout=7"):
            critical_point_count(self.G, U, self.POINT, timeout=7)


class TestFeynmanIntegralTorusCount:
    def test_massive_bubble(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        count = fi.torus_count(seed=0)
        m1, m2 = (edge.get_mass() for edge in fi.graph.get_internal_edges())
        assert [key for key, _ in count.point] == [m1**2, m2**2, sp.Symbol("s", real=True)]
        assert count.variables == tuple(fi.symanzik.lp_parameters)
        assert count.on_shell == ()
        assert count.candidate_master_count == 3

    def test_on_shell_substitutes_the_momentum_products(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        legs = {sp.Symbol(f"p{i}^2", real=True): 0 for i in range(1, 5)}
        count = fi.torus_count(on_shell=legs)
        assert [str(key) for key, _ in count.point] == ["s12", "s23"]
        assert count.on_shell == tuple(legs.items())
        assert count.candidate_master_count == 3

    def test_on_shell_names_symbols_of_the_momentum_products(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        # The invariants are real symbols; a symbol without that assumption is another symbol.
        with pytest.raises(ValidationError, match=r"on_shell names p1\^2, which"):
            fi.torus_count(on_shell={sp.Symbol("p1^2"): 0})

    def test_massive_tadpole_has_one_variable_and_one_master(self) -> None:
        count = FeynmanIntegral.from_cnickel("0|:n", use_mandelstam=False).torus_count()
        assert len(count.variables) == 1
        assert count.candidate_polynomial == (1,)
        assert count.candidate_master_count == 1

    def test_kinematic_constraints_are_not_applied(self) -> None:
        s = sp.Symbol("s", real=True)
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])
        with pytest.raises(ValidationError, match="kinematic_constraints"):
            fi.torus_count()

    def test_a_given_landau_analysis_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nz")
        analysis = landau_analysis(fi)
        expected = fi.torus_count(seed=4)

        def fail(*args: object, **kwargs: object) -> None:
            raise AssertionError("the Landau analysis was repeated")

        monkeypatch.setattr(landau_module, "landau_analysis", fail)
        assert fi.torus_count(seed=4, landau=analysis) == expected

    def test_a_landau_analysis_without_on_shell_raises(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        legs = {sp.Symbol(f"p{i}^2", real=True): 0 for i in range(1, 5)}
        with pytest.raises(ValidationError, match="with on_shell applied"):
            fi.torus_count(on_shell=legs, landau=landau_analysis(fi))

    @pytest.mark.parametrize("key", [sp.Symbol("p1^2"), "p1^2"], ids=["symbol", "string"])
    def test_a_key_without_the_assumptions_of_the_products_is_explained(self, key: object) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        hint = "keys carry their assumptions, as Symbol('p1^2', real=True)"
        with pytest.raises(ValidationError, match=re.escape(hint)):
            fi.torus_count(on_shell={key: 0})  # type: ignore[dict-item]
        with pytest.raises(ValidationError) as info:
            fi.torus_count(on_shell={sp.Symbol("t", real=True): 0})
        assert "assumptions" not in str(info.value)

    def test_on_shell_values_must_not_contain_its_keys(self) -> None:
        # Substituted one after the other, {p2^2: p1^2, p1^2: 0} would depend on the order.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        p1, p2 = (sp.Symbol(f"p{i}^2", real=True) for i in (1, 2))
        for on_shell, key in (({p2: p1, p1: 0}, "p1"), ({p1: p2, p2: 0}, "p2")):
            with pytest.raises(ValidationError, match=rf"values contain its keys {key}\^2;"):
                fi.torus_count(on_shell=on_shell)

    def test_on_shell_values_become_sympy_expressions(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        p1, p2 = (sp.Symbol(f"p{i}^2", real=True) for i in (1, 2))
        expected = fi.torus_count(on_shell={p1: sp.Rational(3, 2)})
        assert expected.on_shell == ((p1, sp.Rational(3, 2)),)
        for value in (Fraction(3, 2), "3/2"):
            assert fi.torus_count(on_shell={p1: value}) == expected  # type: ignore[dict-item]
        for value in (0, p2 / 4):
            ((_, recorded),) = fi.torus_count(on_shell={p1: value}).on_shell
            assert isinstance(recorded, sp.Expr)
            assert recorded == value

    @pytest.mark.parametrize(
        "value",
        [0.5, "0.5", sp.Float(0.5), sp.Symbol("s12", real=True) / 2.0, None, [0]],
        ids=["float", "string", "sympy-float", "expression", "none", "list"],
    )
    def test_on_shell_values_must_be_exact_expressions(self, value: object) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        with pytest.raises(ValidationError, match=r"gives p1\^2 the value .* not an exact"):
            fi.torus_count(on_shell={sp.Symbol("p1^2", real=True): value})  # type: ignore[dict-item]

    @pytest.mark.parametrize("value", ["s12", sp.Symbol("s12")], ids=["string", "symbol"])
    def test_on_shell_values_must_use_the_symbols_of_the_integral(self, value: object) -> None:
        # sympify makes "s12" a symbol without assumptions, which would be drawn separately.
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        hint = "carry their assumptions, as Symbol('s12', real=True)"
        with pytest.raises(ValidationError, match=re.escape(hint)):
            fi.torus_count(on_shell={sp.Symbol("p1^2", real=True): value})  # type: ignore[dict-item]

    def test_on_shell_values_may_use_the_masses(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        p1 = sp.Symbol("p1^2", real=True)
        m1 = fi.graph.get_internal_edges()[0].get_mass()
        assert fi.torus_count(on_shell={p1: m1**2}).on_shell == ((p1, m1**2),)
        hint = "as Symbol('m_1', nonnegative=True, real=True)"
        with pytest.raises(ValidationError, match=re.escape(hint)):
            fi.torus_count(on_shell={p1: "m_1**2"})  # type: ignore[dict-item]


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide.md"
GUIDE_HEADING = "\n## Torus point counts\n"


def test_guide_example_prints_what_the_guide_says(capsys: pytest.CaptureFixture[str]) -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    assert GUIDE_HEADING in guide
    section = guide.split(GUIDE_HEADING, 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    printed = section.split("prints\n\n```\n", 1)[1].split("```", 1)[0]
    exec(compile(code, "docs/guide.md", "exec"), {})
    assert capsys.readouterr().out == printed
    assert printed.splitlines()[2:] == [
        "(-4, 1) 3",
        "None the polynomial through the fit counts has non-integer coefficients",
        "True 2",
    ]


def _later_guide_examples() -> list[str]:
    """The Python blocks of the guide's point-count section after the first."""
    guide = GUIDE.read_text(encoding="utf-8")
    if GUIDE_HEADING not in guide:
        return []
    section = guide.split(GUIDE_HEADING, 1)[1].split("\n## ", 1)[0]
    return re.findall(r"```python\n(.*?)```", section, re.DOTALL)[1:]


def test_guide_has_later_examples() -> None:
    assert len(_later_guide_examples()) == 2


@pytest.mark.parametrize(
    "code",
    [pytest.param(code, id=f"example-{i}") for i, code in enumerate(_later_guide_examples(), 2)],
)
def test_later_guide_examples_print_their_comments(
    code: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Each print line ends with a comment giving exactly what it prints.
    if "critical_point_count(" in code and _singular_binary() is None:
        pytest.skip("Singular not installed")
    expected = [line.rsplit("# ", 1)[1] for line in code.splitlines() if line.startswith("print(")]
    assert expected
    exec(compile(code, "docs/guide.md", "exec"), {})
    assert capsys.readouterr().out.splitlines() == expected
