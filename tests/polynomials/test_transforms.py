"""Tests for the variable transformations in feynkit.polynomials.transforms.

The reference values are hand computations on small polynomials and on the
Lee-Pomeransky polynomial of the massless bubble,
G = u_1 + u_2 - s u_1 u_2 / mu^2.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.polynomials import (
    homogenise_polynomial,
    invert_variables,
    projective_transformation,
    rescale_variables,
)


@pytest.fixture  # type: ignore[misc]
def alphas() -> Generator[tuple[sp.Symbol, sp.Symbol, sp.Symbol], None, None]:
    """Three Schwinger-like parameters for the small worked examples."""
    yield sp.symbols("a1 a2 a3", positive=True)


@pytest.fixture  # type: ignore[misc]
def bubble_g() -> Generator[tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol], None, None]:
    """G of the massless bubble, its variables, and the symbols s and mu."""
    integral = FeynmanIntegral.from_cnickel("11e|e|:zz")
    g = sp.expand(integral.symanzik.g)
    variables = list(integral.symanzik.lp_parameters)
    yield g, variables, sp.Symbol("s", real=True), integral.graph.energy_scale


class TestInvertVariables:
    def test_inverts_every_listed_variable(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """a1 + a2 becomes 1/a1 + 1/a2."""
        a1, a2, _ = alphas

        result = invert_variables(a1 + a2, [a1, a2])

        assert sp.simplify(result - (1 / a1 + 1 / a2)) == 0

    def test_leaves_unlisted_variables_alone(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """Inverting only a1 in a1 + a2 gives 1/a1 + a2."""
        a1, a2, _ = alphas

        result = invert_variables(a1 + a2, [a1])

        assert sp.simplify(result - (1 / a1 + a2)) == 0

    def test_inversion_is_an_involution(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """Applying x -> 1/x twice returns the original expression."""
        a1, a2, _ = alphas
        expr = a1**2 + a1 * a2 + a2**2

        once = invert_variables(expr, [a1, a2])
        twice = invert_variables(once, [a1, a2])

        assert sp.simplify(twice - expr) == 0
        assert sp.simplify(once - expr) != 0

    def test_bubble_inversion_clears_to_a_constant_quadratic_term(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """Under u_i -> 1/u_i the bubble G becomes

            1/u_1 + 1/u_2 - s/(mu^2 u_1 u_2),

        so multiplying through by u_1 u_2 gives u_1 + u_2 - s/mu^2: the linear
        terms swap places and the quadratic term collapses to the constant
        -s/mu^2. This is the inversion that exchanges U and F for a one-loop
        graph.
        """
        g, variables, s, mu = bubble_g
        u1, u2 = variables

        result = invert_variables(g, variables)

        assert sp.simplify(result - (1 / u1 + 1 / u2 - s / (mu**2 * u1 * u2))) == 0
        assert sp.simplify(result * u1 * u2 - (u1 + u2 - s / mu**2)) == 0


class TestRescaleVariables:
    def test_uniform_rescaling_of_a_homogeneous_polynomial(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """a1^2 + a2^2 is homogeneous of degree 2, so a_i -> lam a_i pulls out lam^2."""
        a1, a2, _ = alphas
        lam = sp.Symbol("lambda", positive=True)

        result = rescale_variables(a1**2 + a2**2, {a1: lam * a1, a2: lam * a2})

        assert sp.expand(result - lam**2 * (a1**2 + a2**2)) == 0

    def test_uniform_rescaling_grades_the_bubble_by_degree(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """G is not homogeneous, so u_i -> lam u_i grades it:

            lam u_1 + lam u_2 - lam^2 s u_1 u_2 / mu^2.

        The coefficient of lam is U = u_1 + u_2 and the coefficient of lam^2 is
        F = -s u_1 u_2 / mu^2.
        """
        g, variables, s, mu = bubble_g
        u1, u2 = variables
        lam = sp.Symbol("lambda", positive=True)

        result = sp.expand(rescale_variables(g, {u1: lam * u1, u2: lam * u2}))

        assert sp.expand(result - (lam * u1 + lam * u2 - lam**2 * s * u1 * u2 / mu**2)) == 0
        poly = sp.Poly(result, lam)
        assert sp.expand(poly.coeff_monomial(lam) - (u1 + u2)) == 0
        assert sp.expand(poly.coeff_monomial(lam**2) + s * u1 * u2 / mu**2) == 0

    def test_torus_action_of_weight_one_minus_one_fixes_the_quadratic_term(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """Sending u_1 -> t u_1 and u_2 -> u_2/t gives

            t u_1 + u_2/t - s u_1 u_2 / mu^2,

        because the product u_1 u_2 has total weight 1 - 1 = 0 and so is fixed.
        """
        g, variables, s, mu = bubble_g
        u1, u2 = variables
        t = sp.Symbol("t", positive=True)

        result = sp.expand(rescale_variables(g, {u1: t * u1, u2: u2 / t}))

        assert sp.simplify(result - (t * u1 + u2 / t - s * u1 * u2 / mu**2)) == 0

    def test_empty_map_is_the_identity(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """Substituting nothing leaves the expression alone."""
        a1, a2, _ = alphas
        expr = a1**2 + a2**2

        assert rescale_variables(expr, {}) == expr


class TestProjectiveTransformation:
    def test_sets_the_normalisation_variable_to_one(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """a1^2 + a2^2 + a3^2 with a3 = 1 gives a1^2 + a2^2 + 1."""
        a1, a2, a3 = alphas

        result = projective_transformation(a1**2 + a2**2 + a3**2, [a1, a2, a3], a3)

        assert sp.expand(result - (a1**2 + a2**2 + 1)) == 0
        assert a3 not in result.free_symbols

    def test_variables_argument_is_ignored(
        self, alphas: tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    ) -> None:
        """Only the normalisation variable matters; the list is unused.

        This documents the signature: the middle argument is named with a
        leading underscore in the implementation and never read.
        """
        a1, a2, a3 = alphas
        expr = a1**2 + a2**2 + a3**2

        with_list = projective_transformation(expr, [a1, a2, a3], a3)
        without_list = projective_transformation(expr, [], a3)

        assert with_list == without_list

    def test_homogenise_then_normalise_recovers_the_bubble_polynomial(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """Homogenising G gives h u_1 + h u_2 - s u_1 u_2 / mu^2.

        Setting h = 1 undoes the homogenisation exactly, so the round trip
        returns G = u_1 + u_2 - s u_1 u_2 / mu^2.
        """
        g, variables, s, mu = bubble_g
        u1, u2 = variables
        h = sp.Symbol("h", positive=True)

        homogeneous = homogenise_polynomial(g, variables)
        assert sp.expand(homogeneous - (h * u1 + h * u2 - s * u1 * u2 / mu**2)) == 0

        result = projective_transformation(homogeneous, [u1, u2, h], h)

        assert sp.expand(result - g) == 0

    def test_normalising_a_lee_pomeransky_variable_drops_it(
        self, bubble_g: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, sp.Symbol]
    ) -> None:
        """Setting u_2 = 1 in G leaves u_1 + 1 - s u_1 / mu^2."""
        g, variables, s, mu = bubble_g
        u1, u2 = variables

        result = projective_transformation(g, variables, u2)

        assert sp.expand(result - (u1 + 1 - s * u1 / mu**2)) == 0
