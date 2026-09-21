"""Tests for the Euler operators of feynkit.systems.euler.

The worked example throughout is the massless bubble. Its Lee-Pomeransky
polynomial is G = u_1 + u_2 - s u_1 u_2 / mu^2, whose three monomials
u_1 u_2, u_1 and u_2 give the A-matrix

    A = [[1, 1, 1],
         [1, 1, 0],
         [1, 0, 1]]

with the homogenising row on top and one row per Lee-Pomeransky variable. The
parameter vector is beta = (-D/2, -nu_1, -nu_2).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.systems import (
    create_euler_equations,
    create_euler_operators,
    format_euler_equation,
)

# Hand-written A-matrix of the massless bubble; see the module docstring.
BUBBLE_A = sp.Matrix([[1, 1, 1], [1, 1, 0], [1, 0, 1]])


@pytest.fixture  # type: ignore[misc]
def bubble_data() -> (
    Generator[tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr], None, None]
):
    """A-matrix, z variables, beta parameters and Phi for the massless bubble."""
    z_variables = list(sp.symbols("z_1 z_2 z_3"))
    dimension = sp.Symbol("D")
    nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
    beta = [-dimension / 2, -nu1, -nu2]
    phi = sp.Function("Phi")(*z_variables)
    yield BUBBLE_A, z_variables, beta, phi


class TestCreateEulerOperators:
    def test_one_operator_per_row(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        a_matrix, z_variables, _, _ = bubble_data

        operators = create_euler_operators(a_matrix, z_variables)

        assert len(operators) == a_matrix.rows == 3

    def test_operators_match_the_rows_by_hand(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        """Row r gives Ehat_r = sum_j A[r,j] z_j d/dz_j.

        Row 0 is (1, 1, 1), so Ehat_0 is the total degree operator; row 1 is
        (1, 1, 0) and row 2 is (1, 0, 1), each of which drops one variable.
        """
        a_matrix, z_variables, _, phi = bubble_data
        z1, z2, z3 = z_variables
        d1 = sp.Derivative(phi, z1)
        d2 = sp.Derivative(phi, z2)
        d3 = sp.Derivative(phi, z3)

        operators = create_euler_operators(a_matrix, z_variables)

        assert operators[0] == z1 * d1 + z2 * d2 + z3 * d3
        assert operators[1] == z1 * d1 + z2 * d2
        assert operators[2] == z1 * d1 + z3 * d3

    def test_integer_weights_appear_as_coefficients(self) -> None:
        """For A = [[1, 1, 1], [2, 1, 0], [0, 1, 2]] row 1 gives 2 z_1 d_1 + z_2 d_2."""
        a_matrix = sp.Matrix([[1, 1, 1], [2, 1, 0], [0, 1, 2]])
        z1, z2, z3 = sp.symbols("z_1 z_2 z_3")
        phi = sp.Function("Phi")(z1, z2, z3)

        operators = create_euler_operators(a_matrix, [z1, z2, z3])

        assert operators[1] == 2 * z1 * sp.Derivative(phi, z1) + z2 * sp.Derivative(phi, z2)
        assert operators[2] == z2 * sp.Derivative(phi, z2) + 2 * z3 * sp.Derivative(phi, z3)

    def test_wrong_number_of_variables_raises(self) -> None:
        with pytest.raises(ValidationError, match="must match"):
            create_euler_operators(BUBBLE_A, list(sp.symbols("z_1 z_2")))


class TestCreateEulerEquations:
    def test_one_equation_per_row(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        a_matrix, z_variables, beta, _ = bubble_data

        equations = create_euler_equations(a_matrix, beta, z_variables)

        assert len(equations) == 3

    def test_left_hand_sides_are_the_euler_operators(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        a_matrix, z_variables, beta, _ = bubble_data

        operators = create_euler_operators(a_matrix, z_variables)
        equations = create_euler_equations(a_matrix, beta, z_variables)

        for operator, equation in zip(operators, equations, strict=True):
            assert equation.lhs == operator

    def test_right_hand_sides_are_beta_times_phi(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        """beta = (-D/2, -nu_1, -nu_2), so the three right-hand sides are
        -D Phi / 2, -nu_1 Phi and -nu_2 Phi."""
        a_matrix, z_variables, beta, phi = bubble_data
        dimension = sp.Symbol("D")
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)

        equations = create_euler_equations(a_matrix, beta, z_variables)

        assert equations[0].rhs == -dimension * phi / 2
        assert equations[1].rhs == -nu1 * phi
        assert equations[2].rhs == -nu2 * phi

    def test_full_bubble_system_by_hand(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        """The three equations of the massless bubble, written out."""
        a_matrix, z_variables, beta, phi = bubble_data
        z1, z2, z3 = z_variables
        d1 = sp.Derivative(phi, z1)
        d2 = sp.Derivative(phi, z2)
        d3 = sp.Derivative(phi, z3)
        dimension = sp.Symbol("D")
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)

        equations = create_euler_equations(a_matrix, beta, z_variables)

        assert equations[0] == sp.Eq(z1 * d1 + z2 * d2 + z3 * d3, -dimension * phi / 2)
        assert equations[1] == sp.Eq(z1 * d1 + z2 * d2, -nu1 * phi)
        assert equations[2] == sp.Eq(z1 * d1 + z3 * d3, -nu2 * phi)

    def test_monomial_with_exponents_solving_a_c_equals_beta_is_a_solution(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        """Phi = z_1^c_1 z_2^c_2 z_3^c_3 satisfies the system when A c = beta.

        Solving the bubble system by hand,

            c_1 + c_2 + c_3 = -D/2,  c_1 + c_2 = -nu_1,  c_1 + c_3 = -nu_2,

        gives c = (D/2 - nu_1 - nu_2, nu_2 - D/2, nu_1 - D/2). The first
        exponent is the familiar overall power of the Mandelstam invariant for
        the massless bubble.
        """
        a_matrix, z_variables, beta, phi = bubble_data
        z1, z2, z3 = z_variables
        dimension = sp.Symbol("D")
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
        exponents = [dimension / 2 - nu1 - nu2, nu2 - dimension / 2, nu1 - dimension / 2]

        assert list(a_matrix * sp.Matrix(exponents)) == [sp.expand(b) for b in beta]

        solution = z1 ** exponents[0] * z2 ** exponents[1] * z3 ** exponents[2]
        equations = create_euler_equations(a_matrix, beta, z_variables)

        for equation in equations:
            substituted = equation.subs(phi, solution).doit()
            assert sp.simplify(substituted.lhs - substituted.rhs) == 0

    def test_a_monomial_violating_a_c_equals_beta_is_not_a_solution(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        """Shifting c_1 by one breaks every equation.

        The first column of A is (1, 1, 1), so a unit shift in c_1 moves each
        component of A c by one and no row can still match beta.
        """
        a_matrix, z_variables, beta, phi = bubble_data
        z1, z2, z3 = z_variables
        dimension = sp.Symbol("D")
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
        wrong = z1 ** (dimension / 2 - nu1 - nu2 + 1) * z2 ** (nu2 - dimension / 2)
        wrong = wrong * z3 ** (nu1 - dimension / 2)

        equations = create_euler_equations(a_matrix, beta, z_variables)
        substituted = equations[1].subs(phi, wrong).doit()

        assert sp.simplify(substituted.lhs - substituted.rhs) != 0

    def test_wrong_number_of_variables_raises(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        a_matrix, _, beta, _ = bubble_data

        with pytest.raises(ValidationError, match="A-matrix columns"):
            create_euler_equations(a_matrix, beta, list(sp.symbols("z_1 z_2")))

    def test_wrong_number_of_beta_parameters_raises(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        a_matrix, z_variables, beta, _ = bubble_data

        with pytest.raises(ValidationError, match="A-matrix rows"):
            create_euler_equations(a_matrix, beta[:2], z_variables)


class TestIntegralEulerEquations:
    def test_bubble_gkz_system_reproduces_the_helper(self) -> None:
        """FeynmanIntegral.gkz builds its equations with the same helper.

        The A-matrix that the integral computes has the same columns as the
        hand-written BUBBLE_A, up to the order in which the monomials of G are
        enumerated.
        """
        system = FeynmanIntegral.from_cnickel("11e|e|:zz").gkz

        assert {tuple(system.a_matrix.col(j)) for j in range(system.a_matrix.cols)} == {
            tuple(BUBBLE_A.col(j)) for j in range(BUBBLE_A.cols)
        }

        rebuilt = create_euler_equations(
            system.a_matrix, system.beta_parameters, system.z_variables
        )
        assert rebuilt == system.euler_equations

    def test_beta_is_minus_half_d_and_minus_the_propagator_exponents(self) -> None:
        """beta = (-D/2, -nu_1, -nu_2) for the two-propagator bubble."""
        system = FeynmanIntegral.from_cnickel("11e|e|:zz").gkz

        expected = [-system.dimension / 2] + [-nu for nu in system.propagator_exponents]
        assert [sp.expand(b) for b in system.beta_parameters] == [sp.expand(e) for e in expected]


class TestFormatEulerEquation:
    def test_shows_both_sides_separated_by_an_equals_sign(
        self,
        bubble_data: tuple[sp.Matrix, list[sp.Symbol], list[sp.Expr], sp.Expr],
    ) -> None:
        a_matrix, z_variables, beta, _ = bubble_data

        equations = create_euler_equations(a_matrix, beta, z_variables)
        text = format_euler_equation(equations[1])

        lhs_text, _, rhs_text = text.partition(" = ")
        assert lhs_text == str(equations[1].lhs)
        assert rhs_text == "-nu_1*Phi(z_1, z_2, z_3)"

    def test_simplify_lhs_reduces_the_left_hand_side(self) -> None:
        """With (z^2 - z)/(z - 1) on the left, simplification gives z.

        Passing simplify_lhs=False keeps the unreduced quotient, so the two
        renderings differ.
        """
        z1, z2 = sp.symbols("z_1 z_2")
        phi = sp.Function("Phi")(z1, z2)
        lhs = (z1**2 - z1) / (z1 - 1) * sp.Derivative(phi, z1)
        equation = sp.Eq(lhs, sp.Symbol("beta") * phi)

        simplified = format_euler_equation(equation, simplify_lhs=True)
        verbatim = format_euler_equation(equation, simplify_lhs=False)

        assert simplified == "z_1*Derivative(Phi(z_1, z_2), z_1) = beta*Phi(z_1, z_2)"
        assert verbatim != simplified
        assert "(z_1**2 - z_1)" in verbatim
