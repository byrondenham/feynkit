"""Tests for the Milnor fibres and Lê numbers of feynkit.milnor.

Oracles: the Milnor numbers of the simple singularities A_k, D_4 and E_6
(k, 4 and 6, by hand from the Jacobian ideals), a submersion, and the worked
examples of D. B. Massey, "Non-isolated hypersurface singularities and Lê
cycles", arXiv:1410.3312: Ex. 3.12 (p. 21) and Ex. 3.13 (pp. 21-22), whose
Lê numbers are stated there. Their Euler characteristics are not printed;
Ex. 4.2 (p. 23) asks for them, and the formula
chi(F) = 1 + sum_k (-1)^(n-k) lambda^k on p. 23 gives 2 and 5.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp

from feynkit.landau import _singular_binary
from feynkit.milnor import LeNumbers, le_numbers, milnor_fibre_euler_characteristic

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

T, U, V, W, X, Y, Z = sp.symbols("t u v w x y z")


def origin(variables: list[sp.Symbol]) -> list[int]:
    return [0] * len(variables)


def massey_3_12(a: int, b: int) -> sp.Expr:
    """Massey's Ex. 3.12 (p. 21), y^2 - x^a - t x^b with a > b > 1, in (t, x, y)."""
    return Y**2 - X**a - T * X**b


# Massey's Ex. 3.13 (p. 21), in (u, v, w, x, y).
MASSEY_3_13 = Y**2 - X**3 - (U**2 + V**2 + W**2) * X**2


# --- isolated critical points and submersions -------------------------------------------------


@requires_singular
class TestIsolated:
    """chi~(F) = (-1)^n mu at an isolated critical point, n + 1 the number of variables."""

    @pytest.mark.parametrize(
        ("name", "f", "variables", "mu"),
        [
            ("A_1", X**2 + Y**2, [X, Y], 1),
            ("A_3", X**4 + Y**2, [X, Y], 3),
            ("A_4", X**5 + Y**2, [X, Y], 4),
            ("D_4", X**2 * Y + Y**3, [X, Y], 4),
            ("E_6", X**3 + Y**4, [X, Y], 6),
            ("A_2 in three variables", X**3 + Y**2 + Z**2, [X, Y, Z], 2),
        ],
    )
    def test_milnor_numbers(
        self, name: str, f: sp.Expr, variables: list[sp.Symbol], mu: int
    ) -> None:
        le = le_numbers(f, variables, origin(variables))
        n = len(variables) - 1
        assert le.method == "isolated"
        assert le.critical_dimension == 0
        assert le.numbers == (mu,)
        assert le.polar_numbers == ()
        assert le.reduced_euler_characteristic == (-1) ** n * mu
        assert milnor_fibre_euler_characteristic(f, variables, origin(variables)) == (-1) ** n * mu

    def test_isolated_coordinates_are_the_variables(self) -> None:
        le = le_numbers(X**3 + Y**4, [X, Y], [0, 0])
        assert le.flag == ((1, 0), (0, 1))
        assert le.prime is not None and le.prime < 2**29

    def test_a_rational_point(self) -> None:
        half = sp.Rational(1, 2)
        f = (X - half) ** 3 + (Y + 2) ** 4
        assert milnor_fibre_euler_characteristic(f, [X, Y], [half, -2]) == -6
        assert le_numbers(f, [X, Y], [Fraction(1, 2), -2]).numbers == (6,)


class TestSubmersion:
    def test_a_submersion_has_contractible_fibre(self) -> None:
        f = X + Y**2
        assert milnor_fibre_euler_characteristic(f, [X, Y], [0, 0]) == 0
        le = le_numbers(f, [X, Y], [0, 0])
        assert le == LeNumbers((), (), -1, ((1, 0), (0, 1)), None, "submersion")
        assert le.reduced_euler_characteristic == 0

    def test_a_smooth_point_of_a_singular_hypersurface(self) -> None:
        # x y vanishes at (1, 0), where its gradient is (0, 1).
        assert milnor_fibre_euler_characteristic(X * Y, [X, Y], [1, 0]) == 0


# --- Massey's examples ------------------------------------------------------------------------


@requires_singular
class TestMasseyExamples:
    @pytest.mark.parametrize(("a", "b"), [(3, 2), (4, 2), (5, 3)])
    def test_example_3_12(self, a: int, b: int) -> None:
        # p. 21: lambda^0 = b, lambda^1 = b - 1; n = 2, so chi(F) = 1 + b - (b - 1) = 2.
        f = massey_3_12(a, b)
        le = le_numbers(f, [T, X, Y], [0, 0, 0])
        assert le.method == "attaching"
        assert le.critical_dimension == 1
        assert le.numbers == (b, b - 1)
        assert le.reduced_euler_characteristic == 2 - 1
        assert milnor_fibre_euler_characteristic(f, [T, X, Y], [0, 0, 0]) == 2 - 1

    def test_example_3_13(self) -> None:
        # pp. 21-22: lambda^3 = 1, lambda^2 = 4, lambda^1 = 4, lambda^0 = 5; n = 4, so
        # chi(F) = 1 + 5 - 4 + 4 - 1 = 5.
        variables = [U, V, W, X, Y]
        le = le_numbers(MASSEY_3_13, variables, origin(variables))
        assert le.critical_dimension == 3
        assert le.numbers == (5, 4, 4, 1)
        assert le.reduced_euler_characteristic == 5 - 1
        assert milnor_fibre_euler_characteristic(MASSEY_3_13, variables, origin(variables)) == 4
