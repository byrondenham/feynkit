"""Tests for the Cayley GKZ system of the Schwinger representation.

Oracles are the worked examples of Jimenez-Santacruz, Lopez-Arcos and
Quintero Velez, arXiv:2609.16107: eq. 53 (two-mass bubble), the massless
triangle (4 x 6) and the on-shell massless box (5 x 6). Columns are compared
as sets because the paper orders them differently.
"""

from __future__ import annotations

import sympy as sp

from feynkit.systems.cayley import CayleyGKZSystem, cayley_matrix, create_cayley_system

u, u1, u2, u3 = sp.symbols("u u1 u2 u3")
s, t, m1, m2 = sp.symbols("s t m_1 m_2")
D = sp.Symbol("D", positive=True)


def _columns(matrix: sp.Matrix) -> set[tuple[int, ...]]:
    return {tuple(int(x) for x in matrix[:, j]) for j in range(matrix.cols)}


class TestCayleyMatrix:
    def test_two_mass_bubble_matches_eq_53(self) -> None:
        u_tilde = 1 + u
        f_tilde = m2**2 + (s + m1**2 + m2**2) * u + m1**2 * u**2
        A, u_support, f_support = cayley_matrix(u_tilde, f_tilde, [u])
        assert A.shape == (3, 5)
        assert _columns(A) == {(1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, 2)}
        assert [e for e, _ in u_support] == [(1,), (0,)]
        assert [e for e, _ in f_support] == [(2,), (1,), (0,)]

    def test_massless_triangle_matches_paper(self) -> None:
        u_tilde = 1 + u1 + u2
        f_tilde = s * u1 + t * u2 + sp.Symbol("u_kin") * u1 * u2
        A, _, _ = cayley_matrix(u_tilde, f_tilde, [u1, u2])
        expected = sp.Matrix(
            [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]]
        )
        assert _columns(A) == _columns(expected)

    def test_on_shell_massless_box_matches_paper(self) -> None:
        u_tilde = 1 + u1 + u2 + u3
        f_tilde = -s * u1 * u3 - t * u2
        A, _, _ = cayley_matrix(u_tilde, f_tilde, [u1, u2, u3])
        expected = sp.Matrix(
            [
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ]
        )
        assert _columns(A) == _columns(expected)

    def test_u_block_precedes_f_block(self) -> None:
        A, u_support, f_support = cayley_matrix(1 + u, m2**2 + u, [u])
        n = len(u_support)
        assert list(A[0, :n]) == [1] * n and list(A[0, n:]) == [0] * len(f_support)
        assert list(A[1, :n]) == [0] * n and list(A[1, n:]) == [1] * len(f_support)


class TestParameters:
    def _bubble(self, nu1: sp.Expr, nu2: sp.Expr) -> CayleyGKZSystem:
        return create_cayley_system(
            1 + u,
            m2**2 + (s + m1**2 + m2**2) * u + m1**2 * u**2,
            [u],
            dimension=D,
            propagator_exponents=[nu1, nu2],
            loop_count=1,
        )

    def test_bubble_beta_matches_eq_53(self) -> None:
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
        sys_ = self._bubble(nu1, nu2)
        beta = D / 2
        expected = [nu1 + nu2 - 2 * beta, beta - nu1 - nu2, -nu1]
        assert [
            sp.simplify(a - b) for a, b in zip(sys_.beta_parameters, expected, strict=True)
        ] == [0, 0, 0]

    def test_triangle_beta_at_unit_exponents(self) -> None:
        sys_ = create_cayley_system(
            1 + u1 + u2,
            s * u1 + t * u2 + sp.Symbol("u_kin") * u1 * u2,
            [u1, u2],
            dimension=D,
            propagator_exponents=[1, 1, 1],
            loop_count=1,
        )
        beta = D / 2
        expected = [3 - 2 * beta, beta - 3, -1, -1]
        assert [
            sp.simplify(a - b) for a, b in zip(sys_.beta_parameters, expected, strict=True)
        ] == [0] * 4

    def test_general_loop_number_enters_first_two_entries(self) -> None:
        sys_ = create_cayley_system(
            u1 + u2 + u1 * u2,
            u1 * u2 * (u1 + u2 + 1),
            [u1, u2],
            dimension=D,
            propagator_exponents=[1, 1, 1],
            loop_count=2,
            prefactor=sp.Integer(1),
        )
        beta = D / 2
        assert sp.simplify(sys_.beta_parameters[0] - (3 - 3 * beta)) == 0
        assert sp.simplify(sys_.beta_parameters[1] - (2 * beta - 3)) == 0
        # Verify loop_count L=2 enters Gamma(nu - L*D/2) where nu=3
        assert sp.simplify(sys_.prefactor - sp.gamma(3 - 2 * D / 2)) == 0

    def test_euler_equations_pin_rows_of_a_and_beta(self) -> None:
        sys_ = self._bubble(sp.Integer(1), sp.Integer(1))
        assert len(sys_.euler_equations) == sys_.a_matrix.rows
        variables = sys_.variables
        phi = sp.Function("Phi")(*variables)
        for r, eq in enumerate(sys_.euler_equations):
            assert sp.simplify(eq.rhs - sys_.beta_parameters[r] * phi) == 0
            lhs = sp.expand(eq.lhs)
            for j, var in enumerate(variables):
                coefficient = lhs.coeff(sp.Derivative(phi, var))
                assert (
                    sp.simplify(coefficient - sys_.a_matrix[r, j] * var) == 0
                ), f"Row {r}, column {j} mismatch"

    def test_prefactor_multiplies_gamma_of_nu_minus_l_half_d(self) -> None:
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
        base = sp.Symbol("P")
        sys_ = create_cayley_system(
            1 + u,
            m2**2 + u,
            [u],
            dimension=D,
            propagator_exponents=[nu1, nu2],
            loop_count=1,
            prefactor=base,
        )
        assert sp.simplify(sys_.prefactor - base * sp.gamma(nu1 + nu2 - D / 2)) == 0

    def test_wrong_exponent_count_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            create_cayley_system(1 + u, u, [u], dimension=D, propagator_exponents=[1], loop_count=1)
