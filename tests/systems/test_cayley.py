"""Tests for the Cayley GKZ system of the Schwinger representation.

Oracles are the worked examples of Jimenez-Santacruz, Lopez-Arcos and
Quintero Velez, arXiv:2609.16107: eq. 53 (two-mass bubble), the massless
triangle (4 x 6) and the on-shell massless box (5 x 6). Columns are compared
as sets because the paper orders them differently.
"""

from __future__ import annotations

import warnings

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.systems.cayley import (
    CayleyGKZSystem,
    cayley_matrix,
    create_cayley_system,
    lp_to_cayley,
)

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
        A, u_support, f_support = cayley_matrix(u_tilde, f_tilde, [u1, u2])
        expected = sp.Matrix(
            [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 1, 1]]
        )
        assert _columns(A) == _columns(expected)
        assert [e for e, _ in u_support] == [(1, 0), (0, 1), (0, 0)]
        assert [e for e, _ in f_support] == [(1, 1), (1, 0), (0, 1)]

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

    def test_on_shell_box_beta_at_unit_exponents(self) -> None:
        sys_ = create_cayley_system(
            1 + u1 + u2 + u3,
            -s * u1 * u3 - t * u2,
            [u1, u2, u3],
            dimension=D,
            propagator_exponents=[1, 1, 1, 1],
            loop_count=1,
        )
        expected = [4 - 2 * D / 2, D / 2 - 4, -1, -1, -1]
        assert [
            sp.simplify(a - b) for a, b in zip(sys_.beta_parameters, expected, strict=True)
        ] == [0] * 5

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

        from feynkit.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            create_cayley_system(1 + u, u, [u], dimension=D, propagator_exponents=[1], loop_count=1)


class TestReduction:
    def test_bubble_reduction_matches_eq_54(self) -> None:
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
        sys_ = create_cayley_system(
            1 + u,
            m2**2 + (s + m1**2 + m2**2) * u + m1**2 * u**2,
            [u],
            dimension=D,
            propagator_exponents=[nu1, nu2],
            loop_count=1,
        )
        with pytest.warns(UserWarning, match="does not lie in the span"):
            red = sys_.restrict_to_f_block()
        assert _columns(red.a_matrix) == {(1, 0), (1, 1), (1, 2)}
        expected = [D / 2 - nu1 - nu2, -nu1]
        assert [sp.simplify(a - b) for a, b in zip(red.beta_parameters, expected, strict=True)] == [
            0,
            0,
        ]
        assert red.z_variables == sys_.z_variables
        assert len(red.euler_equations) == 2

    def test_off_shell_box_reduction_matches_eq_red_box(self) -> None:
        p1, p2, p3, p4 = sp.symbols("p1^2 p2^2 p3^2 p4^2")
        f_tilde = p4 * u1 + t * u2 + p1 * u1 * u2 + p3 * u3 + s * u1 * u3 + p2 * u2 * u3
        sys_ = create_cayley_system(
            1 + u1 + u2 + u3,
            f_tilde,
            [u1, u2, u3],
            dimension=D,
            propagator_exponents=[1, 1, 1, 1],
            loop_count=1,
        )
        with pytest.warns(UserWarning, match="does not lie in the span"):
            red = sys_.restrict_to_f_block()
        expected = sp.Matrix(
            [[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1]]
        )
        assert _columns(red.a_matrix) == _columns(expected)
        assert [sp.simplify(b) for b in red.beta_parameters] == [D / 2 - 4, -1, -1, -1]


class TestToricIdeal:
    def test_reduced_bubble_kernel_is_z1_z3_minus_z2_squared(self) -> None:
        nu1, nu2 = sp.symbols("nu_1 nu_2", positive=True)
        sys_ = create_cayley_system(
            1 + u,
            m2**2 + (s + m1**2 + m2**2) * u + m1**2 * u**2,
            [u],
            dimension=D,
            propagator_exponents=[nu1, nu2],
            loop_count=1,
        )
        from feynkit.algebra.toric import compute_toric_ideal_generators

        with pytest.warns(UserWarning, match="does not lie in the span"):
            red = sys_.restrict_to_f_block()
        gens = compute_toric_ideal_generators(red.a_matrix)
        z1, z2, z3 = red.z_variables
        assert len(gens) == 1
        g = sp.expand(gens[0].subs(dict(zip(sp.symbols("z_1:4"), (z1, z2, z3), strict=True))))
        assert g in (sp.expand(z1 * z3 - z2**2), sp.expand(z2**2 - z1 * z3))

    def test_full_bubble_toric_ideal_uses_w_and_z(self) -> None:
        from feynkit.algebra.toric import compute_toric_ideal_generators

        sys_ = create_cayley_system(
            1 + u,
            m2**2 + (s + m1**2 + m2**2) * u + m1**2 * u**2,
            [u],
            dimension=D,
            propagator_exponents=[1, 1],
            loop_count=1,
        )
        raw = compute_toric_ideal_generators(sys_.a_matrix)
        standard = sp.symbols(f"z_1:{sys_.a_matrix.cols + 1}")
        mapping = dict(zip(standard, sys_.variables, strict=True))
        expected = {sp.expand(g.subs(mapping, simultaneous=True)) for g in raw}
        got = {sp.expand(g) for g in sys_.toric_ideal()}
        assert got == expected
        assert raw  # the Cayley matrix of the bubble has a non-trivial kernel
        # The first n columns are the U~ block: their variables are the w's.
        n = len(sys_.u_support)
        assert sys_.variables[:n] == sys_.w_variables
        assert sys_.variables[n:] == sys_.z_variables


class TestAdmissibleRestriction:
    """The F~ block is a true subsystem where beta lies in the span of its columns.

    For a block whose columns span the hyperplane y_0 = 0 that is nu = (L+1) D/2, where
    the exponent of U~ vanishes (Britto, Grimm and Hoefnagels, arXiv:2606.09978, p. 46).
    """

    def _bubble(self, dimension: sp.Expr, powers: list[sp.Expr]) -> CayleyGKZSystem:
        return create_cayley_system(
            1 + u,
            m2**2 + (s + m1**2 + m2**2) * u + m1**2 * u**2,
            [u],
            dimension=dimension,
            propagator_exponents=powers,
            loop_count=1,
        )

    def test_warns_off_the_admissible_point(self) -> None:
        with pytest.warns(UserWarning, match=r"nu = \(L\+1\) D/2"):
            self._bubble(sp.Integer(4), [sp.Integer(1), sp.Integer(1)]).restrict_to_f_block()

    @pytest.mark.parametrize(
        ("dimension", "powers"),
        [
            (sp.Integer(2), [sp.Integer(1), sp.Integer(1)]),
            (sp.Symbol("nu_1") + sp.Symbol("nu_2"), [sp.Symbol("nu_1"), sp.Symbol("nu_2")]),
        ],
        ids=["D=2", "D=nu"],
    )
    def test_silent_where_nu_is_l_plus_one_d_over_two(
        self, dimension: sp.Expr, powers: list[sp.Expr]
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._bubble(dimension, powers).restrict_to_f_block()


class TestLpToCayley:
    @pytest.mark.parametrize("cnickel", ["11e|e|:nn", "111e|e|:nnz", "12e|3e|3e|e|:nnnn"])
    def test_maps_the_columns_and_the_parameter(self, cnickel: str) -> None:
        fi = FeynmanIntegral.from_cnickel(cnickel)
        n = len(fi.symanzik.lp_parameters)
        t = lp_to_cayley(n, fi.loop_count)
        # det T = 1 with the columns ordered (ones, alpha_N, alpha_1, ...), so (-1)^(N-1) here.
        assert t.det() == (-1) ** (n - 1)
        assert _columns(t * fi.gkz.a_matrix) == _columns(fi.schwinger_gkz.a_matrix)
        mapped = t * sp.Matrix(fi.gkz.beta_parameters)
        cayley = fi.schwinger_gkz.beta_parameters
        assert [sp.expand(a - b) for a, b in zip(mapped, cayley, strict=True)] == [0] * (n + 1)
