"""
Exact check of the identity ``fk`` prints for a map between two configurations.

If M a_j + t = b_P(j) for every column a_j of A, with P a bijection onto the
columns of B, then substituting u_i = prod_k v_k^(M_ki) gives

    I_A(beta, z_P) = |det M| I_B(T beta, z),   T = [[1, 0], [t, M]],

where I_X(beta, z) = int u^(-beta') (sum_j z_j u^(x_j))^(beta_0) du/u carries
no Gamma prefactors.  Here A = {(0,0), (1,0), (0,1)} and
B = {(0,0), (2,0), (0,2)}, whose integrals have closed forms.  For A,
u_1 = (z1/z2) x_1 and u_2 = (z1/z3) x_2 reduce the integral to a Dirichlet
integral:

    I_A(beta, z) = Gamma(-beta_1) Gamma(-beta_2) Gamma(beta_1 + beta_2 - beta_0)
                   / Gamma(-beta_0) * z1^(beta_0 - beta_1 - beta_2) z2^beta_1 z3^beta_2

for z_j > 0 and each Gamma argument positive.  For B, w_i = u_i^2 gives
du_i/u_i = dw_i/(2 w_i), so I_B(g, z) = I_A((g_0, g_1/2, g_2/2), z)/4.
"""

from __future__ import annotations

import sympy as sp

from feynkit.cli import _gkz_identity

SIMPLEX = [(0, 0), (1, 0), (0, 1)]
DOUBLED = [(0, 0), (2, 0), (0, 2)]


def _i_a(beta: list[sp.Expr], z: list[sp.Expr]) -> sp.Expr:
    b0, b1, b2 = beta
    z1, z2, z3 = z
    return (
        sp.gamma(-b1)
        * sp.gamma(-b2)
        * sp.gamma(b1 + b2 - b0)
        / sp.gamma(-b0)
        * z1 ** (b0 - b1 - b2)
        * z2**b1
        * z3**b2
    )


def _i_b(g: list[sp.Expr], z: list[sp.Expr]) -> sp.Expr:
    g0, g1, g2 = g
    z1, z2, z3 = z
    return (
        sp.gamma(-g1 / 2)
        * sp.gamma(-g2 / 2)
        * sp.gamma((g1 + g2) / 2 - g0)
        / sp.gamma(-g0)
        * z1 ** (g0 - g1 / 2 - g2 / 2)
        * z2 ** (g1 / 2)
        * z3 ** (g2 / 2)
        / 4
    )


Z = list(sp.symbols("z1:4", positive=True))
S, NU1, NU2 = sp.symbols("s nu1 nu2")
BETA = [-S, -NU1, -NU2]


def test_scaling_puts_the_factor_on_the_target_side() -> None:
    identity = _gkz_identity(SIMPLEX, DOUBLED, sp.Matrix([[2, 0], [0, 2]]), sp.zeros(2, 1), BETA)
    assert identity is not None
    P, factor, t_beta = identity
    assert P == [0, 1, 2]
    assert factor == 4
    assert t_beta == [-S, -2 * NU1, -2 * NU2]
    lhs = _i_a(BETA, Z)
    assert sp.simplify(lhs - 4 * _i_b(t_beta, Z)) == 0
    assert sp.simplify(lhs - _i_b(t_beta, Z)) != 0


def test_three_cycle_puts_the_permutation_on_the_source_side() -> None:
    # (0,0) -> (2,0), (1,0) -> (0,2), (0,1) -> (0,0): det M = 4, P a 3-cycle.
    M = sp.Matrix([[-2, -2], [2, 0]])
    identity = _gkz_identity(SIMPLEX, DOUBLED, M, sp.Matrix([2, 0]), BETA)
    assert identity is not None
    P, factor, t_beta = identity
    assert P == [1, 2, 0]
    assert factor == 4
    rhs = _i_b(t_beta, Z)
    assert sp.simplify(_i_a(BETA, [Z[k] for k in P]) - 4 * rhs) == 0
    P_inv = [P.index(j) for j in range(3)]
    assert sp.simplify(_i_a(BETA, [Z[k] for k in P_inv]) - 4 * rhs) != 0
    assert sp.simplify(_i_a(BETA, [Z[k] for k in P]) - rhs) != 0


def test_no_identity_without_a_column_bijection() -> None:
    # (0,0) -> (0,0), (1,0) -> (1,0), (0,1) -> (0,1) misses (2,0) and (0,2).
    assert _gkz_identity(SIMPLEX, DOUBLED, sp.eye(2), sp.Matrix([0, 0]), BETA) is None
    assert _gkz_identity(SIMPLEX, DOUBLED, sp.zeros(2, 2), sp.Matrix([0, 0]), BETA) is None
