"""Tests for the Cayley GKZ system of the Schwinger representation.

Oracles are the worked examples of Jimenez-Santacruz, Lopez-Arcos and
Quintero Velez, arXiv:2609.16107: eq. 53 (two-mass bubble), the massless
triangle (4 x 6) and the on-shell massless box (5 x 6). Columns are compared
as sets because the paper orders them differently.
"""

from __future__ import annotations

import sympy as sp

from feynkit.systems.cayley import cayley_matrix

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
