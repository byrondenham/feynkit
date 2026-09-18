"""
Standard A-configurations used in the dissertation on GKZ systems.

Each factory function returns an :class:`~feynkit.AConfiguration` built from
a known A-matrix.  Matrices are given in column-= monomial form with the
first row being the homogenisation row of all-ones.

Configurations
--------------
triangle_a_config()
    Massless triangle (feynkit-computed 4×6 A-matrix).
triple_k_a_config()
    Triple-K conformal 3-point integral (4×6).
four_point_simplex_a_config()
    Standard 4-simplex (5×5) — the simplest non-trivial GKZ example.
banana3_a_config()
    Massless 3-propagator banana (feynkit-computed 4×4).
"""

from __future__ import annotations

import sympy as sp

from ..a_configuration import AConfiguration


def triangle_a_config() -> AConfiguration:
    """
    Massless triangle GKZ A-matrix (4×6).

    Computed by feynkit from ``FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")``.
    The six columns correspond to the six monomials of the Lee–Pomeransky
    polynomial G = U + F for the 1-loop massless triangle.

    Matrix (rows = coordinates, columns = monomials)::

        1  1  1  1  1  1     ← homogenisation
        1  1  1  0  0  0     ← u1 exponent
        1  0  0  1  1  0     ← u2 exponent
        0  1  0  1  0  1     ← u3 exponent
    """
    M = sp.Matrix(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ]
    )
    return AConfiguration(M, is_homogenized=True)


def triple_k_a_config() -> AConfiguration:
    """
    Triple-K conformal 3-point integral GKZ A-matrix (4×6).

    The Lee–Pomeransky-style polynomial for the conformal 3-point integral
    (triple-K integral) is::

        G = u1²u2u3 + u1u2²u3 + u1u2u3² + u2u3 + u1u3 + u1u2

    Monomial exponent vectors (columns)::

        (2,1,1), (1,2,1), (1,1,2), (0,1,1), (1,0,1), (1,1,0)

    With homogenisation::

        1  1  1  1  1  1
        2  1  1  0  1  1
        1  2  1  1  0  1
        1  1  2  1  1  0
    """
    M = sp.Matrix(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 1, 1, 0, 1, 1],
            [1, 2, 1, 1, 0, 1],
            [1, 1, 2, 1, 1, 0],
        ]
    )
    return AConfiguration(M, is_homogenized=True)


def four_point_simplex_a_config() -> AConfiguration:
    """
    Standard 4-simplex GKZ A-matrix (5×5).

    The simplest GKZ example: the 4-simplex Δ₄ = conv(e₁,…,e₄,0) in ℝ⁴.
    With homogenisation the A-matrix is the (5×5) identity-like matrix::

        1  1  1  1  1
        1  0  0  0  0
        0  1  0  0  0
        0  0  1  0  0
        0  0  0  1  0

    (The last column corresponds to the origin point (0,0,0,0).)
    """
    M = sp.Matrix(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    return AConfiguration(M, is_homogenized=True)


def banana3_a_config() -> AConfiguration:
    """
    Massless 3-propagator banana GKZ A-matrix (4×4).

    Computed by feynkit from ``FeynmanIntegral.from_cnickel("111e|e|:zzz")``.

    Matrix::

        1  1  1  1
        1  1  1  0
        1  1  0  1
        1  0  1  1
    """
    M = sp.Matrix(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
        ]
    )
    return AConfiguration(M, is_homogenized=True)
