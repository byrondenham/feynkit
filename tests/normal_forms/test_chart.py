"""The frame of a lattice chart, and the lift of chart maps to Z^n."""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit.normal_forms._chart import chart_frame, lift_linear
from feynkit.polytope import polytope_data

POINT_SETS = [
    [(0, 0), (2, 2)],
    [(0, 0, 0), (2, 0, 0), (0, 1, 0)],
    [(0, 0, 0), (2, 2, 0), (0, 1, 1)],
    [(1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1)],
    [(3, 1, -2), (1, 1, 1), (5, 1, -5)],
]


@pytest.mark.parametrize("points", POINT_SETS)
def test_the_frame(points: list[tuple[int, ...]]) -> None:
    frame = chart_frame(points)
    n, d = len(points[0]), frame.dimension
    Q = frame.frame
    assert abs(Q.det()) == 1
    assert Q * frame.coframe == sp.eye(n)
    # B = S H^T, with S the last d columns of Q.
    B = sp.Matrix(frame.chart.basis).T
    assert Q[:, n - d :] * frame.hermite.T == B
    assert frame.index == polytope_data(points).sublattice_index
    assert frame.coordinates.shape == (len(points), d)


@pytest.mark.parametrize("points", POINT_SETS)
def test_the_identity_lifts_to_the_identity(points: list[tuple[int, ...]]) -> None:
    frame = chart_frame(points)
    assert lift_linear(frame, frame, sp.eye(frame.dimension)) == sp.eye(len(points[0]))


def test_the_reflection_of_a_segment() -> None:
    frame = chart_frame([(0, 0), (2, 2)])
    U = lift_linear(frame, frame, sp.Matrix([[-1]]))
    assert U is not None
    assert abs(U.det()) == 1
    assert U * sp.Matrix([2, 2]) == sp.Matrix([-2, -2])


def test_a_map_that_does_not_preserve_the_integer_points() -> None:
    # (0, 0, 0), (2, 0, 0) and (0, 1, 0): swapping the last two points preserves the lattice
    # their differences span but not the integer points of their plane.
    points = [(0, 0, 0), (2, 0, 0), (0, 1, 0)]
    frame = chart_frame(points)
    c = [sp.Matrix(x) for x in frame.chart.coordinates]
    swap = (
        sp.Matrix.hstack(c[2] - c[0], c[1] - c[0])
        * sp.Matrix.hstack(c[1] - c[0], c[2] - c[0]).inv()
    )
    assert lift_linear(frame, frame, swap) is None
    lifted = lift_linear(frame, frame, swap, integral=False)
    assert lifted is not None
    assert lifted * sp.Matrix([2, 0, 0]) == sp.Matrix([0, 1, 0])
