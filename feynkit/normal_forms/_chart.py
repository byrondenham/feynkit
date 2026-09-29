"""
The lattice chart of a point set, with a frame of Z^n adapted to it.

A point set S in Z^n of affine dimension d < n is full-dimensional in its
lattice chart x = o + B c (feynkit.polytope.lattice_chart), where the columns
of B, n x d, are a basis of the lattice L spanned by the differences of the
points, and the chart points generate Z^d affinely. The normal forms search
for maps there and lift them back to Z^n.

The Hermite normal form of B^T with its transform gives B^T W = [0 | H], with
W unimodular and H of size d x d. Put Q = W^-T. Then B = S H^T, where S, the
last d columns of Q, is a basis of sat(L), the integer points of the linear
span of L, and the first n - d columns of Q span a complement. In the chart,
sat(L) is the lattice H^-T Z^d, which contains Z^d with index [sat(L) : L] =
|det H|. A chart map c -> M c + s, with M in GL_d(Q), is the restriction of
an integral affine map of Z^n exactly when H^T M H^-T is an integer matrix,
and then

    Q diag(I_(n-d), H^T M H^-T) Q^-1

is one such map: it acts as B M B^-1 on the span of L and as the identity on
the complement. Between two frames the same formula, with H' and Q' of the
target, lifts a map from one chart to the other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .. import _exact
from ..polytope import LatticeChart, lattice_chart


@dataclass(frozen=True)
class ChartFrame:
    """The lattice chart of a point set and a frame of Z^n adapted to it.

    Attributes
    ----------
    chart
        The chart x = o + B c of the points, from lattice_chart.
    frame
        Q in GL_n(Z); its last d columns are a basis of sat(L).
    coframe
        Q^-1, the transpose of the transform of the Hermite normal form.
    hermite
        H, d x d, with B = Q[:, n - d:] H^T.
    """

    chart: LatticeChart
    frame: sp.ImmutableMatrix
    coframe: sp.ImmutableMatrix
    hermite: sp.ImmutableMatrix

    @property
    def dimension(self) -> int:
        """d, the affine dimension of the points."""
        return len(self.chart.basis)

    @property
    def coordinates(self) -> np.ndarray:
        """The chart points as an N x d integer array, in the order of the points."""
        count = len(self.chart.coordinates)
        return np.array(self.chart.coordinates, dtype=np.int64).reshape(count, self.dimension)

    @property
    def index(self) -> int:
        """[sat(L) : L], which is |det H|."""
        return abs(int(self.hermite.det()))


def chart_frame(points: Sequence[Sequence[int]] | np.ndarray) -> ChartFrame:
    """The lattice chart of the points with its frame; see the module docstring."""
    chart = lattice_chart(points)
    ambient = len(chart.origin)
    d = len(chart.basis)
    form = _exact.hermite_normal_form_with_transform([list(b) for b in chart.basis], ambient)
    transform = sp.Matrix(form.transform)
    return ChartFrame(
        chart=chart,
        frame=sp.ImmutableMatrix(transform.inv().T),
        coframe=sp.ImmutableMatrix(transform.T),
        hermite=sp.ImmutableMatrix(d, d, [x for row in form.hermite for x in row]),
    )


def lift_linear(
    source: ChartFrame, target: ChartFrame, m: sp.Matrix, *, integral: bool = True
) -> sp.ImmutableMatrix | None:
    """Lift the linear part m of a chart map from the source chart to the target chart.

    Returns Q' diag(I, H'^T m H^-T) Q^-1, which maps B c to B' m c, or None
    when integral is true and H'^T m H^-T is not an integer matrix, that is
    when m does not map sat(L) into sat(L'). For m in GL_d(Z) the image is
    all of sat(L') exactly when the indices [sat(L) : L] and [sat(L') : L']
    agree, which the callers check or, with one frame, have. With integral
    false m may be any invertible rational matrix, and so is the lift.
    """
    n = source.frame.rows
    d = source.dimension
    inner = target.hermite.T * sp.Matrix(m) * source.hermite.T.inv()
    if integral and not all(x.is_Integer for x in inner):
        return None
    block = sp.diag(sp.eye(n - d), inner) if n > d else inner
    return sp.ImmutableMatrix(target.frame * block * source.coframe)
