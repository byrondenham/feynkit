"""
Exact integer routines behind feynkit.polytope.

Points are tuples of Python integers, matrices are lists of rows, and a face
is stored as a bitmask over point indices. Nothing here depends on a
floating-point tolerance.
"""

from __future__ import annotations

import math
import numbers
import operator
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, NamedTuple, SupportsIndex, cast

import numpy as np

from .core.exceptions import ComputationError, ValidationError

IntMatrix = list[list[int]]


# --- input -------------------------------------------------------------------


def _as_int(value: object, where: int) -> int:
    if isinstance(value, SupportsIndex):
        return operator.index(value)
    if isinstance(value, numbers.Rational) and value.denominator == 1:
        return int(value.numerator)
    if (
        isinstance(value, (float, np.floating))
        and math.isfinite(value)
        and float(value).is_integer()
    ):
        return int(value)
    raise ValidationError(f"point {where} has the non-integer coordinate {value!r}")


def integer_points(points: object) -> list[tuple[int, ...]]:
    """The points as tuples of Python integers.

    Accepts a two-dimensional NumPy array or a sequence of sequences.
    Integral floats are accepted; any other non-integer raises.

    Raises
    ------
    ValidationError
        If a coordinate is not an integer, or the points do not all have the
        same number of coordinates.
    """
    if isinstance(points, np.ndarray):
        if points.ndim == 2:
            rows: list[Any] = points.tolist()
        elif points.size == 0:
            return []
        else:
            raise ValidationError(
                "points must form a two-dimensional array, one row per point; "
                f"got shape {points.shape}"
            )
    else:
        rows = list(cast(Iterable[Any], points))
    out: list[tuple[int, ...]] = []
    for i, row in enumerate(rows):
        try:
            coordinates = tuple(row)
        except TypeError:
            raise ValidationError(f"point {i} is not a sequence of coordinates: {row!r}") from None
        point = tuple(_as_int(x, i) for x in coordinates)
        if out and len(point) != len(out[0]):
            raise ValidationError(
                f"point {i} has {len(point)} coordinates, point 0 has {len(out[0])}"
            )
        out.append(point)
    return out


def mask_indices(mask: int) -> list[int]:
    """The indices of the set bits of mask, in increasing order."""
    out: list[int] = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def dot(u: Sequence[int], v: Sequence[int]) -> int:
    """The dot product of two integer vectors of equal length."""
    return sum(a * b for a, b in zip(u, v, strict=True))


# --- determinants and rank ---------------------------------------------------


def determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Determinant of a square integer matrix by Bareiss elimination; 1 for 0 x 0."""
    a = [[int(x) for x in row] for row in matrix]
    n = len(a)
    sign, previous = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if a[i][k]), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) // previous
        previous = pivot
    return sign * a[n - 1][n - 1] if n else 1


def rank(rows: Sequence[Sequence[int]]) -> int:
    """Rank of an integer matrix by fraction-free (Bareiss) elimination."""
    a = [[int(x) for x in row] for row in rows]
    if not a:
        return 0
    m, n = len(a), len(a[0])
    r, previous = 0, 1
    for c in range(n):
        pivot_row = next((i for i in range(r, m) if a[i][c]), None)
        if pivot_row is None:
            continue
        a[r], a[pivot_row] = a[pivot_row], a[r]
        pivot = a[r][c]
        for i in range(r + 1, m):
            for j in range(c + 1, n):
                a[i][j] = (pivot * a[i][j] - a[i][c] * a[r][j]) // previous
            a[i][c] = 0
        previous = pivot
        r += 1
        if r == m:
            break
    return r


class AffineSpan:
    """The affine span of points added one at a time, in exact integer arithmetic.

    The differences from the first point are kept as reduced rows, each divided
    by the gcd of its entries, with the column of its first non-zero entry as
    pivot; every row is zero in the pivot columns of the rows before it.
    """

    def __init__(self) -> None:
        self._base: tuple[int, ...] | None = None
        self._rows: list[tuple[int, list[int]]] = []

    @property
    def dimension(self) -> int:
        """Affine dimension of the points so far; -1 before the first point."""
        return -1 if self._base is None else len(self._rows)

    def add(self, point: Sequence[int]) -> bool:
        """Add a point; return whether it lies outside the span so far."""
        if self._base is None:
            self._base = tuple(point)
            return True
        vector = [a - b for a, b in zip(point, self._base, strict=True)]
        for pivot, row in self._rows:
            if vector[pivot]:
                p, q = row[pivot], vector[pivot]
                vector = [p * x - q * y for x, y in zip(vector, row, strict=True)]
        if not any(vector):
            return False
        g = math.gcd(*vector)
        vector = [x // g for x in vector]
        self._rows.append((next(i for i, x in enumerate(vector) if x), vector))
        return True


def affine_rank(points: Iterable[Sequence[int]], bound: int | None = None) -> int:
    """Affine dimension of the points; 0 for none or one.

    With bound, stop as soon as the dimension reaches it; the result is exact
    whenever the true dimension is at most bound.
    """
    span = AffineSpan()
    for point in points:
        span.add(point)
        if bound is not None and span.dimension >= bound:
            break
    return max(span.dimension, 0)


def affine_basis(points: Sequence[Sequence[int]], indices: Iterable[int], size: int) -> list[int]:
    """Up to size indices whose points are affinely independent, chosen greedily in order."""
    chosen: list[int] = []
    if size <= 0:
        return chosen
    span = AffineSpan()
    for i in indices:
        if span.add(points[i]):
            chosen.append(i)
            if len(chosen) == size:
                break
    return chosen


def hyperplane_normal(points: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """The primitive normal of the hyperplane through d affinely independent points of Z^d.

    Its entries are the signed maximal minors of the (d - 1) x d matrix of
    differences from the first point, divided by their gcd.

    Raises
    ------
    ComputationError
        If there are not d points, or they do not span a hyperplane.
    """
    d = len(points[0])
    if len(points) != d:
        raise ComputationError(f"a hyperplane in Z^{d} needs {d} points, got {len(points)}")
    base = points[0]
    diffs = [[a - b for a, b in zip(p, base, strict=True)] for p in points[1:]]
    normal = [(-1) ** j * determinant([row[:j] + row[j + 1 :] for row in diffs]) for j in range(d)]
    g = math.gcd(*normal)
    if g == 0:
        raise ComputationError("the points do not span a hyperplane")
    return tuple(x // g for x in normal)


# --- Hermite and Smith normal forms ------------------------------------------


class HermiteForm(NamedTuple):
    """A Hermite normal form with its transform: rows @ transform = [0 | hermite]."""

    hermite: IntMatrix
    transform: IntMatrix
    rank: int

    @property
    def kernel(self) -> IntMatrix:
        """A basis of the integer kernel: the first ncols - rank columns of the transform."""
        n = len(self.transform)
        return [[self.transform[i][t] for i in range(n)] for t in range(n - self.rank)]

    @property
    def image(self) -> IntMatrix:
        """The last rank columns of the transform, which the matrix maps to the columns of hermite."""
        n = len(self.transform)
        return [[self.transform[i][t] for i in range(n)] for t in range(n - self.rank, n)]


def _gcdex(a: int, b: int) -> tuple[int, int, int]:
    """x, y, g with x*a + y*b = g = gcd(a, b) >= 0, and y = 0 when a divides b, as in SymPy."""
    if a != 0 and b % a == 0:
        return (-1 if a < 0 else 1), 0, abs(a)
    x0, y0, r0 = 1, 0, a
    x1, y1, r1 = 0, 1, b
    while r1:
        q = r0 // r1
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
        r0, r1 = r1, r0 - q * r1
    if r0 < 0:
        return -x0, -y0, -r0
    return x0, y0, r0


def _hermite(
    rows: Sequence[Sequence[int]], ncols: int, track: bool
) -> tuple[IntMatrix, IntMatrix, int]:
    """Cohen's Algorithm 2.4.5 by column operations, bottom row first, as SymPy runs it.

    Returns the reduced matrix, the accumulated transform (empty unless track)
    and k: columns k onwards form the Hermite normal form, the first k are zero.
    """
    a = [[int(x) for x in row] for row in rows]
    u = [[int(i == j) for j in range(ncols)] for i in range(ncols)] if track else []
    mats: tuple[IntMatrix, ...] = (a, u) if track else (a,)

    def mix(k: int, j: int, p: int, q: int, r: int, s: int) -> None:
        # column k <- p*column k + q*column j; column j <- r*column k + s*column j
        for mat in mats:
            for row in mat:
                x, y = row[k], row[j]
                row[k], row[j] = p * x + q * y, r * x + s * y

    def negate(k: int) -> None:
        for mat in mats:
            for row in mat:
                row[k] = -row[k]

    def subtract(j: int, k: int, q: int) -> None:
        # column j <- column j - q*column k
        for mat in mats:
            for row in mat:
                row[j] -= q * row[k]

    k = ncols
    for i in range(len(a) - 1, -1, -1):
        if k == 0:
            break
        k -= 1
        for j in range(k - 1, -1, -1):
            if a[i][j]:
                x, y, g = _gcdex(a[i][k], a[i][j])
                mix(k, j, x, y, -(a[i][j] // g), a[i][k] // g)
        pivot = a[i][k]
        if pivot < 0:
            negate(k)
            pivot = -pivot
        if pivot == 0:
            k += 1
            continue
        for j in range(k + 1, ncols):
            q = a[i][j] // pivot
            if q:
                subtract(j, k, q)
    return a, u, k


def hermite_normal_form(rows: Sequence[Sequence[int]], ncols: int) -> IntMatrix:
    """The column Hermite normal form of an m x ncols integer matrix, in SymPy's convention.

    Its columns are a basis of the lattice the columns of the matrix span. The
    pivot of each column is its last non-zero entry, the pivot rows increase
    from left to right, the pivots are positive, and the entries to the right
    of a pivot in its row lie in [0, pivot). The result equals
    sympy.matrices.normalforms.hermite_normal_form.
    """
    a, _, k = _hermite(rows, ncols, track=False)
    return [row[k:] for row in a]


def hermite_normal_form_with_transform(rows: Sequence[Sequence[int]], ncols: int) -> HermiteForm:
    """The Hermite normal form H with a unimodular U such that rows @ U = [0 | H]."""
    a, u, k = _hermite(rows, ncols, track=True)
    return HermiteForm([row[k:] for row in a], u, ncols - k)


def pivot_rows(hermite: Sequence[Sequence[int]]) -> list[int]:
    """The pivot row of each column of a column Hermite normal form: its last non-zero entry."""
    width = len(hermite[0]) if hermite else 0
    return [max(i for i, row in enumerate(hermite) if row[t]) for t in range(width)]


def solve_hermite(hermite: Sequence[Sequence[int]], target: Sequence[int]) -> list[Fraction] | None:
    """The rational y with hermite @ y = target, or None if there is none.

    hermite is a column Hermite normal form of full column rank; y is found by
    back substitution along the pivot rows, from the last column to the first.
    """
    width = len(hermite[0]) if hermite else 0
    pivots = pivot_rows(hermite)
    y = [Fraction(0)] * width
    for t in range(width - 1, -1, -1):
        i = pivots[t]
        rest = sum((hermite[i][s] * y[s] for s in range(t + 1, width)), Fraction(0))
        y[t] = (target[i] - rest) / hermite[i][t]
    for i, row in enumerate(hermite):
        if sum((row[s] * y[s] for s in range(width)), Fraction(0)) != target[i]:
            return None
    return y


def reduce_modulo(vector: Sequence[int], hermite: Sequence[Sequence[int]]) -> list[int]:
    """The canonical representative of vector modulo the lattice spanned by the columns of hermite.

    Working from the last column to the first, each pivot entry is brought
    into [0, pivot); two vectors that differ by a lattice vector reduce to the
    same result.
    """
    width = len(hermite[0]) if hermite else 0
    pivots = pivot_rows(hermite)
    v = list(vector)
    for t in range(width - 1, -1, -1):
        i = pivots[t]
        q = v[i] // hermite[i][t]
        if q:
            v = [x - q * row[t] for x, row in zip(v, hermite, strict=True)]
    return v


def smith_invariants(rows: Sequence[Sequence[int]], ncols: int) -> list[int]:
    """The non-zero invariant factors d_1 | d_2 | ... of an integer matrix.

    These are the non-zero diagonal entries of its Smith normal form, found by
    elimination with an entry of least absolute value as pivot. For the
    difference matrix of a point configuration their product is the index of
    the lattice L the differences span in its saturation.
    """
    a = [[int(x) for x in row] for row in rows]
    m, n = len(a), ncols
    out: list[int] = []
    for t in range(min(m, n)):
        entries = [(abs(a[i][j]), i, j) for i in range(t, m) for j in range(t, n) if a[i][j]]
        if not entries:
            break
        _, r, c = min(entries)
        a[t], a[r] = a[r], a[t]
        for row in a:
            row[t], row[c] = row[c], row[t]
        while True:
            pivot = a[t][t]
            for i in range(t + 1, m):
                q = a[i][t] // pivot
                if q:
                    a[i] = [x - q * y for x, y in zip(a[i], a[t], strict=True)]
            for j in range(t + 1, n):
                q = a[t][j] // pivot
                if q:
                    for row in a:
                        row[j] -= q * row[t]
            remainders = [(abs(a[i][t]), i, t) for i in range(t + 1, m) if a[i][t]]
            remainders += [(abs(a[t][j]), t, j) for j in range(t + 1, n) if a[t][j]]
            if remainders:
                _, r, c = min(remainders)
                if c == t:
                    a[t], a[r] = a[r], a[t]
                else:
                    for row in a:
                        row[t], row[c] = row[c], row[t]
                continue
            stray = next(
                (i for i in range(t + 1, m) for j in range(t + 1, n) if a[i][j] % pivot), None
            )
            if stray is None:
                break
            a[t] = [x + y for x, y in zip(a[t], a[stray], strict=True)]
        out.append(abs(a[t][t]))
    return out
