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


def _as_int(value: object, where: str) -> int:
    """value as a Python integer; where names its source for the error message.

    Raises
    ------
    ValidationError
        If value is not an integer. Integral floats and rationals are accepted.
    """
    if type(value) is int:
        return value
    if isinstance(value, SupportsIndex):
        return operator.index(value)
    if isinstance(value, numbers.Rational):
        if value.denominator == 1:
            return int(value.numerator)
    elif isinstance(value, numbers.Real):
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            return int(as_float)
    raise ValidationError(f"{where} has the non-integer coordinate {value!r}")


def integer_points(points: object) -> list[tuple[int, ...]]:
    """The points as tuples of Python integers.

    Accepts a two-dimensional NumPy array or a sequence of sequences.
    Integral floats are accepted; any other non-integer raises.

    Raises
    ------
    ValidationError
        If points is not a sequence, a coordinate is not an integer, or the
        points do not all have the same number of coordinates.
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
        try:
            rows = list(cast(Iterable[Any], points))
        except TypeError:
            raise ValidationError(
                f"points must be a sequence of coordinate sequences, got {points!r}"
            ) from None
    out: list[tuple[int, ...]] = []
    for i, row in enumerate(rows):
        try:
            coordinates = tuple(row)
        except TypeError:
            raise ValidationError(f"point {i} is not a sequence of coordinates: {row!r}") from None
        point = tuple(_as_int(x, f"point {i}") for x in coordinates)
        if out and len(point) != len(out[0]):
            raise ValidationError(
                f"point {i} has {len(point)} coordinates, point 0 has {len(out[0])}"
            )
        out.append(point)
    return out


def mask_indices(mask: int) -> list[int]:
    """The indices of the set bits of mask, in increasing order.

    Raises
    ------
    ValidationError
        If mask is negative.
    """
    if mask < 0:
        raise ValidationError(f"mask must be non-negative, got {mask}")
    out: list[int] = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def dot(u: Sequence[int], v: Sequence[int]) -> int:
    """The dot product of two integer vectors of equal length.

    Raises
    ------
    ValidationError
        If u and v have different lengths or an entry is not an integer.
    """
    where = "a vector passed to dot"
    try:
        return sum(_as_int(a, where) * _as_int(b, where) for a, b in zip(u, v, strict=True))
    except ValueError:
        raise ValidationError(
            f"dot needs vectors of equal length, got {len(u)} and {len(v)}"
        ) from None


def _subtract(u: Sequence[int], v: Sequence[int]) -> list[int]:
    """u - v as Python integers, entry by entry.

    Raises
    ------
    ValidationError
        If u and v have different lengths or an entry is not an integer.
    """
    try:
        return [_as_int(a, "a point") - _as_int(b, "a point") for a, b in zip(u, v, strict=True)]
    except ValueError:
        raise ValidationError(
            f"points have different numbers of coordinates: {len(u)} and {len(v)}"
        ) from None


def _int_rows(rows: Iterable[Iterable[object]]) -> IntMatrix:
    """The rows as lists of Python integers.

    Raises
    ------
    ValidationError
        If an entry is not an integer.
    """
    return [[_as_int(x, f"row {i}") for x in row] for i, row in enumerate(rows)]


# --- determinants and rank ---------------------------------------------------


def determinant(matrix: Sequence[Sequence[int]]) -> int:
    """Determinant of a square integer matrix by Bareiss elimination; 1 for 0 x 0.

    Raises
    ------
    ValidationError
        If the matrix is not square or an entry is not an integer.
    """
    a = _int_rows(matrix)
    n = len(a)
    widths = [len(row) for row in a]
    if any(w != n for w in widths):
        raise ValidationError(
            f"determinant needs a square matrix, got {n} rows with widths {widths}"
        )
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
    """Rank of an integer matrix by fraction-free (Bareiss) elimination.

    Raises
    ------
    ValidationError
        If the rows do not all have the same length or an entry is not an
        integer.
    """
    a = _int_rows(rows)
    if not a:
        return 0
    m, n = len(a), len(a[0])
    bad = next((i for i, row in enumerate(a) if len(row) != n), None)
    if bad is not None:
        raise ValidationError(f"row {bad} has {len(a[bad])} entries, row 0 has {n}")
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
        """Add a point; return whether it lies outside the span so far.

        Raises
        ------
        ValidationError
            If point has a different number of coordinates from the points
            already added, or a coordinate is not an integer.
        """
        if self._base is None:
            self._base = tuple(_as_int(x, "a point") for x in point)
            return True
        vector = _subtract(point, self._base)
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
        If there are no points, not d points, or they do not span a
        hyperplane.
    ValidationError
        If the points do not all have the same number of coordinates, or a
        coordinate is not an integer.
    """
    if not points:
        raise ComputationError("hyperplane_normal needs at least one point")
    d = len(points[0])
    if len(points) != d:
        raise ComputationError(f"a hyperplane in Z^{d} needs {d} points, got {len(points)}")
    base = tuple(_as_int(x, "a point") for x in points[0])
    diffs = [_subtract(p, base) for p in points[1:]]
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

    Raises
    ------
    ValidationError
        If a row does not have ncols entries or an entry is not an integer.
    """
    a = _int_rows(rows)
    bad = next((i for i, row in enumerate(a) if len(row) != ncols), None)
    if bad is not None:
        raise ValidationError(f"row {bad} has {len(a[bad])} entries, expected ncols={ncols}")
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

    Raises
    ------
    ValidationError
        If a row does not have ncols entries.
    """
    a, _, k = _hermite(rows, ncols, track=False)
    return [row[k:] for row in a]


def hermite_normal_form_with_transform(rows: Sequence[Sequence[int]], ncols: int) -> HermiteForm:
    """The Hermite normal form H with a unimodular U such that rows @ U = [0 | H].

    Raises
    ------
    ValidationError
        If a row does not have ncols entries.
    """
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

    Raises
    ------
    ValidationError
        If a row does not have ncols entries or an entry is not an integer.
    """
    a = _int_rows(rows)
    bad = next((i for i, row in enumerate(a) if len(row) != ncols), None)
    if bad is not None:
        raise ValidationError(f"row {bad} has {len(a[bad])} entries, expected ncols={ncols}")
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


# --- facets ------------------------------------------------------------------


class Halfspace(NamedTuple):
    """A facet normal . x <= offset of the hull of a point list; mask is its tight set."""

    normal: tuple[int, ...]
    offset: int
    mask: int


def verify_facet(
    points: Sequence[Sequence[int]], associated: Iterable[int], among: Iterable[int]
) -> Halfspace | None:
    """Check a facet candidate exactly against the points indexed by among.

    Picks d affinely independent points from associated, greedily in the given
    order, and takes the primitive normal of the hyperplane through them. The
    candidate is rejected (None) if there are fewer than d such points, or if
    the points of among lie on both sides of the hyperplane or all on it.
    Otherwise the normal is oriented so that every point of among satisfies
    normal . x <= offset, and the tight set among them is recorded.
    """
    d = len(points[0])
    chosen = affine_basis(points, associated, d)
    if len(chosen) < d:
        return None
    normal = hyperplane_normal([points[i] for i in chosen])
    offset = dot(normal, points[chosen[0]])
    indices = list(among)
    heights = [dot(normal, points[j]) - offset for j in indices]
    if any(h > 0 for h in heights):
        if any(h < 0 for h in heights):
            return None
        normal = tuple(-x for x in normal)
        offset = -offset
        heights = [-h for h in heights]
    elif not any(heights):
        return None
    mask = 0
    for j, h in zip(indices, heights, strict=True):
        if h == 0:
            mask |= 1 << j
    return Halfspace(normal, offset, mask)


def beneath_beyond(points: Sequence[Sequence[int]]) -> list[Halfspace]:
    """The facets of conv(points), for full-dimensional points in Z^d with d >= 2.

    Starts from the simplex on the first d + 1 affinely independent points in
    index order, then inserts the other points in index order. A point beneath
    or on every current facet joins the tight sets of the facets it lies on.
    Otherwise the facets it is beyond are dropped and those it lies on gain it.
    For every ridge shared by a facet it is beyond and a facet it is beneath,
    the facet through that ridge and the point is added, verified against the
    points inserted so far; its tight set is the ridge's plus the point. The
    facet list is complete at every step, so ridges are the pairwise
    intersections of tight sets with at least d - 1 points and affine
    dimension d - 2.

    Raises
    ------
    ComputationError
        If there are no points, d < 2 or the points are not full-dimensional,
        or if a new facet fails verification, which would be a bug.
    """
    if not points:
        raise ComputationError("beneath-beyond needs at least one point")
    d = len(points[0])
    if d < 2:
        raise ComputationError(f"beneath-beyond needs dimension at least 2, got {d}")
    simplex = affine_basis(points, range(len(points)), d + 1)
    if len(simplex) != d + 1:
        raise ComputationError(
            "beneath-beyond needs full-dimensional points; these span dimension "
            f"{len(simplex) - 1} in Z^{d}"
        )
    inserted = list(simplex)
    facets: list[Halfspace] = []
    for omitted in simplex:
        facet = verify_facet(points, [i for i in simplex if i != omitted], simplex)
        if facet is None:
            raise ComputationError(
                "beneath-beyond: a facet of the initial simplex failed verification"
            )
        facets.append(facet)
    initial = set(simplex)
    for p in range(len(points)):
        if p in initial:
            continue
        bit = 1 << p
        inserted.append(p)
        heights = [dot(f.normal, points[p]) - f.offset for f in facets]
        if all(h <= 0 for h in heights):
            facets = [
                f._replace(mask=f.mask | bit) if h == 0 else f
                for f, h in zip(facets, heights, strict=True)
            ]
            continue
        visible = [f for f, h in zip(facets, heights, strict=True) if h > 0]
        beneath = [f for f, h in zip(facets, heights, strict=True) if h < 0]
        kept = beneath + [
            f._replace(mask=f.mask | bit) for f, h in zip(facets, heights, strict=True) if h == 0
        ]
        seen = {(f.normal, f.offset) for f in kept}
        for f in visible:
            for g in beneath:
                ridge = f.mask & g.mask
                if ridge.bit_count() < d - 1:
                    continue
                basis = affine_basis(points, mask_indices(ridge), d - 1)
                if len(basis) < d - 1:
                    continue
                new = verify_facet(points, [*basis, p], inserted)
                if new is None or new.mask != ridge | bit:
                    raise ComputationError(
                        "beneath-beyond: the facet through the ridge on points "
                        f"{tuple(mask_indices(ridge))} and point {p} failed verification"
                    )
                if (new.normal, new.offset) not in seen:
                    seen.add((new.normal, new.offset))
                    kept.append(new)
        facets = kept
    return facets


# --- face lattice and certificate --------------------------------------------


class CertificateError(ComputationError):
    """A facet list failed the completeness certificate (C1)-(C3)."""


class FaceLattice(NamedTuple):
    """A certified face lattice.

    top is the mask of all points; dims maps every face, as a mask, to its
    dimension; phi maps every face of dimension at least 1 to its facets.
    """

    top: int
    dims: dict[int, int]
    phi: dict[int, tuple[int, ...]]


def face_dimensions(
    points: Sequence[Sequence[int]], facet_masks: Iterable[int], dimension: int
) -> dict[int, int]:
    """Every non-empty intersection of the facets' tight sets, and P itself, with its exact dimension.

    Every face is an intersection of facets, so the lattice is closed by
    intersecting the frontier with the facets only. A new face c = a & f with
    c != a is a proper face of a, so its dimension is below that of a, and the
    rank computation stops once it reaches that bound.
    """
    facets = sorted(set(facet_masks))
    dims = {(1 << len(points)) - 1: dimension}
    for f in facets:
        dims[f] = affine_rank([points[j] for j in mask_indices(f)], bound=dimension - 1)
    frontier = facets
    while frontier:
        bounds: dict[int, int] = {}
        for a in frontier:
            for f in facets:
                c = a & f
                if c and c not in dims:
                    bounds[c] = min(bounds.get(c, dims[a] - 1), dims[a] - 1)
        for c, bound in bounds.items():
            dims[c] = affine_rank([points[j] for j in mask_indices(c)], bound=bound)
        frontier = list(bounds)
    return dims


def certify(dims: dict[int, int], facet_masks: Iterable[int]) -> dict[int, tuple[int, ...]]:
    """Check the certificate and return phi, the facets of every face of dimension at least 1.

    phi(Q) = {Q & F : F a candidate facet, dim(Q & F) = dim Q - 1}. The list
    is accepted if and only if (C1) every edge has exactly two members of
    phi, (C2) every face of dimension at least 2 has one, and (C3) for every
    such Q, every F in phi(Q) and every R in phi(F), exactly two members of
    phi(Q) contain R. Then, by induction on dimension, phi(Q) holds every
    facet of Q, since the facet-ridge graph of a polytope is connected.

    Raises
    ------
    CertificateError
        Naming the condition that fails and the faces involved.
    """
    facets = sorted(set(facet_masks))
    phi: dict[int, tuple[int, ...]] = {}
    for q, k in dims.items():
        if k >= 1:
            phi[q] = tuple(sorted({q & f for f in facets if q & f and dims[q & f] == k - 1}))
    order = sorted(phi, key=lambda q: (dims[q], mask_indices(q)))

    def on(mask: int) -> tuple[int, ...]:
        return tuple(mask_indices(mask))

    for q in order:
        if dims[q] == 1 and len(phi[q]) != 2:
            raise CertificateError(
                "completeness certificate condition (C1) fails: the edge on points "
                f"{on(q)} has {len(phi[q])} vertices among the candidate faces, not 2"
            )
    for q in order:
        if dims[q] >= 2 and not phi[q]:
            raise CertificateError(
                f"completeness certificate condition (C2) fails: the {dims[q]}-dimensional "
                f"face on points {on(q)} has no facet among the candidate faces"
            )
    for q in order:
        if dims[q] < 2:
            continue
        for f in phi[q]:
            for r in phi[f]:
                count = sum(1 for g in phi[q] if r & g == r)
                if count != 2:
                    raise CertificateError(
                        "completeness certificate condition (C3) fails: in the "
                        f"{dims[q]}-dimensional face on points {on(q)}, the face on points "
                        f"{on(r)} of its facet on points {on(f)} lies in {count} of its "
                        "facets, not 2"
                    )
    return phi


def certified_lattice(
    points: Sequence[Sequence[int]], facet_masks: Iterable[int], dimension: int
) -> FaceLattice:
    """The face lattice generated by verified facets, certified complete.

    Raises
    ------
    CertificateError
        If the facet list fails (C1), (C2) or (C3).
    """
    masks = list(facet_masks)
    dims = face_dimensions(points, masks, dimension)
    return FaceLattice((1 << len(points)) - 1, dims, certify(dims, masks))


# --- pulling triangulation ---------------------------------------------------


def pulling_simplices(
    lattice: FaceLattice, points: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    """The simplices of the pulling triangulation of the top face, as point indices.

    Vertices are ordered lexicographically by the coordinates in points, so
    the triangulation does not depend on the input order. A vertex is its own
    simplex. A face Q of dimension at least 1 is triangulated by joining its
    least vertex v(Q) to each simplex of each facet of Q not containing v(Q).
    A vertex holding several repeated points is represented by the least of
    their indices. Simplices are memoised per face.
    """

    def lowest(mask: int) -> int:
        return (mask & -mask).bit_length() - 1

    vertices = sorted(
        (v for v, k in lattice.dims.items() if k == 0), key=lambda v: tuple(points[lowest(v)])
    )
    memo: dict[int, list[tuple[int, ...]]] = {}

    def simplices(q: int) -> list[tuple[int, ...]]:
        if q not in memo:
            v = next(w for w in vertices if w & q)
            apex = lowest(v)
            if lattice.dims[q] == 0:
                memo[q] = [(apex,)]
            else:
                memo[q] = [(apex, *s) for f in lattice.phi[q] if not f & v for s in simplices(f)]
        return memo[q]

    return simplices(lattice.top)


def simplex_volume(
    coordinates: Sequence[Sequence[int]], simplices: Iterable[tuple[int, ...]], index: int
) -> int:
    """The sum of |det(s_1 - s_0, ..., s_d - s_0)| over the simplices, divided by index.

    Raises
    ------
    ComputationError
        If a determinant is zero or not a multiple of index, which would be a bug.
    """
    total = 0
    for simplex in simplices:
        base = coordinates[simplex[0]]
        det = abs(
            determinant(
                [[a - b for a, b in zip(coordinates[i], base, strict=True)] for i in simplex[1:]]
            )
        )
        if det == 0 or det % index:
            raise ComputationError(
                f"the simplex on points {simplex} has determinant {det}, not a positive "
                f"multiple of the sublattice index {index}"
            )
        total += det
    return total // index
