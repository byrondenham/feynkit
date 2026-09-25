"""
Lattice polytopes: the face lattice and the facet inequalities.

A polytope P = conv(points) is described here by its faces, given as the
indices of the points lying on them, and, when P is full-dimensional in its
ambient lattice, by its facet inequalities m . x <= b. Each normal m is an
integer vector that points outwards, so every point of P satisfies the
inequality and the points of that facet attain equality, and it is primitive:
the greatest common divisor of its entries is 1, which makes the pair (m, b)
unique for a given facet. A lower-dimensional polytope has no such
description in the ambient lattice, since the inequalities would not cut out
its affine hull, so :attr:`PolytopeData.facets` is empty unless
:attr:`PolytopeData.is_full_dimensional` holds. The normalised volume is
measured in the lattice the differences of the points span, so, whatever the
ambient dimension, it is the holonomic rank of the associated GKZ system for
generic coefficients and non-resonant beta.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sympy as sp

from . import _exact
from .a_configuration import AConfiguration
from .core.exceptions import ComputationError, ValidationError

__all__ = [
    "Facet",
    "LatticeChart",
    "PolytopeData",
    "faces",
    "lattice_chart",
    "lattice_coordinates",
    "polytope_data",
]


# --- data types --------------------------------------------------------------


@dataclass(frozen=True)
class LatticeChart:
    """Lattice coordinates x = o + B c for conv(points).

    Attributes
    ----------
    origin
        o, an integer point of the affine hull.
    basis
        The d columns of B, each of length n: the Hermite-normal-form basis, in
        SymPy's convention, of the lattice L spanned by the differences of the
        points.
    coordinates
        c_j for each point, with alpha_j = o + B c_j. Every coordinate is
        non-negative with minimum 0, and the c_j generate Z^d affinely.
    """

    origin: tuple[int, ...]
    basis: tuple[tuple[int, ...], ...]
    coordinates: tuple[tuple[int, ...], ...]

    def to_ambient(self, c: Sequence[int]) -> tuple[int, ...]:
        """The ambient point o + B c.

        Raises
        ------
        ValidationError
            If c does not have one entry per basis vector.
        """
        if len(c) != len(self.basis):
            raise ValidationError(f"chart coordinates have length {len(self.basis)}, got {len(c)}")
        return tuple(
            o + sum(ct * b[k] for ct, b in zip(c, self.basis, strict=True))
            for k, o in enumerate(self.origin)
        )


@dataclass(frozen=True)
class Facet:
    """One facet of a full-dimensional lattice polytope: m . x <= b for x in P.

    Attributes
    ----------
    normal
        The primitive integer outward normal m.
    offset
        The right-hand side b.
    point_indices
        Indices into :attr:`PolytopeData.points` of the points on the facet,
        that is, the points with m . x = b.
    """

    normal: tuple[int, ...]
    offset: int
    point_indices: tuple[int, ...]


@dataclass(frozen=True)
class PolytopeData:
    """The face lattice, facet inequalities and volume of conv(points).

    Attributes
    ----------
    points
        The points, in the order they were given; not only the vertices.
    ambient_dimension
        Number of coordinates of each point.
    dimension
        Affine dimension of the polytope.
    vertex_indices
        Indices of the points that are vertices.
    faces
        Every face as (dimension, indices of the points on it), including the
        vertices and the polytope itself.
    facets
        The codimension-one faces as inequalities; empty unless the polytope
        is full-dimensional.
    normalized_volume
        Volume normalised so that a unimodular simplex has volume 1,
        measured in the lattice the differences of the points span.
    """

    points: tuple[tuple[int, ...], ...]
    ambient_dimension: int
    dimension: int
    vertex_indices: tuple[int, ...]
    faces: tuple[tuple[int, tuple[int, ...]], ...]
    facets: tuple[Facet, ...]
    normalized_volume: int

    @property
    def is_full_dimensional(self) -> bool:
        """Whether the polytope spans its ambient space."""
        return self.dimension == self.ambient_dimension

    @property
    def f_vector(self) -> tuple[int, ...]:
        """Number of faces of each dimension, from 0 up to the dimension."""
        counts = [0] * (self.dimension + 1)
        for d, _idx in self.faces:
            counts[d] += 1
        return tuple(counts)

    @property
    def vertices(self) -> tuple[tuple[int, ...], ...]:
        """The vertices, in the order of :attr:`vertex_indices`."""
        return tuple(self.points[i] for i in self.vertex_indices)


# --- face enumeration --------------------------------------------------------


def _affine_rank(pts: np.ndarray) -> int:
    if len(pts) <= 1:
        return 0
    diffs = (pts[1:] - pts[0]).astype(float)
    return int(np.linalg.matrix_rank(diffs, tol=1e-9))


def faces(pts: np.ndarray) -> list[tuple[int, tuple[int, ...]]]:
    """All faces of conv(pts) as (dimension, indices of the points on the face).

    Includes the vertices and the polytope itself. Points that lie on a face
    without being vertices are included in that face.
    """
    n_pts = len(pts)
    if n_pts == 0:
        return []
    rank = _affine_rank(pts)
    if rank == 0:
        return [(0, tuple(range(n_pts)))]

    # Work in coordinates of the affine hull so the hull is full-dimensional.
    diffs = (pts - pts[0]).astype(float)
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    coords = diffs @ vt[:rank].T

    if rank == 1:
        order = np.argsort(coords[:, 0])
        lo, hi = int(order[0]), int(order[-1])
        return [(0, (lo,)), (0, (hi,)), (1, tuple(range(n_pts)))]

    from scipy.spatial import ConvexHull

    hull = ConvexHull(coords)
    facets: set[frozenset[int]] = set()
    for eq in hull.equations:
        normal, offset = eq[:-1], eq[-1]
        on = frozenset(int(i) for i in range(n_pts) if abs(coords[i] @ normal + offset) < 1e-7)
        facets.add(on)

    all_faces: set[frozenset[int]] = set(facets)
    frontier = set(facets)
    while frontier:
        new: set[frozenset[int]] = set()
        for a in frontier:
            for b in all_faces:
                c = a & b
                if c and c not in all_faces and c not in new:
                    new.add(c)
        all_faces |= new
        frontier = new
    all_faces.add(frozenset(range(n_pts)))

    out: list[tuple[int, tuple[int, ...]]] = []
    for face in all_faces:
        idx = tuple(sorted(face))
        out.append((_affine_rank(pts[list(idx)]), idx))
    out.sort(key=lambda f: (f[0], f[1]))
    return out


def _differences(points: Sequence[Sequence[int]]) -> list[list[int]]:
    """The difference matrix: one row p - points[0] for each later point p."""
    base = points[0]
    return [[a - b for a, b in zip(p, base, strict=True)] for p in points[1:]]


def lattice_chart(points: Sequence[Sequence[int]] | np.ndarray) -> LatticeChart:
    """The lattice chart x = o + B c of conv(points).

    The columns of B are the Hermite normal form, in SymPy's convention, of
    the matrix whose columns are the differences alpha_j - alpha_1, so they
    are a basis of the lattice L those differences span. Each point is
    alpha_j = o + B c_j with c_j in Z^d, shifted so that every coordinate has
    minimum 0; in the chart the points generate Z^d affinely.

    Raises
    ------
    ValidationError
        If there are no points or a coordinate is not an integer.
    """
    pts = _exact.integer_points(points)
    if not pts:
        raise ValidationError("lattice_chart needs at least one point")
    base = pts[0]
    ambient = len(base)
    columns = _differences(pts)
    hermite = _exact.hermite_normal_form(
        [[row[k] for row in columns] for k in range(ambient)], len(columns)
    )
    rank = len(hermite[0]) if hermite else 0
    basis = tuple(tuple(hermite[k][t] for k in range(ambient)) for t in range(rank))
    raw: list[list[int]] = []
    for point in pts:
        solution = _exact.solve_hermite(hermite, [a - b for a, b in zip(point, base, strict=True)])
        if solution is None or any(c.denominator != 1 for c in solution):
            raise ComputationError(
                f"point {point} is not in the lattice spanned by the differences"
            )
        raw.append([int(c) for c in solution])
    shift = [min(c[t] for c in raw) for t in range(rank)]
    coordinates = tuple(tuple(c[t] - shift[t] for t in range(rank)) for c in raw)
    origin = tuple(
        base[k] + sum(s * b[k] for s, b in zip(shift, basis, strict=True)) for k in range(ambient)
    )
    return LatticeChart(origin=origin, basis=basis, coordinates=coordinates)


def lattice_coordinates(pts: np.ndarray) -> list[tuple[int, ...]]:
    """Integer coordinates of the points in the lattice their differences span.

    The coordinates of lattice_chart(pts): non-negative, in the
    Hermite-normal-form basis of the difference lattice.
    """
    return list(lattice_chart(pts).coordinates)


# --- public API --------------------------------------------------------------


def polytope_data(points: Sequence[Sequence[int]]) -> PolytopeData:
    """Face lattice, facet inequalities and normalised volume of conv(points)."""
    pts = np.asarray([tuple(int(x) for x in p) for p in points], dtype=np.int64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValidationError("polytope_data needs at least one point")
    all_faces = tuple((d, tuple(idx)) for d, idx in faces(pts))
    dimension = max(d for d, _ in all_faces)
    ambient = int(pts.shape[1])
    vertex_indices = tuple(sorted(idx[0] for d, idx in all_faces if d == 0))
    facets: tuple[Facet, ...] = ()
    if dimension == ambient:
        facets = tuple(_facet(pts, idx) for d, idx in all_faces if d == dimension - 1)
    return PolytopeData(
        points=tuple(tuple(int(x) for x in p) for p in pts),
        ambient_dimension=ambient,
        dimension=dimension,
        vertex_indices=vertex_indices,
        faces=all_faces,
        facets=facets,
        normalized_volume=_normalized_volume(pts, dimension),
    )


def _facet(pts: np.ndarray, idx: tuple[int, ...]) -> Facet:
    """Primitive integer outward normal of the facet through pts[idx]."""
    base = pts[idx[0]]
    ambient = int(pts.shape[1])
    diffs = sp.Matrix(
        len(idx) - 1, ambient, [int(x) for i in idx[1:] for x in (pts[i] - base).tolist()]
    )
    null = diffs.nullspace()
    if len(null) != 1:
        raise ValidationError("facet does not span a hyperplane")
    vec = null[0]
    lcm = sp.ilcm(*[sp.Rational(x).q for x in vec], 1)
    ints = [int(x * lcm) for x in vec]
    g = math.gcd(*ints)
    normal = [x // g for x in ints]
    centroid = pts.mean(axis=0)
    offset = int(sum(m * int(x) for m, x in zip(normal, base, strict=True)))
    if sum(m * c for m, c in zip(normal, centroid, strict=True)) > offset:
        normal = [-m for m in normal]
        offset = -offset
    return Facet(tuple(normal), offset, tuple(idx))


def _normalized_volume(pts: np.ndarray, dimension: int) -> int:
    """Normalised volume of conv(pts) in the lattice its differences span.

    ``AConfiguration.normalized_volume`` measures a full-dimensional
    configuration, so a polytope of dimension 1, or one lying in a proper
    affine subspace, is first rewritten in coordinates of that lattice.
    """
    if dimension == 0:
        return 1
    if dimension == 1:
        along = [c[0] for c in lattice_coordinates(pts)]
        return max(along) - min(along)
    if dimension < int(pts.shape[1]):
        pts = np.asarray(lattice_coordinates(pts), dtype=np.int64)
    return int(AConfiguration(pts.T, is_homogenized=False).normalized_volume)
