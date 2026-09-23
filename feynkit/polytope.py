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

from .a_configuration import AConfiguration
from .core.exceptions import ValidationError

__all__ = [
    "Facet",
    "PolytopeData",
    "faces",
    "lattice_coordinates",
    "polytope_data",
]


# --- data types --------------------------------------------------------------


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


def lattice_coordinates(pts: np.ndarray) -> list[tuple[int, ...]]:
    """Integer coordinates of the points in the lattice their differences span.

    A Z-basis of the difference lattice comes from the Hermite normal form of
    the difference matrix; coordinates are shifted to be non-negative.
    """
    if len(pts) == 1:
        return [()]
    from sympy.matrices.normalforms import hermite_normal_form

    diffs = sp.Matrix([[int(x) for x in row] for row in (pts[1:] - pts[0])])
    hnf = hermite_normal_form(diffs.T).T  # rows: Z-basis of the row lattice of diffs
    basis = sp.Matrix([row for row in hnf.tolist() if any(x != 0 for x in row)])
    if basis.rows == 0:
        return [() for _ in pts]
    coords: list[tuple[int, ...]] = []
    for row in pts - pts[0]:
        target = sp.Matrix([[int(x) for x in row]])
        sol = (
            basis.T.solve_least_squares(target.T)
            if basis.rows < basis.cols
            else basis.T.solve(target.T)
        )
        c = [sp.nsimplify(x) for x in sol]
        if any(not x.is_integer for x in c):
            raise ValueError("Point is not in the lattice spanned by the differences")
        coords.append(tuple(int(x) for x in c))
    mins = [min(c[i] for c in coords) for i in range(basis.rows)]
    return [tuple(c[i] - mins[i] for i in range(basis.rows)) for c in coords]


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
