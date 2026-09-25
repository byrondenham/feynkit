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

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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
            If c does not have one entry per basis vector, or an entry is not
            an integer.
        """
        if len(c) != len(self.basis):
            raise ValidationError(f"chart coordinates have length {len(self.basis)}, got {len(c)}")
        entries = [_exact._as_int(x, "the chart point") for x in c]
        return tuple(
            o + sum(e * b[k] for e, b in zip(entries, self.basis, strict=True))
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


# --- backends ----------------------------------------------------------------

_GENERATORS = ("python", "qhull", "normaliz")


def _pynormaliz_available() -> bool:
    """Whether PyNormaliz can be imported; the analogue of _4ti2_binary and _singular_binary."""
    return importlib.util.find_spec("PyNormaliz") is not None


def _resolve_backend(backend: str) -> str:
    """The facet generator for backend; "auto" is "python" until Normaliz has been timed."""
    if backend == "auto":
        return "python"
    if backend not in _GENERATORS:
        raise ComputationError(
            f"Unknown backend {backend!r}. Choose 'auto', 'python', 'qhull' or 'normaliz'."
        )
    if backend == "normaliz" and not _pynormaliz_available():
        raise ComputationError(
            "normaliz backend requested but PyNormaliz is not installed. "
            "Install it with: pip install PyNormaliz"
        )
    return backend


def _qhull_candidates(coords: list[tuple[int, ...]]) -> list[_exact.Halfspace] | None:
    """Facet candidates from Qhull, each verified exactly; None when Qhull fails.

    Each Qhull facet contributes its simplex and the points within 1e-7 of its
    equation, and verify_facet picks d affinely independent points from them:
    Qhull triangulates non-simplicial facets and may return flat simplices, so
    the simplex alone will not do. Rejected candidates are dropped, and a set
    of associated points already verified is skipped. Coordinates too large
    for a float count as a Qhull failure.
    """
    from scipy.spatial import ConvexHull, QhullError

    try:
        x = np.asarray(coords, dtype=float)
        hull = ConvexHull(x)
    except (QhullError, ValueError, OverflowError):
        return None
    everything = range(len(coords))
    seen: set[frozenset[int]] = set()
    found: dict[tuple[tuple[int, ...], int], _exact.Halfspace] = {}
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        near = np.flatnonzero(np.abs(x @ equation[:-1] + equation[-1]) < 1e-7)
        associated = list(dict.fromkeys([*map(int, simplex), *map(int, near)]))
        key = frozenset(associated)
        if key in seen:
            continue
        seen.add(key)
        halfspace = _exact.verify_facet(coords, associated, everything)
        if halfspace is not None:
            found.setdefault((halfspace.normal, halfspace.offset), halfspace)
    return list(found.values())


def _normaliz_candidates(coords: list[tuple[int, ...]]) -> list[_exact.Halfspace] | None:
    """Facet candidates from PyNormaliz's support hyperplanes, verified exactly.

    The points go in as vertices with the homogenising coordinate 1 last, and
    each hyperplane lambda, with lambda . (x, 1) >= 0 on P, contributes the
    points on which it vanishes. Returns None when the import fails.
    """
    try:
        import PyNormaliz
    except ImportError:
        return None
    cone = PyNormaliz.Cone(vertices=[[*p, 1] for p in coords])
    everything = range(len(coords))
    found: dict[tuple[tuple[int, ...], int], _exact.Halfspace] = {}
    for form in cone.SupportHyperplanes():
        if len(form) != len(coords[0]) + 1:
            continue
        *linear, constant = (int(v) for v in form)
        associated = [j for j, p in enumerate(coords) if _exact.dot(linear, p) + constant == 0]
        halfspace = _exact.verify_facet(coords, associated, everything)
        if halfspace is not None:
            found.setdefault((halfspace.normal, halfspace.offset), halfspace)
    return list(found.values())


def _certified_hull(
    coords: list[tuple[int, ...]], backend: str
) -> tuple[list[_exact.Halfspace], _exact.FaceLattice, str]:
    """Verified, certified facets of full-dimensional coords, the lattice, and the generator used.

    "qhull" and "normaliz" fall back to beneath-beyond when their generator
    fails or their list fails the certificate. On the "python" path a failed
    certificate is a bug and raises.
    """
    dimension = len(coords[0])
    if backend in ("qhull", "normaliz"):
        candidates = (
            _qhull_candidates(coords) if backend == "qhull" else _normaliz_candidates(coords)
        )
        if candidates is not None:
            try:
                lattice = _exact.certified_lattice(coords, [h.mask for h in candidates], dimension)
            except _exact.CertificateError:
                pass
            else:
                return candidates, lattice, backend
    halfspaces = _exact.beneath_beyond(coords)
    try:
        lattice = _exact.certified_lattice(coords, [h.mask for h in halfspaces], dimension)
    except _exact.CertificateError as exc:
        raise ComputationError(
            "the facets from beneath-beyond fail the completeness certificate, which is a bug "
            f"in feynkit: {exc}"
        ) from exc
    return halfspaces, lattice, "python"


# --- lattice chart -----------------------------------------------------------


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
        If there are no points, a coordinate is not an integer, or the points
        do not all have the same number of coordinates.
    ComputationError
        If a point is not in the lattice spanned by the differences, which
        would be a bug.
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


# --- certified hull ----------------------------------------------------------


@dataclass(frozen=True)
class _Hull:
    """The certified facets and face lattice from which every result is derived.

    coordinates are ambient when in_chart is false (P full-dimensional of
    dimension at least 2, or a point) and chart coordinates otherwise; the
    halfspaces are facets in those coordinates.
    """

    points: tuple[tuple[int, ...], ...]
    dimension: int
    in_chart: bool
    coordinates: tuple[tuple[int, ...], ...]
    halfspaces: tuple[_exact.Halfspace, ...]
    lattice: _exact.FaceLattice
    generator: str

    @property
    def faces(self) -> list[tuple[int, tuple[int, ...]]]:
        """Every face as (dimension, indices), sorted by dimension and then indices."""
        return sorted(
            (k, tuple(_exact.mask_indices(mask))) for mask, k in self.lattice.dims.items()
        )


def _segment(
    coordinates: list[tuple[int, ...]],
) -> tuple[list[_exact.Halfspace], _exact.FaceLattice]:
    """A segment 0 <= c <= c_max in its chart coordinate c: two vertices, -c <= 0, c <= c_max."""
    top = (1 << len(coordinates)) - 1
    c_max = max(c[0] for c in coordinates)
    low = sum(1 << j for j, c in enumerate(coordinates) if c[0] == 0)
    high = sum(1 << j for j, c in enumerate(coordinates) if c[0] == c_max)
    lattice = _exact.FaceLattice(top, {top: 1, low: 0, high: 0}, {top: tuple(sorted((low, high)))})
    return [_exact.Halfspace((-1,), 0, low), _exact.Halfspace((1,), c_max, high)], lattice


def _hull(points: list[tuple[int, ...]], backend: str, chart: LatticeChart | None = None) -> _Hull:
    """The certified hull of the non-empty points.

    A full-dimensional polytope of dimension at least 2 is handled in ambient
    coordinates, every other polytope of dimension at least 1 in its lattice
    chart, where it is full-dimensional. Points and segments are closed-form.
    """
    backend = _resolve_backend(backend)
    ambient = len(points[0])
    dimension = _exact.affine_rank(points)
    top = (1 << len(points)) - 1
    if dimension == 0:
        lattice = _exact.FaceLattice(top, {top: 0}, {})
        return _Hull(tuple(points), 0, False, tuple(points), (), lattice, "closed form")
    in_chart = dimension < ambient or dimension == 1
    if in_chart:
        coordinates = list((chart if chart is not None else lattice_chart(points)).coordinates)
    else:
        coordinates = points
    if dimension == 1:
        halfspaces, lattice = _segment(coordinates)
        generator = "closed form"
    else:
        halfspaces, lattice, generator = _certified_hull(coordinates, backend)
    return _Hull(
        tuple(points),
        dimension,
        in_chart,
        tuple(coordinates),
        tuple(halfspaces),
        lattice,
        generator,
    )


# --- public API --------------------------------------------------------------


def faces(
    pts: np.ndarray | Sequence[Sequence[int]], *, backend: str = "auto"
) -> list[tuple[int, tuple[int, ...]]]:
    """All faces of conv(pts) as (dimension, indices of the points on the face).

    Includes the vertices and the polytope itself, sorted by dimension and
    then indices. Every point on a face is listed in it, repeated points
    included. The facets are computed and certified complete in integer
    arithmetic; see polytope_data for backend.

    Raises
    ------
    ValidationError
        If a coordinate is not an integer, or the points do not all have the
        same number of coordinates.
    ComputationError
        If backend is unknown or unavailable, or the facets from beneath-beyond
        fail the completeness certificate, which would be a bug.
    """
    points = _exact.integer_points(pts)
    if not points:
        _resolve_backend(backend)
        return []
    return _hull(points, backend).faces


def polytope_data(points: Sequence[Sequence[int]], *, backend: str = "auto") -> PolytopeData:
    """Face lattice, facet inequalities and normalised volume of conv(points).

    Parameters
    ----------
    points
        Integer points, one row per point; repeated points are allowed.
    backend
        Where facet candidates come from: "python" (beneath-beyond, the
        reference), "qhull" (scipy's Qhull, falling back to "python" when it
        fails or its list is incomplete), "normaliz" (PyNormaliz, which must
        be installed; same fallback) or "auto", which is "python". Every
        candidate is verified and the list certified complete in integer
        arithmetic, so every backend returns the same PolytopeData.

    Raises
    ------
    ValidationError
        If there are no points, a coordinate is not an integer, or the points
        do not all have the same number of coordinates.
    ComputationError
        If backend is unknown, if "normaliz" is requested without PyNormaliz,
        or if the facets from beneath-beyond fail the completeness
        certificate, which would be a bug.
    """
    pts = _exact.integer_points(points)
    if not pts:
        raise ValidationError("polytope_data needs at least one point")
    hull = _hull(pts, backend)
    ambient = len(pts[0])
    all_faces = tuple(hull.faces)
    facets: list[Facet] = []
    if hull.dimension == ambient and hull.dimension >= 2:
        facets = [
            Facet(h.normal, h.offset, tuple(_exact.mask_indices(h.mask))) for h in hull.halfspaces
        ]
    elif hull.dimension == ambient == 1:
        values = [p[0] for p in pts]
        lo, hi = min(values), max(values)
        facets = [
            Facet((-1,), -lo, tuple(j for j, x in enumerate(values) if x == lo)),
            Facet((1,), hi, tuple(j for j, x in enumerate(values) if x == hi)),
        ]
    facets.sort(key=lambda f: f.point_indices)
    return PolytopeData(
        points=tuple(pts),
        ambient_dimension=ambient,
        dimension=hull.dimension,
        vertex_indices=tuple(sorted(idx[0] for k, idx in all_faces if k == 0)),
        faces=all_faces,
        facets=tuple(facets),
        normalized_volume=_normalized_volume(np.asarray(pts, dtype=np.int64), hull.dimension),
    )


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
