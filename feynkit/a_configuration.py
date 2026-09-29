"""
Arbitrary GKZ A-configurations: equivalence, finite-index maps, intrinsic models.

An A-configuration is a matrix whose columns are (homogenised) exponent vectors
of a polynomial, the fundamental input to a GKZ hypergeometric system.  This
module works with *arbitrary* A-matrices, not only those derived from Feynman
graphs via feynkit's pipeline.

Public surface
--------------
AConfiguration
    Wraps an integer matrix.  Exposes Smith normal-form invariants,
    Newton-polytope vertices, unimodular / affine equivalence tests, and the
    finite-index map search.

FiniteIndexResult
    Result dataclass for :func:`finite_index_map`.

IntrinsicModel
    Intrinsic lattice model produced by :func:`intrinsic_lattice_model`.

finite_index_map(source, target) -> FiniteIndexResult
    Search for an integer affine map x -> Mx + t with det M != 0, not
    necessarily +/-1, taking the source points injectively to target points.

intrinsic_lattice_model(points) -> IntrinsicModel
    Express a point configuration in the Hermite normal form basis of the
    lattice its differences span.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, permutations
from typing import Any, cast

import numpy as np
import sympy as sp

from . import _exact
from .core.exceptions import ValidationError
from .normal_forms._chart import chart_frame, lift_linear
from .normal_forms._invariants import (
    hull_vertex_indices,
    integer_inverse,
    integral_candidate,
    to_integer_points,
)
from .normal_forms.affine_equivalence import (
    affine_maps,
    is_affinely_equivalent,
    is_point_config_equivalent,
    is_unimodular_equivalent,
)
from .normal_forms.polytope_automorphisms import configuration_symmetries
from .polytope import lattice_chart
from .polytope import normalized_volume as _normalized_volume
from .types import PolytopeAutomorphisms, PolytopeEquivalence

# ------------------------------------------------------------------------------
# Value types
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteIndexResult:
    """
    Result of a finite-index map search between two point configurations.

    Attributes
    ----------
    found
        Whether a valid integer affine map was found.
    witness_matrix
        The integer matrix M such that M*x + t maps source onto target.
    translation
        The integer translation vector t (column vector as ImmutableMatrix).
    determinant
        |det(M)|.  Equal to 1 iff the map is unimodular.
    column_permutation
        The permutation of source columns induced by M (i.e. which source
        point maps to which target point).
    is_unimodular
        True iff ``determinant == 1``.
    """

    found: bool
    witness_matrix: sp.ImmutableMatrix | None = None
    translation: sp.ImmutableMatrix | None = None
    determinant: int | None = None
    column_permutation: list[int] | None = None
    is_unimodular: bool = False


@dataclass(frozen=True)
class IntrinsicModel:
    """
    A point configuration in a basis of the lattice its differences span.

    Let L be the lattice spanned by the differences of the points from the
    first point, and d its rank. As intrinsic_lattice_model builds it, point i
    of the configuration is exactly

        base_point + sum_t intrinsic_coords[i][t] * basis[t].

    Attributes
    ----------
    base_point
        The first point; its intrinsic coordinates are all 0.
    smith_invariants
        The non-zero invariant factors of the difference matrix, the values
        AConfiguration.smith_invariants gives. Their product is the index of L
        in its saturation, the integer points of its linear span.
    intrinsic_rank
        d, the affine dimension of the points. It is not the holonomic rank of
        the GKZ system.
    intrinsic_coords
        For each point, its d integer coordinates in basis relative to
        base_point.
    basis
        The d vectors of the Hermite normal form basis of L, as in
        feynkit.polytope.lattice_chart. The default () keeps code that builds
        an IntrinsicModel without it working; intrinsic_lattice_model always
        sets it.
    """

    base_point: tuple[int, ...]
    smith_invariants: list[int]
    intrinsic_rank: int
    intrinsic_coords: tuple[tuple[int, ...], ...]
    basis: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class SymmetryPair:
    """
    One integer affine self-map of a GKZ A-configuration.

    A symmetry pair (M, t, P) satisfies: for every affine point a_j in the
    configuration, M*a_j + t = a_{P(j)}, where P is a bijection on column
    indices.  Equivalently, in homogenised form T*A = A*Pi_P where

        T = [[1,  0^T ],
             [t,  M   ]]

    is an invertible (n+1) x (n+1) integer matrix and Pi_P is the column
    permutation matrix for P (Pi_P[P(j), j] = 1).

    For Feynman integrals each symmetry pair gives the transformation identity
    (FMS 2019; de la Cruz 2024):

        I_A(beta, z_P) = I_A(T beta, z)

    where z_P = (z_{P(0)}, ..., z_{P(N-1)}), I_A is the Euler-Mellin integral
    without Gamma prefactors, and the prefactor R(beta) = 1.

    Attributes
    ----------
    linear_map
        M: n x n integer matrix (the linear part of the affine map).
    translation
        t: n x 1 integer column vector.
    column_permutation
        P as a length-N tuple; P[j] = index that column j maps to.
    determinant
        |det M|.  Equal to 1 iff the map is unimodular.
    is_unimodular
        True iff ``determinant == 1``.
    """

    linear_map: sp.ImmutableMatrix
    translation: sp.ImmutableMatrix
    column_permutation: tuple[int, ...]
    determinant: int
    is_unimodular: bool

    @property
    def homogenized_map(self) -> sp.ImmutableMatrix:
        """T = [[1, 0^T], [t, M]]: the (n+1) x (n+1) homogenised matrix."""
        n = self.linear_map.rows
        rows: list[list] = [[sp.Integer(1)] + [sp.Integer(0)] * n]
        M_list = self.linear_map.tolist()
        for i in range(n):
            rows.append([self.translation[i, 0]] + M_list[i])
        return sp.ImmutableMatrix(rows)

    def transform_beta(self, beta: list[sp.Expr] | sp.Matrix) -> list[sp.Expr]:
        """
        Apply T to the GKZ parameter vector beta.

        Returns T beta as a list, the parameter vector on the right of
        I_A(beta, z_P) = I_A(T beta, z).
        """
        T = sp.Matrix(self.homogenized_map.tolist())
        b = beta if isinstance(beta, sp.Matrix) else sp.Matrix(list(beta))
        return list(T * b)


# ------------------------------------------------------------------------------
# AConfiguration
# ------------------------------------------------------------------------------


class AConfiguration:
    """
    A GKZ A-configuration: an integer matrix whose columns are (optionally
    homogenised) exponent vectors.

    Parameters
    ----------
    matrix
        The A-matrix as a SymPy Matrix, numpy array, or nested list.
        Rows are coordinates, columns are monomials/points.
    is_homogenized
        If ``True``, the first row is all-ones and the affine points are the
        remaining rows.  If ``False``, the matrix is taken as-is.  If
        ``None`` (default), auto-detected: the first row is all-ones and at
        least two rows exist.
    """

    def __init__(
        self,
        matrix: object,
        *,
        is_homogenized: bool | None = None,
    ) -> None:
        if isinstance(matrix, sp.Matrix):
            arr = np.array(matrix.tolist(), dtype=np.int64)
        elif isinstance(matrix, np.ndarray):
            arr = matrix.astype(np.int64)
        else:
            arr = np.array(list(cast(Iterable[Any], matrix)), dtype=np.int64)

        if arr.ndim != 2:
            raise ValueError("A-matrix must be 2-dimensional")

        self._matrix = arr

        if is_homogenized is None:
            # Auto-detect: first row is all-ones, at least 2 rows.
            is_homogenized = bool(arr.shape[0] >= 2 and np.all(arr[0] == 1))

        self._is_homogenized = bool(is_homogenized)

    # -- raw access -------------------------------------------------------------

    @property
    def matrix(self) -> sp.ImmutableMatrix:
        """The full A-matrix as a SymPy ImmutableMatrix (rows x columns)."""
        return sp.ImmutableMatrix(self._matrix.tolist())

    @property
    def is_homogenized(self) -> bool:
        return self._is_homogenized

    # -- affine points (rows = points, columns = coordinates) ------------------

    @property
    def affine_points(self) -> np.ndarray:
        """
        Affine (un-homogenised) points as an n_pts x n_dim integer array.

        If the matrix is homogenised, the first row (all-ones) is stripped.
        The result is transposed so that rows are points.
        """
        if self._is_homogenized:
            return self._matrix[1:].T.copy()
        return self._matrix.T.copy()

    @property
    def n_points(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def ambient_dim(self) -> int:
        """Dimension of the ambient integer lattice."""
        if self._is_homogenized:
            return int(self._matrix.shape[0]) - 1
        return int(self._matrix.shape[0])

    # -- Smith invariants -------------------------------------------------------

    @property
    def smith_invariants(self) -> list[int]:
        """
        Non-zero Smith normal form diagonal entries of the difference matrix.

        These classify the sublattice generated by the configuration (up to
        basis change), and are an intrinsic invariant of the point set.
        """
        pts = self.affine_points.tolist()
        if len(pts) < 2:
            return []
        base = pts[0]
        differences = [[a - b for a, b in zip(p, base, strict=True)] for p in pts[1:]]
        return _exact.smith_invariants(differences, self.ambient_dim)

    # -- Newton polytope --------------------------------------------------------

    @property
    def newton_polytope_points(self) -> list[tuple[int, ...]]:
        """Hull vertices of the Newton polytope (convex hull of affine points)."""
        pts = self.affine_points
        idx = hull_vertex_indices(pts)
        return [tuple(int(x) for x in pts[i]) for i in idx]

    # -- affine dimension -------------------------------------------------------

    @property
    def affine_dim(self) -> int:
        """Affine dimension of the points, computed exactly."""
        return _exact.affine_rank(self.affine_points.tolist())

    # -- normalised lattice volume ----------------------------------------------

    @cached_property
    def normalized_volume(self) -> int:
        """
        Normalised volume of the Newton polytope, in the lattice its points span.

        The volume of conv(affine points), of affine dimension d, normalised so
        that a d-simplex whose edge vectors form a basis of the lattice L
        spanned by the differences of the points has volume 1. It is computed
        exactly from a pulling triangulation of the certified face lattice:
        for a full-dimensional configuration as the sum of |det| over the
        simplices divided by [Z^n : L], the product of the Smith invariants,
        and for a lower-dimensional one in lattice coordinates, where the
        points generate Z^d. See feynkit.polytope.normalized_volume.

        For a full-dimensional configuration, generic coefficients and
        non-resonant beta this is the holonomic rank of the GKZ system
        (Klausen 2020, Theorem 2.2; de la Cruz 2019, section 3). Below full
        dimension the rows of the homogenised A are linearly dependent, and for
        generic beta the system has no non-zero solutions. Two configurations
        related by a finite-index map of index k have volumes differing by k in
        the ambient lattice, but the same volume here.

        Raises
        ------
        ValidationError
            If the configuration has no points.
        ComputationError
            If the face lattice fails its completeness certificate or a simplex
            determinant is not a positive multiple of [Z^n : L]; either would
            be a bug.
        """
        return _normalized_volume(self.affine_points)

    # -- equivalence -----------------------------------------------------------

    def is_unimodular_equivalent_to(self, other: AConfiguration) -> PolytopeEquivalence:
        """Test unimodular equivalence of the Newton polytopes (Liu-Cai)."""
        return is_unimodular_equivalent(
            self.newton_polytope_points,
            other.newton_polytope_points,
        )

    def is_affinely_equivalent_to(self, other: AConfiguration) -> PolytopeEquivalence:
        """
        Test affine equivalence of the Newton polytopes (hull vertices only).

        Returns ``relation="affine_polytope"`` with witness ``M``, translation
        ``t``, and ``determinant=det(M)`` on success.  For the stricter GKZ
        check on all monomials use :meth:`is_point_config_equivalent_to`.
        """
        return is_affinely_equivalent(
            self.newton_polytope_points,
            other.newton_polytope_points,
        )

    def is_point_config_equivalent_to(self, other: AConfiguration) -> PolytopeEquivalence:
        """
        Test affine equivalence of the full A-column sets (all monomials).

        This is the stricter GKZ system condition: the map must send *every*
        monomial exponent vector of self to one of other, not just the convex
        hull vertices.  Returns ``relation="affine_point_config"``.
        """
        return is_point_config_equivalent(
            self.affine_points,
            other.affine_points,
        )

    def finite_index_map_to(self, other: AConfiguration) -> FiniteIndexResult:
        """
        Search for a finite-index integer affine map from self to other.

        Operates on all affine points (all A-columns), not just hull vertices.
        To search on hull vertices only, pass
        ``AConfiguration(np.array(self.newton_polytope_points))`` explicitly.
        """
        return finite_index_map(self, other)

    def intrinsic_model(self) -> IntrinsicModel:
        """The intrinsic lattice model of the affine points; see intrinsic_lattice_model."""
        return intrinsic_lattice_model(self.affine_points)

    def automorphisms(self) -> PolytopeAutomorphisms:
        """The automorphism group of the Newton polytope as a lattice polytope."""
        from .normal_forms.polytope_automorphisms import compute_polytope_automorphisms

        return compute_polytope_automorphisms(self.newton_polytope_points)

    def symmetry_pairs(self) -> list[SymmetryPair]:
        """
        Find all integer affine self-maps of this configuration.

        Each returned :class:`SymmetryPair` (M, t, P) satisfies T*A = A*Pi_P
        where T = [[1, 0^T], [t, M]] and gives the transformation identity
        I_A(beta, z_P) = I_A(T beta, z) for the associated Feynman integral
        without its prefactor (de la Cruz 2024).

        Every returned pair has det M = +/-1: P has finite order k, so
        T^k A = A, and as A has full rank, M^k = I.  Below full dimension the
        pairs are found in the lattice chart of the points and extended to
        Z^n; see :func:`symmetry_pairs`.  Maps with |det M| > 1 relate two
        different configurations; see :func:`finite_index_map`.
        """
        return symmetry_pairs(self)

    def __repr__(self) -> str:
        r, c = self._matrix.shape
        hom = " (homogenised)" if self._is_homogenized else ""
        return f"AConfiguration({r} x {c}{hom}, {self.n_points} points, dim={self.ambient_dim})"


# ------------------------------------------------------------------------------
# finite_index_map
# ------------------------------------------------------------------------------


def finite_index_map(
    source: AConfiguration | Sequence | np.ndarray,
    target: AConfiguration | Sequence | np.ndarray,
) -> FiniteIndexResult:
    """
    Search for an integer affine map x -> Mx + t, with det M != 0, taking
    the source points injectively to target points.

    M is an integer matrix and t an integer vector, and each source point,
    counted with its multiplicity, goes to a different target point; with as
    many points on both sides the map is a bijection of the columns. If
    |det M| = 1, the map is unimodular (a lattice isomorphism). If
    |det M| > 1, the map is a finite-index embedding: the image M Z^n of the
    source lattice is a sublattice of index |det M| in the target lattice Z^n.

    Parameters
    ----------
    source, target
        AConfiguration objects, or raw point arrays (rows = points).

    Returns
    -------
    FiniteIndexResult
        ``found=True`` with the first map the search finds, or
        ``found=False`` when there is none, as when there are more source
        points than target points.

    Raises
    ------
    ValidationError
        If a coordinate is not an integer.
    NotImplementedError
        If the source points are not full-dimensional and there are fewer of
        them than target points.

    Algorithm
    ---------
    Every step is in integer arithmetic. With as many points on both sides
    the map is an affine bijection of the two polytopes, one of the maps of
    feynkit.normal_forms.affine_equivalence.affine_maps, which finds them by
    the neighbour-restricted vertex search, the translation first; the first
    that is integral is returned, and every such map has the same |det M|,
    the ratio of the normalised volumes. Below full dimension that search
    runs in the lattice charts of the two configurations, and a chart map is
    kept when its lift (feynkit.normal_forms._chart.lift_linear) is integral,
    that is when it maps the integer points of one affine hull into those of
    the other. The lift maps a complement of one direction space onto a
    complement of the other, which gives the least |det M| of all integral
    extensions.

    With fewer source points than target points, the images of an affine
    basis of the source are tried among all ordered sets of target points,
    skipping a set unless its determinant is a non-zero multiple of that of
    the basis, since det M is their quotient; this can be slow for large
    configurations.
    """
    src_pts = _to_pts(source)
    tgt_pts = _to_pts(target)
    n_src, n_dim = src_pts.shape
    if tgt_pts.shape[1] != n_dim or n_src == 0 or n_src > tgt_pts.shape[0]:
        return FiniteIndexResult(found=False)
    src = [tuple(int(x) for x in p) for p in src_pts.tolist()]
    tgt = [tuple(int(x) for x in p) for p in tgt_pts.tolist()]

    aff_dim = _exact.affine_rank(src)
    candidates: Iterable[tuple[list[list[int]], list[int]]]
    if aff_dim == 0:
        identity = [[int(i == j) for j in range(n_dim)] for i in range(n_dim)]
        candidates = (
            (identity, [b - a for a, b in zip(src[0], point, strict=True)])
            for point in dict.fromkeys(tgt)
        )
    elif n_src < len(tgt):
        if aff_dim < n_dim:
            raise NotImplementedError(
                "finite_index_map searches below full dimension only between configurations "
                "with the same number of points"
            )
        candidates = _embeddings(src, tgt)
    elif aff_dim < n_dim:
        candidates = _chart_bijections(src_pts, tgt_pts)
    else:
        candidates = _integral_bijections(src_pts, tgt_pts)

    for M, t in candidates:
        perm = _column_map(src, tgt, M, t)
        if perm is not None:
            det = abs(_exact.determinant(M))
            return FiniteIndexResult(
                found=True,
                witness_matrix=sp.ImmutableMatrix(M),
                translation=sp.ImmutableMatrix([[x] for x in t]),
                determinant=det,
                column_permutation=perm,
                is_unimodular=(det == 1),
            )
    return FiniteIndexResult(found=False)


def _column_map(
    src: list[tuple[int, ...]], tgt: list[tuple[int, ...]], M: list[list[int]], t: list[int]
) -> list[int] | None:
    """For each source point x, the index of a target point M x + t, each index used once; or None."""
    free: dict[tuple[int, ...], list[int]] = {}
    for j, point in enumerate(tgt):
        free.setdefault(point, []).append(j)
    perm: list[int] = []
    for x in src:
        image = tuple(
            sum(m * c for m, c in zip(row, x, strict=True)) + s for row, s in zip(M, t, strict=True)
        )
        slots = free.get(image)
        if not slots:
            return None
        perm.append(slots.pop(0))
    return perm


def _integral_bijections(
    src_pts: np.ndarray, tgt_pts: np.ndarray
) -> Iterator[tuple[list[list[int]], list[int]]]:
    """The integral maps among the affine maps of full-dimensional src_pts onto tgt_pts."""
    for linear, shift, q in affine_maps(src_pts, tgt_pts):
        if all(x % q == 0 for row in linear for x in row) and all(s % q == 0 for s in shift):
            yield [[x // q for x in row] for row in linear], [s // q for s in shift]


def _chart_bijections(
    src_pts: np.ndarray, tgt_pts: np.ndarray
) -> Iterator[tuple[list[list[int]], list[int]]]:
    """The affine maps of src_pts onto tgt_pts, below full dimension, whose chart maps lift."""
    frame_a, frame_b = chart_frame(src_pts), chart_frame(tgt_pts)
    if frame_a.dimension != frame_b.dimension:
        return
    first = frame_a.chart.coordinates[0]
    origin = [int(x) for x in src_pts[0]]
    for linear, shift, q in affine_maps(frame_a.coordinates, frame_b.coordinates):
        lift = lift_linear(frame_a, frame_b, sp.Matrix(linear) / q)
        if lift is None:
            continue
        M = [[int(x) for x in row] for row in lift.tolist()]
        # The chart map sends the chart point of src_pts[0] to a chart point of tgt_pts.
        image = frame_b.chart.to_ambient(
            [
                (sum(n * c for n, c in zip(row, first, strict=True)) + s) // q
                for row, s in zip(linear, shift, strict=True)
            ]
        )
        t = [
            b - sum(m * a for m, a in zip(row, origin, strict=True))
            for row, b in zip(M, image, strict=True)
        ]
        yield M, t


def _embeddings(
    src: list[tuple[int, ...]], tgt: list[tuple[int, ...]]
) -> Iterator[tuple[list[list[int]], list[int]]]:
    """
    Integral maps M, t with det M != 0 that send an affine basis of src to target points.

    src is full-dimensional. The basis is src[0] and n more points; its image
    is tried at every ordered set of n + 1 distinct target points, skipping a
    set unless its determinant is a non-zero multiple of that of the basis.
    """
    n = len(src[0])
    deltas = np.array(
        [[a - b for a, b in zip(p, src[0], strict=True)] for p in src], dtype=np.int64
    )
    basis = _basis_indices(deltas, n)
    if basis is None:  # pragma: no cover - src is full-dimensional
        return
    size = 2 * max(abs(x) for p in (*src, *tgt) for x in p) + 1
    adj, det = integer_inverse(sp.Matrix(deltas[basis].T.tolist()), size)
    points = list(dict.fromkeys(tgt))
    for anchor in points:
        others = [p for p in points if p != anchor]
        for combo in combinations(others, n):
            rows = [[a - b for a, b in zip(p, anchor, strict=True)] for p in combo]
            image_det = _exact.determinant(rows)
            if image_det == 0 or image_det % det:
                continue
            for order in permutations(rows):
                M_int = integral_candidate(np.array(order, dtype=adj.dtype).T, adj, det)
                if M_int is None:
                    continue
                M = [[int(x) for x in row] for row in M_int.tolist()]
                t = [
                    b - sum(m * a for m, a in zip(row, src[0], strict=True))
                    for row, b in zip(M, anchor, strict=True)
                ]
                yield M, t


# ------------------------------------------------------------------------------
# symmetry_pairs
# ------------------------------------------------------------------------------


def symmetry_pairs(
    cfg: AConfiguration | Sequence | np.ndarray,
) -> list[SymmetryPair]:
    """
    Find all integer affine self-maps of a GKZ A-configuration.

    An integer affine self-map x -> Mx + t sends every point in the
    configuration to another point in the configuration bijectively, with M
    an invertible integer matrix.  Such a map has det M = +/-1: P has finite
    order k, so T^k A = A, and as A has full rank, M^k = I.  Maps with
    |det M| > 1 relate two different configurations; see
    :func:`finite_index_map`.

    Below full dimension, where A does not have full rank, the pairs are
    those of the points in their lattice chart whose linear part maps the
    integer points of the affine hull onto themselves, each extended to Z^n
    as the automorphisms of :func:`compute_polytope_automorphisms` are: M
    fixes a complement of the direction space of the affine hull, the span of
    the differences of the points, and det M = +/-1. Another extension
    differs only off the affine hull, where T beta changes with it.

    Each result is a :class:`SymmetryPair` encoding the linear map M,
    translation t, induced column permutation P, and determinant |det M|.

    For Feynman integrals, each symmetry pair gives (de la Cruz 2024):

        I_A(beta, z_P) = I_A(T beta, z),   T = [[1, 0^T], [t, M]]

    where z_P = (z_{P(0)}, ..., z_{P(N-1)}), I_A is the Euler-Mellin integral
    without Gamma prefactors, and the prefactor R(beta) = 1.

    Parameters
    ----------
    cfg
        An :class:`AConfiguration`, or any array of affine points (rows =
        points, columns = coordinates).

    Returns
    -------
    list[SymmetryPair]
        All valid self-maps, including the identity.  Empty when the
        configuration has no points, and when a point is repeated but not
        all points are equal, since the columns are matched by their
        coordinates.

    Algorithm
    ---------
    A pair permutes the vertices of conv(A), so it is one of the
    automorphisms of the Newton polytope that
    :func:`compute_polytope_automorphisms` finds, and the pairs are the
    automorphisms that send every point to a point
    (feynkit.normal_forms.polytope_automorphisms.configuration_symmetries).
    They come in the order in which a basis search over all the points,
    anchored at the first with a basis from the rarest label classes, finds
    them.
    """
    pts = _to_pts(cfg)
    N = pts.shape[0]
    n_dim = pts.shape[1]

    if N == 0:
        return []

    deltas = (pts - pts[0]).astype(np.int64)
    aff_dim = _exact.rank(deltas[1:].tolist()) if N > 1 else 0

    if aff_dim == 0:
        return [
            SymmetryPair(
                linear_map=sp.ImmutableMatrix(sp.eye(n_dim)),
                translation=sp.ImmutableMatrix(sp.zeros(n_dim, 1)),
                column_permutation=tuple(range(N)),
                determinant=1,
                is_unimodular=True,
            )
        ]

    if aff_dim != n_dim:
        return _chart_symmetry_pairs(pts)

    # Every pair permutes the vertices of conv(A), so it is an automorphism of
    # the polytope; the pairs are the automorphisms that permute every point.
    return [
        SymmetryPair(
            linear_map=M,
            translation=t,
            column_permutation=perm,
            determinant=1,
            is_unimodular=True,
        )
        for M, t, perm in configuration_symmetries(pts)
    ]


def _chart_symmetry_pairs(pts: np.ndarray) -> list[SymmetryPair]:
    """The symmetry pairs of points that are not full-dimensional, from their lattice chart."""
    frame = chart_frame(pts)
    origin = sp.Matrix(pts[0].tolist())
    results: list[SymmetryPair] = []
    for pair in symmetry_pairs(frame.coordinates):
        M = lift_linear(frame, frame, pair.linear_map)
        if M is None:
            continue
        t = sp.Matrix(pts[pair.column_permutation[0]].tolist()) - M * origin
        det = int(abs(M.det()))
        results.append(
            SymmetryPair(
                linear_map=M,
                translation=sp.ImmutableMatrix(t),
                column_permutation=pair.column_permutation,
                determinant=det,
                is_unimodular=(det == 1),
            )
        )
    return results


# ------------------------------------------------------------------------------
# intrinsic_lattice_model
# ------------------------------------------------------------------------------


def intrinsic_lattice_model(
    points: AConfiguration | Sequence | np.ndarray,
) -> IntrinsicModel:
    """
    Express a point configuration in the Hermite normal form basis of its lattice.

    Let alpha_0, ..., alpha_{N-1} be the points, in Z^n, and L the lattice
    spanned by the differences alpha_j - alpha_0, of rank d. The model is read
    off the lattice chart x = o + B c of feynkit.polytope.lattice_chart:

    - basis is B, the Hermite normal form, in SymPy's convention, of the
      matrix whose columns are the differences, so a basis of L;
    - base_point is alpha_0;
    - intrinsic_coords[j] is c_j - c_0, where c_j are the chart coordinates.

    So for every d from 0 to n, and in integer arithmetic,

        alpha_j = base_point + sum_t intrinsic_coords[j][t] * basis[t],

    and intrinsic_coords[0] is 0. The coordinates are integers, since B is a
    basis of L, and unique, since its columns are linearly independent. They
    can be negative. The chart shifts its coordinates to minimum 0 on each
    axis, so its origin o = alpha_0 - B c_0 need not be a point of the
    configuration, and chart.to_ambient gives alpha_j from c_j, not from
    intrinsic_coords[j].

    intrinsic_rank is d, the affine dimension, and equals len(basis). It is
    not the holonomic rank of the GKZ system, which for generic beta is the
    normalised volume of conv(points) when d = n. Below full dimension, d < n,
    the rows of the homogenised A are linearly dependent, and for generic beta
    the system has no non-zero solutions.

    smith_invariants are the non-zero invariant factors of the (N - 1) x n
    difference matrix, from feynkit._exact.smith_invariants, as in
    AConfiguration.smith_invariants. Their product is the index of L in its
    saturation, the integer points of its linear span: 1 exactly when L is
    saturated, and [Z^n : L] when d = n.

    For d = 0, a single point or copies of one, basis and smith_invariants
    are empty and every coordinate is the empty tuple.

    Parameters
    ----------
    points
        An AConfiguration, whose affine points are used, or the points as a
        two-dimensional array or a sequence of integer sequences, one per
        point.

    Raises
    ------
    ValidationError
        If there are no points, points is not a sequence of coordinate
        sequences or a two-dimensional array, a coordinate is not an integer,
        or the points do not all have the same number of coordinates.
    ComputationError
        If lattice_chart finds a point outside the lattice spanned by the
        differences, which would be a bug.
    """
    source = points.affine_points if isinstance(points, AConfiguration) else points
    pts = _exact.integer_points(source)
    if not pts:
        raise ValidationError("intrinsic_lattice_model needs at least one point")
    chart = lattice_chart(pts)
    first = chart.coordinates[0]
    base = pts[0]
    differences = [[a - b for a, b in zip(p, base, strict=True)] for p in pts[1:]]
    return IntrinsicModel(
        base_point=base,
        smith_invariants=_exact.smith_invariants(differences, len(base)),
        intrinsic_rank=len(chart.basis),
        intrinsic_coords=tuple(
            tuple(a - b for a, b in zip(c, first, strict=True)) for c in chart.coordinates
        ),
        basis=chart.basis,
    )


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------


def _to_pts(obj: object) -> np.ndarray:
    """Coerce to an (n_pts x n_dim) integer numpy array.

    A coordinate that is not an integer raises ValidationError, as in
    to_integer_points; it is never rounded.
    """
    if isinstance(obj, AConfiguration):
        return obj.affine_points
    return to_integer_points(obj)


def _basis_indices(deltas: np.ndarray, aff_dim: int) -> list[int] | None:
    """
    Greedy selection of ``aff_dim`` row indices (skip row 0 = zero vector)
    whose rows are linearly independent.
    """
    chosen: list[int] = []
    chosen_rows: list[list[int]] = []
    for i in range(1, deltas.shape[0]):
        row = [int(x) for x in deltas[i]]
        if _exact.rank([*chosen_rows, row]) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(row)
            if len(chosen) == aff_dim:
                return chosen
    return None
