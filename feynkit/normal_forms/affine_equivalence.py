"""
Equivalence tests for convex (lattice) polytopes.

This module exposes two complementary verbs:

- :func:`is_unimodular_equivalent`, implements the Liu-Cai algorithm
  (arXiv:2506.23846). Decides whether two integer point configurations span
  unimodularly-isomorphic lattice polytopes, and returns a witness
  ``U in GL_n(Z)`` and an integer translation when one exists.
- :func:`is_affinely_equivalent`, broader equivalence over the rationals,
  and :func:`is_point_config_equivalent`, the same for every point. Both use
  the vertex search of the unimodular test with rational maps and labels
  that every affine bijection keeps.

Both return a :class:`feynkit.PolytopeEquivalence` carrying the verdict, a
witness map (when known), and a vertex correspondence (when known).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp

from .. import _exact
from ..core.exceptions import ValidationError
from ..polytope import PolytopeData, polytope_data
from ..types import PolytopeEquivalence
from . import _invariants
from ._chart import ChartFrame, chart_frame, lift_linear
from .polytope_automorphisms import vertex_maps

# ------------------------------------------------------------------------------
# Public API: Liu-Cai unimodular equivalence
# ------------------------------------------------------------------------------


def is_unimodular_equivalent(
    points_a: object,
    points_b: object,
) -> PolytopeEquivalence:
    """
    Decide whether two integer point configurations span unimodularly-
    isomorphic convex lattice polytopes (Liu-Cai, arXiv:2506.23846).

    Two lattice polytopes ``P, P' subset of Z^n`` are *unimodularly isomorphic* if
    there exists ``U in GL_n(Z)`` and ``Z in Z^n`` such that ``P' = UP + Z``.

    Parameters
    ----------
    points_a, points_b
        Point configurations as ``d x n`` integer arrays (rows = lattice
        points). Accepts numpy arrays, sympy matrices, or any iterable of
        integer-coordinate tuples. Non-vertex points (interior or on a
        face) are filtered out automatically by computing the convex hull.
        Unlike the affine tests, this one does not scale rational points to
        integers, since scaling changes the lattice.

    Returns
    -------
    PolytopeEquivalence
        ``equivalent=True`` together with a witness map ``U`` and a vertex
        correspondence on success; ``equivalent=False`` otherwise.

    Raises
    ------
    ValidationError
        If a coordinate is not an integer.

    Notes
    -----
    The algorithm:

    1. Restrict each point set to its vertices, from the certified face
       lattice of :func:`feynkit.polytope.polytope_data`.
    2. Build the labelled vertex/edge graph $\\mathcal{GW}(P)$ on the exact
       1-skeleton, with node label ``lab(v) = det(A_v)`` (Liu-Cai,
       Definition 5.2) and edge weight ``lab(u) + lab(v)``.
    3. Fix an anchor vertex of the first polytope, in its rarest label
       class, and a basis among its neighbours; map them to each vertex of
       the second with the same label and to its neighbours with the same
       labels and edges, solve for ``U`` and verify it on every vertex
       (feynkit.normal_forms.polytope_automorphisms.vertex_maps). The
       translation, when it maps one vertex set onto the other, comes first.

    Below full dimension both polytopes must have the same dimension d, and
    the search runs in the lattice charts of their vertex sets, where they are
    full-dimensional in Z^d. A chart map is accepted only when it maps the
    integer points of one affine hull onto those of the other, which needs the
    same sublattice index for the two vertex sets, and the witness is its lift
    U in GL_n(Z). U maps the direction space of one affine hull onto that of
    the other, and a complement of the first onto a complement of the second;
    see feynkit.normal_forms._chart. Two points are always equivalent, and two
    segments exactly when their lattice lengths agree.

    See Also
    --------
    is_affinely_equivalent : the broader rational equivalence relation.
    """
    pts_a = _invariants.to_integer_points(points_a)
    pts_b = _invariants.to_integer_points(points_b)

    if pts_a.shape[1] != pts_b.shape[1]:
        return PolytopeEquivalence(False, "unimodular")

    data_a, idx_a = _vertices(pts_a)
    data_b, idx_b = _vertices(pts_b)
    V_a = pts_a[idx_a]
    V_b = pts_b[idx_b]

    if V_a.shape != V_b.shape:
        return PolytopeEquivalence(False, "unimodular")

    n_dim = V_a.shape[1]
    n_vert = V_a.shape[0]

    # Trivial cases.
    if data_a is None or data_b is None:
        return PolytopeEquivalence(True, "unimodular", witness_map=sp.eye(n_dim))
    if n_vert == 1:
        # Single point: any unimodular map works; pick the identity translation.
        U = sp.eye(n_dim)
        return PolytopeEquivalence(
            equivalent=True,
            relation="unimodular",
            witness_map=U,
            vertex_correspondence=[0],
        )
    if data_a.dimension != data_b.dimension:
        return PolytopeEquivalence(False, "unimodular")

    frames: tuple[ChartFrame, ChartFrame] | None = None
    accept: Callable[[sp.Matrix], bool] | None = None
    coords_a, coords_b = V_a, V_b
    if not data_a.is_full_dimensional:
        frame_a, frame_b = chart_frame(V_a), chart_frame(V_b)
        if frame_a.index != frame_b.index:
            return PolytopeEquivalence(False, "unimodular")
        frames = (frame_a, frame_b)
        coords_a, coords_b = frame_a.coordinates, frame_b.coordinates

        def lifts(U: sp.Matrix) -> bool:
            return lift_linear(frame_a, frame_b, U) is not None

        accept = lifts

    # Build labelled graphs.
    GW_a = _invariants.label_skeleton(coords_a, _invariants.polytope_skeleton(data_a))
    GW_b = _invariants.label_skeleton(coords_b, _invariants.polytope_skeleton(data_b))

    # Quick filter: vertex-label multisets must match.
    labels_a = sorted(GW_a.nodes[i]["label"] for i in GW_a.nodes())
    labels_b = sorted(GW_b.nodes[i]["label"] for i in GW_b.nodes())
    if labels_a != labels_b:
        return PolytopeEquivalence(False, "unimodular")

    # Edge counts must match too (a 1-skeleton invariant).
    if GW_a.number_of_edges() != GW_b.number_of_edges():
        return PolytopeEquivalence(False, "unimodular")

    result = _direct_basis_search(coords_a, coords_b, GW_a, GW_b, idx_a, idx_b, accept=accept)
    if frames is None or not result.equivalent:
        return result
    return PolytopeEquivalence(
        equivalent=True,
        relation="unimodular",
        witness_map=lift_linear(*frames, result.witness_map),
        vertex_correspondence=result.vertex_correspondence,
    )


def _vertices(pts: np.ndarray) -> tuple[PolytopeData | None, np.ndarray]:
    """The polytope data of the points, None when there are none, and the indices of the vertices."""
    if pts.shape[0] == 0:
        return None, np.arange(0)
    data = polytope_data(pts)
    return data, np.array(data.vertex_indices, dtype=np.int64)


def _direct_basis_search(
    V_a: np.ndarray,
    V_b: np.ndarray,
    GW_a: Any,  # networkx.Graph with integer 'label' node attributes
    GW_b: Any,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    accept: Callable[[sp.Matrix], bool] | None = None,
) -> PolytopeEquivalence:
    """
    The first unimodular map of V_a onto V_b that vertex_maps finds.

    ``GW_a`` and ``GW_b`` are the labelled 1-skeletons, node i being row i.
    Liu-Cai labels are unimodular invariants, so the maps sought keep them. A
    candidate is kept when it is integral, sends every vertex to a vertex and
    has determinant +/-1; a map that ``accept``, when given, rejects is skipped
    and the search goes on.
    """
    n_vert = V_a.shape[0]
    labels_a = [GW_a.nodes[i]["label"] for i in range(n_vert)]
    labels_b = [GW_b.nodes[i]["label"] for i in range(n_vert)]
    neighbours_a = [set(GW_a.neighbors(i)) for i in range(n_vert)]
    neighbours_b = [set(GW_b.neighbors(i)) for i in range(n_vert)]
    for U_int, _, perm in vertex_maps(V_a, V_b, neighbours_a, neighbours_b, labels_a, labels_b):
        U = sp.Matrix(U_int.tolist())
        if abs(U.det()) != 1:
            continue
        if accept is not None and not accept(U):
            continue
        return PolytopeEquivalence(
            equivalent=True,
            relation="unimodular",
            witness_map=sp.ImmutableMatrix(U),
            vertex_correspondence=_lift_correspondence(idx_a, idx_b, list(perm)),
        )
    return PolytopeEquivalence(False, "unimodular")


def _lift_correspondence(
    hull_idx_a: np.ndarray,  # noqa: ARG001 (source indices are implicit in the correspondence order)
    hull_idx_b: np.ndarray,
    hull_correspondence: list[int],
) -> list[int]:
    """
    Translate a vertex correspondence from hull-vertex indices back to the
    original input indices. Non-vertex (interior / on-face) input points
    have no image and are not included.
    """
    return [int(hull_idx_b[j]) for j in hull_correspondence]


# ------------------------------------------------------------------------------
# Public API: affine equivalence (broader, rational)
# ------------------------------------------------------------------------------


def is_affinely_equivalent(
    points_a: object,
    points_b: object,
) -> PolytopeEquivalence:
    """
    Decide whether two **Newton polytopes** are affinely equivalent over the
    rationals: i.e. there is an affine map ``v -> M*v + t`` with ``M in GL_n(Q)``
    taking one *vertex set* onto the other.

    Both inputs are first projected to their convex-hull vertices before the
    search.  Interior lattice points and non-vertex boundary points are
    discarded, use :func:`is_point_config_equivalent` if you need the
    stricter all-columns GKZ check.

    The search is that of :func:`is_unimodular_equivalent`, with rational
    maps: an anchor in the rarest label class and a basis among its
    neighbours, mapped to vertices of the other polytope with the same labels
    and edges. An affine bijection multiplies both Liu-Cai determinants at
    every vertex by det(M)^2, so each polytope's are divided by the sum of the
    second over its vertices, which every such map keeps
    (feynkit.normal_forms._invariants.affine_labels). The translation, when it
    maps one vertex set onto the other, is tried first.

    The coordinates may be rational. Each configuration is scaled by the
    least common denominator of its coordinates, q_a or q_b, and a map
    ``M', t'`` between the scaled vertex sets gives ``M = (q_a / q_b) M'`` and
    ``t = t' / q_b``.

    Parameters
    ----------
    points_a, points_b
        Point configurations as ``n_pts x dim`` arrays (rows = points).
        Accepts numpy arrays, sympy matrices, or nested lists, with
        coordinates that are integers, integral floats, fractions or SymPy
        rationals.

    Returns
    -------
    PolytopeEquivalence
        ``relation="affine_polytope"``.  On success: ``witness_map=M``,
        ``translation=t``, ``determinant=det(M)``.  On failure: all ``None``.
        Below full dimension the vertices do not determine M; it is one
        invertible extension of the map between the affine hulls, and its
        determinant depends on that extension.

    Raises
    ------
    ValidationError
        If a coordinate is not a rational number given exactly; floats that
        are not integers are rejected, not rounded.
    """
    pts_a, q_a = _clear_denominators(_coerce_sympy_points(points_a))
    pts_b, q_b = _clear_denominators(_coerce_sympy_points(points_b))
    pts_a_np = _invariants.to_integer_points(pts_a)
    pts_b_np = _invariants.to_integer_points(pts_b)
    V_a = pts_a_np[_invariants.hull_vertex_indices(pts_a_np)]
    V_b = pts_b_np[_invariants.hull_vertex_indices(pts_b_np)]

    witness = _rescaled(
        _affine_witness(_coerce_sympy_points(V_a.tolist()), _coerce_sympy_points(V_b.tolist())),
        q_a,
        q_b,
    )
    if witness is None:
        return PolytopeEquivalence(equivalent=False, relation="affine_polytope")
    M, t, det = witness
    return PolytopeEquivalence(
        equivalent=True,
        relation="affine_polytope",
        witness_map=M,
        translation=t,
        determinant=det,
    )


def is_point_config_equivalent(
    points_a: object,
    points_b: object,
) -> PolytopeEquivalence:
    """
     Decide whether two **full point configurations** are affinely equivalent:
     i.e. there is an affine map ``v -> M*v + t`` taking every column of A_1
     (as a multiset) to a column of A_2.

     Unlike :func:`is_affinely_equivalent`, this function does **not** filter
     to convex-hull vertices.  It is the correct check for GKZ system
     equivalence, where the full monomial support, not just the Newton polytope
    , determines the hypergeometric system.

     The map is an affine bijection of the two polytopes, so it is found
     among the maps of their vertices, as in :func:`is_affinely_equivalent`,
     and checked on every point. Below full dimension it is found between the
     lattice charts of the two configurations, where it is unique for a
     correspondence of affine bases. Rational points are first scaled to
     integers, each configuration by the least common denominator q_a or q_b
     of its coordinates, and a map ``M', t'`` between the scaled points gives
     ``M = (q_a / q_b) M'`` and ``t = t' / q_b``.

     Parameters
     ----------
     points_a, points_b
         All affine points (rows = points) of the two A-matrices, with the
         homogenisation row stripped.  Size of the point sets must match.
     Returns
     -------
     PolytopeEquivalence
         ``relation="affine_point_config"``.  On success: ``witness_map=M``,
         ``translation=t``, ``determinant=det(M)``.  On failure: all ``None``.
         Below full dimension M is one invertible extension of the map
         between the affine hulls, and its determinant depends on that
         extension.

     Raises
     ------
     ValidationError
         If a coordinate is not a rational number given exactly; floats that
         are not integers are rejected, not rounded.
    """
    pts_a = _coerce_sympy_points(points_a)
    pts_b = _coerce_sympy_points(points_b)
    witness = _affine_witness(pts_a, pts_b)
    if witness is None:
        return PolytopeEquivalence(equivalent=False, relation="affine_point_config")
    M, t, det = witness
    return PolytopeEquivalence(
        equivalent=True,
        relation="affine_point_config",
        witness_map=M,
        translation=t,
        determinant=det,
    )


# ------------------------------------------------------------------------------
# Internal: the affine search, in integer arithmetic
# ------------------------------------------------------------------------------


def _coerce_sympy_points(data: object) -> sp.Matrix:
    matrix = sp.Matrix(data)
    if matrix.rows == 0 or matrix.cols == 0:
        raise ValueError("Point configuration must be non-empty")
    return matrix


def _affine_witness(
    points_a: sp.Matrix,
    points_b: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix, sp.Expr] | None:
    """
    (M, t, det M) of an affine map x -> M x + t sending the rows of points_a onto those of points_b.

    The rows are compared as multisets. The points are first scaled to
    integers, each configuration by the least common denominator of its
    coordinates, and the witness is scaled back. Two configurations of one
    point each, perhaps repeated, get M = I. In full dimension the map is
    found by _find_affine_witness. Below full dimension the points do not
    determine the linear part of a map between them, so it is found between
    the lattice charts, where both configurations are full-dimensional and
    the map is unique for each correspondence of affine bases, and lifted; the
    lift is invertible, and its determinant depends on the complements of the
    two direction spaces that it maps onto each other.
    """
    if points_a.shape != points_b.shape:
        return None
    scaled_a, q_a = _clear_denominators(points_a)
    scaled_b, q_b = _clear_denominators(points_b)
    pts_a = _invariants.to_integer_points(scaled_a)
    pts_b = _invariants.to_integer_points(scaled_b)
    rank = _exact.affine_rank(pts_a.tolist())
    if rank != _exact.affine_rank(pts_b.tolist()):
        return None
    if rank == 0:
        t = sp.Matrix(pts_b[0].tolist()) - sp.Matrix(pts_a[0].tolist())
        return _rescaled((sp.eye(pts_a.shape[1]), t, sp.Integer(1)), q_a, q_b)
    if rank == pts_a.shape[1]:
        return _rescaled(_find_affine_witness(pts_a, pts_b), q_a, q_b)
    frame_a, frame_b = chart_frame(pts_a), chart_frame(pts_b)
    found = _find_affine_witness(frame_a.coordinates, frame_b.coordinates)
    if found is None:
        return None
    m, s, _ = found
    M = lift_linear(frame_a, frame_b, m, integral=False)
    if M is None:  # pragma: no cover - lift_linear returns None only when integral
        return None
    image = frame_b.chart.to_ambient(list(m * sp.Matrix(frame_a.chart.coordinates[0]) + s))
    t = sp.Matrix(image) - M * sp.Matrix(pts_a[0].tolist())
    return _rescaled((M, t, M.det()), q_a, q_b)


def _clear_denominators(points: sp.Matrix) -> tuple[sp.Matrix, int]:
    """
    The points times q, and q, the least positive integer that makes them integers.

    A float counts as an integer when it equals one exactly, at its own
    precision. Raises ValidationError on any other coordinate that is not a
    rational number, a float that is not an integer included: it is rejected,
    not rounded.
    """
    values: list[Any] = []
    for i in range(points.rows):
        for x in points.row(i):
            # The difference, not ==, since SymPy does not call Float(2.0) equal to 2.
            if x.is_Float and x.is_finite and not x - int(x):
                x = sp.Integer(int(x))
            if not x.is_Rational:
                raise ValidationError(
                    f"point {i} has the coordinate {x}, which is not an exact rational number"
                )
            values.append(x)
    q = math.lcm(*(int(x.q) for x in values))
    return sp.Matrix(points.rows, points.cols, [x * q for x in values]), q


def _rescaled(
    witness: tuple[sp.Matrix, sp.Matrix, sp.Expr] | None, q_a: int, q_b: int
) -> tuple[sp.Matrix, sp.Matrix, sp.Expr] | None:
    """A witness from q_a A onto q_b B as one from A onto B: (q_a / q_b) M and t / q_b."""
    if witness is None or q_a == q_b == 1:
        return witness
    M, t, _ = witness
    M = M * sp.Rational(q_a, q_b)
    return M, t / q_b, M.det()


def _find_affine_witness(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> tuple[sp.Matrix, sp.Matrix, sp.Expr] | None:
    """
    (M, t, det M) of the first affine map x -> M x + t sending the rows of pts_a onto those of pts_b.

    Both are full-dimensional integer configurations, compared as multisets
    of rows; None when there is no such map. M and t may be rational. Such a
    map is an affine bijection of the two polytopes, so it sends the vertices
    of one onto those of the other and keeps the labels of affine_labels. The
    candidates are the maps that vertex_maps finds with those labels, the
    translation first, each checked on every row.
    """
    data_a, data_b = polytope_data(pts_a), polytope_data(pts_b)
    V_a = np.array(data_a.vertices, dtype=np.int64)
    V_b = np.array(data_b.vertices, dtype=np.int64)
    if V_a.shape != V_b.shape or data_b.dimension != pts_b.shape[1]:
        return None
    skeleton_a = _invariants.polytope_skeleton(data_a)
    skeleton_b = _invariants.polytope_skeleton(data_b)
    if skeleton_a.number_of_edges() != skeleton_b.number_of_edges():
        return None
    labels_a, _ = _invariants.affine_labels(V_a, skeleton_a)
    labels_b, _ = _invariants.affine_labels(V_b, skeleton_b)
    if sorted(labels_a) != sorted(labels_b):
        return None

    rows_a = [[int(x) for x in row] for row in pts_a.tolist()]
    rows_b = Counter(tuple(int(x) for x in row) for row in pts_b.tolist())
    neighbours_a = [set(skeleton_a.neighbors(i)) for i in range(len(V_a))]
    neighbours_b = [set(skeleton_b.neighbors(i)) for i in range(len(V_b))]
    for N, q, perm in vertex_maps(
        V_a, V_b, neighbours_a, neighbours_b, labels_a, labels_b, integral=False
    ):
        # q t = q V_b[perm[0]] - N V_a[0], and q (M x + t) = N x + q t, in Python integers.
        linear = [[int(x) for x in row] for row in N.tolist()]
        shift = [
            q * int(b) - sum(n * int(a) for n, a in zip(row, V_a[0], strict=True))
            for row, b in zip(linear, V_b[perm[0]], strict=True)
        ]
        images: Counter[tuple[int, ...]] = Counter()
        for x in rows_a:
            scaled = [
                sum(n * c for n, c in zip(row, x, strict=True)) + s
                for row, s in zip(linear, shift, strict=True)
            ]
            if any(y % q for y in scaled):
                break
            images[tuple(y // q for y in scaled)] += 1
        else:
            if images == rows_b:
                M = sp.Matrix(linear) / q
                return M, sp.Matrix(shift) / q, M.det()
    return None
