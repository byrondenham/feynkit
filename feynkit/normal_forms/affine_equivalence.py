"""
Equivalence tests for convex (lattice) polytopes.

This module exposes two complementary verbs:

- :func:`is_unimodular_equivalent`, implements the Liu-Cai algorithm
  (arXiv:2506.23846). Decides whether two integer point configurations span
  unimodularly-isomorphic lattice polytopes, and returns a witness
  ``U in GL_n(Z)`` and an integer translation when one exists.
- :func:`is_affinely_equivalent`, broader equivalence over the rationals.
  Uses brute-force search over affine bases.

Both return a :class:`feynkit.PolytopeEquivalence` carrying the verdict, a
witness map (when known), and a vertex correspondence (when known).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from itertools import combinations, permutations
from itertools import product as _prod
from typing import Any

import numpy as np
import sympy as sp

from .. import _exact
from ..core.exceptions import ValidationError
from ..polytope import PolytopeData, polytope_data
from ..types import PolytopeEquivalence
from . import _invariants
from ._chart import ChartFrame, chart_frame, lift_linear
from .polytope_automorphisms import _select_basis_indices_by_label

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
    3. Fix an affine basis of the first polytope and, for each anchor and
       label-preserving choice of basis images in the second, solve for
       ``U`` and verify it on every vertex.

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

    # Select a basis in the vertices of the first polytope, from the rarest
    # label classes first, so that few combinations of vertices of the second
    # share its labels.
    v_0 = coords_a[0]
    deltas_a = (coords_a - v_0).astype(np.int64)
    labels = [GW_a.nodes[i]["label"] for i in range(n_vert)]
    basis_indices = _select_basis_indices_by_label(
        deltas_a, coords_a.shape[1], labels, Counter(labels)
    )
    if basis_indices is None:
        return PolytopeEquivalence(False, "unimodular")

    W_a = sp.Matrix(deltas_a[basis_indices].T.tolist())
    if W_a.det() == 0:
        return PolytopeEquivalence(False, "unimodular")
    W_a_inv = W_a.inv()

    result = _direct_basis_search(
        coords_a,
        coords_b,
        GW_a,
        GW_b,
        deltas_a,
        basis_indices,
        W_a,
        W_a_inv,
        idx_a,
        idx_b,
        accept=accept,
    )
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


def _label_preserving_orderings(
    combo: tuple[int, ...],
    labels_b: list[int],
    basis_label_seq: list[int],
) -> Iterator[tuple[int, ...]]:
    """
    Yield all column orderings of ``combo`` where the k-th entry has the same
    Liu-Cai label as ``basis_label_seq[k]``.

    Liu-Cai labels are unimodular invariants, so any valid witness map must
    preserve them.  Restricting to label-preserving orderings reduces the
    search space from ``n_dim!`` to ``prod_lab (count_lab)!`` where ``count_lab`` is
    the multiplicity of label ``lab`` in the basis, typically a small constant.
    """
    from collections import defaultdict
    from itertools import product as _prod

    # Group combo entries by their label.
    label_to_entries: dict[int, list[int]] = defaultdict(list)
    for j in combo:
        label_to_entries[labels_b[j]].append(j)

    # Group basis positions by required label.
    label_to_positions: dict[int, list[int]] = defaultdict(list)
    for k, lab in enumerate(basis_label_seq):
        label_to_positions[lab].append(k)

    # Check structural compatibility.
    if set(label_to_entries) != set(label_to_positions):
        return
    for lab in label_to_positions:
        if len(label_to_entries[lab]) != len(label_to_positions[lab]):
            return

    # For each label group, generate all permutations of that group's entries.
    # Then take the Cartesian product across groups to form full orderings.
    groups = [
        (label_to_positions[lab], list(permutations(label_to_entries[lab])))
        for lab in sorted(label_to_positions)
    ]

    result: list[int] = [-1] * len(combo)
    for group_perms in _prod(*[perms for _, perms in groups]):
        for (positions, _), assigned in zip(groups, group_perms, strict=True):
            for pos, entry in zip(positions, assigned, strict=True):
                result[pos] = entry
        yield tuple(result)


def _direct_basis_search(
    V_a: np.ndarray,
    V_b: np.ndarray,
    GW_a: Any,  # networkx.Graph with integer 'label' node attributes
    GW_b: Any,
    deltas_a: np.ndarray,
    basis_indices: list[int],
    W_a: sp.Matrix,
    W_a_inv: sp.Matrix,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    *,
    accept: Callable[[sp.Matrix], bool] | None = None,
) -> PolytopeEquivalence:
    """
    Enumerate candidate (anchor, ordered-basis) pairs in V_b and solve for U.

    For each anchor vertex in V_b whose Liu-Cai label matches V_a[0], try the
    unordered n-subsets of the remaining vertices as the basis image, with:

      1. Label multiset: the subsets are generated class by class, so that
         their labels are those of the basis_indices vertices in V_a (Liu-Cai
         labels are unimodular invariants, so valid maps preserve them). They
         are exactly the subsets whose sorted labels equal the basis's.
      2. |det W_b| = |det W_a|: computed exactly once per unordered subset
         (column permutations only flip the sign, so all orderings share the
         same |det|).
      3. Label-preserving orderings only: instead of all n_dim! column
         permutations, only those where the k-th column label matches the
         required label for that basis position, typically prod_lab (count_lab)!
         orderings rather than n_dim!.

    Each candidate U = W_b W_a^-1 that passes them is found in integer
    arithmetic, as W_b adj(W_a) / det(W_a), and kept when it is integral and
    maps every vertex to a vertex; only survivors reach the exact SymPy step.
    A verified U that ``accept``, when given, rejects is skipped and the
    search goes on.
    """
    n_vert = V_a.shape[0]
    v_0 = V_a[0]

    size = 2 * max(int(np.abs(V_a).max()), int(np.abs(V_b).max())) + 1
    adj_a, det_a = _invariants.integer_inverse(W_a, size)
    deltas_exact = deltas_a.astype(adj_a.dtype)

    labels_a_node = [GW_a.nodes[i]["label"] for i in range(n_vert)]
    labels_b_node = [GW_b.nodes[i]["label"] for i in range(n_vert)]
    label_a0 = labels_a_node[0]
    basis_label_seq = [labels_a_node[k] for k in basis_indices]
    basis_label_needs = Counter(basis_label_seq)
    basis_label_classes = sorted(basis_label_needs)

    for anchor_idx in range(n_vert):
        if labels_b_node[anchor_idx] != label_a0:
            continue

        v_0_image = V_b[anchor_idx]
        deltas_b = (V_b - v_0_image).astype(np.int64)

        # Fast delta->index lookup (hull vertices are distinct, so no collisions).
        delta_to_b_idx = {tuple(int(x) for x in row): i for i, row in enumerate(deltas_b.tolist())}

        # Filter 1: generate only the subsets with the labels of the basis.
        label_to_others: dict[int, list[int]] = defaultdict(list)
        for other in range(n_vert):
            if other != anchor_idx:
                label_to_others[labels_b_node[other]].append(other)
        sub_combo_iters = [
            combinations(label_to_others[lbl], basis_label_needs[lbl])
            for lbl in basis_label_classes
        ]
        for sub_combos in _prod(*sub_combo_iters):
            combo = tuple(v for sub in sub_combos for v in sub)

            # Filter 2: |det(W_b)| must equal |det(W_a)| for U = W_b*W_a^-1 to
            # have det +/-1, in exact integer arithmetic.
            if abs(_exact.determinant(deltas_b[list(combo)].tolist())) != abs(det_a):
                continue

            # Filter 3: only label-preserving column orderings.
            for perm in _label_preserving_orderings(combo, labels_b_node, basis_label_seq):
                U_int = _invariants.integral_candidate(deltas_b[list(perm)].T, adj_a, det_a)
                if U_int is None:
                    continue

                # Integer verification of all points.
                mapped = U_int @ deltas_exact.T  # n_dim x n_vert
                vertex_map: dict[int, int] = {}
                valid = True
                for i in range(n_vert):
                    key = tuple(int(x) for x in mapped[:, i])
                    j = delta_to_b_idx.get(key)
                    if j is None:
                        valid = False
                        break
                    vertex_map[i] = j
                if not valid:
                    continue

                # Exact SymPy verification.
                W_b_sp = sp.Matrix(deltas_b[list(perm)].T.tolist())
                U = W_b_sp * W_a_inv
                if not all(e.is_Integer for e in U):
                    continue
                if abs(U.det()) != 1:
                    continue

                v_0_col = sp.Matrix(v_0.tolist())
                v_0_img_col = sp.Matrix(v_0_image.tolist())
                Z = v_0_img_col - U * v_0_col
                if not all(e.is_Integer for e in Z):
                    continue

                if not _verify_unimodular_witness(U, Z, V_a, V_b, vertex_map):
                    continue
                if accept is not None and not accept(U):
                    continue

                corr = [vertex_map[i] for i in range(n_vert)]
                return PolytopeEquivalence(
                    equivalent=True,
                    relation="unimodular",
                    witness_map=sp.ImmutableMatrix(U),
                    vertex_correspondence=_lift_correspondence(idx_a, idx_b, corr),
                )

    return PolytopeEquivalence(False, "unimodular")


def _verify_unimodular_witness(
    U: sp.Matrix,
    Z: sp.Matrix,
    V_a: np.ndarray,
    V_b: np.ndarray,
    vertex_map: dict[int, int],
) -> bool:
    """Check ``U*v + Z == V_b[vertex_map[i]]`` for every vertex ``v = V_a[i]``."""
    for i in range(V_a.shape[0]):
        v_a_col = sp.Matrix(V_a[i].tolist())
        v_b_target = sp.Matrix(V_b[vertex_map[i]].tolist())
        if U * v_a_col + Z != v_b_target:
            return False
    return True


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

     Below full dimension the map is found between the lattice charts of
     the two configurations, where it is unique for a correspondence of
     affine bases. Rational points are first scaled to integers, each
     configuration by the least common denominator q_a or q_b of its
     coordinates, and a map ``M', t'`` between the scaled points gives
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
         Below full dimension, if a coordinate is not a rational number given
         exactly; floats that are not integers are rejected, not rounded.
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
# Internal: brute-force sympy backend (preserved from the original
# implementation; the algorithm is correct and was hand-tuned).
# ------------------------------------------------------------------------------


def _coerce_sympy_points(data: object) -> sp.Matrix:
    matrix = sp.Matrix(data)
    if matrix.rows == 0 or matrix.cols == 0:
        raise ValueError("Point configuration must be non-empty")
    return matrix


def _affine_rank(points: sp.Matrix) -> int:
    if points.rows <= 1:
        return 0
    base = points[0, :]
    deltas = [points[i, :] - base for i in range(1, points.rows)]
    return int(sp.Matrix(deltas).rank())


def _affine_basis_indices(points: sp.Matrix, rank: int) -> list[tuple[int, ...]]:
    n_points, _dim = points.shape
    if n_points < rank + 1:
        return []

    bases: list[tuple[int, ...]] = []
    for idxs in combinations(range(n_points), rank + 1):
        base = points[idxs[0], :]
        deltas = [points[i, :] - base for i in idxs[1:]]
        if sp.Matrix(deltas).rank() == rank:
            bases.append(idxs)
    return bases


def _solve_affine_map(source_basis: sp.Matrix, target_basis: sp.Matrix) -> sp.Matrix | None:
    dim = source_basis.shape[1]
    variables = sp.symbols(f"w0:{dim * (dim + 1)}")
    affine_map = sp.Matrix(dim + 1, dim, variables)

    equations: list[sp.Expr] = []
    for i in range(source_basis.rows):
        x = source_basis.row(i)
        y = target_basis.row(i)
        lhs = x.row_join(sp.Matrix([[1]])) * affine_map
        equations.extend((lhs[0, j] - y[0, j]) for j in range(dim))

    solution = sp.linsolve(equations, variables)
    if solution == sp.EmptySet:
        return None

    solved = next(iter(solution))
    concrete = [value.subs(dict.fromkeys(value.free_symbols, 0)) for value in solved]
    return sp.Matrix(dim + 1, dim, concrete)


def _apply_affine_map(points: sp.Matrix, affine_map: sp.Matrix) -> sp.Matrix:
    points_aug = points.row_join(sp.ones(points.rows, 1))
    mapped = points_aug * affine_map
    return mapped.applyfunc(sp.simplify)


def _point_key(point: list[sp.Expr]) -> tuple[tuple, ...]:
    return tuple(sp.default_sort_key(sp.simplify(value)) for value in point)


def _same_point_multiset(points_a: sp.Matrix, points_b: sp.Matrix) -> bool:
    if points_a.shape != points_b.shape:
        return False
    keys_a = sorted(_point_key(row) for row in points_a.tolist())
    keys_b = sorted(_point_key(row) for row in points_b.tolist())
    return keys_a == keys_b


def _affine_witness(
    points_a: sp.Matrix,
    points_b: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix, sp.Expr] | None:
    """
    _find_affine_witness, run in the lattice charts below full dimension.

    Below full dimension the points do not determine the linear part of an
    affine map between them, and _find_affine_witness sets its free
    parameters to 0, which can leave it singular. In the charts, where both
    configurations are full-dimensional, the map is unique for each
    correspondence of affine bases, and its lift is invertible. Its
    determinant then depends on the complements of the two direction spaces
    that the lift maps onto each other. Below full dimension the points are
    first scaled to integers, each configuration by the least common
    denominator of its coordinates, and the witness is scaled back.
    """
    if points_a.shape != points_b.shape:
        return None
    rank = _affine_rank(points_a)
    if rank in (0, points_a.cols) or rank != _affine_rank(points_b):
        return _find_affine_witness(points_a, points_b)
    points_a, q_a = _clear_denominators(points_a)
    points_b, q_b = _clear_denominators(points_b)
    frame_a = chart_frame(points_a.tolist())
    frame_b = chart_frame(points_b.tolist())
    found = _find_affine_witness(
        sp.Matrix(frame_a.chart.coordinates), sp.Matrix(frame_b.chart.coordinates)
    )
    if found is None:
        return None
    m, s, _ = found
    M = lift_linear(frame_a, frame_b, m, integral=False)
    if M is None:  # pragma: no cover - lift_linear returns None only when integral
        return None
    image = frame_b.chart.to_ambient(list(m * sp.Matrix(frame_a.chart.coordinates[0]) + s))
    t = sp.Matrix(image) - M * points_a.row(0).T
    return _rescaled((M, t, M.det()), q_a, q_b)


def _clear_denominators(points: sp.Matrix) -> tuple[sp.Matrix, int]:
    """
    The points times q, and q, the least positive integer that makes them integers.

    Integral floats count as integers. Raises ValidationError on any other
    coordinate that is not a rational number, a float that is not an integer
    included: it is rejected, not rounded.
    """
    values: list[Any] = []
    for i in range(points.rows):
        for x in points.row(i):
            if x.is_Float and math.isfinite(float(x)) and float(x).is_integer():
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
    points_a: sp.Matrix,
    points_b: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix, sp.Expr] | None:
    """
    Return ``(M, t, det(M))`` of the first affine map ``v -> M*v + t`` sending
    the multiset ``points_a`` onto ``points_b``, or ``None`` if none exists.

    ``M`` is the linear part (dim x dim), ``t`` is the translation column
    vector (dim x 1).  Both may be rational.
    """
    if points_a.shape != points_b.shape:
        return None

    dim = points_a.cols
    rank_a = _affine_rank(points_a)
    rank_b = _affine_rank(points_b)
    if rank_a != rank_b:
        return None
    if rank_a == 0:
        # Each configuration is one point, perhaps repeated: the translation
        # t = b - a maps one onto the other.
        M = sp.eye(dim)
        t = (points_b.row(0) - points_a.row(0)).T
        return M, t, sp.Integer(1)

    source_bases = _affine_basis_indices(points_a, rank_a)
    if not source_bases:
        return None

    target_bases = _affine_basis_indices(points_b, rank_b)
    if not target_bases:
        return None

    for source_idxs in source_bases:
        source_basis = points_a[list(source_idxs), :]
        for target_idxs in target_bases:
            target_points = points_b[list(target_idxs), :]
            for perm in permutations(range(rank_a + 1)):
                permuted_target = target_points[list(perm), :]
                affine_map = _solve_affine_map(source_basis, permuted_target)
                if affine_map is None:
                    continue
                mapped = _apply_affine_map(points_a, affine_map)
                if _same_point_multiset(mapped, points_b):
                    # affine_map is (dim+1) x dim: rows 0..dim-1 form M^T,
                    # row dim is the translation t^T.
                    M = affine_map[:dim, :].T
                    t = affine_map[dim, :].T
                    det = M.det()
                    return M, t, det
    return None
