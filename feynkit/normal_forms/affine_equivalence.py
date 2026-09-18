"""
Equivalence tests for convex (lattice) polytopes.

This module exposes two complementary verbs:

- :func:`is_unimodular_equivalent` — implements the Liu–Cai algorithm
  (arXiv:2506.23846). Decides whether two integer point configurations span
  unimodularly-isomorphic lattice polytopes, and returns a witness
  ``U ∈ GL_n(ℤ)`` and an integer translation when one exists.
- :func:`is_affinely_equivalent` — broader equivalence over the rationals.
  Uses brute-force search over affine bases.

Both return a :class:`feynkit.PolytopeEquivalence` carrying the verdict, a
witness map (when known), and a vertex correspondence (when known).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations, permutations
from typing import Any

import numpy as np
import sympy as sp

from ..types import PolytopeEquivalence
from . import _invariants

# ──────────────────────────────────────────────────────────────────────────────
# Public API: Liu–Cai unimodular equivalence
# ──────────────────────────────────────────────────────────────────────────────


def is_unimodular_equivalent(
    points_a: object,
    points_b: object,
) -> PolytopeEquivalence:
    """
    Decide whether two integer point configurations span unimodularly-
    isomorphic convex lattice polytopes (Liu–Cai, arXiv:2506.23846).

    Two lattice polytopes ``P, P' ⊂ ℤ^n`` are *unimodularly isomorphic* if
    there exists ``U ∈ GL_n(ℤ)`` and ``Z ∈ ℤ^n`` such that ``P' = UP + Z``.

    Parameters
    ----------
    points_a, points_b
        Point configurations as ``d × n`` integer arrays (rows = lattice
        points). Accepts numpy arrays, sympy matrices, or any iterable of
        integer-coordinate tuples. Non-vertex points (interior or on a
        face) are filtered out automatically by computing the convex hull.

    Returns
    -------
    PolytopeEquivalence
        ``equivalent=True`` together with a witness map ``U`` and a vertex
        correspondence on success; ``equivalent=False`` otherwise.

    Notes
    -----
    The algorithm:

    1. Restrict each point set to its convex-hull vertices.
    2. Build the labelled vertex/edge graph $\\mathcal{GW}(P)$ with
       node label ``lab(v) = det(A_v)`` (Liu–Cai, Definition 5.2) and edge
       weight ``lab(u) + lab(v)``.
    3. Compute one MST of $\\mathcal{GW}(P)$ and all MSTs of
       $\\mathcal{GW}(P')$.
    4. For each label-preserving tree isomorphism ``φ`` and each
       ``χ ∈ Aut_lab(T)``, build the candidate vertex map ``φ ∘ χ`` and try
       to solve for ``U ∈ GL_n(ℤ)`` and the integer translation ``Z``.

    See Also
    --------
    is_affinely_equivalent : the broader rational equivalence relation.
    """
    pts_a = _invariants.to_integer_points(points_a)
    pts_b = _invariants.to_integer_points(points_b)

    if pts_a.shape[1] != pts_b.shape[1]:
        return PolytopeEquivalence(False, "unimodular")

    idx_a = _invariants.hull_vertex_indices(pts_a)
    idx_b = _invariants.hull_vertex_indices(pts_b)
    V_a = pts_a[idx_a]
    V_b = pts_b[idx_b]

    if V_a.shape != V_b.shape:
        return PolytopeEquivalence(False, "unimodular")

    n_dim = V_a.shape[1]
    n_vert = V_a.shape[0]

    # Trivial cases.
    if n_vert == 0:
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

    # Build labelled graphs.
    GW_a = _invariants.labelled_polytope_graph(V_a)
    GW_b = _invariants.labelled_polytope_graph(V_b)

    # Quick filter: vertex-label multisets must match.
    labels_a = sorted(GW_a.nodes[i]["label"] for i in GW_a.nodes())
    labels_b = sorted(GW_b.nodes[i]["label"] for i in GW_b.nodes())
    if labels_a != labels_b:
        return PolytopeEquivalence(False, "unimodular")

    # Edge counts must match too (a 1-skeleton invariant).
    if GW_a.number_of_edges() != GW_b.number_of_edges():
        return PolytopeEquivalence(False, "unimodular")

    # Select a basis in V_a.
    v_0 = V_a[0]
    deltas_a = (V_a - v_0).astype(np.int64)
    basis_indices = _select_basis_indices(deltas_a, n_dim)
    if basis_indices is None:
        return PolytopeEquivalence(False, "unimodular")

    W_a = sp.Matrix(deltas_a[basis_indices].T.tolist())
    if W_a.det() == 0:
        return PolytopeEquivalence(False, "unimodular")
    W_a_inv = W_a.inv()

    return _direct_basis_search(
        V_a,
        V_b,
        GW_a,
        GW_b,
        deltas_a,
        basis_indices,
        W_a,
        W_a_inv,
        idx_a,
        idx_b,
    )


def _select_basis_indices(deltas: np.ndarray, n_dim: int) -> list[int] | None:
    """
    Greedy selection of ``n_dim`` row indices of ``deltas`` whose rows are
    linearly independent. Returns ``None`` if no such basis exists.

    The first row (index 0) corresponds to the anchor vertex itself
    (delta = 0), so it is skipped.
    """
    chosen: list[int] = []
    chosen_rows: list[np.ndarray] = []
    for i in range(1, deltas.shape[0]):
        candidate = np.array(chosen_rows + [deltas[i].astype(float)])
        if np.linalg.matrix_rank(candidate) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(deltas[i].astype(float))
            if len(chosen) == n_dim:
                return chosen
    return None


def _label_preserving_orderings(
    combo: tuple[int, ...],
    labels_b: list[int],
    basis_label_seq: list[int],
) -> Iterator[tuple[int, ...]]:
    """
    Yield all column orderings of ``combo`` where the k-th entry has the same
    Liu–Cai label as ``basis_label_seq[k]``.

    Liu–Cai labels are unimodular invariants, so any valid witness map must
    preserve them.  Restricting to label-preserving orderings reduces the
    search space from ``n_dim!`` to ``∏_lab (count_lab)!`` where ``count_lab`` is
    the multiplicity of label ``lab`` in the basis — typically a small constant.
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
    W_a: sp.Matrix,  # noqa: ARG001 — kept for call-site symmetry with W_a_inv
    W_a_inv: sp.Matrix,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> PolytopeEquivalence:
    """
    Enumerate candidate (anchor, ordered-basis) pairs in V_b and solve for U.

    For each anchor vertex in V_b whose Liu–Cai label matches V_a[0], try all
    C(n_vert-1, n_dim) unordered n-subsets of the remaining vertices as the
    basis image, filtered by:

      1. Label multiset: sorted labels of the subset must equal sorted labels
         of the basis_indices vertices in V_a (Liu–Cai labels are unimodular
         invariants, so valid maps preserve them).
      2. |det| = 1: computed once per unordered subset (column permutations
         only flip the sign, so all orderings share the same |det|).
      3. Label-preserving orderings only: instead of all n_dim! column
         permutations, only those where the k-th column label matches the
         required label for that basis position — typically ∏_lab (count_lab)!
         orderings rather than n_dim!.

    Candidates passing all three filters are verified with numpy integer
    arithmetic; only survivors reach the exact SymPy step.
    """
    n_vert = V_a.shape[0]
    n_dim = V_a.shape[1]
    v_0 = V_a[0]

    # Numpy inverse for fast screening (float arithmetic).
    W_a_np = deltas_a[basis_indices].T.astype(float)
    W_a_inv_np = np.linalg.inv(W_a_np)
    abs_det_W_a = abs(np.linalg.det(W_a_np))

    labels_a_node = [GW_a.nodes[i]["label"] for i in range(n_vert)]
    labels_b_node = [GW_b.nodes[i]["label"] for i in range(n_vert)]
    label_a0 = labels_a_node[0]
    basis_label_multiset = sorted(labels_a_node[k] for k in basis_indices)
    basis_label_seq = [labels_a_node[k] for k in basis_indices]

    for anchor_idx in range(n_vert):
        if labels_b_node[anchor_idx] != label_a0:
            continue

        v_0_image = V_b[anchor_idx]
        deltas_b = (V_b - v_0_image).astype(np.int64)

        # Fast delta→index lookup (hull vertices are distinct, so no collisions).
        delta_to_b_idx = {tuple(int(x) for x in row): i for i, row in enumerate(deltas_b.tolist())}

        others = [j for j in range(n_vert) if j != anchor_idx]

        for combo in combinations(others, n_dim):
            # Filter 1: label multiset.
            if sorted(labels_b_node[j] for j in combo) != basis_label_multiset:
                continue

            # Filter 2: |det(W_b)| must equal |det(W_a)| for U = W_b·W_a⁻¹ to
            # have det ±1.  The original filter checked |det| ≈ 1 which is only
            # correct when W_a itself has det ±1; this is the general form.
            W_cand = deltas_b[list(combo)].T.astype(float)
            if abs(abs(np.linalg.det(W_cand)) - abs_det_W_a) > 0.5:
                continue

            # Filter 3: only label-preserving column orderings.
            for perm in _label_preserving_orderings(combo, labels_b_node, basis_label_seq):
                W_b_np = deltas_b[list(perm)].T.astype(float)
                U_np = W_b_np @ W_a_inv_np
                U_int = np.round(U_np).astype(np.int64)
                if not np.allclose(U_np, U_int.astype(float), atol=1e-6):
                    continue

                # Integer verification of all points.
                mapped = U_int @ deltas_a.T  # n_dim × n_vert
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
    """Check ``U·v + Z == V_b[vertex_map[i]]`` for every vertex ``v = V_a[i]``."""
    for i in range(V_a.shape[0]):
        v_a_col = sp.Matrix(V_a[i].tolist())
        v_b_target = sp.Matrix(V_b[vertex_map[i]].tolist())
        if U * v_a_col + Z != v_b_target:
            return False
    return True


def _lift_correspondence(
    hull_idx_a: np.ndarray,  # noqa: ARG001 — source indices are implicit in the correspondence order
    hull_idx_b: np.ndarray,
    hull_correspondence: list[int],
) -> list[int]:
    """
    Translate a vertex correspondence from hull-vertex indices back to the
    original input indices. Non-vertex (interior / on-face) input points
    have no image and are not included.
    """
    return [int(hull_idx_b[j]) for j in hull_correspondence]


# ──────────────────────────────────────────────────────────────────────────────
# Public API: affine equivalence (broader, rational)
# ──────────────────────────────────────────────────────────────────────────────


def is_affinely_equivalent(
    points_a: object,
    points_b: object,
) -> PolytopeEquivalence:
    """
    Decide whether two **Newton polytopes** are affinely equivalent over the
    rationals: i.e. there is an affine map ``v ↦ M·v + t`` with ``M ∈ GL_n(ℚ)``
    taking one *vertex set* onto the other.

    Both inputs are first projected to their convex-hull vertices before the
    search.  Interior lattice points and non-vertex boundary points are
    discarded — use :func:`is_point_config_equivalent` if you need the
    stricter all-columns GKZ check.

    Parameters
    ----------
    points_a, points_b
        Point configurations as ``n_pts × dim`` arrays (rows = points).
        Accepts numpy arrays, sympy matrices, or nested lists.

    Returns
    -------
    PolytopeEquivalence
        ``relation="affine_polytope"``.  On success: ``witness_map=M``,
        ``translation=t``, ``determinant=det(M)``.  On failure: all ``None``.
    """
    pts_a_np = _invariants.to_integer_points(points_a)
    pts_b_np = _invariants.to_integer_points(points_b)
    V_a = pts_a_np[_invariants.hull_vertex_indices(pts_a_np)]
    V_b = pts_b_np[_invariants.hull_vertex_indices(pts_b_np)]

    pts_a = _coerce_sympy_points(V_a.tolist())
    pts_b = _coerce_sympy_points(V_b.tolist())
    witness = _find_affine_witness(pts_a, pts_b)
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
    i.e. there is an affine map ``v ↦ M·v + t`` taking every column of A_1
    (as a multiset) to a column of A_2.

    Unlike :func:`is_affinely_equivalent`, this function does **not** filter
    to convex-hull vertices.  It is the correct check for GKZ system
    equivalence, where the full monomial support — not just the Newton polytope
    — determines the hypergeometric system.

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
    """
    pts_a = _coerce_sympy_points(points_a)
    pts_b = _coerce_sympy_points(points_b)
    witness = _find_affine_witness(pts_a, pts_b)
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


# ──────────────────────────────────────────────────────────────────────────────
# Internal: brute-force sympy backend (preserved from the original
# implementation; the algorithm is correct and was hand-tuned).
# ──────────────────────────────────────────────────────────────────────────────


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


def _find_affine_witness(
    points_a: sp.Matrix,
    points_b: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix, sp.Expr] | None:
    """
    Return ``(M, t, det(M))`` of the first affine map ``v ↦ M·v + t`` sending
    the multiset ``points_a`` onto ``points_b``, or ``None`` if none exists.

    ``M`` is the linear part (dim × dim), ``t`` is the translation column
    vector (dim × 1).  Both may be rational.
    """
    if points_a.shape != points_b.shape:
        return None

    n_points, dim = points_a.shape
    if n_points == 1:
        # Any translation works; recover t = b - a.
        M = sp.eye(dim)
        t = (points_b.row(0) - points_a.row(0)).T
        return M, t, sp.Integer(1)

    rank_a = _affine_rank(points_a)
    rank_b = _affine_rank(points_b)
    if rank_a != rank_b:
        return None

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
                    # affine_map is (dim+1)×dim: rows 0..dim-1 form M^T,
                    # row dim is the translation t^T.
                    M = affine_map[:dim, :].T
                    t = affine_map[dim, :].T
                    det = M.det()
                    return M, t, det
    return None
