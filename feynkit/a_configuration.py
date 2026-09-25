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
    Search for an integer affine map x -> Mx + t (no det +/-1 constraint)
    taking every source point onto a target point.

intrinsic_lattice_model(points) -> IntrinsicModel
    Express a point configuration in the Hermite normal form basis of the
    lattice its differences span.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from itertools import product as _prod
from typing import Any, cast

import numpy as np
import sympy as sp

from . import _exact
from .core.exceptions import ValidationError
from .normal_forms._invariants import (
    hull_vertex_indices,
    labelled_polytope_graph,
)
from .normal_forms.affine_equivalence import (
    is_affinely_equivalent,
    is_point_config_equivalent,
    is_unimodular_equivalent,
)
from .normal_forms.polytope_automorphisms import (
    _label_preserving_orderings,
    _select_basis_indices_by_label,
)
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
        Non-trivial Smith normal form diagonal entries of the difference matrix.

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
        """Compute the unimodular automorphism group of the Newton polytope."""
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
        T^k A = A, and as A has full rank, M^k = I.  Maps with |det M| > 1
        relate two different configurations; see :func:`finite_index_map`.
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
    Search for an integer affine map x -> Mx + t taking every source point
    onto some target point, with no constraint on |det M|.

    If |det M| = 1, the map is unimodular (a lattice isomorphism).  If
    |det M| > 1, the map is a finite-index embedding: the image M Z^n of the
    source lattice is a sublattice of index |det M| in the target lattice Z^n.

    Parameters
    ----------
    source, target
        AConfiguration objects, or raw point arrays (rows = points).

    Returns
    -------
    FiniteIndexResult
        ``found=True`` with witness data on success, ``found=False`` otherwise.

    Algorithm
    ---------
    For each candidate anchor vertex v' in the target (as image of source[0]):

      1. Compute deltas_src = source_pts - source_pts[0].
      2. Select aff_dim linearly independent rows as a basis.
      3. For each aff_dim-subset of target deltas as candidate basis image:
         - Solve M = W_tgt @ W_src^-1  (rational).
         - Check M is integer.
         - Verify every source point maps to a target point.
         - Return on first success.
    """
    src_pts = _to_pts(source)
    tgt_pts = _to_pts(target)

    n_src = src_pts.shape[0]
    n_tgt = tgt_pts.shape[0]
    n_dim = src_pts.shape[1]

    if tgt_pts.shape[1] != n_dim:
        return FiniteIndexResult(found=False)

    # Build a fast lookup for target points.
    tgt_set = {tuple(int(x) for x in row): i for i, row in enumerate(tgt_pts.tolist())}

    deltas_src = (src_pts - src_pts[0]).astype(np.int64)

    # Find aff_dim linearly independent directions from source.
    aff_dim = int(np.linalg.matrix_rank(deltas_src[1:].astype(float)))
    if aff_dim == 0:
        # Single point: trivially maps to any single target point.
        if n_tgt >= 1:
            t_vec = sp.ImmutableMatrix((tgt_pts[0] - src_pts[0]).reshape(-1, 1).tolist())
            return FiniteIndexResult(
                found=True,
                witness_matrix=sp.ImmutableMatrix(sp.eye(n_dim)),
                translation=t_vec,
                determinant=1,
                column_permutation=[0],
                is_unimodular=True,
            )
        return FiniteIndexResult(found=False)

    basis_idx = _basis_indices(deltas_src, aff_dim)
    if basis_idx is None:
        return FiniteIndexResult(found=False)

    W_src = sp.Matrix(deltas_src[basis_idx].T.tolist())
    W_src_inv = W_src.inv()  # rational

    for anchor_tgt_idx in range(n_tgt):
        v0_tgt = tgt_pts[anchor_tgt_idx]
        deltas_tgt = (tgt_pts - v0_tgt).astype(np.int64)
        tgt_delta_set = {tuple(int(x) for x in row): i for i, row in enumerate(deltas_tgt.tolist())}

        other_tgt = [j for j in range(n_tgt) if j != anchor_tgt_idx]
        for combo in combinations(other_tgt, aff_dim):
            W_tgt_np = deltas_tgt[list(combo)].T.astype(float)
            # Quick float check before exact arithmetic.
            W_src_np = deltas_src[basis_idx].T.astype(float)
            M_np = W_tgt_np @ np.linalg.inv(W_src_np)
            M_int = np.round(M_np).astype(np.int64)
            if not np.allclose(M_np, M_int.astype(float), atol=1e-6):
                continue

            # Verify all source deltas map to target deltas.
            mapped_deltas = M_int @ deltas_src.T  # n_dim x n_src
            col_perm: list[int] = []
            valid = True
            for i in range(n_src):
                key = tuple(int(x) for x in mapped_deltas[:, i])
                j = tgt_delta_set.get(key)
                if j is None:
                    valid = False
                    break
                col_perm.append(anchor_tgt_idx if i == 0 else j)
            if not valid:
                continue

            # Exact SymPy computation for the result.
            W_tgt_sp = sp.Matrix(deltas_tgt[list(combo)].T.tolist())
            M = W_tgt_sp * W_src_inv
            if not all(e.is_Integer for e in M):
                continue

            det = int(abs(M.det()))
            v0_src_col = sp.Matrix(src_pts[0].tolist())
            v0_tgt_col = sp.Matrix(v0_tgt.tolist())
            t = v0_tgt_col - M * v0_src_col
            if not all(e.is_Integer for e in t):
                continue

            # Final full verification.
            all_ok = True
            for i, pt in enumerate(src_pts):
                img = M * sp.Matrix(pt.tolist()) + t
                key = tuple(int(x) for x in img)
                if key not in tgt_set:
                    all_ok = False
                    break
                col_perm[i] = tgt_set[key]
            if not all_ok:
                continue

            return FiniteIndexResult(
                found=True,
                witness_matrix=sp.ImmutableMatrix(M),
                translation=sp.ImmutableMatrix(t),
                determinant=det,
                column_permutation=col_perm,
                is_unimodular=(det == 1),
            )

    return FiniteIndexResult(found=False)


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
        All valid self-maps, including the identity.  Empty only if the
        configuration has no points or is lower-dimensional than its ambient
        space (degenerate case).

    Algorithm
    ---------
    All self-maps of a non-degenerate configuration are unimodular (by the
    affine volume argument).  Liu-Cai vertex labels, invariants of unimodular
    maps, are used to (a) filter anchor candidates and (b) generate only
    label-valid basis combinations.  Fix a label-diverse canonical basis
    {p_0, p_{b_1}, ..., p_{b_n}} (rarest Liu-Cai class first).  For each
    anchor image p_a whose label matches p_0, iterate only over unordered
    target basis combos drawn from same-label buckets, then try all
    label-consistent orderings via :func:`_label_preserving_orderings`.

    Complexity: O(|Aut| * d!) in the best case (K_4: ~144 checks vs 13 B
    naive).  Falls back to the full P(N-1, n) count when all labels coincide.
    """
    pts = _to_pts(cfg)
    N = pts.shape[0]
    n_dim = pts.shape[1]

    if N == 0:
        return []

    deltas = (pts - pts[0]).astype(np.int64)
    aff_dim = int(np.linalg.matrix_rank(deltas[1:].astype(float))) if N > 1 else 0

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
        # Lower-dimensional configuration: M is under-determined in the ambient
        # space.  This case is not supported (same limitation as finite_index_map).
        return []

    # All self-maps of a non-degenerate configuration are unimodular:
    # |det M| = Vol(M*conv(A)) / Vol(conv(A)) = Vol(conv(A)) / Vol(conv(A)) = 1.
    # Liu-Cai labels (det-of-moment-matrix at each hull vertex) are unimodular
    # invariants, so they can filter both anchors and basis combinations.
    hull_idx = hull_vertex_indices(pts)
    pts_hull = pts[hull_idx]
    GW = labelled_polytope_graph(pts_hull)
    hull_label = {int(hull_idx[i]): GW.nodes[i]["label"] for i in range(len(hull_idx))}
    # Interior points (not hull vertices) get label 0; they form a separate orbit.
    labels: list[int] = [hull_label.get(i, 0) for i in range(N)]
    label_a0 = labels[0]
    label_count = Counter(labels)

    # Label-diverse source basis: rarest-class rows first, minimising combo count.
    basis_idx = _select_basis_indices_by_label(deltas, aff_dim, labels, label_count)
    if basis_idx is None:
        basis_idx = _basis_indices(deltas, aff_dim)
    if basis_idx is None:
        return []

    W_src_np = deltas[basis_idx].T.astype(float)
    W_src_inv_np = np.linalg.inv(W_src_np)
    W_src_sp = sp.Matrix(deltas[basis_idx].T.tolist())
    W_src_inv_sp = W_src_sp.inv()

    basis_label_multiset = sorted(labels[k] for k in basis_idx)
    basis_label_seq = [labels[k] for k in basis_idx]
    basis_label_needs = Counter(basis_label_multiset)
    basis_label_classes = sorted(basis_label_needs)

    results: list[SymmetryPair] = []
    seen: set[tuple] = set()  # deduplicate by (M_int_flat, t_int_flat)

    for anchor_idx in range(N):
        # Liu-Cai labels are unimodular invariants: skip anchors whose label
        # differs from pts[0] (they cannot be images of pts[0] under any valid map).
        if labels[anchor_idx] != label_a0:
            continue

        v0 = pts[anchor_idx]
        deltas_from = (pts - v0).astype(np.int64)
        delta_to_idx: dict[tuple, int] = {
            tuple(int(x) for x in row): i for i, row in enumerate(deltas_from.tolist())
        }

        others = [j for j in range(N) if j != anchor_idx]

        # Group non-anchor points by label for direct combo generation.
        label_to_others: dict[int, list[int]] = defaultdict(list)
        for j in others:
            label_to_others[labels[j]].append(j)

        # Skip anchor if any needed label class is underrepresented.
        if any(len(label_to_others[lbl]) < basis_label_needs[lbl] for lbl in basis_label_classes):
            continue

        # Generate label-valid unordered combos, then try all orderings that
        # match the basis label sequence.  This replaces permutations(others, d)
        # which is P(N-1, d), catastrophically large for K_4 (N=31, d=6 -> 427M).
        sub_combo_iters = [
            combinations(label_to_others[lbl], basis_label_needs[lbl])
            for lbl in basis_label_classes
        ]
        for sub_combos in _prod(*sub_combo_iters):
            combo = tuple(v for sub in sub_combos for v in sub)

            for perm in _label_preserving_orderings(combo, labels, basis_label_seq):
                W_tgt_np = deltas_from[list(perm)].T.astype(float)
                M_np = W_tgt_np @ W_src_inv_np
                M_int = np.round(M_np).astype(np.int64)
                if not np.allclose(M_np, M_int.astype(float), atol=1e-6):
                    continue

                if abs(np.linalg.det(M_int.astype(float))) < 0.5:
                    continue

                # Verify all N source deltas map to target deltas (bijection).
                mapped = M_int @ deltas.T  # n_dim x N
                col_perm: list[int] = [-1] * N
                seen_tgts: set[int] = set()
                valid = True
                for i in range(N):
                    key = tuple(int(x) for x in mapped[:, i])
                    j_opt = delta_to_idx.get(key)
                    if j_opt is None or j_opt in seen_tgts:
                        valid = False
                        break
                    col_perm[i] = j_opt
                    seen_tgts.add(j_opt)
                if not valid:
                    continue

                # Exact integer t (no floating point: all operands are integers).
                t_int = v0.astype(np.int64) - M_int @ pts[0].astype(np.int64)
                dedup_key = tuple(M_int.flatten()) + tuple(int(x) for x in t_int)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # Exact SymPy reconstruction and final check.
                W_tgt_sp = sp.Matrix(deltas_from[list(perm)].T.tolist())
                M = W_tgt_sp * W_src_inv_sp
                if not all(e.is_Integer for e in M):
                    continue

                v0_src_col = sp.Matrix(pts[0].tolist())
                v0_tgt_col = sp.Matrix(v0.tolist())
                t = v0_tgt_col - M * v0_src_col
                if not all(e.is_Integer for e in t):
                    continue

                det = int(abs(M.det()))
                if det == 0:
                    continue

                results.append(
                    SymmetryPair(
                        linear_map=sp.ImmutableMatrix(M),
                        translation=sp.ImmutableMatrix(t),
                        column_permutation=tuple(col_perm),
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
    not the holonomic rank of the GKZ system: below full dimension, d < n,
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
    """Coerce to an (n_pts x n_dim) integer numpy array."""
    if isinstance(obj, AConfiguration):
        return obj.affine_points
    if isinstance(obj, np.ndarray):
        return obj.astype(np.int64)
    return np.array(list(cast(Iterable[Any], obj)), dtype=np.int64)


def _basis_indices(deltas: np.ndarray, aff_dim: int) -> list[int] | None:
    """
    Greedy selection of ``aff_dim`` row indices (skip row 0 = zero vector)
    whose rows are linearly independent.
    """
    chosen: list[int] = []
    chosen_rows: list[np.ndarray] = []
    for i in range(1, deltas.shape[0]):
        candidate = np.array(chosen_rows + [deltas[i].astype(float)])
        if np.linalg.matrix_rank(candidate) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(deltas[i].astype(float))
            if len(chosen) == aff_dim:
                return chosen
    return None
