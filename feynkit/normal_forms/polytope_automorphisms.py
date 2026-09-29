"""
Unimodular automorphism group of a convex lattice polytope.

The automorphism group Aut(P) consists of all unimodular affine maps
(U, t) with U in GL_n(Z), |det U| = 1, t in Z ^n, that send P to itself.
The identity (I, 0) is always a member; for a generic polytope it is the
only member.  Symmetric integrals (triangles, bananas, boxes) have larger
groups that are directly visible in their GKZ structure.

The algorithm extends the Liu-Cai basis-search used for equivalence testing
(_direct_basis_search in affine_equivalence.py): instead of returning on the
first valid witness, every valid (U, t) is collected and deduplicated via the
induced vertex permutation.

The vertices, and the edges on which the Liu-Cai labels are computed, come from
the certified face lattice of feynkit.polytope.polytope_data.

Basis selection is done with a label-diversity strategy: vertices from smaller
Liu-Cai label classes are chosen as basis vectors first.  For highly-symmetric
polytopes (e.g. K_4 Newton polytope: 31 vertices with label classes of sizes
12, 12, 4 and 3) this reduces the number of candidate basis combinations per
anchor from C(30,6) ~ 594 000 to a few hundred.

Public API
----------
compute_polytope_automorphisms(points) -> PolytopeAutomorphisms
    Full unimodular automorphism group of conv(points).

compute_graph_automorphisms(graph) -> list[list[int]]
    Vertex permutations of the Feynman graph (topology + masses) as a
    subgroup of the polytope automorphisms.

coefficient_preserving_indices(fi, auts) -> list[int]
    Indices into auts.maps of automorphisms that also preserve the
    coefficient multiset of the G polynomial (relevant for functional eqs).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import combinations, permutations
from itertools import product as _prod
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import sympy as sp

from ..polytope import polytope_data
from ..types import PolytopeAutomorphisms
from . import _invariants

if TYPE_CHECKING:
    from ..core.graph import Graph
    from ..integral import FeynmanIntegral


# ------------------------------------------------------------------------------
# Public: polytope automorphism group
# ------------------------------------------------------------------------------


def compute_polytope_automorphisms(points: object) -> PolytopeAutomorphisms:
    """
    Compute the full unimodular automorphism group of conv(points).

    Parameters
    ----------
    points
        Integer point configuration (n x d array, rows = points). Non-vertex
        points are filtered out before the computation.

    Returns
    -------
    PolytopeAutomorphisms
        Contains all (U, t) pairs, the group order, induced vertex
        permutations, and vertex orbits. The vertices are indexed in the
        order of ``polytope_data(points).vertices``.

    Notes
    -----
    The vertices and the edges that the Liu-Cai labels are built on come from
    the certified face lattice of :func:`feynkit.polytope.polytope_data`. The
    algorithm is the Liu-Cai basis-search: for each candidate image of a
    fixed anchor vertex, enumerate all possible images of a fixed basis and
    verify that the implied affine map sends the vertices onto themselves.
    The basis is chosen to maximise Liu-Cai label diversity (rarest-label
    vertices first), which for polytopes with a few large label orbits
    dramatically reduces the number of candidate basis combinations.
    """
    pts = _invariants.to_integer_points(points)
    n_dim = pts.shape[1]
    if pts.shape[0] == 0:
        return PolytopeAutomorphisms(
            maps=[(sp.ImmutableMatrix(sp.eye(n_dim)), sp.ImmutableMatrix(sp.zeros(n_dim, 1)))],
            order=1,
            vertex_permutations=[[]],
            vertex_orbits=[],
        )

    data = polytope_data(pts)
    V_arr = np.array(data.vertices, dtype=np.int64).reshape(len(data.vertices), n_dim)
    if V_arr.shape[0] == 1:
        return _identity_only(V_arr, n_dim)

    GW = _invariants.label_skeleton(V_arr, _invariants.polytope_skeleton(data))
    found_maps, found_vperms = _basis_search(V_arr, GW)
    if not found_maps:
        return _identity_only(V_arr, n_dim)

    return PolytopeAutomorphisms(
        maps=found_maps,
        order=len(found_maps),
        vertex_permutations=found_vperms,
        vertex_orbits=_compute_orbits(V_arr.shape[0], found_vperms),
    )


def _basis_search(
    V_arr: np.ndarray, GW: nx.Graph
) -> tuple[list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]], list[list[int]]]:
    """
    Every unimodular affine map permuting the rows of ``V_arr``, with its permutation.

    ``V_arr`` holds the vertices of a full-dimensional lattice polytope and
    ``GW`` its labelled 1-skeleton, whose node i is row i. Returns no maps
    when the vertices do not span their ambient space.
    """
    n_vert, n_dim = V_arr.shape
    labels = [GW.nodes[i]["label"] for i in range(n_vert)]
    label_a0 = labels[0]
    label_count = Counter(labels)

    # Select a label-diverse basis from V_arr[0] as anchor: vertices from
    # smaller label classes are chosen first so that the basis label multiset
    # is as diverse as possible, minimising C(class_size, k) per class.
    v_0 = V_arr[0]
    deltas_a = (V_arr - v_0).astype(np.int64)
    basis_indices = _select_basis_indices_by_label(deltas_a, n_dim, labels, label_count)
    if basis_indices is None:
        return [], []

    W_a = sp.Matrix(deltas_a[basis_indices].T.tolist())
    if W_a.det() == 0:
        return [], []

    basis_label_multiset = sorted(labels[k] for k in basis_indices)
    basis_label_seq = [labels[k] for k in basis_indices]

    W_a_np = deltas_a[basis_indices].T.astype(float)
    W_a_inv_np = np.linalg.inv(W_a_np)
    abs_det_W_a = abs(np.linalg.det(W_a_np))

    found_maps: list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]] = []
    found_vperms: list[list[int]] = []
    seen_vperms: set[tuple[int, ...]] = set()

    # Pre-group non-anchor vertices by label for direct (non-filtering) combo gen.
    # We sort the label classes so the ordering in _label_preserving_orderings is
    # deterministic.
    basis_label_needs = Counter(basis_label_multiset)
    basis_label_classes = sorted(basis_label_needs)

    for anchor_idx in range(n_vert):
        if labels[anchor_idx] != label_a0:
            continue

        v_0_image = V_arr[anchor_idx]
        deltas_b = (V_arr - v_0_image).astype(np.int64)
        delta_to_b_idx = {tuple(int(x) for x in row): i for i, row in enumerate(deltas_b.tolist())}

        others = [j for j in range(n_vert) if j != anchor_idx]

        # Group others by label, build once per anchor.
        label_to_others: dict[int, list[int]] = defaultdict(list)
        for j in others:
            label_to_others[labels[j]].append(j)

        # Generate only label-valid combos directly (avoids C(n-1, d) filtering).
        sub_combo_iters = [
            combinations(label_to_others[lbl], basis_label_needs[lbl])
            for lbl in basis_label_classes
        ]
        for sub_combos in _prod(*sub_combo_iters):
            combo = tuple(v for sub in sub_combos for v in sub)

            W_cand = deltas_b[list(combo)].T.astype(float)
            if abs(abs(np.linalg.det(W_cand)) - abs_det_W_a) > 0.5:
                continue

            for perm in _label_preserving_orderings(combo, labels, basis_label_seq):
                W_b_np = deltas_b[list(perm)].T.astype(float)
                U_np = W_b_np @ W_a_inv_np
                U_int = np.round(U_np).astype(np.int64)
                if not np.allclose(U_np, U_int.astype(float), atol=1e-6):
                    continue

                # Check det +/-1 using the already-computed float det of W_cand.
                if abs(abs(np.linalg.det(W_b_np)) / abs_det_W_a - 1.0) > 0.05:
                    continue

                mapped = U_int @ deltas_a.T
                vertex_map: dict[int, int] = {}
                valid = True
                for i in range(n_vert):
                    key = tuple(int(x) for x in mapped[:, i])
                    j_opt = delta_to_b_idx.get(key)
                    if j_opt is None:
                        valid = False
                        break
                    vertex_map[i] = j_opt
                if not valid:
                    continue

                # The numpy check above is exact for integer polytope vertices; the
                # sympy checks below are only needed to construct the output objects.
                vperm = tuple(vertex_map[i] for i in range(n_vert))
                if vperm in seen_vperms:
                    continue

                # Compute exact integer translation: Z = v_0_image - U * v_0.
                Z_int = v_0_image.astype(np.int64) - U_int @ v_0.astype(np.int64)

                seen_vperms.add(vperm)
                found_maps.append(
                    (
                        sp.ImmutableMatrix(U_int.tolist()),
                        sp.ImmutableMatrix(Z_int.reshape(-1, 1).tolist()),
                    )
                )
                found_vperms.append(list(vperm))
    return found_maps, found_vperms


# ------------------------------------------------------------------------------
# Public: graph automorphisms
# ------------------------------------------------------------------------------


def compute_graph_automorphisms(graph: Graph) -> list[list[int]]:
    """
    All vertex permutations of the Feynman graph that preserve topology and
    mass colouring.

    Each permutation is a list ``sigma`` of length ``V`` (internal vertices)
    where ``sigma[i-1]`` is the new label for internal vertex ``i``. Only
    permutations of internal vertices are considered (external legs are not
    permuted). The colouring compared is massless or massive, and a mass of
    None counts as massive.

    Returns a list of permutations (as 1-indexed vertex lists), always
    including the identity, in lexicographic order.

    The automorphisms of the graph with its legs map one least labelling of
    the canonical Nickel search onto each of the others. Each such map is
    checked in full, the legs at every vertex and the colours between every
    pair of vertices, and kept if it preserves them.
    """
    from ..core.graph import _canonical_labellings

    V = graph.internal_vertices
    multiplicity = [[0] * V for _ in range(V)]
    colours: dict[tuple[int, int], list[str]] = defaultdict(list)
    for e in graph.get_internal_edges():
        u, w = sorted((e.v1 - 1, e.v2 - 1))
        multiplicity[u][w] += 1
        if u != w:
            multiplicity[w][u] += 1
        colours[u, w].append("z" if (e.mass is not None and e.mass == sp.Integer(0)) else "n")
    legs = [0] * V
    for e in graph.get_external_edges():
        legs[e.v1 - 1] += 1

    # A map is a bijection on pairs of vertices, so one that sends every pair
    # with propagators to a pair with the same colours sends the pairs without
    # propagators to pairs without them.
    expected = {pair: sorted(cs) for pair, cs in colours.items()}
    _, labellings = _canonical_labellings(multiplicity, legs)
    found = []
    for order in labellings:
        image = dict(zip(labellings[0], order, strict=True))
        if all(legs[v] == legs[image[v]] for v in range(V)) and all(
            expected.get((min(image[u], image[w]), max(image[u], image[w]))) == cs
            for (u, w), cs in expected.items()
        ):
            found.append([image[v] + 1 for v in range(V)])
    return sorted(found)


# ------------------------------------------------------------------------------
# Public: coefficient-preserving automorphisms
# ------------------------------------------------------------------------------


def coefficient_preserving_indices(
    fi: FeynmanIntegral,
    auts: PolytopeAutomorphisms,
) -> list[int]:
    """
    Indices into ``auts.maps`` of automorphisms that also preserve the
    multiset of G-polynomial coefficients.

    A unimodular automorphism (U, t) is *coefficient-preserving* if, when
    applied to the support of G, each monomial maps to another monomial with
    the same coefficient.  Every automorphism that maps the monomials of G
    onto themselves, with sigma the induced permutation, gives the identity
    I_A(beta, z_sigma) = I_A(T beta, z) for the integral without Gamma
    prefactors.  For a coefficient-preserving one z_sigma = z at the physical
    point, so the identity relates the integral at beta and at T beta; it
    does not relate different kinematic points.

    Parameters
    ----------
    fi
        The Feynman integral whose G polynomial coefficients are used.
    auts
        Automorphism group as returned by :func:`compute_polytope_automorphisms`.

    Returns
    -------
    list[int]
        Indices ``k`` such that ``auts.maps[k]`` is coefficient-preserving.
        Index 0 (identity) is always included.
    """
    support = fi.newton_polytope.support  # list of (exponent_tuple, coeff)
    coeff_map: dict[tuple[int, ...], sp.Expr] = dict(support)

    result = []
    for k, (U, t) in enumerate(auts.maps):
        preserves = True
        for exp, coeff in support:
            p = sp.Matrix(list(exp))
            img = U * p + t
            img_key = tuple(int(x) for x in img)
            mapped_coeff = coeff_map.get(img_key)
            if mapped_coeff is None:
                preserves = False
                break
            if sp.simplify(mapped_coeff - coeff) != 0:
                preserves = False
                break
        if preserves:
            result.append(k)

    return result


# ------------------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------------------


def _select_basis_indices(deltas: np.ndarray, n_dim: int) -> list[int] | None:
    """Greedy basis selection by index order (fallback)."""
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


def _select_basis_indices_by_label(
    deltas: np.ndarray,
    n_dim: int,
    labels: list[int],
    label_count: Counter,
) -> list[int] | None:
    """
    Greedy basis selection ordered by Liu-Cai label class size (rarest first).

    Choosing vertices from smaller label classes as basis vectors minimises
    the number of eligible basis combinations during the anchor loop: if the
    basis label multiset uses only small-class labels, the per-anchor
    combination count is C(s_1, k_1) x C(s_2, k_2) x ... rather than C(N-1, n).
    For K_4 this reduces ~106 000 combos/anchor to ~4, giving ~700 x  speedup.
    """
    sorted_indices = sorted(
        range(1, deltas.shape[0]),
        key=lambda i: (label_count[labels[i]], i),
    )
    chosen: list[int] = []
    chosen_rows: list[np.ndarray] = []
    for i in sorted_indices:
        candidate = np.array(chosen_rows + [deltas[i].astype(float)])
        if np.linalg.matrix_rank(candidate) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(deltas[i].astype(float))
            if len(chosen) == n_dim:
                return chosen
    return None


def _label_preserving_orderings(
    combo: tuple[int, ...],
    labels: list[int],
    basis_label_seq: list[int],
) -> Iterator[tuple[int, ...]]:
    """Yield all orderings of ``combo`` consistent with ``basis_label_seq``."""
    label_to_entries: dict[int, list[int]] = defaultdict(list)
    for j in combo:
        label_to_entries[labels[j]].append(j)

    label_to_positions: dict[int, list[int]] = defaultdict(list)
    for k, ell in enumerate(basis_label_seq):
        label_to_positions[ell].append(k)

    if set(label_to_entries) != set(label_to_positions):
        return
    for ell in label_to_positions:
        if len(label_to_entries[ell]) != len(label_to_positions[ell]):
            return

    groups = [
        (label_to_positions[ell], list(permutations(label_to_entries[ell])))
        for ell in sorted(label_to_positions)
    ]

    result: list[int] = [-1] * len(combo)
    for group_perms in _prod(*[perms for _, perms in groups]):
        for (positions, _), assigned in zip(groups, group_perms, strict=True):
            for pos, entry in zip(positions, assigned, strict=True):
                result[pos] = entry
        yield tuple(result)


def _verify_witness(
    U: sp.Matrix,
    Z: sp.Matrix,
    V_arr: np.ndarray,
    vertex_map: dict[int, int],
) -> bool:
    for i in range(V_arr.shape[0]):
        v_col = sp.Matrix(V_arr[i].tolist())
        target = sp.Matrix(V_arr[vertex_map[i]].tolist())
        if U * v_col + Z != target:
            return False
    return True


def _compute_orbits(n_vert: int, vperms: list[list[int]]) -> list[list[int]]:
    parent = list(range(n_vert))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for vperm in vperms:
        for i, j in enumerate(vperm):
            union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n_vert):
        r = find(i)
        buckets.setdefault(r, []).append(i)

    return sorted(sorted(orb) for orb in buckets.values())


def _identity_only(V_arr: np.ndarray, n_dim: int) -> PolytopeAutomorphisms:
    n_vert = V_arr.shape[0]
    ident = sp.ImmutableMatrix(sp.eye(n_dim))
    zero = sp.ImmutableMatrix(sp.zeros(n_dim, 1))
    return PolytopeAutomorphisms(
        maps=[(ident, zero)],
        order=1,
        vertex_permutations=[list(range(n_vert))],
        vertex_orbits=[[i] for i in range(n_vert)],
    )
