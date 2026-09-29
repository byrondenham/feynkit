"""
Automorphism group of a convex lattice polytope.

The automorphism group Aut(P) is the group of P as a lattice polytope in the
affine lattice aff(P) cap Z^n: the affine bijections of the affine hull of P
that map its integer points onto themselves and P onto itself. When P is
full-dimensional these are the unimodular affine maps (U, t), U in GL_n(Z),
|det U| = 1, t in Z^n, that send P to itself. Below full dimension each
extends to such a map of Z^n, and ``maps`` holds one extension per element.
The identity (I, 0) is always a member; for a generic polytope it is the
only member.  Symmetric integrals (triangles, bananas, boxes) have larger
groups that are directly visible in their GKZ structure.

The algorithm extends the Liu-Cai basis-search used for equivalence testing
(_direct_basis_search in affine_equivalence.py): instead of returning on the
first valid witness, every valid (U, t) is collected and deduplicated via the
induced vertex permutation.

The vertices, and the edges on which the Liu-Cai labels are computed, come from
the certified face lattice of feynkit.polytope.polytope_data.

The search takes its basis among the neighbours of an anchor vertex in the
1-skeleton and maps it only to neighbours of each candidate image, with the
same labels and the same edges among them, since an automorphism maps the
neighbours of a vertex onto the neighbours of its image. This prunes where
the labels separate nothing: the massless pentagon's group, of order 720,
takes about 0.3 s, and the hexagon's, of order 5040, about 2 s.

Public API
----------
compute_polytope_automorphisms(points) -> PolytopeAutomorphisms
    Automorphism group of conv(points) as a lattice polytope.

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
from itertools import permutations
from itertools import product as _prod
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import sympy as sp

from .. import _exact
from ..core.exceptions import ComputationError
from ..polytope import polytope_data
from ..types import PolytopeAutomorphisms
from . import _invariants
from ._chart import chart_frame, lift_linear

if TYPE_CHECKING:
    from ..core.graph import Graph
    from ..integral import FeynmanIntegral


# ------------------------------------------------------------------------------
# Public: polytope automorphism group
# ------------------------------------------------------------------------------


def compute_polytope_automorphisms(points: object) -> PolytopeAutomorphisms:
    """
    Compute the automorphism group of conv(points) as a lattice polytope.

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
    fixed anchor vertex, enumerate the images of a fixed basis and verify
    that the implied affine map sends the vertices onto themselves. The
    anchor is taken in the rarest label class and the basis among its
    neighbours in the 1-skeleton, whose images must be neighbours of the
    anchor's image with the same labels and the same edges among them. The
    maps come in the order in which a search over all vertices, anchored at
    vertex 0 with a basis from the rarest label classes, finds them.

    A polytope of dimension d < n is searched in the lattice chart of its
    vertices, where it is full-dimensional in Z^d and its labels are d x d
    determinants. A chart map is kept when it maps the integer points of the
    affine hull onto themselves, and ``maps`` holds its lift (U, t), whose
    linear part U in GL_n(Z) fixes a complement of the direction space of the
    affine hull, the span of the differences of the vertices; see
    feynkit.normal_forms._chart. ``order`` counts the chart maps kept, that is
    the vertex permutations; the unimodular maps of Z^n taking P to itself
    form an infinite group when d < n and n >= 2.
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

    skeleton = _invariants.polytope_skeleton(data)
    if data.is_full_dimensional:
        found_maps, found_vperms = _basis_search(V_arr, _invariants.label_skeleton(V_arr, skeleton))
    else:
        found_maps, found_vperms = _chart_search(V_arr, skeleton)
    if not found_maps:
        return _identity_only(V_arr, n_dim)

    return PolytopeAutomorphisms(
        maps=found_maps,
        order=len(found_maps),
        vertex_permutations=found_vperms,
        vertex_orbits=_compute_orbits(V_arr.shape[0], found_vperms),
    )


def _chart_search(
    V_arr: np.ndarray, skeleton: nx.Graph
) -> tuple[list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]], list[list[int]]]:
    """
    The automorphisms of vertices that are not full-dimensional, lifted from their chart.

    Row i of ``V_arr`` is node i of ``skeleton``. The search runs on the chart
    coordinates of the vertices, a chart map is kept when its lift is integral,
    and the lift (U, t) sends every vertex where the chart map sends it.
    """
    frame = chart_frame(V_arr)
    coordinates = frame.coordinates
    chart_maps, chart_vperms = _basis_search(
        coordinates, _invariants.label_skeleton(coordinates, skeleton)
    )
    found_maps: list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]] = []
    found_vperms: list[list[int]] = []
    for (M, _s), vperm in zip(chart_maps, chart_vperms, strict=True):
        U = lift_linear(frame, frame, M)
        if U is None:
            continue
        t = sp.Matrix(V_arr[vperm[0]].tolist()) - U * sp.Matrix(V_arr[0].tolist())
        found_maps.append((U, sp.ImmutableMatrix(t)))
        found_vperms.append(vperm)
    return found_maps, found_vperms


def _basis_search(
    V_arr: np.ndarray, GW: nx.Graph
) -> tuple[list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]], list[list[int]]]:
    """
    Every unimodular affine map permuting the rows of ``V_arr``, with its permutation.

    ``V_arr`` holds the vertices of a full-dimensional lattice polytope and
    ``GW`` its labelled 1-skeleton, whose node i is row i. Returns no maps
    when the vertices do not span their ambient space.

    An automorphism maps each vertex to a vertex with the same label and
    degree, and its neighbours onto the neighbours of the image, keeping their
    labels and the edges among them. The search fixes an anchor in the rarest
    label class and a basis among its neighbours, whose edge directions span
    the space at a vertex of a full-dimensional polytope. For each vertex that
    can be the anchor's image it maps the basis only to neighbours of that
    vertex with the same labels and the same edges among them. Every
    candidate U = W_b W_a^-1 is found in integer arithmetic, as
    W_b adj(W_a) / det(W_a), and kept only when it is integral and permutes the
    vertices. The maps come in the order in which a search over all vertices,
    anchored at vertex 0 with a label-diverse basis, finds them (see
    _search_order), so the result does not depend on how it is found.
    """
    n_vert, n_dim = V_arr.shape
    labels = [GW.nodes[i]["label"] for i in range(n_vert)]
    label_count = Counter(labels)

    # The basis that fixes the order of the result, as the former search chose it.
    order_basis = _select_basis_indices_by_label(
        (V_arr - V_arr[0]).astype(np.int64), n_dim, labels, label_count
    )
    if order_basis is None:
        return [], []

    neighbours = [set(GW.neighbors(i)) for i in range(n_vert)]
    anchor = min(range(n_vert), key=lambda i: (label_count[labels[i]], i))
    deltas_a = (V_arr - V_arr[anchor]).astype(np.int64)
    rarest_first = sorted(neighbours[anchor], key=lambda j: (label_count[labels[j]], j))
    basis = _independent_rows(deltas_a, rarest_first, n_dim)
    if basis is None:  # pragma: no cover - the edges at a vertex span the space
        raise ComputationError("the edges at a vertex of the polytope do not span its space")

    W_a = sp.Matrix(deltas_a[basis].T.tolist())
    adj_a, det_a = _invariants.integer_inverse(W_a, 2 * int(np.abs(V_arr).max()) + 1)
    deltas_exact = deltas_a.astype(adj_a.dtype)
    basis_labels = [labels[b] for b in basis]
    adjacent = [[basis[m] in neighbours[b] for m in range(n_dim)] for b in basis]

    found: dict[tuple[int, ...], tuple[sp.ImmutableMatrix, sp.ImmutableMatrix]] = {}
    for image in range(n_vert):
        if labels[image] != labels[anchor] or len(neighbours[image]) != len(neighbours[anchor]):
            continue

        deltas_b = (V_arr - V_arr[image]).astype(np.int64)
        delta_to_b_idx = {tuple(int(x) for x in row): i for i, row in enumerate(deltas_b.tolist())}
        options = [
            [j for j in sorted(neighbours[image]) if labels[j] == label] for label in basis_labels
        ]
        for images in _matching_neighbours(options, adjacent, neighbours):
            U_int = _invariants.integral_candidate(deltas_b[list(images)].T, adj_a, det_a)
            if U_int is None:
                continue

            mapped = U_int @ deltas_exact.T
            vertex_map: dict[int, int] = {}
            used: set[int] = set()
            valid = True
            for i in range(n_vert):
                key = tuple(int(x) for x in mapped[:, i])
                j_opt = delta_to_b_idx.get(key)
                if j_opt is None or j_opt in used:
                    valid = False
                    break
                vertex_map[i] = j_opt
                used.add(j_opt)
            if not valid:
                continue

            # The integer checks above are exact; SymPy only builds the output.
            vperm = tuple(vertex_map[i] for i in range(n_vert))
            if vperm in found:
                continue

            # Compute exact integer translation: Z = image - U * anchor.
            Z_int = V_arr[image].astype(adj_a.dtype) - U_int @ V_arr[anchor].astype(adj_a.dtype)
            found[vperm] = (
                sp.ImmutableMatrix(U_int.tolist()),
                sp.ImmutableMatrix(Z_int.reshape(-1, 1).tolist()),
            )

    order = sorted(found, key=lambda vperm: _search_order(vperm, order_basis, labels))
    return [found[vperm] for vperm in order], [list(vperm) for vperm in order]


def configuration_symmetries(
    points: np.ndarray,
) -> list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix, tuple[int, ...]]]:
    """
    The unimodular affine maps that permute the rows of ``points``, with their permutations.

    ``points`` is a full-dimensional integer configuration. With a repeated
    row there are none, since the rows are matched by their coordinates. A
    map that permutes the points permutes the vertices of conv(points), so it
    is one of the automorphisms of :func:`compute_polytope_automorphisms`, and
    the maps sought are those automorphisms that send every point to a point.
    They come in the order in which a search over all the points, anchored at
    point 0 with a basis from the rarest label classes, finds them; the hull
    vertices carry the labels of the automorphism search and the other points
    the label 0.
    """
    n_points, n_dim = points.shape
    rows = [tuple(int(x) for x in p) for p in points.tolist()]
    index = {p: i for i, p in enumerate(rows)}
    if len(index) < n_points:
        return []

    # The automorphisms, as compute_polytope_automorphisms finds them in full
    # dimension, with the labels of the vertices for the order below.
    data = polytope_data(points)
    V_arr = np.array(data.vertices, dtype=np.int64).reshape(len(data.vertices), n_dim)
    GW = _invariants.label_skeleton(V_arr, _invariants.polytope_skeleton(data))
    maps, _ = _basis_search(V_arr, GW)

    size = 2 * int(np.abs(points).max()) + 1
    found: list[tuple[sp.ImmutableMatrix, sp.ImmutableMatrix, tuple[int, ...]]] = []
    for U, t in maps:
        linear = [[int(x) for x in row] for row in U.tolist()]
        shift = [int(x) for x in t]
        largest = max(abs(x) for row in linear for x in row)
        # An image has entries below n_dim max|U| size / 2 + |t|, and |t| is below that plus size.
        dtype = _invariants.exact_dtype(n_dim * largest * size + size)
        matrix: np.ndarray = np.array(linear, dtype=dtype)
        images: np.ndarray = points.astype(dtype) @ matrix.T + np.array(shift, dtype=dtype)
        perm: list[int] = []
        for image in images.tolist():
            j = index.get(tuple(int(x) for x in image))
            if j is None:
                break
            perm.append(j)
        else:
            found.append((U, t, tuple(perm)))

    vertex_label = {v: GW.nodes[i]["label"] for i, v in enumerate(data.vertex_indices)}
    labels = [vertex_label.get(i, 0) for i in range(n_points)]
    basis = _select_basis_indices_by_label(
        (points - points[0]).astype(np.int64), n_dim, labels, Counter(labels)
    )
    if basis is None:  # pragma: no cover - the points are full-dimensional
        raise ComputationError("the points do not span their ambient space")
    found.sort(key=lambda pair: _search_order(pair[2], basis, labels))
    return found


def _matching_neighbours(
    options: list[list[int]], adjacent: list[list[bool]], neighbours: list[set[int]]
) -> Iterator[tuple[int, ...]]:
    """
    The distinct choices of one entry of each ``options[k]`` with the edges of ``adjacent``.

    Entry k and entry m of a choice must be joined by an edge, in
    ``neighbours``, exactly when ``adjacent[k][m]`` holds.
    """
    chosen: list[int] = []

    def extend(k: int) -> Iterator[tuple[int, ...]]:
        if k == len(options):
            yield tuple(chosen)
            return
        for j in options[k]:
            if j in chosen:
                continue
            if any((j in neighbours[chosen[m]]) != adjacent[k][m] for m in range(k)):
                continue
            chosen.append(j)
            yield from extend(k + 1)
            chosen.pop()

    yield from extend(0)


def _search_order(
    perm: tuple[int, ...], basis: list[int], labels: list[int]
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """
    Where a search anchored at point 0 with the given basis finds the map of ``perm``.

    That search tries the images of point 0 in index order; for each, the
    sets of basis images class by class, each class in the order of
    itertools.combinations; and for each set the orderings of each class in
    the order of itertools.permutations. The map with permutation ``perm``
    sends point 0 to perm[0] and basis point b to perm[b], so its position is
    the key returned, compared lexicographically.
    """
    classes = sorted({labels[b] for b in basis})
    images = [[perm[b] for b in basis if labels[b] == label] for label in classes]
    return perm[0], tuple(tuple(sorted(c)) for c in images), tuple(tuple(c) for c in images)


def _independent_rows(deltas: np.ndarray, candidates: list[int], n_dim: int) -> list[int] | None:
    """The first ``n_dim`` of ``candidates`` whose rows of ``deltas`` are independent, greedily."""
    chosen: list[int] = []
    chosen_rows: list[list[int]] = []
    for i in candidates:
        row = [int(x) for x in deltas[i]]
        if _exact.rank([*chosen_rows, row]) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(row)
            if len(chosen) == n_dim:
                return chosen
    return None


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
    chosen_rows: list[list[int]] = []
    for i in range(1, deltas.shape[0]):
        row = [int(x) for x in deltas[i]]
        if _exact.rank([*chosen_rows, row]) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(row)
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
    chosen_rows: list[list[int]] = []
    for i in sorted_indices:
        row = [int(x) for x in deltas[i]]
        if _exact.rank([*chosen_rows, row]) == len(chosen) + 1:
            chosen.append(i)
            chosen_rows.append(row)
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
