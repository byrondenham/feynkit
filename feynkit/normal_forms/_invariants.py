"""
Combinatorial invariants and helpers for the Liu–Cai unimodular isomorphism
algorithm (arXiv:2506.23846).

The functions in this module are the building blocks used by
:func:`feynkit.normal_forms.affine_equivalence.is_unimodular_equivalent`.
They are also useful in their own right for callers that want to inspect
the labelled vertex-edge graph of a lattice polytope, the Liu–Cai vertex
labels, or the label-preserving automorphism group.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, cast

import networkx as nx
import numpy as np
import sympy as sp
from scipy.spatial import ConvexHull, QhullError

# ──────────────────────────────────────────────────────────────────────────────
# Coercion / hull
# ──────────────────────────────────────────────────────────────────────────────


def to_integer_points(points: object) -> np.ndarray:
    """
    Coerce a point configuration to a ``d × n`` integer numpy array.

    Accepts numpy arrays, sympy matrices, lists of tuples, and similar.
    Raises :class:`ValueError` on non-integer entries — Liu–Cai is defined
    only for lattice polytopes.
    """
    if isinstance(points, np.ndarray):
        arr = points
    elif isinstance(points, sp.Matrix):
        arr = np.array(points.tolist(), dtype=object)
    else:
        arr = np.array(list(cast(Iterable[Any], points)), dtype=object)

    if arr.ndim != 2:
        raise ValueError(f"Point configuration must be 2-dimensional; got shape {arr.shape}")

    try:
        out = np.array(arr.tolist(), dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Liu–Cai requires integer points: {exc}") from exc

    # Re-coerce via Python int to detect non-integer floats that np accepted.
    for row in arr.tolist():
        for v in row:
            if isinstance(v, float) and v != int(v):
                raise ValueError(f"Liu–Cai requires integer points; got float {v}")

    return out


def hull_vertex_indices(points: np.ndarray) -> np.ndarray:
    """
    Return indices into ``points`` that are extreme points of conv(points).

    For full-dimensional input this is :attr:`scipy.spatial.ConvexHull.vertices`.
    For collinear / lower-dimensional input we project to the affine hull,
    run scipy there, and lift the result back. Returns the input indices in
    sorted order.
    """
    n_pts = points.shape[0]
    if n_pts <= 1:
        return np.arange(n_pts)

    centroid = points.mean(axis=0)
    deltas = points - centroid
    rank = int(np.linalg.matrix_rank(deltas.astype(float)))

    if rank == 0:
        # All points coincide.
        return np.array([0])

    if rank == points.shape[1]:
        try:
            hull = ConvexHull(points.astype(float))
        except QhullError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not compute convex hull: {exc}") from exc
        return np.sort(hull.vertices)

    # Degenerate: project to affine hull, run hull there, lift back.
    # Right singular vectors span the row space of deltas (= affine hull).
    _u, _s, vt = np.linalg.svd(deltas.astype(float), full_matrices=False)
    basis = vt[:rank, :].T  # n_dim × rank
    coords = deltas.astype(float) @ basis  # n_pts × rank
    if rank == 1:
        # Convex hull of a 1-d set: just min and max.
        idx_min = int(np.argmin(coords[:, 0]))
        idx_max = int(np.argmax(coords[:, 0]))
        return np.array(sorted({idx_min, idx_max}))
    try:
        hull = ConvexHull(coords)
    except QhullError as exc:  # pragma: no cover
        raise ValueError(f"Could not compute convex hull: {exc}") from exc
    return np.sort(hull.vertices)


# ──────────────────────────────────────────────────────────────────────────────
# Vertex/edge graph and Liu–Cai labels
# ──────────────────────────────────────────────────────────────────────────────


def _facet_incidence(vertices: np.ndarray, hull: ConvexHull, tol: float = 1e-9) -> list[set[int]]:
    """For each vertex i, return the set of facet indices containing it."""
    n_vert = vertices.shape[0]
    incidence: list[set[int]] = [set() for _ in range(n_vert)]
    verts_f = vertices.astype(float)
    for j, eq in enumerate(hull.equations):
        a = eq[:-1]
        b = eq[-1]
        residuals = verts_f @ a + b
        for i in range(n_vert):
            if abs(residuals[i]) < tol:
                incidence[i].add(j)
    return incidence


def vertex_edge_graph(vertices: np.ndarray) -> nx.Graph:
    """
    1-skeleton of the convex hull of ``vertices``.

    All input rows must be extreme points of conv(vertices); use
    :func:`hull_vertex_indices` to filter first if necessary. The returned
    graph's nodes are ``0..d-1`` indexing into ``vertices``, and edges are
    the 1-faces of the polytope.

    Two extreme vertices ``u, v`` form an edge iff at least ``n - 1``
    facets pass through both (``n`` is the ambient dimension).
    """
    n_dim = vertices.shape[1]
    n_vert = vertices.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n_vert))

    if n_vert <= 1:
        return G

    if n_dim == 1 or n_vert == 2:
        # Trivial cases: line segment or two-vertex degenerate.
        for i in range(n_vert):
            for k in range(i + 1, n_vert):
                G.add_edge(i, k)
        return G

    # Standard case: full-dim polytope in R^n.
    centroid = vertices.mean(axis=0).astype(float)
    deltas = vertices.astype(float) - centroid
    rank = int(np.linalg.matrix_rank(deltas))

    if rank < n_dim:
        # Lower-dimensional polytope embedded in higher-dim ambient space.
        # Build the 1-skeleton in the affine hull and lift back.
        _u, _s, vt = np.linalg.svd(deltas, full_matrices=False)
        basis = vt[:rank, :].T  # n_dim × rank
        coords = deltas @ basis
        if rank == 1:
            order = np.argsort(coords[:, 0])
            for a, b in zip(order, order[1:], strict=True):
                G.add_edge(int(a), int(b))
            return G
        try:
            hull = ConvexHull(coords)
        except QhullError as exc:  # pragma: no cover
            raise ValueError(f"Could not compute convex hull: {exc}") from exc
        incidence = _facet_incidence(coords, hull)
        for i in range(n_vert):
            for k in range(i + 1, n_vert):
                if len(incidence[i] & incidence[k]) >= rank - 1:
                    G.add_edge(i, k)
        return G

    try:
        hull = ConvexHull(vertices.astype(float))
    except QhullError as exc:  # pragma: no cover
        raise ValueError(f"Could not compute convex hull: {exc}") from exc

    incidence = _facet_incidence(vertices, hull)
    for i in range(n_vert):
        for k in range(i + 1, n_vert):
            if len(incidence[i] & incidence[k]) >= n_dim - 1:
                G.add_edge(i, k)
    return G


def vertex_label(v: np.ndarray, neighbours: np.ndarray) -> int:
    """
    Liu–Cai vertex label

        ℓ(v) = det(A_v),   A_v = Σ_w (w − v)(w − v)^T

    where the sum is over the rows of ``neighbours``. Returns an exact
    Python ``int``; Liu–Cai shows that for any unimodular ``U``,
    ``A_{U v + Z} = U A_v U^T``, so ``det(A_v) = det(A_{U v + Z})``.
    """
    diffs = neighbours.astype(np.int64) - v.astype(np.int64)  # m × n
    A_v = diffs.T @ diffs  # n × n integer
    return int(sp.Matrix(A_v.tolist()).det())


def labelled_polytope_graph(vertices: np.ndarray) -> nx.Graph:
    """
    Build the Liu–Cai labelled vertex-edge graph $\\mathcal{GW}(P)$:

    - nodes ``0..d-1`` index into ``vertices`` (assumed to be extreme points),
    - node attribute ``"label"`` is the Liu–Cai vertex label ``det(A_v)``,
    - edge attribute ``"weight"`` is the sum of endpoint labels.
    """
    G = vertex_edge_graph(vertices)

    labels: dict[int, int] = {}
    for i in G.nodes():
        nbrs = list(G.neighbors(i))
        if not nbrs:
            labels[i] = 0
            continue
        labels[i] = vertex_label(vertices[i], vertices[np.array(nbrs)])
    nx.set_node_attributes(G, labels, "label")

    for u, v in G.edges():
        G[u][v]["weight"] = labels[u] + labels[v]
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Spanning trees and label-preserving isomorphisms
# ──────────────────────────────────────────────────────────────────────────────


def _copy_node_labels(src: nx.Graph, dst: nx.Graph) -> nx.Graph:
    for n in dst.nodes():
        if "label" in src.nodes[n]:
            dst.nodes[n]["label"] = src.nodes[n]["label"]
    return dst


def iter_minimum_spanning_trees(G: nx.Graph) -> Iterator[nx.Graph]:
    """
    Lazily yield minimum spanning trees of a connected weighted graph in
    weight order, stopping as soon as the weight strictly exceeds the
    minimum.

    Each yielded tree is a fresh ``nx.Graph``; the node ``"label"``
    attribute from ``G`` is propagated onto each tree.
    """
    if G.number_of_nodes() == 0:
        return
    if G.number_of_nodes() == 1:
        T = nx.Graph()
        T.add_node(next(iter(G.nodes())))
        _copy_node_labels(G, T)
        yield T
        return

    iterator = nx.algorithms.tree.mst.SpanningTreeIterator(G, weight="weight")
    min_weight: float | None = None
    for tree in iterator:
        weight = sum(d.get("weight", 1) for _, _, d in tree.edges(data=True))
        if min_weight is None:
            min_weight = weight
        elif weight > min_weight + 1e-9:
            return
        _copy_node_labels(G, tree)
        yield tree


def all_minimum_spanning_trees(G: nx.Graph) -> list[nx.Graph]:
    """
    Materialise every minimum spanning tree as a list (eager).

    Prefer :func:`iter_minimum_spanning_trees` when the caller can return
    early on a match — for highly-symmetric graphs the MST count can be
    very large.
    """
    return list(iter_minimum_spanning_trees(G))


def _node_label_match(a: dict, b: dict) -> bool:
    return a.get("label") == b.get("label")


def label_preserving_isomorphisms(T1: nx.Graph, T2: nx.Graph) -> Iterator[dict[int, int]]:
    """
    Yield all isomorphisms ``T1 → T2`` that preserve the ``"label"`` node
    attribute. Each yielded value is a dict ``{i_in_T1: j_in_T2}``.
    """
    if T1.number_of_nodes() != T2.number_of_nodes():
        return
    if T1.number_of_edges() != T2.number_of_edges():
        return
    matcher = nx.algorithms.isomorphism.GraphMatcher(T1, T2, node_match=_node_label_match)
    yield from matcher.isomorphisms_iter()


def label_preserving_automorphisms(T: nx.Graph) -> list[dict[int, int]]:
    """All label-preserving automorphisms of ``T`` (= $\\mathrm{Aut}_\\ell(T)$)."""
    return list(label_preserving_isomorphisms(T, T))
