"""
Combinatorial invariants and helpers for the Liu-Cai unimodular isomorphism
algorithm (arXiv:2506.23846).

The functions in this module are the building blocks used by
:func:`feynkit.normal_forms.affine_equivalence.is_unimodular_equivalent`.
They are also useful in their own right for callers that want to inspect
the labelled vertex-edge graph of a lattice polytope, the Liu-Cai vertex
labels, or the label-preserving automorphism group.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, cast

import networkx as nx
import numpy as np
import sympy as sp

from ..polytope import PolytopeData, polytope_data

# ------------------------------------------------------------------------------
# Coercion / hull
# ------------------------------------------------------------------------------


def to_integer_points(points: object) -> np.ndarray:
    """
    Coerce a point configuration to a ``d x n`` integer numpy array.

    Accepts numpy arrays, sympy matrices, lists of tuples, and similar.
    Raises :class:`ValueError` on non-integer entries, Liu-Cai is defined
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
        raise ValueError(f"Liu-Cai requires integer points: {exc}") from exc

    # Re-coerce via Python int to detect non-integer floats that np accepted.
    for row in arr.tolist():
        for v in row:
            if isinstance(v, float) and v != int(v):
                raise ValueError(f"Liu-Cai requires integer points; got float {v}")

    return out


def hull_vertex_indices(points: np.ndarray) -> np.ndarray:
    """
    Return indices into ``points`` that are extreme points of conv(points).

    The vertices are those of :func:`feynkit.polytope.polytope_data`, found in
    integer arithmetic in every dimension. Returns the input indices in sorted
    order; of repeated points, the first.
    """
    if points.shape[0] == 0:
        return np.arange(0)
    return np.array(polytope_data(points).vertex_indices, dtype=np.int64)


# ------------------------------------------------------------------------------
# Vertex/edge graph and Liu-Cai labels
# ------------------------------------------------------------------------------


def polytope_skeleton(data: PolytopeData) -> nx.Graph:
    """
    The 1-skeleton of the polytope ``data`` describes.

    Node ``i`` is the vertex ``data.vertices[i]``, and two nodes are joined
    when they are the vertices of a face of dimension 1 of the certified face
    lattice. A segment is its own edge; a point has none.
    """
    position = {v: i for i, v in enumerate(data.vertex_indices)}
    G = nx.Graph()
    G.add_nodes_from(range(len(position)))
    for dimension, indices in data.faces:
        if dimension == 1:
            G.add_edge(*(position[i] for i in indices if i in position))
    return G


def vertex_edge_graph(vertices: np.ndarray) -> nx.Graph:
    """
    1-skeleton of the convex hull of ``vertices``.

    All input rows must be extreme points of conv(vertices); use
    :func:`hull_vertex_indices` to filter first if necessary. The returned
    graph's nodes are ``0..d-1`` indexing into ``vertices``, and its edges are
    the faces of dimension 1 of :func:`feynkit.polytope.polytope_data`, in
    every dimension.
    """
    if vertices.shape[0] == 0:
        return nx.Graph()
    return polytope_skeleton(polytope_data(vertices))


def vertex_label(v: np.ndarray, neighbours: np.ndarray) -> int:
    """
    Liu-Cai vertex label

        l(v) = det(A_v),   A_v = sum_w (w - v)(w - v)^T

    where the sum is over the rows of ``neighbours``. Returns an exact
    Python ``int``; Liu-Cai shows that for any unimodular ``U``,
    ``A_{U v + Z} = U A_v U^T``, so ``det(A_v) = det(A_{U v + Z})``.
    """
    diffs = neighbours.astype(np.int64) - v.astype(np.int64)  # m x n
    A_v = diffs.T @ diffs  # n x n integer
    return int(sp.Matrix(A_v.tolist()).det())


def label_skeleton(coordinates: np.ndarray, skeleton: nx.Graph) -> nx.Graph:
    """
    A copy of ``skeleton`` with the Liu-Cai labels of the vertices at ``coordinates``.

    Node ``i`` of ``skeleton`` is the vertex ``coordinates[i]``. The node
    attribute ``"label"`` is its label, 0 when it has no neighbour, and the edge
    attribute ``"weight"`` the sum of the labels of the two ends. The
    coordinates may be those of a lattice chart, in which a polytope that is
    not full-dimensional is full-dimensional; in the ambient coordinates every
    label of such a polytope is 0.
    """
    G = skeleton.copy()
    labels: dict[int, int] = {}
    for i in G.nodes():
        nbrs = list(G.neighbors(i))
        labels[i] = vertex_label(coordinates[i], coordinates[np.array(nbrs)]) if nbrs else 0
    nx.set_node_attributes(G, labels, "label")
    for u, v in G.edges():
        G[u][v]["weight"] = labels[u] + labels[v]
    return G


def labelled_polytope_graph(vertices: np.ndarray) -> nx.Graph:
    """
    Build the Liu-Cai labelled vertex-edge graph $\\mathcal{GW}(P)$:

    - nodes ``0..d-1`` index into ``vertices`` (assumed to be extreme points),
    - node attribute ``"label"`` is the Liu-Cai vertex label ``det(A_v)``,
    - edge attribute ``"weight"`` is the sum of endpoint labels.
    """
    return label_skeleton(vertices, vertex_edge_graph(vertices))


# ------------------------------------------------------------------------------
# Spanning trees and label-preserving isomorphisms
# ------------------------------------------------------------------------------


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
    early on a match, for highly-symmetric graphs the MST count can be
    very large.
    """
    return list(iter_minimum_spanning_trees(G))


def _node_label_match(a: dict, b: dict) -> bool:
    return a.get("label") == b.get("label")


def label_preserving_isomorphisms(T1: nx.Graph, T2: nx.Graph) -> Iterator[dict[int, int]]:
    """
    Yield all isomorphisms ``T1 -> T2`` that preserve the ``"label"`` node
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
