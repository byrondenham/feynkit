"""The exact vertices and 1-skeleton that the Liu-Cai labels are built on."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, ValidationError
from feynkit.normal_forms._invariants import (
    hull_vertex_indices,
    label_skeleton,
    labelled_polytope_graph,
    polytope_skeleton,
    to_integer_points,
    vertex_edge_graph,
    vertex_label,
)
from feynkit.polytope import polytope_data


@pytest.mark.parametrize(
    ("cnickel", "edges"),
    [
        # Qhull triangulates the non-simplicial facets, and the graph built from its
        # facets counted their diagonals as edges: 15 and 55.
        ("12e|2e|e|:nnn", 9),
        ("12e|3e|3e|e|:nzzz", 28),
    ],
)
def test_edges_are_the_faces_of_dimension_one(cnickel: str, edges: int) -> None:
    data = polytope_data(FeynmanIntegral.from_cnickel(cnickel).newton_polytope.points)
    graph = vertex_edge_graph(np.array(data.vertices))
    assert graph.number_of_edges() == edges == data.f_vector[1]
    assert sorted(graph.nodes) == list(range(len(data.vertices)))


def test_a_point_on_an_edge_is_not_a_vertex() -> None:
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1]])
    assert hull_vertex_indices(points).tolist() == [0, 1, 2, 3]


def test_segments_and_points() -> None:
    assert hull_vertex_indices(np.array([[3], [0], [1]])).tolist() == [0, 1]
    assert list(vertex_edge_graph(np.array([[0, 0], [2, 2]])).edges) == [(0, 1)]
    assert vertex_edge_graph(np.array([[1, 2]])).number_of_edges() == 0
    assert hull_vertex_indices(np.array([[1, 2], [1, 2]])).tolist() == [0]


def test_labels_are_zero_below_full_dimension() -> None:
    # The triangle (0, 0, 1), (0, 1, 0), (1, 0, 0) spans a plane in R^3: each 3 x 3 moment
    # matrix has rank 2.
    graph = labelled_polytope_graph(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
    assert [graph.nodes[i]["label"] for i in graph.nodes] == [0, 0, 0]


@pytest.mark.parametrize(
    "coordinate",
    [sp.Rational(1, 2), Fraction(1, 2), 0.5, sp.Float(0.5), sp.Float("1.00000000000000000001", 30)],
)
def test_non_integer_points_raise(coordinate: object) -> None:
    # They were truncated: int(Rational(1, 2)) is 0. A float of 30 digits that rounds to 1 as a
    # double is not an integer either.
    with pytest.raises(ValidationError, match="non-integer"):
        to_integer_points([(0, 0), (coordinate, 1)])


def test_integers_in_other_types() -> None:
    points = to_integer_points([(sp.Integer(2), 1.0), (Fraction(4, 2), sp.Rational(3, 1))])
    assert points.dtype == np.int64
    assert points.tolist() == [[2, 1], [2, 3]]
    assert to_integer_points([(sp.Float(2.0), sp.Float("-3", 30))]).tolist() == [[2, -3]]
    assert to_integer_points(np.zeros((0, 3), dtype=np.int64)).shape == (0, 3)


def test_labels_of_large_coordinates() -> None:
    # A_v = [[25 * 10^18 + 1, 10^10], [10^10, 25 * 10^18 + 1]] has entries beyond int64,
    # where the products wrapped.
    v = np.array([0, 0], dtype=np.int64)
    neighbours = np.array([[5 * 10**9, 1], [1, 5 * 10**9]], dtype=np.int64)
    assert vertex_label(v, neighbours) == (25 * 10**18 + 1) ** 2 - 10**20


def test_the_label_pairs_two_invariants() -> None:
    # Over its neighbours the labels of the vertices of 112|3|4e|5e|5e|e|:nnnnzzz fall into
    # 6 classes; paired with the determinant over all the other vertices, into 9, the orbits
    # of its group. Both parts are unchanged by a unimodular map.
    points = FeynmanIntegral.from_cnickel("112|3|4e|5e|5e|e|:nnnnzzz").newton_polytope.points
    data = polytope_data(points)
    V = np.array(data.vertices, dtype=np.int64)
    skeleton = polytope_skeleton(data)
    graph = label_skeleton(V, skeleton)
    local = {i: vertex_label(V[i], V[list(graph.neighbors(i))]) for i in graph.nodes}
    for i in graph.nodes:
        whole = vertex_label(V[i], np.delete(V, i, axis=0))
        pair = local[i] + whole
        assert graph.nodes[i]["label"] == pair * (pair + 1) // 2 + whole
    assert len(set(local.values())) == 6
    assert len({graph.nodes[i]["label"] for i in graph.nodes}) == 9
    U = np.eye(V.shape[1], dtype=np.int64)
    U[0, 1], U[3, 6], U[5, 2] = 1, -2, 3
    image = label_skeleton(V @ U.T + 5, skeleton)
    assert [image.nodes[i]["label"] for i in image.nodes] == [
        graph.nodes[i]["label"] for i in graph.nodes
    ]
