"""The exact vertices and 1-skeleton that the Liu-Cai labels are built on."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, ValidationError
from feynkit.normal_forms._invariants import (
    hull_vertex_indices,
    labelled_polytope_graph,
    to_integer_points,
    vertex_edge_graph,
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


@pytest.mark.parametrize("coordinate", [sp.Rational(1, 2), Fraction(1, 2), 0.5, sp.Float(0.5)])
def test_non_integer_points_raise(coordinate: object) -> None:
    # They were truncated: int(Rational(1, 2)) is 0.
    with pytest.raises(ValidationError, match="non-integer"):
        to_integer_points([(0, 0), (coordinate, 1)])


def test_integers_in_other_types() -> None:
    points = to_integer_points([(sp.Integer(2), 1.0), (Fraction(4, 2), sp.Rational(3, 1))])
    assert points.dtype == np.int64
    assert points.tolist() == [[2, 1], [2, 3]]
    assert to_integer_points(np.zeros((0, 3), dtype=np.int64)).shape == (0, 3)
