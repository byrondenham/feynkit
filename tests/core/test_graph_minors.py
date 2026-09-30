"""Graph.contract and Graph.delete: Feynman-graph minors.

Contracting a set of propagators identifies the ends of each and removes it;
deleting removes them. Legs move with their vertices, and a vertex left
without propagators is removed with its legs. The tests check the minors
against deletion-contraction for U, Kirchhoff's spanning-tree count and the
loop number.
"""

from __future__ import annotations

import networkx as nx
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from feynkit.core import Edge, Graph
from feynkit.core.exceptions import GraphTopologyError, ValidationError
from feynkit.polynomials.spanning_trees import spanning_tree_poly
from feynkit.polynomials.symanzik import _reverse_monomials

# Graphs with multiple edges, self-loops and bridges among them.
GRAPHS = {
    "bubble": "11e|e|:nn",
    "triangle": "12e|2e|e|:nnn",
    "box": "12e|3e|3e|e|:zzzz",
    "sunrise": "111e|e|:nnn",
    "parachute": "12e|22e|e|:nnnn",
    "kite": "12e|23|3|e|:nnnnn",
    "bubble_on_a_bridge": "1e|22|e|:nnn",
    "bubbles_on_a_bridge": "11e|2|33|e|:nnnnn",
    "tadpole_on_a_bridge": "01e|e|:nn",
    "figure_eight": "00e|:nn",
}


def _shape(graph: Graph) -> tuple[int, tuple, tuple]:
    """The graph as plain data: vertex count, propagators and legs."""
    internal = tuple(
        (e.idx, frozenset((e.v1, e.v2)), e.mass, e.nu) for e in graph.get_internal_edges()
    )
    legs = tuple((e.idx, e.v1, e.v2) for e in graph.get_external_edges())
    return graph.internal_vertices, internal, legs


def _u(graph: Graph) -> sp.Expr:
    """U of a connected graph, the complements of its spanning trees, as feynkit computes it."""
    edges = graph.get_internal_edges()
    pairs = [(e.v1 - 1, e.v2 - 1) for e in edges]
    a = [graph.schwinger_parameters[e.idx] for e in edges]
    return _reverse_monomials(spanning_tree_poly(graph.internal_vertices, pairs, a), a)


def _trees(graph: Graph) -> int:
    """The number of spanning trees, by Kirchhoff's theorem; self-loops do not enter."""
    ones = dict.fromkeys(graph.schwinger_parameters.values(), 1)
    laplacian = graph.calculate_laplacian(include_external=False).subs(ones)
    n = graph.internal_vertices
    return int(laplacian[: n - 1, : n - 1].det())


def _loops(graph: Graph) -> int:
    """E - V + c, the loop number of a graph with c connected components."""
    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(range(1, graph.internal_vertices + 1))
    multigraph.add_edges_from((e.v1, e.v2) for e in graph.get_internal_edges())
    components = nx.number_connected_components(multigraph)
    return len(graph.get_internal_edges()) - graph.internal_vertices + components


def _is_bridge(graph: Graph, idx: int) -> bool:
    return _loops(graph.delete([idx])) == _loops(graph)


class TestContract:
    def test_contracting_a_bubble_edge_gives_a_tadpole(self) -> None:
        graph = Graph.from_cnickel("11e|e|:nn")
        minor = graph.contract([1])
        assert minor.internal_vertices == 1
        (edge,) = minor.get_internal_edges()
        assert (edge.idx, edge.v1, edge.v2) == (2, 1, 1)
        assert edge.mass == sp.Symbol("m_2", nonnegative=True, real=True)
        assert [(e.idx, e.v1, e.v2) for e in minor.get_external_edges()] == [(3, 1, 2), (4, 1, 3)]

    def test_legs_move_with_their_vertices(self) -> None:
        graph = Graph.from_cnickel("12e|3e|3e|e|:zzzz")
        minor = graph.contract([2])
        # Edge 2 joins vertices 1 and 3 of the box; they become vertex 1, and vertex 4 becomes 3.
        assert minor.internal_vertices == 3
        assert [(e.idx, e.v1) for e in minor.get_external_edges()] == [
            (5, 1),
            (6, 2),
            (7, 1),
            (8, 3),
        ]
        assert [(e.idx, e.v1, e.v2) for e in minor.get_internal_edges()] == [
            (1, 1, 2),
            (3, 2, 3),
            (4, 1, 3),
        ]

    def test_a_contracted_cycle_leaves_self_loops_of_parallel_edges(self) -> None:
        graph = Graph.from_cnickel("111e|e|:nnn")
        minor = graph.contract([1])
        assert minor.internal_vertices == 1
        assert all(e.is_self_loop() for e in minor.get_internal_edges())
        assert minor.get_loop_count() == 2

    def test_contracting_a_cycle_contracts_its_vertices_to_one(self) -> None:
        graph = Graph.from_cnickel("12e|22e|e|:nnnn")
        minor = graph.contract([3, 4])
        assert minor.internal_vertices == 2
        assert [e.idx for e in minor.get_internal_edges()] == [1, 2]
        assert minor.get_loop_count() == 1

    def test_contracting_nothing_copies_the_graph(self) -> None:
        graph = Graph.from_cnickel("12e|23|3|e|:nnnnn")
        assert _shape(graph.contract([])) == _shape(graph)

    def test_the_energy_scale_is_kept(self) -> None:
        mu = sp.Symbol("Q", positive=True)
        edges = [
            Edge(idx=1, v1=1, v2=2, is_internal=True),
            Edge(idx=2, v1=1, v2=2, is_internal=True),
            Edge(idx=3, v1=1, v2=3, is_internal=False),
        ]
        graph = Graph(internal_vertices=2, external_legs=1, edges=edges, energy_scale=mu)
        assert graph.contract([1]).energy_scale == mu
        assert graph.delete([1]).energy_scale == mu


class TestDelete:
    def test_deleting_a_triangle_edge_leaves_a_path(self) -> None:
        graph = Graph.from_cnickel("12e|2e|e|:nnn")
        minor = graph.delete([3])
        assert _shape(minor)[0] == 3
        assert [e.idx for e in minor.get_internal_edges()] == [1, 2]
        assert minor.get_loop_count() == 0
        assert minor.external_legs == 3

    def test_a_vertex_left_without_propagators_is_removed_with_its_legs(self) -> None:
        graph = Graph.from_cnickel("12e|2e|e|:nnn")
        # Edges 1 and 2 are the two propagators at vertex 1, which carries leg 4.
        minor = graph.delete([1, 2])
        assert minor.internal_vertices == 2
        (edge,) = minor.get_internal_edges()
        assert (edge.idx, edge.v1, edge.v2) == (3, 1, 2)
        assert [(e.idx, e.v1, e.v2) for e in minor.get_external_edges()] == [(5, 1, 3), (6, 2, 4)]

    def test_deleting_a_bridge_disconnects(self) -> None:
        graph = Graph.from_cnickel(GRAPHS["bubbles_on_a_bridge"])
        minor = graph.delete([3])
        assert minor.internal_vertices == 4
        assert _loops(minor) == 2

    def test_deleting_a_bridge_to_a_leaf_removes_the_leaf(self) -> None:
        graph = Graph.from_cnickel("01e|e|:nn")
        minor = graph.delete([2])
        assert minor.internal_vertices == 1
        assert [(e.idx, e.v1, e.v2) for e in minor.get_external_edges()] == [(3, 1, 2)]


class TestErrors:
    @pytest.mark.parametrize("operation", ["contract", "delete"])
    def test_an_unknown_index_is_rejected(self, operation: str) -> None:
        graph = Graph.from_cnickel("11e|e|:nn")
        with pytest.raises(ValidationError, match="7"):
            getattr(graph, operation)([7])

    @pytest.mark.parametrize("operation", ["contract", "delete"])
    def test_a_leg_is_not_a_propagator(self, operation: str) -> None:
        graph = Graph.from_cnickel("11e|e|:nn")
        with pytest.raises(ValidationError, match="3"):
            getattr(graph, operation)([3])

    @pytest.mark.parametrize("operation", ["contract", "delete"])
    def test_removing_every_propagator_is_rejected(self, operation: str) -> None:
        graph = Graph.from_cnickel("11e|e|:nn")
        with pytest.raises(GraphTopologyError, match="no propagator"):
            getattr(graph, operation)([1, 2])


class TestIdentities:
    @pytest.mark.parametrize("name", GRAPHS)
    def test_deletion_contraction_for_u(self, name: str) -> None:
        graph = Graph.from_cnickel(GRAPHS[name])
        u = _u(graph)
        for edge in graph.get_internal_edges():
            a = graph.schwinger_parameters[edge.idx]
            if edge.is_self_loop():
                expected = a * _u(graph.delete([edge.idx]))
            elif _is_bridge(graph, edge.idx):
                expected = _u(graph.contract([edge.idx]))
            else:
                expected = a * _u(graph.delete([edge.idx])) + _u(graph.contract([edge.idx]))
            assert sp.expand(u - expected) == 0, (name, edge.idx)

    @pytest.mark.parametrize("name", GRAPHS)
    def test_spanning_tree_counts(self, name: str) -> None:
        graph = Graph.from_cnickel(GRAPHS[name])
        trees = _trees(graph)
        for edge in graph.get_internal_edges():
            if edge.is_self_loop():
                assert trees == _trees(graph.delete([edge.idx]))
            elif _is_bridge(graph, edge.idx):
                assert trees == _trees(graph.contract([edge.idx]))
            else:
                deleted, contracted = graph.delete([edge.idx]), graph.contract([edge.idx])
                assert trees == _trees(deleted) + _trees(contracted), (name, edge.idx)

    @pytest.mark.parametrize("name", GRAPHS)
    def test_loop_numbers(self, name: str) -> None:
        graph = Graph.from_cnickel(GRAPHS[name])
        loops = graph.get_loop_count()
        for edge in graph.get_internal_edges():
            loop = edge.is_self_loop()
            bridge = not loop and _is_bridge(graph, edge.idx)
            assert graph.contract([edge.idx]).get_loop_count() == loops - loop
            assert _loops(graph.delete([edge.idx])) == loops - (not bridge)

    def test_a_self_loop_contracts_as_it_deletes(self) -> None:
        graph = Graph.from_cnickel("01e|e|:nn")
        assert _shape(graph.contract([1])) == _shape(graph.delete([1]))


@st.composite
def _disjoint_sets(draw: st.DrawFn) -> tuple[str, frozenset[int], frozenset[int]]:
    name = draw(st.sampled_from(sorted(GRAPHS)))
    edges = [e.idx for e in Graph.from_cnickel(GRAPHS[name]).get_internal_edges()]
    fate = draw(st.lists(st.sampled_from("ckd"), min_size=len(edges), max_size=len(edges)))
    contract = frozenset(e for e, f in zip(edges, fate, strict=True) if f == "c")
    delete = frozenset(e for e, f in zip(edges, fate, strict=True) if f == "d")
    return name, contract, delete


@given(_disjoint_sets())
def test_contraction_and_deletion_commute(case: tuple[str, frozenset[int], frozenset[int]]) -> None:
    name, contract, delete = case
    graph = Graph.from_cnickel(GRAPHS[name])
    try:
        first = graph.delete(delete).contract(contract)
    except GraphTopologyError:
        with pytest.raises(GraphTopologyError):
            graph.contract(contract).delete(delete)
        return
    assert _shape(first) == _shape(graph.contract(contract).delete(delete))


@given(_disjoint_sets())
def test_contraction_is_one_step(case: tuple[str, frozenset[int], frozenset[int]]) -> None:
    name, first, second = case
    graph = Graph.from_cnickel(GRAPHS[name])
    try:
        together = graph.contract(first | second)
    except GraphTopologyError:
        return
    assert _shape(together) == _shape(graph.contract(first).contract(second))
