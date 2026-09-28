"""
Properties of the generated graphs that hold under any relabelling.

An example is a graph from the pool, a permutation of its vertices, one of its
propagator indices and one propagator. The relabelled graph is built from Edge
objects: propagators keep their mass symbols, their ends are mapped, and the
legs stay in order on the images of their vertices, so the kinematics are
unchanged. The pool is ordered by loops, propagators and legs, so shrinking
moves towards small graphs and identity permutations.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import NamedTuple, TypeVar

import pytest
import sympy as sp
from hypothesis import example, given
from hypothesis import strategies as st

from feynkit import FeynmanIntegral
from feynkit.core import Edge, Graph
from feynkit.generate import generate_graphs
from feynkit.polytope import polytope_data


class Relabelling(NamedTuple):
    cnickel: str
    vertices: tuple[int, ...]  # vertex v + 1 becomes vertex vertices[v] + 1
    edges: tuple[int, ...]  # propagator k + 1 becomes propagator edges[k] + 1
    propagator: int  # a propagator of the relabelled graph, from 1


# The graphs of tier 1 and those of tier 2 with at most six propagators, self-loops allowed.
POOL = [
    *generate_graphs(1, range(2, 7), self_loops=True),
    *generate_graphs(2, range(2, 5), edges=range(1, 7), self_loops=True),
]


def massless_self_loop(cnickel: str) -> bool:
    """Whether a massless propagator joins a vertex to itself; the Newton polytope is then
    not full-dimensional."""
    topology, colours = cnickel.split(":")
    k = 0
    for i, entry in enumerate(topology.split("|")):
        for c in entry.replace("e", ""):
            if int(c) == i and colours[k] == "z":
                return True
            k += 1
    return False


# The graphs of the pool with at most five propagators and a full-dimensional Newton polytope.
SMALL_POOL = [s for s in POOL if len(s.split(":")[1]) <= 5 and not massless_self_loop(s)]
# The graphs of both tiers with at most five propagators and equal masses.
SHARED_POOL = [
    *generate_graphs(1, range(2, 6), masses="shared"),
    *generate_graphs(2, range(2, 5), edges=range(1, 6), masses="shared"),
]


def test_pools() -> None:
    assert (len(POOL), len(SMALL_POOL), len(SHARED_POOL)) == (591, 237, 936)
    loops = [s for s in POOL if any(str(i) in e for i, e in enumerate(s.split("|")))]
    assert len(loops) == 110


@st.composite
def relabellings(draw: st.DrawFn, pool: list[str]) -> Relabelling:
    cnickel = draw(st.sampled_from(pool))
    topology, colours = cnickel.split(":")
    vertices = draw(st.permutations(range(topology.count("|"))))
    edges = draw(st.permutations(range(len(colours))))
    propagator = draw(st.integers(1, len(colours)))
    return Relabelling(cnickel, tuple(vertices), tuple(edges), propagator)


# Tier 1 up to the box, and the two-loop graphs with two legs and with four legs and
# seven propagators, massless and fully massive.
TWO_LOOPS = (
    "111e|e| 112e|2|e| 1122|e|e| 112|3|3e|e| 123|23|e|e| "
    "123|45|4e|5e|e|e| 123|4e|4e|5e|5|e| 123|24|e|5e|5e|e| 112|3|4e|5e|5e|e|"
)
EXAMPLE_GRAPHS = [*generate_graphs(1, range(2, 5))] + [
    f"{t}:{c * (len(t) - t.count('|') - t.count('e'))}" for t in TWO_LOOPS.split() for c in "zn"
]


def _examples() -> list[Relabelling]:
    """Three relabellings of each example graph, drawn once."""
    rng = random.Random(0)
    found = []
    for cnickel in EXAMPLE_GRAPHS:
        topology, colours = cnickel.split(":")
        V, E = topology.count("|"), len(colours)
        for _ in range(3):
            vertices = tuple(rng.sample(range(V), V))
            edges = tuple(rng.sample(range(E), E))
            found.append(Relabelling(cnickel, vertices, edges, rng.randint(1, E)))
    return found


EXAMPLES = _examples()
SMALL_EXAMPLES = [r for r in EXAMPLES if len(r.edges) <= 5]

Test = TypeVar("Test", bound=Callable[..., None])


def with_examples(examples: list[Relabelling]) -> Callable[[Test], Test]:
    def attach(test: Test) -> Test:
        for r in examples:
            test = example(r)(test)
        return test

    return attach


def relabel(r: Relabelling) -> tuple[Graph, Graph]:
    """The graph of the string, and the graph relabelled."""
    graph = Graph.from_cnickel(r.cnickel)
    V = graph.internal_vertices
    edges = [
        Edge(
            idx=r.edges[e.idx - 1] + 1,
            v1=r.vertices[e.v1 - 1] + 1,
            v2=r.vertices[e.v2 - 1] + 1,
            is_internal=True,
            mass=e.mass,
        )
        for e in graph.get_internal_edges()
    ]
    E = len(edges)
    edges += [
        Edge(idx=E + j + 1, v1=r.vertices[e.v1 - 1] + 1, v2=V + j + 1, is_internal=False)
        for j, e in enumerate(graph.get_external_edges())
    ]
    return graph, Graph(internal_vertices=V, external_legs=graph.external_legs, edges=edges)


def delete_edge(graph: Graph, idx: int) -> Graph:
    """The graph without propagator idx; indices, masses and legs are kept."""
    kept = [e for e in graph.get_internal_edges() if e.idx != idx]
    return Graph(
        internal_vertices=graph.internal_vertices,
        external_legs=graph.external_legs,
        edges=kept + graph.get_external_edges(),
    )


def contract_edge(graph: Graph, idx: int) -> Graph:
    """The graph with propagator idx contracted: its ends merge, their legs move to the
    merged vertex, and propagators parallel to it become self-loops. Indices and masses
    are kept."""
    e = next(e for e in graph.get_internal_edges() if e.idx == idx)
    keep, gone = min(e.v1, e.v2), max(e.v1, e.v2)

    def image(v: int) -> int:
        return keep if v == gone else v - (v > gone)

    edges = [
        Edge(idx=f.idx, v1=image(f.v1), v2=image(f.v2), is_internal=True, mass=f.mass)
        for f in graph.get_internal_edges()
        if f.idx != idx
    ]
    edges += [
        Edge(idx=f.idx, v1=image(f.v1), v2=image(f.v2), is_internal=False)
        for f in graph.get_external_edges()
    ]
    return Graph(
        internal_vertices=graph.internal_vertices - 1,
        external_legs=graph.external_legs,
        edges=edges,
    )


def integral(graph: Graph, like: FeynmanIntegral | None = None) -> FeynmanIntegral:
    """The integral of a graph, with the momentum products of another if given."""
    if like is not None:
        return FeynmanIntegral(graph, momentum_products=like.momentum_products)
    return FeynmanIntegral(graph, use_mandelstam=graph.external_legs >= 2)


def propagator(graph: Graph, idx: int) -> Edge:
    return next(e for e in graph.get_internal_edges() if e.idx == idx)


@given(relabellings(POOL + SHARED_POOL))
@with_examples(EXAMPLES)
def test_cnickel_is_invariant(r: Relabelling) -> None:
    _, relabelled = relabel(r)
    assert relabelled.cnickel() == r.cnickel


@given(relabellings(POOL))
@with_examples(EXAMPLES)
def test_the_newton_polytope_is_permuted(r: Relabelling) -> None:
    graph, relabelled = relabel(r)
    points = integral(graph).newton_polytope.points
    moved = integral(relabelled).newton_polytope.points
    # Coordinate k of a point is propagator k + 1.
    source = {new: old for old, new in enumerate(r.edges)}
    assert sorted(moved) == sorted(tuple(p[source[k]] for k in range(len(p))) for p in points)
    before, after = polytope_data(points), polytope_data(moved)
    assert after.normalized_volume == before.normalized_volume
    assert after.f_vector == before.f_vector


@given(relabellings(SMALL_POOL))
@with_examples(SMALL_EXAMPLES)
def test_relabelled_integrals_are_unimodularly_equivalent(r: Relabelling) -> None:
    graph, relabelled = relabel(r)
    assert integral(graph).is_unimodular_equivalent_to(integral(relabelled)).equivalent


@pytest.mark.xfail(
    strict=True,
    reason="the floating hull finds 53 and 55 vertices, not 52; "
    "passes once the normal forms use the exact vertices",
)
def test_the_massive_non_planar_double_box_is_equivalent_to_a_relabelling() -> None:
    r = Relabelling("123|4e|4e|5e|5|e|:nnnnnnz", (4, 2, 1, 0, 5, 3), (1, 0, 4, 5, 6, 2, 3), 1)
    graph, relabelled = relabel(r)
    assert integral(graph).is_unimodular_equivalent_to(integral(relabelled)).equivalent


@pytest.mark.xfail(
    strict=True,
    reason="the equivalence test needs a full-dimensional polytope; "
    "passes once the normal forms work in the lattice chart",
)
def test_a_scaleless_graph_is_equivalent_to_itself() -> None:
    r = Relabelling("012e|3e|3e|e|:znnnn", (0, 1, 2, 3), (0, 1, 2, 3, 4), 1)
    graph, relabelled = relabel(r)
    points = integral(graph).newton_polytope.points
    assert not polytope_data(points).is_full_dimensional
    assert integral(graph).is_unimodular_equivalent_to(integral(relabelled)).equivalent


@given(relabellings(POOL))
@with_examples(EXAMPLES)
def test_deletion_contraction(r: Relabelling) -> None:
    _, graph = relabel(r)
    fi = integral(graph)
    e = propagator(graph, r.propagator)
    a = graph.schwinger_parameters[e.idx]
    deleted = integral(delete_edge(graph, e.idx), fi).symanzik.u
    if e.v1 == e.v2:
        assert sp.expand(fi.symanzik.u - a * deleted) == 0
    else:
        contracted = integral(contract_edge(graph, e.idx), fi).symanzik.u
        assert sp.expand(fi.symanzik.u - a * deleted - contracted) == 0


@given(relabellings(POOL))
@with_examples(EXAMPLES)
def test_a_coordinate_face_is_the_contracted_graph(r: Relabelling) -> None:
    _, graph = relabel(r)
    e = propagator(graph, r.propagator)
    if e.v1 == e.v2:
        return
    fi = integral(graph)
    contracted = integral(contract_edge(graph, e.idx), fi)
    u = fi.symanzik.lp_parameters[e.idx - 1]
    assert u.name == f"u_{e.idx}"
    assert sp.expand(fi.symanzik.g.subs(u, 0) - contracted.symanzik.g) == 0
    points = fi.newton_polytope.points
    face = tuple(i for i, p in enumerate(points) if p[e.idx - 1] == 0)
    faces = {
        tuple(sorted(indices)): dimension for dimension, indices in polytope_data(points).faces
    }
    assert faces[face] == polytope_data(contracted.newton_polytope.points).dimension
