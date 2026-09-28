"""
The canonical Nickel and CNickel strings against exhaustive search.

scan_cnickel is the definition of the CNickel string: the least (topology,
colours) pair over all V! vertex labellings and all names of the shared masses,
with parallel propagators sorted. It shares no code with feynkit.core.graph.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import pytest
import sympy as sp

from feynkit.core import Edge, Graph
from feynkit.core.constants import MASS_ASSUMPTIONS

LETTERS = "abcdefghijklmopqrtuvwxy"


def scan_cnickel(graph: Graph) -> str:
    """The least (topology, colours) pair over every labelling and every naming."""
    V = graph.internal_vertices
    edges = graph.get_internal_edges()
    zero = sp.Integer(0)
    masses = [zero if e.mass is None else e.mass for e in edges]
    count = Counter(m for m in masses if m != zero)
    classes = [m for m in count if count[m] > 1]
    legs = Counter(e.v1 for e in graph.get_external_edges())
    best: tuple[str, str] | None = None
    for names in itertools.permutations(LETTERS[: len(classes)]):
        code = dict(zip(classes, names, strict=True))
        colour = ["z" if m == zero else code.get(m, "n") for m in masses]
        for perm in itertools.permutations(range(1, V + 1)):
            label = {v: i for i, v in enumerate(perm)}
            groups: dict[tuple[int, int], list[str]] = defaultdict(list)
            for e, c in zip(edges, colour, strict=True):
                i, j = sorted((label[e.v1], label[e.v2]))
                groups[i, j].append(c)
            topology = "".join(
                "".join(str(j) * len(groups[i, j]) for j in range(i, V) if (i, j) in groups)
                + "e" * legs[perm[i]]
                + "|"
                for i in range(V)
            )
            colours = "".join("".join(sorted(groups[key])) for key in sorted(groups))
            if best is None or (topology, colours) < best:
                best = (topology, colours)
    assert best is not None
    return f"{best[0]}:{best[1]}"


# One and two loops, with parallel propagators, a self-loop and automorphism groups
# of orders 1 to 8.
TOPOLOGIES = [
    "11e|e|",
    "12e|2e|e|",
    "111e|e|",
    "12e|3e|3e|e|",
    "123|23|e|e|",
    "112e|3|3e|e|",
    "011e|e|",
]


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_cnickel_is_the_least_string_over_labellings_and_names(topology: str) -> None:
    edges = len(Graph.from_cnickel(topology).get_internal_edges())
    for word in itertools.product("znab", repeat=edges):
        graph = Graph.from_cnickel(f"{topology}:{''.join(word)}")
        cnickel = graph.cnickel()
        assert cnickel == scan_cnickel(graph)
        assert Graph.from_cnickel(cnickel).cnickel() == cnickel


@pytest.mark.parametrize(
    ("given", "canonical"),
    [
        ("12e|2e|e|:bbn", "12e|2e|e|:aan"),
        ("12e|2e|e|:aan", "12e|2e|e|:aan"),
        ("12e|2e|e|:nan", "12e|2e|e|:nnn"),
        ("11e|e|:ab", "11e|e|:nn"),
        ("111e|e|:bba", "111e|e|:aan"),
        ("123|23|e|e|:bbaaz", "123|23|e|e|:aabbz"),
    ],
)
def test_shared_masses_are_renamed(given: str, canonical: str) -> None:
    assert Graph.from_cnickel(given).cnickel() == canonical


def test_shared_masses_are_named_in_order_of_first_appearance() -> None:
    for word in itertools.product("zabc", repeat=5):
        colours = Graph.from_cnickel("123|23|e|e|:" + "".join(word)).cnickel().split(":")[1]
        letters = [c for c in dict.fromkeys(colours) if c not in "zn"]
        assert letters == list("abc"[: len(letters)])


def _triangle(masses: list[sp.Expr]) -> Graph:
    ends = [(1, 2), (2, 3), (3, 1)]
    edges = [
        Edge(idx=k + 1, v1=v1, v2=v2, is_internal=True, mass=m)
        for k, ((v1, v2), m) in enumerate(zip(ends, masses, strict=True))
    ]
    edges += [Edge(idx=4 + k, v1=k + 1, v2=4 + k, is_internal=False) for k in range(3)]
    return Graph(internal_vertices=3, external_legs=3, edges=edges)


@pytest.mark.parametrize("name", ["M", "m_1", "m_s"])
def test_equal_masses_share_a_letter_whatever_their_symbol(name: str) -> None:
    m = sp.Symbol(name, **MASS_ASSUMPTIONS)
    assert _triangle([m, m, sp.Integer(0)]).cnickel() == "12e|2e|e|:aaz"
    assert _triangle([m, sp.Integer(0), sp.Integer(0)]).cnickel() == "12e|2e|e|:nzz"


def test_more_classes_than_letters_is_not_implemented() -> None:
    # 48 self-loops at one vertex, in 24 pairs of equal masses.
    edges = [
        Edge(idx=k + 1, v1=1, v2=1, is_internal=True, mass=sp.Symbol(f"M{k // 2}"))
        for k in range(48)
    ]
    graph = Graph(internal_vertices=1, external_legs=0, edges=edges)
    assert graph.nickel_index() == "0" * 48 + "|"
    with pytest.raises(NotImplementedError, match="23 letters"):
        graph.cnickel()
