"""
The canonical Nickel and CNickel strings against exhaustive search.

scan_cnickel is the definition of the CNickel string: the least (topology,
colours) pair over all V! vertex labellings and all names of the shared masses,
with parallel propagators sorted. scan_labellings is the scan over all V!
labellings that feynkit made before its branch and bound, and
scan_graph_automorphisms the scan compute_graph_automorphisms made. None of
them shares code with feynkit.
"""

from __future__ import annotations

import itertools
import random
import time
import tracemalloc
from collections import Counter, defaultdict

import pytest
import sympy as sp

from feynkit.core import Edge, Graph
from feynkit.core.constants import MASS_ASSUMPTIONS
from feynkit.core.graph import _canonical_labellings, _least_naming, _tie_choices
from feynkit.normal_forms import compute_graph_automorphisms

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


def scan_labellings(
    multiplicity: list[list[int]], legs: list[int]
) -> tuple[str, set[tuple[int, ...]]]:
    """The least Nickel string over all V! labellings, and the labellings attaining it."""
    V = len(legs)
    best: str | None = None
    found: set[tuple[int, ...]] = set()
    for order in itertools.permutations(range(V)):
        label = {v: i for i, v in enumerate(order)}
        text = ""
        for i, v in enumerate(order):
            digits = sorted(
                label[w] for w in range(V) if label[w] >= i for _ in range(multiplicity[v][w])
            )
            text += "".join(map(str, digits)) + "e" * legs[v] + "|"
        if best is None or text < best:
            best, found = text, {order}
        elif text == best:
            found.add(order)
    assert best is not None
    return best, found


def scan_graph_automorphisms(graph: Graph) -> list[list[int]]:
    """compute_graph_automorphisms as it was: every vertex permutation, in order."""
    V = graph.internal_vertices
    adj: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for e in graph.get_internal_edges():
        mc = "z" if (e.mass is not None and e.mass == sp.Integer(0)) else "n"
        adj[e.v1].append((e.v2, mc))
        adj[e.v2].append((e.v1, mc))
    legs = Counter(e.v1 for e in graph.get_external_edges())

    def signature(perm: tuple[int, ...]) -> list[tuple[int, tuple[tuple[int, str], ...]]]:
        new = {old: k for k, old in enumerate(perm, start=1)}
        return [(legs[v], tuple(sorted((new[w], c) for w, c in adj[v]))) for v in perm]

    identity = signature(tuple(range(1, V + 1)))
    return [
        list(perm)
        for perm in itertools.permutations(range(1, V + 1))
        if signature(perm) == identity
    ]


def _matrix(graph: Graph) -> tuple[list[list[int]], list[int]]:
    V = graph.internal_vertices
    multiplicity = [[0] * V for _ in range(V)]
    for e in graph.get_internal_edges():
        multiplicity[e.v1 - 1][e.v2 - 1] += 1
        if e.v1 != e.v2:
            multiplicity[e.v2 - 1][e.v1 - 1] += 1
    legs = [0] * V
    for e in graph.get_external_edges():
        legs[e.v1 - 1] += 1
    return multiplicity, legs


def _random_graph(rng: random.Random, V: int) -> Graph:
    """A random multigraph, not always connected, with self-loops, legs and masses."""
    masses = [sp.Integer(0), sp.Symbol("M", **MASS_ASSUMPTIONS), sp.Symbol("K", **MASS_ASSUMPTIONS)]
    edges = []
    for k in range(rng.randint(V - 1, 2 * V)):
        v1, v2 = rng.randint(1, V), rng.randint(1, V)
        if v1 == v2 and rng.random() < 0.7:
            v2 = v2 % V + 1
        edges.append(Edge(idx=k + 1, v1=v1, v2=v2, is_internal=True, mass=rng.choice(masses)))
    legs = [rng.randint(1, V) for _ in range(rng.randint(0, V))]
    edges += [
        Edge(idx=len(edges) + j + 1, v1=v, v2=V + j + 1, is_internal=False)
        for j, v in enumerate(legs)
    ]
    return Graph(internal_vertices=V, external_legs=len(legs), edges=edges)


RANDOM_GRAPHS = [_random_graph(random.Random(k), 1 + k % 7) for k in range(200)]


@pytest.mark.parametrize("k", range(len(RANDOM_GRAPHS)))
def test_branch_and_bound_finds_every_least_labelling(k: int) -> None:
    multiplicity, legs = _matrix(RANDOM_GRAPHS[k])
    nickel, labellings = _canonical_labellings(multiplicity, legs)
    assert len(set(labellings)) == len(labellings)
    assert (nickel, set(labellings)) == scan_labellings(multiplicity, legs)


# The random graphs have small automorphism groups, so the search rarely meets
# long ties deep down. These graphs with eight vertices have large groups: the
# six cubic graphs, named by their triangles where that tells them apart (the
# 4-prism is the cube), and the 8-cycle with alternate propagators doubled. Each
# comes with the orders of its group without legs and with a leg at vertex 0.
# The scan takes about 0.6 s on each, so only the three with the largest groups
# run by default, a tie going to the larger group with the leg.
EIGHT_VERTICES = {
    "two-K4": ("01 02 03 12 13 23 45 46 47 56 57 67", 1152, 144),
    "cube": ("01 02 03 14 15 24 26 35 36 47 57 67", 48, 6),
    "two-diamonds": ("01 02 03 12 13 24 35 46 47 56 57 67", 16, 4),
    "mobius-ladder": ("01 04 12 15 23 26 34 37 45 56 67 70", 16, 2),
    "one-triangle": ("01 02 03 12 14 25 36 37 46 47 56 57", 12, 4),
    "two-triangles": ("01 02 03 12 14 25 34 36 47 56 57 67", 4, 1),
    "doubled-8-cycle": ("01 01 12 23 23 34 45 45 56 67 67 70", 8, 1),
}
MOST_SYMMETRIC = sorted(EIGHT_VERTICES, key=lambda name: EIGHT_VERTICES[name][1:])[-3:]


@pytest.mark.parametrize(
    ("ends", "legs", "automorphisms"),
    [
        pytest.param(
            ends,
            [leg, 0, 0, 0, 0, 0, 0, 0],
            automorphisms,
            id=f"{name}-{'leg' if leg else 'vacuum'}",
            marks=[] if name in MOST_SYMMETRIC else [pytest.mark.slow],
        )
        for name, (ends, *orders) in EIGHT_VERTICES.items()
        for leg, automorphisms in enumerate(orders)
    ],
)
def test_branch_and_bound_on_symmetric_graphs_with_eight_vertices(
    ends: str, legs: list[int], automorphisms: int
) -> None:
    multiplicity = [[0] * 8 for _ in range(8)]
    for u, w in ((int(pair[0]), int(pair[1])) for pair in ends.split()):
        multiplicity[u][w] += 1
        if u != w:
            multiplicity[w][u] += 1
    assert all(sum(row) == 3 for row in multiplicity)
    nickel, labellings = _canonical_labellings(multiplicity, legs)
    assert len(labellings) == len(set(labellings)) == automorphisms
    assert (nickel, set(labellings)) == scan_labellings(multiplicity, legs)


@pytest.mark.parametrize("k", range(0, len(RANDOM_GRAPHS), 4))
def test_random_graphs_against_the_definition(k: int) -> None:
    graph = RANDOM_GRAPHS[k]
    assert graph.cnickel() == scan_cnickel(graph)
    assert compute_graph_automorphisms(graph) == scan_graph_automorphisms(graph)


# Every string the rest of the test suite parses.
SUITE = [
    "00|:nz",
    "011e|e|:znn",
    "011e|e|:zzz",
    "012e|2e|e|:znnn",
    "012e|2e|e|:zzzz",
    "01e|e|:zn",
    "0|",
    "0|:n",
    "0|:z",
    "0|:z|",
    "111e|e|:nnn",
    "111e|e|:zzz",
    "11e|e|",
    "11e|e|:nn",
    "11e|e|:nz",
    "11e|e|:zn",
    "11e|e|:zz",
    "12ee|22e|e|:nnnn",
    "12ee|22e|e|:zzzz",
    "12e|22|e|:nzzz",
    "12e|23|3|e|:nnnnn",
    "12e|23|3|e|:nzzzz",
    "12e|23|3|e|:zzzzz",
    "12e|2e|e|",
    "12e|2e|e|:aab",
    "12e|2e|e|:nnn",
    "12e|2e|e|:nzz",
    "12e|2e|e|:sss",
    "12e|2e|e|:znz",
    "12e|2e|e|:zzz",
    "12e|3e|3e|e|:nnnn",
    "12e|3e|3e|e|:nnzz",
    "12e|3e|3e|e|:zzzz",
    "13e|2e|3e|e|:zzzz",
    "145|26|3e|4e|e|6e|e|:zzzzzzzz",
    "15e|24|3e|4e|5|e|:zzzzzzz",
    "e11|e|:n00|n|",
    "e11|e|:zzz|z|",
    "e|e|",
    "111e||:zzz",
    "111||:nnn",
    "112|2||:zzzz",
    "123|4e|4e|4e||:zzzzzz",
    "11122||e|:zzzzz",
]


@pytest.mark.parametrize("cnickel", SUITE)
def test_every_string_of_the_suite_against_the_definition(cnickel: str) -> None:
    graph = Graph.from_cnickel(cnickel)
    assert graph.cnickel() == scan_cnickel(graph)
    assert compute_graph_automorphisms(graph) == scan_graph_automorphisms(graph)


@pytest.mark.parametrize(
    ("given", "nickel"),
    [
        ("13e|2e|3e|e|", "12e|3e|3e|e|"),  # the box
        ("12e|23|3|e|", "123|23|e|e|"),  # the kite
        ("15e|24|3e|4e|5|e|", "123|45|4e|5e|e|e|"),  # the planar double box
        ("112|2|34|4e|e|", "112|2|34|4e|e|"),  # vertex 0 has degree 3, vertex 2 degree 4
        ("19e|2e|3e|4e|5e|6e|7e|8e|9e|e|", "12e|3e|4e|5e|6e|7e|8e|9e|9e|e|"),  # the 10-gon
    ],
)
def test_known_nickel_strings(given: str, nickel: str) -> None:
    assert Graph.from_cnickel(given).nickel_index() == nickel


def test_ten_vertices_are_labelled_and_eleven_are_not() -> None:
    ring = [Edge(idx=k + 1, v1=k + 1, v2=(k + 1) % 10 + 1, is_internal=True) for k in range(10)]
    graph = Graph(internal_vertices=10, external_legs=0, edges=ring)
    assert graph.cnickel() == "12|3|4|5|6|7|8|9|9||:zzzzzzzzzz"
    assert len(compute_graph_automorphisms(graph)) == 20
    ring = [Edge(idx=k + 1, v1=k + 1, v2=(k + 1) % 11 + 1, is_internal=True) for k in range(11)]
    graph = Graph(internal_vertices=11, external_legs=0, edges=ring)
    with pytest.raises(NotImplementedError, match="V <= 10"):
        graph.nickel_index()
    with pytest.raises(NotImplementedError, match="V <= 10"):
        graph.cnickel()
    assert len(compute_graph_automorphisms(graph)) == 22


def test_tie_orders_come_one_at_a_time() -> None:
    ties = [[4, 0, 2], [1, 3], [5]]
    assert list(_tie_choices(ties, itertools.permutations)) == [
        tuple(itertools.chain.from_iterable(choice))
        for choice in itertools.product(*(itertools.permutations(tie) for tie in ties))
    ]
    # product() would build all 10! orders of a tie of 10 before the first.
    tracemalloc.start()
    try:
        first = next(_tie_choices([list(range(10))], itertools.permutations))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert first == tuple(range(10))
    assert peak < 2**18


@pytest.mark.parametrize(
    "graph",
    [
        Graph.from_cnickel("12e|3|3e|e|:zzzz"),  # only the legs break the symmetry of the box
        Graph.from_cnickel("12e|3e|3e|e|:nnzz"),  # only the masses break it
        Graph.from_cnickel("112|2|34|4e|e|:nzzznzz"),
        *RANDOM_GRAPHS[::20],
    ],
)
def test_graph_automorphisms_check_every_candidate(
    graph: Graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offered every labelling, not only the least ones, the function still keeps
    # exactly the automorphisms.
    def every_labelling(
        multiplicity: list[list[int]], legs: list[int]
    ) -> tuple[str, list[tuple[int, ...]]]:
        return "", list(itertools.permutations(range(len(legs))))

    monkeypatch.setattr("feynkit.core.graph._canonical_labellings", every_labelling)
    assert compute_graph_automorphisms(graph) == scan_graph_automorphisms(graph)


def test_the_hub_of_a_wheel_does_not_build_its_tie_orders_at_once() -> None:
    # The seven neighbours of the hub tie. Building their 7! orders at once
    # took 0.7 MiB here and about 6 GB for the wheel with eleven spokes.
    rim = 7
    ends = [(1, k + 2) for k in range(rim)] + [(k + 2, (k + 1) % rim + 2) for k in range(rim)]
    edges = [Edge(idx=j + 1, v1=u, v2=w, is_internal=True) for j, (u, w) in enumerate(ends)]
    graph = Graph(internal_vertices=rim + 1, external_legs=0, edges=edges)
    tracemalloc.start()
    try:
        automorphisms = compute_graph_automorphisms(graph)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    dihedral = [
        [1, *((r + s * p) % rim + 2 for p in range(rim))] for r in range(rim) for s in (1, -1)
    ]
    assert automorphisms == sorted(dihedral)
    assert peak < 2**18


def exhaustive_naming(groups: list[list[str | int]]) -> str:
    """_least_naming as it was: every order of the classes that first appear in a group with
    equal counts, with no pruning of interchangeable classes."""
    best: list[str] = []

    def walk(g: int, names: dict[int, str], text: str) -> None:
        if best and text > best[0][: len(text)]:
            return
        if g == len(groups):
            best[:] = [text]
            return
        new = Counter(c for c in groups[g] if isinstance(c, int) and c not in names)
        ties = [
            sorted(c for c in new if new[c] == k) for k in sorted(set(new.values()), reverse=True)
        ]
        for choice in itertools.product(*(itertools.permutations(tie) for tie in ties)):
            named = dict(names)
            for c in itertools.chain.from_iterable(choice):
                named[c] = LETTERS[len(named)]
            codes = sorted(named[c] if isinstance(c, int) else c for c in groups[g])
            walk(g + 1, named, text + "".join(codes))

    walk(0, {}, "")
    return best[0]


def scan_naming(groups: list[list[str | int]]) -> str:
    """The least colour string over every name of every class."""
    classes = sorted({c for group in groups for c in group if isinstance(c, int)})
    return min(
        "".join(
            "".join(sorted(dict(zip(classes, names, strict=True)).get(c, c) for c in group))
            for group in groups
        )
        for names in itertools.permutations(LETTERS[: len(classes)])
    )


def _random_groups(rng: random.Random) -> list[list[str | int]]:
    """Groups of parallel propagators whose classes often tie, some of them interchangeable."""
    size = rng.randint(1, 5)
    profiles: list[tuple[int, ...]] = []
    for _ in range(rng.randint(1, 6)):
        if profiles and rng.random() < 0.4:
            profiles.append(rng.choice(profiles))
        else:
            profiles.append(tuple(rng.choice([0, 0, 1, 1, 2]) for _ in range(size)))
    ids = rng.sample(range(50), len(profiles))
    groups: list[list[str | int]] = []
    for g in range(size):
        group: list[str | int] = [
            c for c, profile in zip(ids, profiles, strict=True) for _ in range(profile[g])
        ]
        group += rng.choice(["", "z", "n", "zn", "nn"])
        rng.shuffle(group)
        groups.append(group or ["z"])
    return groups


@pytest.mark.parametrize("seed", range(300))
def test_least_naming_against_every_order_and_every_name(seed: int) -> None:
    groups = _random_groups(random.Random(seed))
    assert _least_naming(groups) == exhaustive_naming(groups) == scan_naming(groups)


@pytest.mark.parametrize("vertices", [1, 2])
def test_ten_pairs_of_equal_masses_are_named_at_once(vertices: int) -> None:
    # Ten classes first appear together, twice each, among parallel propagators.
    edges = [
        Edge(idx=k + 1, v1=1, v2=vertices, is_internal=True, mass=sp.Symbol(f"M{k // 2}"))
        for k in range(20)
    ]
    graph = Graph(internal_vertices=vertices, external_legs=0, edges=edges)
    start = time.perf_counter()
    cnickel = graph.cnickel()
    assert time.perf_counter() - start < 1
    assert cnickel.split(":")[1] == "aabbccddeeffgghhiijj"
