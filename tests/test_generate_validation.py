"""
The graph generator against an independent enumeration, closed forms and OEIS A000029.

The enumeration shares only the rules with feynkit.generate: no cores, no
subdivision, no Nickel strings and no lexicographic minimum. For each block it
fills the multiplicity matrices of each vector of internal degrees row by row,
keeps the connected, bridgeless ones and removes isomorphic copies with
networkx. The colourings are the orbits of the colour words under the
automorphisms networkx finds, and Burnside's lemma counts them again.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from fractions import Fraction
from functools import cache
from typing import NamedTuple

import networkx as nx
import pytest
from networkx.algorithms.isomorphism import GraphMatcher

from feynkit.core import Graph
from feynkit.core.graph import _canonical_labellings
from feynkit.generate import generate_graphs, mass_colourings
from tests.core.test_canonical import scan_labellings

LETTERS = "abcdefghijklmopqrtuvwxy"


class Representative(NamedTuple):
    vertices: int
    pairs: tuple[tuple[int, int], ...]  # the ends of each propagator, u <= w
    legs: tuple[int, ...]


# --- topologies ------------------------------------------------------------------


def leg_vectors(vertices: int, legs: int, cap: int) -> Iterator[tuple[int, ...]]:
    """Non-increasing leg vectors with the given total and entries at most cap."""

    def rest(i: int, remaining: int, upper: int) -> Iterator[tuple[int, ...]]:
        if i == vertices:
            if remaining == 0:
                yield ()
            return
        for x in range(min(upper, remaining), -1, -1):
            for tail in rest(i + 1, remaining - x, x):
                yield (x, *tail)

    yield from rest(0, legs, cap)


def degree_vectors(edges: int, legs: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Internal degrees of at least 2 and at least 3 minus the legs, with total 2E,
    non-increasing among vertices with equal legs."""
    low = [max(2, 3 - x) for x in legs]

    def rest(i: int, remaining: int, upper: int) -> Iterator[tuple[int, ...]]:
        if i == len(legs):
            if remaining == 0:
                yield ()
            return
        top = remaining - sum(low[i + 1 :])
        if i > 0 and legs[i] == legs[i - 1]:
            top = min(top, upper)
        for d in range(low[i], top + 1):
            for tail in rest(i + 1, remaining - d, d):
                yield (d, *tail)

    yield from rest(0, 2 * edges, 2 * edges)


def matrices(degrees: tuple[int, ...], self_loops: bool) -> Iterator[list[list[int]]]:
    """The symmetric multiplicity matrices with these degrees, filled row by row; a diagonal
    entry, the number of self-loops, counts twice."""
    V = len(degrees)
    A = [[0] * V for _ in range(V)]
    rest = list(degrees)

    def fill(i: int, j: int) -> Iterator[list[list[int]]]:
        if i == V:
            yield [row[:] for row in A]
            return
        if j == V:
            if rest[i] == 0:
                yield from fill(i + 1, i + 1)
            return
        if i == j:
            options = range(rest[i] // 2 if self_loops else 0, -1, -1)
            step = 2
        else:
            later = sum(rest[k] for k in range(j + 1, V))
            options = range(min(rest[i], rest[j]), max(0, rest[i] - later) - 1, -1)
            step = 1
        for k in options:
            A[i][j] = A[j][i] = k
            rest[i] -= step * k
            if i != j:
                rest[j] -= k
            yield from fill(i, j + 1)
            rest[i] += step * k
            if i != j:
                rest[j] += k
        A[i][j] = A[j][i] = 0

    yield from fill(0, 0)


def multigraph(rep: Representative) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_nodes_from((v, {"legs": rep.legs[v]}) for v in range(rep.vertices))
    G.add_edges_from(rep.pairs)
    return G


def bridgeless(G: nx.MultiGraph) -> bool:
    """Connected, and no single propagator is a bridge: a bridge of the simple graph
    underneath that carries one propagator."""
    simple = nx.Graph(G)
    if not nx.is_connected(simple):
        return False
    return all(G.number_of_edges(u, w) > 1 for u, w in nx.bridges(simple))


def same_legs(a: dict[str, int], b: dict[str, int]) -> bool:
    return a["legs"] == b["legs"]


def signature(rep: Representative) -> tuple[object, ...]:
    """An isomorphism invariant: each vertex's degree, legs and self-loops, with those of
    its neighbours and the number of propagators to each."""
    degree = Counter(v for pair in rep.pairs for v in pair)
    joins = Counter(rep.pairs)
    own = [(degree[v], rep.legs[v], joins[v, v]) for v in range(rep.vertices)]
    around: dict[int, list[tuple[object, ...]]] = defaultdict(list)
    for (u, w), m in joins.items():
        if u != w:
            around[u].append((*own[w], m))
            around[w].append((*own[u], m))
    return tuple(sorted((own[v], tuple(sorted(around[v]))) for v in range(rep.vertices)))


@cache
def brute_topologies(
    loops: int, edges: int, legs: int, self_loops: bool, max_legs: int | None
) -> tuple[Representative, ...]:
    V = edges - loops + 1
    if V < 1:
        return ()
    buckets: dict[tuple[tuple[int, int], ...], list[nx.MultiGraph]] = defaultdict(list)
    found = []
    for placement in leg_vectors(V, legs, legs if max_legs is None else max_legs):
        for degrees in degree_vectors(edges, placement):
            for A in matrices(degrees, self_loops):
                pairs = tuple((u, w) for u in range(V) for w in range(u, V) for _ in range(A[u][w]))
                rep = Representative(V, pairs, placement)
                G = multigraph(rep)
                if not bridgeless(G):
                    continue
                bucket = buckets[signature(rep)]
                if any(nx.is_isomorphic(G, H, node_match=same_legs) for H in bucket):
                    continue
                bucket.append(G)
                found.append(rep)
    return tuple(found)


# --- colourings ----------------------------------------------------------------------


def subdivided(rep: Representative) -> nx.Graph:
    """One node per vertex and one per propagator, joined to its ends; self-loops marked."""
    S = nx.Graph()
    for v in range(rep.vertices):
        S.add_node(("v", v), kind=("v", rep.legs[v]))
    for k, (u, w) in enumerate(rep.pairs):
        S.add_node(("e", k), kind=("e", u == w))
        S.add_edge(("e", k), ("v", u))
        S.add_edge(("e", k), ("v", w))
    return S


def same_kind(a: dict[str, object], b: dict[str, object]) -> bool:
    return a["kind"] == b["kind"]


@cache
def edge_group(rep: Representative) -> tuple[tuple[int, ...], ...]:
    """The permutations of the propagators induced by automorphisms, parallel ones swapping."""
    S = subdivided(rep)
    maps = GraphMatcher(S, S, node_match=same_kind).isomorphisms_iter()
    return tuple(sorted({tuple(m["e", k][1] for k in range(len(rep.pairs))) for m in maps}))


def set_partitions(items: list[int]) -> Iterator[list[list[int]]]:
    if not items:
        yield []
        return
    for part in set_partitions(items[1:]):
        yield [[items[0]], *part]
        for i in range(len(part)):
            yield [*part[:i], [items[0], *part[i]], *part[i + 1 :]]


def words(edges: int, masses: str) -> Iterator[tuple[str, ...]]:
    if masses == "zn":
        yield from itertools.product("zn", repeat=edges)
        return
    for part in set_partitions(list(range(edges + 1))):
        word = ["z"] * edges
        letters = iter(LETTERS)
        for block in part:
            if edges not in block:
                code = "n" if len(block) == 1 else next(letters)
                for k in block:
                    word[k] = code
        yield tuple(word)


def rename(word: tuple[str, ...]) -> str:
    """Shared-mass letters renamed in order of first appearance."""
    names: dict[str, str] = {}
    return "".join(c if c in "zn" else names.setdefault(c, LETTERS[len(names)]) for c in word)


def orbit_key(rep: Representative, word: tuple[str, ...]) -> str:
    keys = []
    for g in edge_group(rep):
        image = [""] * len(word)
        for k, c in enumerate(word):
            image[g[k]] = c
        keys.append(rename(tuple(image)))
    return min(keys)


def cycles(g: tuple[int, ...]) -> int:
    seen: set[int] = set()
    count = 0
    for k in range(len(g)):
        if k not in seen:
            count += 1
            while k not in seen:
                seen.add(k)
                k = g[k]
    return count


def burnside(rep: Representative, masses: str) -> int:
    """The number of orbits: the mean number of words, or of set partitions of the
    propagators and a fixed marker for the massless ones, that each permutation fixes."""
    group = edge_group(rep)
    E = len(rep.pairs)
    if masses == "zn":
        fixed = sum(2 ** cycles(g) for g in group)
    else:
        fixed = 0
        for part in set_partitions(list(range(E + 1))):
            blocks = {frozenset(b) for b in part}
            for g in group:
                full = (*g, E)
                fixed += all(frozenset(full[k] for k in b) in blocks for b in blocks)
    count = Fraction(fixed, len(group))
    assert count.denominator == 1
    return int(count)


def brute_orbits(rep: Representative, masses: str) -> set[str]:
    return {orbit_key(rep, word) for word in words(len(rep.pairs), masses)}


def representative_of(cnickel: str) -> tuple[Representative, str]:
    """The graph of a CNickel string, and its colours in propagator order."""
    graph = Graph.from_cnickel(cnickel)
    legs = [0] * graph.internal_vertices
    for e in graph.get_external_edges():
        legs[e.v1 - 1] += 1
    pairs = tuple((min(e.v1, e.v2) - 1, max(e.v1, e.v2) - 1) for e in graph.get_internal_edges())
    return Representative(graph.internal_vertices, pairs, tuple(legs)), cnickel.partition(":")[2]


@cache
def place(topology: str, reps: tuple[Representative, ...]) -> tuple[int, tuple[int, ...]]:
    """The index of the one representative isomorphic to the topology, legs matched, and
    the representative's propagator that each propagator of the topology becomes."""
    rep, _ = representative_of(topology)
    S = subdivided(rep)
    found = []
    for index, other in enumerate(reps):
        if signature(other) == signature(rep):
            match = GraphMatcher(S, subdivided(other), node_match=same_kind)
            if match.is_isomorphic():
                image = tuple(match.mapping["e", k][1] for k in range(len(rep.pairs)))
                found.append((index, image))
    assert len(found) == 1, (topology, found)
    return found[0]


def locate(cnickel: str, reps: tuple[Representative, ...]) -> tuple[int, str]:
    """The representative of the graph, and the orbit key of its colouring carried over."""
    topology, _, colours = cnickel.partition(":")
    index, image = place(topology, reps)
    word = [""] * len(colours)
    for k, c in enumerate(colours):
        word[image[k]] = c
    return index, orbit_key(reps[index], tuple(word))


# --- the tiers, with and without self-loops and a limit on legs --------------------

# (loops, legs): tier 1 from the tadpole to the hexagon, tier 2 up to four legs,
# vacuum and one-leg graphs included.
TIER_BLOCKS = [(1, n) for n in range(7)] + [(2, n) for n in range(5)]
OPTIONS = [
    pytest.param(False, 1, id="default"),
    pytest.param(True, 1, id="self-loops"),
    pytest.param(False, None, id="any-legs"),
]


def block_edges(loops: int, legs: int, self_loops: bool) -> range:
    if loops == 1:
        return range(1 if self_loops else 2, legs + 1)
    return range(loops, legs + 3 * loops - 2)


@pytest.mark.parametrize("masses", ["zn", "shared"])
@pytest.mark.parametrize(("self_loops", "max_legs"), OPTIONS)
@pytest.mark.parametrize(("loops", "legs"), TIER_BLOCKS)
def test_tiers_agree_with_the_brute_force(
    loops: int, legs: int, self_loops: bool, max_legs: int | None, masses: str
) -> None:
    for edges in block_edges(loops, legs, self_loops):
        reps = brute_topologies(loops, edges, legs, self_loops, max_legs)
        generated = list(
            generate_graphs(
                loops,
                legs,
                edges=edges,
                masses=masses,
                self_loops=self_loops,
                max_legs_per_vertex=max_legs,
            )
        )
        expected = {(i, key) for i, rep in enumerate(reps) for key in brute_orbits(rep, masses)}
        located = Counter(locate(s, reps) for s in generated)
        assert set(located) == expected, (loops, edges, legs)
        assert max(located.values(), default=1) == 1
        assert len(expected) == sum(burnside(rep, masses) for rep in reps)


# --- three loops -----------------------------------------------------------------


# The blocks with eight vertices take about 10 s.
THREE_LOOP_BLOCKS = [
    pytest.param(edges, legs, marks=[pytest.mark.slow] if edges == 10 else [], id=f"{edges}-{legs}")
    for legs in (2, 3, 4)
    for edges in block_edges(3, legs, False)
]


@pytest.mark.parametrize(("edges", "legs"), THREE_LOOP_BLOCKS)
def test_three_loops_agree_with_the_brute_force(edges: int, legs: int) -> None:
    reps = brute_topologies(3, edges, legs, False, 1)
    topologies = list(generate_graphs(3, legs, edges=edges, masses="z"))
    assert Counter(locate(s, reps)[0] for s in topologies) == Counter(range(len(reps)))
    zn = sum(1 for _ in generate_graphs(3, legs, edges=edges))
    assert zn == sum(burnside(rep, "zn") for rep in reps)


@pytest.mark.slow
def test_three_loop_totals() -> None:
    graphs = list(generate_graphs(3, range(2, 5)))
    assert len(graphs) == 74480
    assert len({s.split(":")[0] for s in graphs}) == 412


# --- closed forms ---------------------------------------------------------------

# OEIS A000029, binary bracelets of n beads, from n = 0.
A000029 = [1, 2, 3, 4, 6, 8, 13, 18, 30, 46, 78]


def bracelets(n: int) -> int:
    """Burnside over the dihedral group of order 2n acting on the n propagators."""
    phi = [sum(math.gcd(k, d) == 1 for k in range(1, d + 1)) for d in range(n + 1)]
    rotations = Fraction(sum(phi[d] * 2 ** (n // d) for d in range(1, n + 1) if n % d == 0), 2 * n)
    reflections = Fraction(2 ** ((n - 1) // 2)) if n % 2 else Fraction(3 * 2 ** (n // 2), 4)
    count = rotations + reflections
    assert count.denominator == 1
    return int(count)


@pytest.mark.parametrize("n", range(2, 11))
def test_one_loop_colourings_are_bracelets(n: int) -> None:
    graphs = list(generate_graphs(1, n))
    assert len({s.split(":")[0] for s in graphs}) == 1
    assert len(graphs) == bracelets(n) == A000029[n]


def partitions_into_three(E: int) -> int:
    return sum(1 for a in range(1, E) for b in range(a, E) if E - a - b >= b)


def partitions_into_two_of_at_least_two(E: int) -> int:
    return sum(1 for a in range(2, E) if E - a >= a)


def two_loop_topologies(E: int, n: int) -> int:
    return partitions_into_three(E) * (0 <= n - E + 3 <= 2) + partitions_into_two_of_at_least_two(
        E
    ) * (0 <= n - E + 2 <= 1)


def test_two_loop_topologies_follow_the_closed_form() -> None:
    blocks = 0
    for n in range(9):
        for E in range(2, 12):
            count = len({s.split(":")[0] for s in generate_graphs(2, n, edges=E, masses="z")})
            assert count == two_loop_topologies(E, n), (E, n)
            blocks += 1
    assert blocks == 90


TWO_LOOPS = {
    2: "111e|e| 112e|2|e| 1122|e|e| 112|3|3e|e| 123|23|e|e|",
    3: "112e|2e|e| 1122e|e|e| 112e|3|3e|e| 123e|23|e|e| 1123|e|3e|e| 112|3|4e|4e|e| "
    "123|24|e|4e|e| 123|4e|4e|4e||",
    4: "123|45|4e|5e|e|e| 123|4e|4e|5e|5|e| 123|24|e|5e|5e|e| 112|3|4e|5e|5e|e|",
}


@pytest.mark.parametrize("n", [2, 3, 4])
def test_two_loop_lists(n: int) -> None:
    edges = 7 if n == 4 else None
    found = {s.split(":")[0] for s in generate_graphs(2, n, edges=edges, masses="z")}
    assert found == set(TWO_LOOPS[n].split())


@pytest.mark.parametrize(
    ("topology", "count"), [("111e|e|", 4), ("123|23|e|e|", 14), ("112|3|3e|e|", 18)]
)
def test_colourings_by_burnside(topology: str, count: int) -> None:
    rep, _ = representative_of(topology)
    assert len(mass_colourings(topology)) == burnside(rep, "zn") == count


# --- every string -----------------------------------------------------------------


# The "shared" strings take about 6 s for each set of options.
ROUND_TRIPS = [
    pytest.param(False, 1, "zn", id="default-zn"),
    pytest.param(True, 1, "zn", id="self-loops-zn"),
    pytest.param(False, None, "zn", id="any-legs-zn"),
    pytest.param(False, 1, "shared", id="default-shared"),
    pytest.param(True, 1, "shared", id="self-loops-shared", marks=pytest.mark.slow),
    pytest.param(False, None, "shared", id="any-legs-shared", marks=pytest.mark.slow),
]


@pytest.mark.parametrize(("self_loops", "max_legs", "masses"), ROUND_TRIPS)
def test_every_string_round_trips_and_obeys_the_rules(
    self_loops: bool, max_legs: int | None, masses: str
) -> None:
    checked = set()
    for loops, legs in TIER_BLOCKS:
        for edges in block_edges(loops, legs, self_loops):
            for s in generate_graphs(
                loops,
                legs,
                edges=edges,
                masses=masses,
                self_loops=self_loops,
                max_legs_per_vertex=max_legs,
            ):
                assert Graph.from_cnickel(s).cnickel() == s
                topology = s.partition(":")[0]
                if topology in checked:
                    continue
                checked.add(topology)
                rep, _ = representative_of(topology)
                assert (rep.vertices, len(rep.pairs), sum(rep.legs)) == (
                    edges - loops + 1,
                    edges,
                    legs,
                )
                assert bridgeless(multigraph(rep))
                degree = Counter(v for pair in rep.pairs for v in pair)
                assert all(degree[v] + rep.legs[v] >= 3 for v in range(rep.vertices))
                assert max_legs is None or max(rep.legs) <= max_legs
                assert self_loops or all(u != w for u, w in rep.pairs)


def _digest() -> str:
    text = "\n".join(
        generate_graphs([1, 2], range(5), masses="shared", self_loops=True, edges=range(1, 6))
    )
    return hashlib.sha256(text.encode()).hexdigest()


def test_output_does_not_depend_on_the_hash_seed() -> None:
    code = "from tests.test_generate_validation import _digest; print(_digest())"
    digests = {_digest()}
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        run = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        digests.add(run.stdout.strip())
    assert len(digests) == 1


# --- the canonical labelling against the full scan ------------------------------


# The topologies of the tiers, with self-loops and any legs per vertex or neither, and
# of three loops with 2 to 4 legs: 410 with at most six vertices, 126 with seven or eight.
COUNT_UP_TO_SIX = 410
COUNT_SEVEN_AND_EIGHT = 126


def generated_topologies(max_vertices: int, min_vertices: int = 1) -> list[str]:
    found = set()
    for self_loops, max_legs in [(True, None), (False, 1)]:
        for loops, legs in TIER_BLOCKS:
            found |= {
                s.split(":")[0]
                for s in generate_graphs(
                    loops, legs, masses="z", self_loops=self_loops, max_legs_per_vertex=max_legs
                )
            }
    found |= {s.split(":")[0] for s in generate_graphs(3, range(2, 5), masses="z")}
    return sorted(
        t for t in found if min_vertices <= Graph.from_cnickel(t).internal_vertices <= max_vertices
    )


def check_against_the_scan(topologies: list[str]) -> None:
    rng = random.Random(0)
    for t in topologies:
        rep, _ = representative_of(t)
        perm = list(range(rep.vertices))
        rng.shuffle(perm)
        V = rep.vertices
        multiplicity = [[0] * V for _ in range(V)]
        for u, w in rep.pairs:
            multiplicity[perm[u]][perm[w]] += 1
            if u != w:
                multiplicity[perm[w]][perm[u]] += 1
        legs = [0] * V
        for v in range(V):
            legs[perm[v]] = rep.legs[v]
        nickel, labellings = _canonical_labellings(multiplicity, legs)
        assert nickel == t
        assert (nickel, set(labellings)) == scan_labellings(multiplicity, legs)


def test_canonical_labellings_match_the_scan_up_to_six_vertices() -> None:
    topologies = generated_topologies(6)
    assert len(topologies) == COUNT_UP_TO_SIX
    check_against_the_scan(topologies)


@pytest.mark.slow
def test_canonical_labellings_match_the_scan_at_seven_and_eight_vertices() -> None:
    topologies = generated_topologies(8, 7)
    assert len(topologies) == COUNT_SEVEN_AND_EIGHT
    check_against_the_scan(topologies)
