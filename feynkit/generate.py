"""
One-particle-irreducible Feynman graphs as canonical CNickel strings.

generate_graphs yields, once each, every graph with L loops, E propagators and
n legs that is connected, bridgeless (deleting one propagator leaves the
propagators connected; legs do not count) and of degree at least 3 at every
vertex, counting a self-loop twice and a leg once. Each comes with every
assignment of masses up to the automorphisms of the graph. The strings are
those Graph.cnickel() returns.

A vertex of internal degree 2 carries a leg, and suppressing those vertices
leaves a core of degree at least 3 with L loops, at most 2(L - 1) vertices and
3(L - 1) propagators, or for one loop a single vertex with a self-loop. The
generator runs this backwards: it enumerates the cores, replaces each core
propagator by a chain, places the legs, and keeps one graph per canonical
Nickel string.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import cache
from itertools import product

from .core.exceptions import ValidationError
from .core.graph import Graph, _canonical_colours, _canonical_labellings, _Colour, _pair_order

__all__ = ["MASS_ALPHABETS", "MAX_VERTICES", "generate_graphs", "mass_colourings"]

# "z": massless propagators only. "zn": each propagator massless or with a mass
# of its own. "shared": equal masses as well, written a, b, c, ...
MASS_ALPHABETS = ("z", "zn", "shared")

# CNickel labels vertices with one digit each.
MAX_VERTICES = 10

# The two ends (u, w), u <= w, of a propagator; vertices count from 0.
_Pair = tuple[int, int]


def _multiplicity(vertices: int, pairs: Iterable[_Pair]) -> list[list[int]]:
    """The number of propagators joining each pair of vertices, self-loops on the diagonal."""
    multiplicity = [[0] * vertices for _ in range(vertices)]
    for u, w in pairs:
        multiplicity[u][w] += 1
        if u != w:
            multiplicity[w][u] += 1
    return multiplicity


def _connected(vertices: int, pairs: Sequence[_Pair], skip: int = -1) -> bool:
    """Whether the propagators other than pairs[skip] connect all the vertices."""
    neighbours: list[set[int]] = [set() for _ in range(vertices)]
    for k, (u, w) in enumerate(pairs):
        if k != skip:
            neighbours[u].add(w)
            neighbours[w].add(u)
    seen = {0}
    stack = [0]
    while stack:
        for w in neighbours[stack.pop()] - seen:
            seen.add(w)
            stack.append(w)
    return len(seen) == vertices


def _bridgeless(vertices: int, pairs: Sequence[_Pair]) -> bool:
    """Connected, and still connected after deleting any one propagator."""
    return _connected(vertices, pairs) and all(
        u == w or _connected(vertices, pairs, k) for k, (u, w) in enumerate(pairs)
    )


def _has_cut_vertex(vertices: int, pairs: Sequence[_Pair]) -> bool:
    """Whether deleting some vertex disconnects the others; for graphs without self-loops."""
    if vertices < 3:
        return False
    for v in range(vertices):
        index = {w: i for i, w in enumerate(w for w in range(vertices) if w != v)}
        kept = [(index[a], index[b]) for a, b in pairs if v not in (a, b)]
        if not _connected(vertices - 1, kept):
            return True
    return False


def _multisets(pairs: Sequence[_Pair], count: int, vertices: int) -> Iterator[tuple[_Pair, ...]]:
    """The multisets of count pairs, listed in the order of pairs, in which every vertex has
    degree at least 3. pairs is sorted, so a vertex below the first vertex of the next pair
    has its final degree."""
    degree = [0] * vertices
    chosen: list[_Pair] = []

    def extend(start: int) -> Iterator[tuple[_Pair, ...]]:
        if len(chosen) == count:
            if min(degree) >= 3:
                yield tuple(chosen)
            return
        if sum(max(0, 3 - d) for d in degree) > 2 * (count - len(chosen)):
            return
        for k in range(start, len(pairs)):
            u, w = pairs[k]
            if any(degree[v] < 3 for v in range(u)):
                return
            chosen.append((u, w))
            degree[u] += 1
            degree[w] += 1
            yield from extend(k)
            chosen.pop()
            degree[u] -= 1
            degree[w] -= 1

    yield from extend(0)


@cache
def _cores(loops: int) -> tuple[tuple[int, tuple[_Pair, ...]], ...]:
    """The cores of a number of loops of at least 2, one per Nickel string, as (vertices,
    propagators): connected, bridgeless, self-loops allowed, every degree at least 3."""
    found: dict[str, tuple[int, tuple[_Pair, ...]]] = {}
    for vertices in range(1, 2 * loops - 1):
        pairs = [(u, w) for u in range(vertices) for w in range(u, vertices)]
        for core in _multisets(pairs, vertices + loops - 1, vertices):
            if _bridgeless(vertices, core):
                multiplicity = _multiplicity(vertices, core)
                nickel = _canonical_labellings(multiplicity, [0] * vertices)[0]
                found.setdefault(nickel, (vertices, core))
    return tuple(found[nickel] for nickel in sorted(found))


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """The tuples of parts positive integers with the given total."""
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _placements(low: Sequence[int], high: int, total: int) -> Iterator[list[int]]:
    """The leg vectors x with low[v] <= x[v] <= high and the given total."""
    if not low:
        if total == 0:
            yield []
        return
    for x in range(low[0], min(high, total) + 1):
        for rest in _placements(low[1:], high, total - x):
            yield [x, *rest]


def _candidates(
    loops: int,
    edges: int,
    legs: int,
    self_loops: bool,
    max_legs: int | None,
    one_vertex_irreducible: bool,
) -> Iterator[tuple[int, list[_Pair], list[int]]]:
    """Every graph of the block, with repeats, as (vertices, propagators, legs at each vertex):
    each core propagator becomes a chain, whose inner vertices carry at least one leg."""
    cores = ((1, ((0, 0),)),) if loops == 1 else _cores(loops)
    high = legs if max_legs is None else max_legs
    for core_vertices, core in cores:
        if (
            one_vertex_irreducible
            and loops > 1
            and (any(u == w for u, w in core) or _has_cut_vertex(core_vertices, core))
        ):
            continue
        degree = Counter(v for pair in core for v in pair)
        for lengths in _compositions(edges, len(core)):
            if not self_loops and any(
                u == w and k == 1 for (u, w), k in zip(core, lengths, strict=True)
            ):
                continue
            vertices = core_vertices
            pairs: list[_Pair] = []
            for (u, w), k in zip(core, lengths, strict=True):
                chain = [u, *range(vertices, vertices + k - 1), w]
                vertices += k - 1
                pairs.extend(
                    (min(a, b), max(a, b)) for a, b in zip(chain[:-1], chain[1:], strict=True)
                )
            low = [max(0, 3 - degree[v]) for v in range(core_vertices)]
            low += [1] * (vertices - core_vertices)
            for placement in _placements(low, high, legs):
                yield vertices, pairs, placement


def _words(edges: int, masses: str) -> Iterator[Sequence[_Colour]]:
    """Every colour word: "z" or "n" for each propagator, or for "shared" the number of its
    class of equal masses, a class of one being "n"."""
    if masses == "z":
        yield ["z"] * edges
        return
    if masses == "zn":
        yield from product("nz", repeat=edges)
        return

    # The set partitions of the propagators and a marker, as restricted growth
    # strings; the propagators in the marker's block are massless.
    def grow(blocks: list[int], used: int) -> Iterator[list[int]]:
        if len(blocks) == edges + 1:
            yield blocks
            return
        for b in range(used + 1):
            yield from grow([*blocks, b], max(used, b + 1))

    for blocks in grow([], 0):
        size = Counter(blocks[:edges])
        yield ["z" if b == blocks[edges] else "n" if size[b] == 1 else b for b in blocks[:edges]]


def _colourings(
    pairs: Sequence[_Pair],
    multiplicity: Sequence[Sequence[int]],
    labellings: Sequence[tuple[int, ...]],
    masses: str,
) -> set[str]:
    """The canonical colour strings of every colour word, computed as Graph.cnickel() does."""
    pair_orders = [_pair_order(order, multiplicity) for order in labellings]
    found = set()
    for word in _words(len(pairs), masses):
        colours: dict[_Pair, list[_Colour]] = defaultdict(list)
        for pair, colour in zip(pairs, word, strict=True):
            colours[pair].append(colour)
        found.add(_canonical_colours(pair_orders, colours))
    return found


def _block(
    loops: int,
    edges: int,
    legs: int,
    masses: str,
    self_loops: bool,
    max_legs: int | None,
    one_vertex_irreducible: bool,
) -> list[str]:
    """The canonical CNickel strings of one block, sorted."""
    topologies: dict[str, tuple[list[_Pair], list[list[int]], list[tuple[int, ...]]]] = {}
    for vertices, pairs, placement in _candidates(
        loops, edges, legs, self_loops, max_legs, one_vertex_irreducible
    ):
        multiplicity = _multiplicity(vertices, pairs)
        nickel, labellings = _canonical_labellings(multiplicity, placement)
        topologies.setdefault(nickel, (pairs, multiplicity, labellings))
    return sorted(
        f"{nickel}:{colours}"
        for nickel, (pairs, multiplicity, labellings) in topologies.items()
        for colours in _colourings(pairs, multiplicity, labellings, masses)
    )


def _values(value: int | Iterable[int], name: str, minimum: int) -> list[int]:
    """The distinct integers given, sorted; ValidationError for anything else."""
    values = [value] if isinstance(value, int) else list(value)
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
            raise ValidationError(f"{name} must be integers of at least {minimum}, got {v!r}")
    return sorted(set(values))


def _check_masses(masses: str) -> None:
    if masses not in MASS_ALPHABETS:
        raise ValidationError(f"masses must be one of {MASS_ALPHABETS}, got {masses!r}")


def generate_graphs(
    loops: int | Iterable[int],
    legs: int | Iterable[int],
    *,
    edges: int | Iterable[int] | None = None,
    masses: str = "zn",
    self_loops: bool = False,
    max_legs_per_vertex: int | None = 1,
    one_vertex_irreducible: bool = False,
) -> Iterator[str]:
    """
    The 1PI graphs with the given loops, legs and propagators, as canonical CNickel strings.

    A graph is kept when it is connected, bridgeless and of degree at least 3
    at every vertex, counting a self-loop twice and each leg once. Legs are
    unlabelled, so graphs that differ only in which momentum enters where are
    one string. Every graph comes once with each mass colouring up to its
    automorphisms, and each string s equals ``Graph.from_cnickel(s).cnickel()``.

    Parameters
    ----------
    loops, legs
        The numbers of loops L >= 1 and legs n >= 0, one or several each.
        Vacuum (n = 0) and one-leg graphs come only when asked for.
    edges
        The numbers of propagators E, or None for every E the rules allow:
        L <= E <= n + 3(L - 1) for L >= 2, and 2 <= E <= n for L = 1
        (E = 1 with self-loops).
    masses
        ``"zn"``: each propagator massless (z) or with a mass of its own (n).
        ``"z"``: massless only. ``"shared"``: equal masses as well, each class
        of two or more propagators with a common mass a letter, a, b, c in
        order of first appearance.
    self_loops
        Whether a propagator may join a vertex to itself. Massless self-loops
        are scaleless and still kept.
    max_legs_per_vertex
        The most legs at one vertex, or None for no limit. At generic
        kinematics several legs at a vertex give the Newton polytope of one.
    one_vertex_irreducible
        Keep only graphs whose propagators cannot be split into two non-empty
        sets sharing one vertex; the others have integrals that factorise.

    Returns
    -------
    Iterator[str]
        Blocks of (L, E, n) in increasing order, each sorted by code point.
        A block is computed when the iterator reaches it.

    Raises
    ------
    ValidationError
        When called, not at the first string: loops below 1, legs below 0,
        max_legs_per_vertex below 1, an unknown alphabet, or a block with more
        than MAX_VERTICES vertices, which CNickel cannot label.
    """
    loop_values = _values(loops, "loops", 1)
    leg_values = _values(legs, "legs", 0)
    edge_values = None if edges is None else _values(edges, "edges", 1)
    _check_masses(masses)
    if max_legs_per_vertex is not None:
        _values(max_legs_per_vertex, "max_legs_per_vertex", 1)

    blocks = []
    for L in loop_values:
        for n in leg_values:
            allowed = range(1 if self_loops else 2, n + 1) if L == 1 else range(L, n + 3 * L - 2)
            for E in allowed if edge_values is None else edge_values:
                if E in allowed:
                    blocks.append((L, E, n))
    for L, E, n in blocks:
        if E - L + 1 > MAX_VERTICES:
            raise ValidationError(
                f"the block of {L} loops, {E} propagators and {n} legs has {E - L + 1} "
                f"vertices; CNickel labels at most {MAX_VERTICES}"
            )

    def run() -> Iterator[str]:
        for L, E, n in sorted(blocks):
            yield from _block(
                L, E, n, masses, self_loops, max_legs_per_vertex, one_vertex_irreducible
            )

    return run()


def mass_colourings(topology: str, *, masses: str = "zn") -> list[str]:
    """
    The canonical CNickel strings of every mass colouring of a topology, sorted.

    ``topology`` is a Nickel string without colours, canonical or not. The
    colourings are counted up to the automorphisms of the graph, and ``masses``
    is as for :func:`generate_graphs`.
    """
    _check_masses(masses)
    if ":" in topology:
        raise ValidationError(f"mass_colourings takes a topology without colours, got {topology!r}")
    graph = Graph.from_cnickel(topology)
    vertices = graph.internal_vertices
    if vertices > MAX_VERTICES:
        raise ValidationError(
            f"{topology!r} has {vertices} vertices; CNickel labels at most {MAX_VERTICES}"
        )
    pairs = [(min(e.v1, e.v2) - 1, max(e.v1, e.v2) - 1) for e in graph.get_internal_edges()]
    placement = [0] * vertices
    for e in graph.get_external_edges():
        placement[e.v1 - 1] += 1
    multiplicity = _multiplicity(vertices, pairs)
    nickel, labellings = _canonical_labellings(multiplicity, placement)
    return sorted(
        f"{nickel}:{colours}" for colours in _colourings(pairs, multiplicity, labellings, masses)
    )
