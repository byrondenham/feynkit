"""
Graph representation for Feynman diagrams.

This module provides the Graph class representing complete Feynman diagrams
with methods to compute Laplacians and related polynomials.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain, permutations, product

import sympy as sp

from .constants import (
    DEFAULT_ENERGY_SCALE,
    EXTERNAL_PARAM_PREFIX,
    SCALE_ASSUMPTIONS,
    SCHWINGER_PARAM_PREFIX,
)
from .edge import Edge
from .exceptions import GraphTopologyError
from .validation import (
    validate_edge_connectivity,
    validate_nonnegative_integer,
    validate_positive_integer,
)

# -----------------------------------------------------------------------------
# Mass-code helpers
# -----------------------------------------------------------------------------

# Codes that map to zero mass.
_MASSLESS_CODES: frozenset[str] = frozenset(("0", "z"))

# Single-character codes that denote a *shared* symbolic mass m_{code}: the
# lowercase letters a-y, excluding the reserved codes 'n' (unique non-zero) and
# 's' (special shared). Digits are not labels: m_1 is also the mass that 'n'
# gives propagator 1.
_LABELED_CODES: frozenset[str] = frozenset(
    c for c in "abcdefghijklmnopqrstuvwxy" if c not in ("n", "s")
)

_ALL_VALID_CODES: frozenset[str] = _MASSLESS_CODES | {"n", "s"} | _LABELED_CODES


def _mass_from_code(mc: str, edge_idx: int, mass_assumptions: dict) -> sp.Expr:
    """Map a single mass-code character to a SymPy mass expression."""
    if mc in _MASSLESS_CODES:
        return sp.Integer(0)
    if mc == "n":
        return sp.Symbol(f"m_{edge_idx}", **mass_assumptions)
    if mc == "s":
        return sp.Symbol("m_s", **mass_assumptions)
    # Labeled code: all edges sharing the same label share the same symbol.
    return sp.Symbol(f"m_{mc}", **mass_assumptions)


# -----------------------------------------------------------------------------
# Canonical labelling
# -----------------------------------------------------------------------------

# Letters for the classes of propagators that share a mass, in increasing order:
# a to y without the reserved codes n and s.
_SHARED_LETTERS = "abcdefghijklmopqrtuvwxy"

# The colour of a propagator before the shared masses are named: "z" if it is
# massless, "n" if no other propagator has its mass, otherwise the number of its
# class of equal masses.
_Colour = str | int


def _nickel_entry(
    v: int, label: Mapping[int, int], multiplicity: Sequence[Sequence[int]], legs: Sequence[int]
) -> str:
    """The Nickel entry of vertex v: the labels of its neighbours with labels at least its own,
    ascending and repeated for parallel propagators, then one 'e' per leg. Only v and its
    neighbours need labels."""
    i = label[v]
    digits = sorted(
        label[w] for w, m in enumerate(multiplicity[v]) if m and label[w] >= i for _ in range(m)
    )
    return "".join(map(str, digits)) + "e" * legs[v]


def _canonical_labellings(
    multiplicity: Sequence[Sequence[int]], legs: Sequence[int]
) -> tuple[str, list[tuple[int, ...]]]:
    """The canonical Nickel string and every labelling that attains it.

    Vertices count from 0. multiplicity[u][w] is the number of propagators
    joining u and w, and multiplicity[v][v] the number of self-loops at v;
    legs[v] is the number of legs at v. A labelling is a tuple whose entry i is
    the vertex given label i. The canonical string is the least, by code point,
    over all V! labellings.

    The search visits only breadth-first labellings. Vertices are processed in
    label order, and the unlabelled neighbours of the vertex being processed
    take the next labels, those joined to it by more propagators first, every
    order within a tie being tried; when every labelled vertex has been
    processed, any unlabelled vertex may take the next label. Any other
    labelling can be changed, from the first vertex where it departs from this
    rule, into one whose earlier entries are equal and whose entry there is
    smaller, so every least labelling is breadth-first. A branch is abandoned
    only when its entries so far exceed the least complete string found, so
    none is lost. The cost is at least the number of least labellings, the
    order of the automorphism group of the graph with its legs.

    The argument for minimality needs single-digit labels, so it holds for
    V <= 10. Above that, compute_graph_automorphisms relies only on the set of
    least labellings being one coset of the automorphism group.
    """
    V = len(legs)
    best: list[str] = []
    found: list[tuple[int, ...]] = []

    def search(order: tuple[int, ...], text: str, i: int) -> None:
        if best and text > best[0][: len(text)]:
            return
        if i == V:
            if not best or text < best[0]:
                best[:] = [text]
                found.clear()
            found.append(order)
            return
        if i == len(order):
            for v in range(V):
                if v not in order:
                    search(order + (v,), text, i)
            return
        v = order[i]
        new: dict[int, list[int]] = defaultdict(list)
        for w, m in enumerate(multiplicity[v]):
            if m and w not in order:
                new[m].append(w)
        ties = [new[m] for m in sorted(new, reverse=True)]
        for choice in product(*(permutations(tie) for tie in ties)):
            longer = order + tuple(chain.from_iterable(choice))
            label = {u: k for k, u in enumerate(longer)}
            search(longer, text + _nickel_entry(v, label, multiplicity, legs) + "|", i + 1)

    search((), "", 0)
    return best[0], found


def _pair_order(
    order: Sequence[int], multiplicity: Sequence[Sequence[int]]
) -> list[tuple[int, int]]:
    """The vertex pairs (u, w), u <= w, joined by propagators, in the order in which the Nickel
    string of the labelling lists them."""
    pairs: list[tuple[int, int]] = []
    for i, v in enumerate(order):
        pairs.extend((min(v, w), max(v, w)) for w in order[i:] if multiplicity[v][w])
    return pairs


def _tie_orders(
    tie: Sequence[int], profile: Mapping[int, tuple[int, ...]]
) -> Iterator[tuple[int, ...]]:
    """The orders of the tied classes, one for each sequence of their profiles."""
    if len({profile[c] for c in tie}) == len(tie):
        yield from permutations(tie)
        return
    seen: set[tuple[int, ...]] = set()
    for i, c in enumerate(tie):
        if profile[c] not in seen:
            seen.add(profile[c])
            for rest in _tie_orders([*tie[:i], *tie[i + 1 :]], profile):
                yield (c, *rest)


def _least_naming(groups: Sequence[Sequence[_Colour]]) -> str:
    """The least colour string of the groups of parallel propagators, over the names of the
    classes of shared masses; each group is written sorted.

    In the least string a class takes the next free letter where it first
    appears, classes with more propagators in that group first, since any other
    naming makes that group larger. Every order within a tie is tried, up to
    interchangeable classes: two tied classes with the same number of
    propagators in every group, their profile, give the same string when their
    letters are swapped, so only one of their orders is tried.
    """
    counts = [Counter(group) for group in groups]
    classes = {c for group in groups for c in group if isinstance(c, int)}
    profile = {c: tuple(count[c] for count in counts) for c in classes}
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
        for choice in product(*(_tie_orders(tie, profile) for tie in ties)):
            named = dict(names)
            for c in chain.from_iterable(choice):
                named[c] = _SHARED_LETTERS[len(named)]
            codes = sorted(named[c] if isinstance(c, int) else c for c in groups[g])
            walk(g + 1, named, text + "".join(codes))

    walk(0, {}, "")
    return best[0]


def _canonical_colours(
    pair_orders: Iterable[Sequence[tuple[int, int]]],
    colours: Mapping[tuple[int, int], Sequence[_Colour]],
) -> str:
    """The least colour string over the labellings, given by their pair orders, and the names
    of the shared masses."""
    return min(_least_naming([colours[pair] for pair in pairs]) for pairs in pair_orders)


class Graph:
    """
    A Feynman graph with internal and external edges.

    Represents a Feynman diagram as a graph structure with internal vertices
    and external legs. Provides methods to compute the graph Laplacian and
    related polynomials used in parametric integration.

    Attributes
    ----------
    internal_vertices : int
        Number of internal vertices in the graph (r_int).
    external_legs : int
        Number of external legs (external momenta, n_ext).
    edges : List[Edge]
        Complete list of all edges (both internal and external).
    energy_scale : sp.Symbol
        Energy scale parameter for dimensional analysis (typically mu).
    schwinger_parameters : Dict[int, sp.Symbol]
        Schwinger/Feynman parameters for internal edges, indexed by edge idx.
        Parameters are named 'a_{idx}' by default.
    external_parameters : Dict[int, sp.Symbol]
        Parameters for external legs, indexed 1 to n_ext.
        Parameters are named 'b_{j}' by default.

    Methods
    -------
    calculate_laplacian(include_external=True)
        Compute the graph Laplacian matrix.
    calculate_w_polynomial()
        Compute the W polynomial (determinant of the modified Laplacian).
    expand_w_by_external_parameters()
        Expand W polynomial by external leg parameters.
    get_internal_edges()
        Get list of internal edges.
    get_external_edges()
        Get list of external edges.
    get_loop_count()
        Calculate number of independent loops: L = E - V + 1.

    Examples
    --------
    >>> import sympy as sp
    >>> from feynkit.core import Edge, Graph
    >>>
    >>> # Create a simple bubble diagram (2 vertices, 2 internal edges, 2 external legs)
    >>> m1, m2 = sp.symbols('m1 m2', nonnegative=True)
    >>> nu1, nu2, = sp.symbols('nu1 nu2', positive=True)
    >>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
    >>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
    >>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    >>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    >>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    >>>
    >>> # Compute Laplacian
    >>> W = graph.calculate_w_polynomial()
    >>> print(W)
    >>>
    >>> # Get loop count
    >>> loops = graph.get_loop_count()
    >>> print(f"Number of loops: {loops}")

    Notes
    -----
    The vertex indexing convention follows:
    - Internal vertices: 1 to r_int
    - External vertices: (r_int + 1) to (r_int + n_ext)

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    """

    def __init__(
        self,
        internal_vertices: int,
        external_legs: int,
        edges: list[Edge],
        energy_scale: sp.Symbol | None = None,
    ):
        """
        Initialise a Feynman graph.

        Parameters
        ----------
        internal_vertices : int
            Number of internal vertices (must be positive).
        external_legs : int
            Number of external legs (can be zero).
        edges : List[Edge]
            List of all edges in the graph (internal propagators and external legs).
        energy_scale : Optional[sp.Symbol], default None
            Energy scale parameter. If None, creates symbol 'mu' with positive=True.

        Raises
        ------
        ValidationError
            If internal_vertices is not positive or external_legs is negative.
        EdgeConnectivityError
            If any edge violates the vertex indexing constraints.
        GraphTopologyError
            If graph topology is inconsistent.
        """
        # Validate basic parameters
        self.internal_vertices = validate_positive_integer(internal_vertices, "internal_vertices")
        self.external_legs = validate_nonnegative_integer(external_legs, "external_legs")
        self.edges = list(edges)

        # Set or create energy scale
        if energy_scale is not None:
            self.energy_scale = energy_scale
        else:
            self.energy_scale = sp.Symbol(DEFAULT_ENERGY_SCALE, **SCALE_ASSUMPTIONS)

        # Validate edge connectivity
        self._validate_edges()

        # Separate and sort internal and external edges by index
        self._internal_edges = sorted([e for e in self.edges if e.is_internal], key=lambda e: e.idx)
        self._external_edges = sorted(
            [e for e in self.edges if not e.is_internal], key=lambda e: e.idx
        )

        # Validate we have the expected number of external legs
        if len(self._external_edges) != self.external_legs:
            raise GraphTopologyError(
                f"Expected {self.external_legs} external edges, "
                f"but found {len(self._external_edges)}"
            )

        # Create Schwinger/Feynman parameters for internal edges
        self.schwinger_parameters: dict[int, sp.Symbol] = {
            e.idx: sp.Symbol(f"{SCHWINGER_PARAM_PREFIX}_{e.idx}", nonnegative=True)
            for e in self._internal_edges
        }

        # Create parameters for external legs
        self.external_parameters: dict[int, sp.Symbol] = {
            j: sp.Symbol(f"{EXTERNAL_PARAM_PREFIX}_{j}", nonnegative=True)
            for j in range(1, self.external_legs + 1)
        }

    def _validate_edges(self) -> None:
        """
        Validate all edges in the graph.

        Raises
        ------
        EdgeConnectivityError
            If any edge violates connectivity constraints.
        GraphTopologyError
            If edge indices are not unique.
        """
        # Check for duplicate edge indices
        edge_indices = [e.idx for e in self.edges]
        if len(edge_indices) != len(set(edge_indices)):
            duplicates = [idx for idx in edge_indices if edge_indices.count(idx) > 1]
            raise GraphTopologyError(f"Duplicate edge indices found: {set(duplicates)}")

        # Validate each edge's connectivity
        for edge in self.edges:
            validate_edge_connectivity(
                edge_v1=edge.v1,
                edge_v2=edge.v2,
                is_internal=edge.is_internal,
                internal_vertex_count=self.internal_vertices,
                external_leg_count=self.external_legs,
                edge_idx=edge.idx,
            )

    def get_internal_edges(self) -> list[Edge]:
        """
        Get list of internal edges (propagators).

        Returns
        -------
        List[Edge]
            List of internal edges, sorted by index.
        """
        return self._internal_edges.copy()

    def get_external_edges(self) -> list[Edge]:
        """
        Get list of external edges (external legs).

        Returns
        -------
        List[Edge]
            List of external edges, sorted by index.
        """
        return self._external_edges.copy()

    def get_loop_count(self) -> int:
        """
        Calculate the number of independent loops in the graph.

        The loop count L is given by the formula:
            L = E - V + 1
        where E is the number of internal edges and V is the number of internal vertices.

        Returns
        -------
        int
            Number of independent loops.

        Examples
        --------
        >>> # Bubble diagram: 2 edges, 2 vertices -> L = 2 - 2 + 1 = 1
        >>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        >>> graph.get_loop_count()
        1
        """
        return len(self._internal_edges) - self.internal_vertices + 1

    # -- Nickel / CNickel index ------------------------------------------------

    def _nickel_data(
        self,
    ) -> tuple[list[list[int]], list[int], dict[tuple[int, int], list[_Colour]]]:
        """
        The multiplicity matrix, the legs at each vertex and the colours of the
        propagators joining each pair of vertices, in the graph's own numbering
        of the vertices: the input to the search for the canonical labelling.

        Vertices count from 0, and a pair (u, w) has u <= w. A propagator's
        colour is "z" if it is massless, "n" if no other propagator has its
        mass, and otherwise the number of its class of equal masses. A mass of
        None counts as zero.
        """
        zero = sp.Integer(0)
        masses = [zero if e.mass is None else e.mass for e in self._internal_edges]
        count = Counter(m for m in masses if m != zero)
        shared = {m: k for k, m in enumerate(m for m in count if count[m] > 1)}

        V = self.internal_vertices
        multiplicity = [[0] * V for _ in range(V)]
        colours: dict[tuple[int, int], list[_Colour]] = defaultdict(list)
        for e, m in zip(self._internal_edges, masses, strict=True):
            u, w = sorted((e.v1 - 1, e.v2 - 1))
            multiplicity[u][w] += 1
            if u != w:
                multiplicity[w][u] += 1
            colours[(u, w)].append("z" if m == zero else shared.get(m, "n"))

        legs = [0] * V
        for e in self._external_edges:
            legs[e.v1 - 1] += 1
        return multiplicity, legs, dict(colours)

    def nickel_index(self) -> str:
        """
        Canonical Nickel topology string for this graph.

        The Nickel index encodes the graph topology as a string that is
        invariant under vertex relabelling (canonical = lex minimum over
        all V! relabellings of internal vertices).

        For each vertex i (0-indexed), the entry lists higher-numbered
        internal neighbours in ascending order followed by 'e' for each
        external leg; entries are separated by '|'.

        Examples
        --------
        Massless bubble:   ``"11e|e|"``
        Massless triangle: ``"12e|2e|e|"``
        Massless box:      ``"12e|3e|3e|e|"``
        3-prop banana:     ``"111e|e|"``

        Notes
        -----
        Supports up to 10 internal vertices (single-digit vertex labels). The
        least string is found by an exact branch and bound, not a scan of all
        V! labellings. Its time grows with the size of the automorphism group,
        and with wide ties at vertices of high degree. At V = 10 the star takes
        14.4 s, a hub with nine double-edge petals 14.7 s (32.9 s in
        :meth:`cnickel`) and the complete graph 208 s (874 s in
        :meth:`cnickel`), while the graphs that :func:`feynkit.generate_graphs`
        emits at its tiers take milliseconds.
        """
        V = self.internal_vertices
        if V > 10:
            raise NotImplementedError(
                f"Nickel index requires V <= 10; this graph has {V} internal vertices"
            )
        multiplicity, legs, _ = self._nickel_data()
        return _canonical_labellings(multiplicity, legs)[0]

    def cnickel(self) -> str:
        """
        Canonical Colored Nickel (CNickel) index: topology + mass colouring.

        Extends :meth:`nickel_index` with a mass-colour suffix separated by
        ``':'``, one code per propagator in the order the propagators appear
        left-to-right in the topology string: ``'z'`` for a massless
        propagator, ``'n'`` for a mass no other propagator has, and a letter
        ``'a'``, ``'b'``, ... for each class of propagators sharing a mass.

        The canonical form is the least ``(topology, colouring)`` pair over the
        vertex labellings and the names of the shared masses, so it absorbs
        the automorphisms of the graph. Shared masses are named in order of
        first appearance, and a mass is written ``'n'`` whenever one propagator
        alone carries it, whatever its symbol.

        Raises
        ------
        NotImplementedError
            If the graph has more than 10 internal vertices, or more than 23
            classes of propagators sharing a mass, the number of letters
            ``'a'`` to ``'y'`` without ``'n'`` and ``'s'``.

        Notes
        -----
        The time is that of :meth:`nickel_index`, whose Notes give examples,
        and one colour string for each least labelling.

        Examples
        --------
        Massless triangle:     ``"12e|2e|e|:zzz"``
        One-massive triangle:  ``"12e|2e|e|:nzz"``
        All-massive triangle:  ``"12e|2e|e|:nnn"``
        Two equal masses:      ``"12e|2e|e|:aan"``
        Massless bubble:       ``"11e|e|:zz"``
        """
        V = self.internal_vertices
        if V > 10:
            raise NotImplementedError(
                f"CNickel index requires V <= 10; this graph has {V} internal vertices"
            )
        multiplicity, legs, colours = self._nickel_data()
        classes = {c for group in colours.values() for c in group if isinstance(c, int)}
        if len(classes) > len(_SHARED_LETTERS):
            raise NotImplementedError(
                f"CNickel has {len(_SHARED_LETTERS)} letters for shared masses; "
                f"this graph has {len(classes)} classes"
            )
        nickel, labellings = _canonical_labellings(multiplicity, legs)
        pair_orders = (_pair_order(order, multiplicity) for order in labellings)
        return f"{nickel}:{_canonical_colours(pair_orders, colours)}"

    @classmethod
    def from_cnickel(cls, cnickel: str) -> Graph:
        """
        Construct a :class:`Graph` from a CNickel string.

        Accepts both the full ``"<topology>:<colours>"`` form and a bare
        topology string (all edges default to massless when colours are absent).

        Vertex labels in the string are 0-indexed; feynkit's internal vertex
        numbering (1-indexed) is assigned in the same order.  External legs are
        numbered sequentially in the order they appear reading left-to-right
        through the topology entries.

        The topology has one entry per vertex, each ending in ``'|'``. An entry
        may be empty, as the last one of ``"111e||"``, the sunrise with one leg,
        since it lists only the higher neighbours of its vertex. Only the one
        terminating bar is stripped, so a superfluous bar adds a vertex without
        propagators or legs, which is rejected.

        Massive edges (colour ``'n'``) receive a symbolic mass
        ``m_<idx>`` with assumptions ``{nonnegative: True, real: True}``.
        Massless edges (colour ``'z'`` or ``'0'``) receive ``mass = 0``. A
        letter other than ``'n'`` and ``'z'`` gives the shared mass
        ``m_<letter>``. Digits other than ``'0'`` are rejected: ``m_1`` is
        also the mass ``'n'`` gives propagator 1.

        Parameters
        ----------
        cnickel
            CNickel string, e.g. ``"12e|2e|e|:nzz"`` or ``"12e|2e|e|"``.

        Returns
        -------
        Graph

        Raises
        ------
        ValueError
            If the string is malformed, contains out-of-range vertex labels,
            a digit mass code other than ``'0'`` or, with more than one vertex,
            a vertex without propagators and legs, or the mass-colour count does
            not match the internal edge count.

        Examples
        --------
        >>> Graph.from_cnickel("11e|e|:zz")          # massless bubble
        >>> Graph.from_cnickel("12e|2e|e|:zzz")      # massless triangle
        >>> Graph.from_cnickel("12e|2e|e|:nzz")      # one-mass triangle
        >>> Graph.from_cnickel("111e|e|:zzz")        # massless 3-prop banana
        >>> Graph.from_cnickel("12e|2e|e|")           # bare topology -> massless
        """
        from .constants import MASS_ASSUMPTIONS

        if ":" in cnickel:
            topology, color_part = cnickel.rsplit(":", 1)
            if "|" in color_part:
                # Structured format: colour string mirrors the topology structure
                # (one colour per topology character, '|' as separator).
                # Extract only the colours that sit at digit (internal-edge) positions.
                topo_chars = topology.rstrip("|").replace("|", "")
                color_chars = color_part.rstrip("|").replace("|", "")
                mass_str = "".join(
                    cc for tc, cc in zip(topo_chars, color_chars, strict=True) if tc.isdigit()
                )
            else:
                mass_str = color_part
        else:
            topology = cnickel
            mass_str = ""

        # Every entry ends in one '|'; the entries before it may be empty.
        topology = topology.removesuffix("|")
        if not topology:
            raise ValueError(f"Empty topology in CNickel string: {cnickel!r}")

        parts = topology.split("|")
        V = len(parts)

        internal_edges: list[Edge] = []
        external_edges: list[Edge] = []
        edge_idx = 1
        mass_idx = 0

        # -- internal edges ------------------------------------------------
        for i, entry in enumerate(parts):
            for ch in entry:
                if ch.isdigit():
                    j = int(ch)
                    if j >= V:
                        raise ValueError(
                            f"Vertex label {j} out of range [0, {V - 1}] "
                            f"in entry {i} of {cnickel!r}"
                        )
                    if j >= i:
                        mc = mass_str[mass_idx] if mass_idx < len(mass_str) else "z"
                        if mc in "123456789":
                            raise ValueError(
                                f"Digit mass code {mc!r} in {cnickel!r}: m_{mc} is also the "
                                f"mass 'n' gives propagator {mc}; name shared masses with letters"
                            )
                        if mc not in _ALL_VALID_CODES:
                            raise ValueError(f"Unknown mass code {mc!r} in {cnickel!r}")
                        mass_idx += 1
                        mass = _mass_from_code(mc, edge_idx, MASS_ASSUMPTIONS)
                        internal_edges.append(
                            Edge(
                                idx=edge_idx,
                                v1=i + 1,
                                v2=j + 1,
                                is_internal=True,
                                mass=mass,
                            )
                        )
                        edge_idx += 1
                elif ch != "e":
                    raise ValueError(
                        f"Unexpected character {ch!r} in entry {i} " f"of topology {topology!r}"
                    )

        if mass_str and mass_idx != len(mass_str):
            raise ValueError(
                f"Mass-colour length {len(mass_str)} does not match "
                f"internal edge count {mass_idx} in {cnickel!r}"
            )

        # -- external edges ------------------------------------------------
        ext_v = V + 1
        for i, entry in enumerate(parts):
            for ch in entry:
                if ch == "e":
                    external_edges.append(
                        Edge(
                            idx=edge_idx,
                            v1=i + 1,
                            v2=ext_v,
                            is_internal=False,
                        )
                    )
                    edge_idx += 1
                    ext_v += 1

        # An entry lists only the higher neighbours, so a propagator in any entry
        # can touch a vertex; legs touch only the vertex of their own entry.
        touched = {e.v1 for e in internal_edges + external_edges}
        touched |= {e.v2 for e in internal_edges}
        bare = [v - 1 for v in range(1, V + 1) if v not in touched]
        if V > 1 and bare:
            raise ValueError(
                f"Vertex {bare[0]} of {cnickel!r} has no propagator and no leg; "
                "check for a superfluous '|' or an entry nothing refers to"
            )

        total_ext = ext_v - (V + 1)
        return cls(
            internal_vertices=V,
            external_legs=total_ext,
            edges=internal_edges + external_edges,
        )

    @classmethod
    def from_nickel(cls, nickel: str) -> Graph:
        """
        Construct a massless :class:`Graph` from a bare Nickel topology string.

        Equivalent to ``Graph.from_cnickel(nickel)``, the mass-color suffix is
        omitted, so all propagators default to zero mass.

        Examples
        --------
        >>> Graph.from_nickel("11e|e|")       # massless bubble
        >>> Graph.from_nickel("12e|2e|e|")    # massless triangle
        >>> Graph.from_nickel("13e|2e|3e|e|") # massless box
        """
        return cls.from_cnickel(nickel)

    def calculate_laplacian(self, include_external: bool = True) -> sp.Matrix:
        """
        Compute the graph Laplacian matrix.

        The Laplacian matrix encodes the connectivity structure of the graph. Each edge
        contributes to the Laplacian based on its Schwinger parameter (a for internal
        edges, b for external legs).

        For an edge connecting vertices i and j with parameter p:
        - L[i,i] += p (diagonal: sum of parameters at vertex i)
        - L[j,j] += p (diagonal: sum of parameters at vertex j)
        - L[i,j] -= p (off-diagonal: negative parameter)
        - L[j,i] -= p (symmetric)

        Parameters
        ----------
        include_external : bool, default True
            If True, include external legs with parameters b_j in the Laplacian.
            If False, only internal edges with parameters a_i are included.

        Returns
        -------
        sp.Matrix
            The Laplacian matrix of size (r_int + n_ext) x (r_int + n_ext) if
            include_external is True, or r_int x r_int if False.

        Notes
        -----
        Self-loops (tadpoles, where i == j) are skipped as they don't contribute
        to the Laplacian in the standard formulation.

        Examples
        --------
        >>> L = graph.calculate_laplacian()
        >>> print(L)
        >>> L_internal = graph.calculate_laplacian(include_external=False)
        >>> print(L_internal)
        """
        # Determine matrix size based on whether external vertices are included
        size = self.internal_vertices + (self.external_legs if include_external else 0)
        L = sp.zeros(size, size)

        # Add contributions from internal edges with Schwinger parameters a_i
        for edge in self._internal_edges:
            param = self.schwinger_parameters[edge.idx]
            i, j = edge.v1, edge.v2

            # Skip self-loops (tadpoles)
            if i == j:
                continue

            # Standard Laplacian construction (vertex indices are 1-indexed, matrix is 0-indexed)
            L[i - 1, i - 1] += param  # Diagonal element at vertex i
            L[j - 1, j - 1] += param  # Diagonal element at vertex j
            L[i - 1, j - 1] -= param  # Off-diagonal coupling
            L[j - 1, i - 1] -= param  # Symmetric off-diagonal

        # Add contributions from external legs with parameters b_j
        if include_external and self.external_legs > 0:
            for edge in self._external_edges:
                # Map external vertex to leg number
                leg_number = edge.v2 - self.internal_vertices
                param = self.external_parameters[leg_number]
                i, ext = edge.v1, edge.v2

                # External legs connect internal vertex i to external vertex ext
                L[i - 1, i - 1] += param
                L[ext - 1, ext - 1] += param
                L[i - 1, ext - 1] -= param
                L[ext - 1, i - 1] -= param

        return sp.Matrix(L)

    def calculate_w_polynomial(self) -> sp.Expr:
        """
        Compute the W polynomial from the graph Laplacian.

        The W polynomial is obtained by taking the determinant of the Laplacian
        restricted to internal vertices (after removing external vertex rows/columns).
        This polynomial appears in the Symanzik parametrisation of Feynman integrals.

        Returns
        -------
        sp.Expr
            The W polynomial, factored for simplified form. This is a polynomial
            in the Schwinger parameters {a_i} and external leg parameters {b_j}.

        Notes
        -----
        For graphs with external legs (n_ext > 0), we compute det(L_internal) where
        L_internal is the Laplacian with external vertex rows/columns removed.
        For graphs with no external legs, we compute det(L) directly.

        The W polynomial encodes the spanning trees of the graph and is related
        to the Symanzik U polynomial through a variable transformation.

        Examples
        --------
        >>> W = graph.calculate_w_polynomial()
        >>> print(W)
        """
        # Get full Laplacian including external vertices
        L = self.calculate_laplacian(include_external=True)

        if self.external_legs > 0:
            # Remove rows and columns corresponding to external vertices
            # External vertices are indexed r_int to r_int + n_ext - 1 (0-indexed)
            rows_to_keep = list(range(self.internal_vertices))
            M = L.extract(rows_to_keep, rows_to_keep)
        else:
            M = L

        return sp.expand(M.det(method="bareiss"))

    def expand_w_by_external_parameters(self) -> dict[tuple[int, ...], sp.Expr]:
        """
        Expand the W polynomial as a polynomial in external parameters b_j.

        Expands W as a sum over monomials in the b parameters and returns a dictionary
        mapping each monomial signature to its coefficient (which depends on a parameters).

        Returns
        -------
        Dict[tuple[int, ...], sp.Expr]
            Dictionary mapping tuples of existing leg indices to polynomial coefficients.
            Each key is a sorted tuple representing a monomial in b variable.
            For example:
            - (1,) represents b_1 with coefficient in a's
            - (1, 2) represents b_1 * b_2 with coefficient
            - () represents the constant term (no b's)

        Notes
        -----
        This expansion is used to extract the Symanzik U and F polynomials.
        The coefficient of a degree-1 term gives U (after variable transformation).
        The coefficients of degree-2 terms give contributions to F related to
        external momentum invariants.

        Examples
        --------
        >>> coeffs = graph.expand_w_by_external_parameters()
        >>> for monomial, coeff in coeffs.items():
        ...     print(f"b_{monomial}: {coeff}")
        """
        W = sp.expand(self.calculate_w_polynomial())

        if self.external_legs == 0:
            return {(): W}

        b_symbols = [self.external_parameters[j] for j in range(1, self.external_legs + 1)]

        # Convert to polynomial in b variables
        P = sp.Poly(W, *b_symbols)
        coefficients = {}

        # Extract each monomial and its coefficient
        for monomial, coeff in P.terms():
            # Convert monomial exponents to list of leg indices
            # monomial is a tuple of exponents (e_1, e_2, ..., e_n_ext)
            indices = []
            for j, exp in enumerate(monomial, start=1):
                if exp > 0:
                    # Add leg index j, repeated exp times (for products like b_1^2)
                    indices.extend([j] * int(exp))

            # Store coefficient with sorted tuple key
            coefficients[tuple(sorted(indices))] = coeff

        return coefficients

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the graph.

        Returns
        -------
        str
            String representation showing graph structure.
        """
        return (
            f"Graph(internal_vertices={self.internal_vertices}, "
            f"external_legs={self.external_legs}, "
            f"edges={len(self.edges)}, "
            f"loops={self.get_loop_count()})"
        )

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the graph.

        Returns
        -------
        str
            Concise description of the graph.
        """
        return (
            f"Feynman Graph: {self.internal_vertices} internal vertices, "
            f"{self.external_legs} external legs, "
            f"{len(self._internal_edges)} propagators, "
            f"{self.get_loop_count()} loops"
        )
