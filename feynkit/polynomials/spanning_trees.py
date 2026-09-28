"""
Spanning-tree and spanning-2-forest enumeration for Feynman graphs.

Used to compute Symanzik polynomials without symbolic matrix determinants.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from typing import Any

import sympy as sp


def _build_uf(
    n_vertices: int, edge_subset: tuple[int, ...], edge_pairs: list[tuple[int, int]]
) -> tuple[list[int] | None, int]:
    """
    Run union-find over the given edge subset.

    Returns ``(parent, n_components)`` on success (acyclic subgraph), or
    ``(None, 0)`` if a cycle is detected.
    """
    parent = list(range(n_vertices))

    for i in edge_subset:
        u, v = edge_pairs[i]
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            return None, 0
        parent[ru] = rv

    roots = len({_find(parent, i) for i in range(n_vertices)})
    return parent, roots


def _find(parent: list[int], x: int) -> int:
    """Union-find root lookup with path halving."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _spanning_trees(
    n_vertices: int, edge_pairs: list[tuple[int, int]]
) -> Iterator[tuple[int, ...]]:
    """Yield edge-index tuples for every spanning tree of the graph."""
    n_edges = len(edge_pairs)
    for subset in combinations(range(n_edges), n_vertices - 1):
        parent, n_comp = _build_uf(n_vertices, subset, edge_pairs)
        if parent is not None and n_comp == 1:
            yield subset


def _separating_2forests(
    n_vertices: int,
    edge_pairs: list[tuple[int, int]],
    v1: int,
    v2: int,
) -> Iterator[tuple[int, ...]]:
    """
    Yield edge-index tuples for every spanning 2-forest in which v1 and v2
    are in different components.
    """
    n_edges = len(edge_pairs)
    for subset in combinations(range(n_edges), n_vertices - 2):
        parent, n_comp = _build_uf(n_vertices, subset, edge_pairs)
        if parent is None or n_comp != 2:
            continue
        if _find(parent, v1) != _find(parent, v2):
            yield subset


def _spanning_2forests(
    n_vertices: int, edge_pairs: list[tuple[int, int]]
) -> Iterator[tuple[tuple[int, ...], list[bool]]]:
    """
    Yield every spanning 2-forest with the side of each vertex: True on the
    tree holding vertex 0, False on the other.
    """
    if n_vertices < 2:
        return
    for subset in combinations(range(len(edge_pairs)), n_vertices - 2):
        parent, n_comp = _build_uf(n_vertices, subset, edge_pairs)
        if parent is None or n_comp != 2:
            continue
        root = _find(parent, 0)
        yield subset, [_find(parent, v) == root for v in range(n_vertices)]


def _momentum_product(momentum_products: dict[tuple[int, int], Any], j: int, k: int) -> Any:
    """p_j . p_k, given under (j, k) or (k, j), or 0 when neither is given."""
    value = momentum_products.get((j, k))
    if value is None:
        value = momentum_products.get((k, j))
    return 0 if value is None else value


def gkz_exponent_vectors(
    n_vertices: int,
    edge_pairs: list[tuple[int, int]],
    leg_to_vertex: dict[int, int],
    momentum_products: dict[tuple[int, int], Any],
    internal_edge_masses: list[Any],
) -> list[tuple[int, ...]]:
    """
    Return the distinct exponent vectors of G = U_lp + F_lp without expanding
    G.  Each entry is an n-tuple of non-negative integers representing one
    column of the GKZ A-matrix (excluding the homogenising row).

    The monomials are those of G with a non-zero coefficient.  A spanning
    2-forest with legs A and B on its two trees and edges C joining them gives
    the complement of the forest, with coefficient

        c = sum_{j in A, l in B} p_j . p_l + sum_{e in C} m_e^2

    in F, up to the factor 1/mu^2.  It is kept when sp.expand(c) != 0, the
    zero test of extract_monomial_support, since c can vanish while its terms
    do not: on shell, or where p_1^2 = m_1^2.  Each edge e whose m_e^2 is
    non-zero by the same test, and spanning tree T without e, give u_e^2 times
    the U monomial of T, with coefficient m_e^2.

    Parameters
    ----------
    n_vertices : int
        Number of internal vertices.
    edge_pairs : list of (int, int)
        0-indexed internal-vertex pairs for each internal edge (length n).
    leg_to_vertex : dict[int, int]
        Maps external-leg number (1-based) to 0-indexed internal vertex.
    momentum_products : dict[(int,int), Any]
        Symbolic momentum dot-products; pairs (j, k) with j < k.  A pair
        given as (k, j) is read the same way, and a missing pair is 0.
    internal_edge_masses : list
        Mass expression for each internal edge (0 = massless).
    """
    n = len(edge_pairs)
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []

    def _add(vec: tuple[int, ...]) -> None:
        if vec not in seen:
            seen.add(vec)
            result.append(vec)

    # U monomials: complement of each spanning tree
    for tree in _spanning_trees(n_vertices, edge_pairs):
        tree_set = set(tree)
        _add(tuple(0 if i in tree_set else 1 for i in range(n)))

    # 2-forest monomials: complement of each spanning 2-forest whose coefficient is not 0
    legs = sorted(leg_to_vertex.items())
    for forest, side in _spanning_2forests(n_vertices, edge_pairs):
        coefficient = sp.Integer(0)
        for j, vj in legs:
            for k, vk in legs:
                if side[vj] and not side[vk]:
                    coefficient += sp.sympify(_momentum_product(momentum_products, j, k))
        for i, (v1, v2) in enumerate(edge_pairs):
            if side[v1] != side[v2]:
                coefficient += sp.sympify(internal_edge_masses[i]) ** 2
        if sp.expand(coefficient) != 0:
            forest_set = set(forest)
            _add(tuple(0 if i in forest_set else 1 for i in range(n)))

    # u_k^2 monomials: u_k * (U monomial for T), for each tree T and massive edge k not in T
    mass_edge_indices = [
        i for i, m in enumerate(internal_edge_masses) if sp.expand(sp.sympify(m) ** 2) != 0
    ]
    if mass_edge_indices:
        for tree in _spanning_trees(n_vertices, edge_pairs):
            tree_set = set(tree)
            u_vec = [0 if i in tree_set else 1 for i in range(n)]
            for k in mass_edge_indices:
                if k in tree_set:
                    continue
                vec = list(u_vec)
                vec[k] += 1
                _add(tuple(vec))

    return result


def spanning_tree_poly(
    n_vertices: int,
    edge_pairs: list[tuple[int, int]],
    params: list[sp.Symbol],
) -> sp.Expr:
    """
    Sum of Schwinger-parameter products over all spanning trees.

    This is the coefficient C that appears in the W-polynomial expansion;
    the Symanzik U is then obtained by applying :func:`reverse_monomials`.
    """
    total = sp.Integer(0)
    for subset in _spanning_trees(n_vertices, edge_pairs):
        total += sp.Mul(*[params[i] for i in subset]) if subset else sp.Integer(1)
    return total


def separating_2forest_poly(
    n_vertices: int,
    edge_pairs: list[tuple[int, int]],
    v1: int,
    v2: int,
    params: list[sp.Symbol],
) -> sp.Expr:
    """
    Sum of Schwinger-parameter products over all spanning 2-forests
    that separate v1 from v2 (0-indexed).
    """
    total = sp.Integer(0)
    for subset in _separating_2forests(n_vertices, edge_pairs, v1, v2):
        total += sp.Mul(*[params[i] for i in subset]) if subset else sp.Integer(1)
    return total
