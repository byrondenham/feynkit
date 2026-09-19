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


def gkz_exponent_vectors(
    n_vertices: int,
    edge_pairs: list[tuple[int, int]],
    leg_to_vertex: dict[int, int],
    momentum_products: dict[tuple[int, int], Any],
    internal_edge_masses: list[Any],
) -> list[tuple[int, ...]]:
    """
    Return the distinct exponent vectors of G = U_lp + F_lp without any SymPy
    polynomial manipulation.  Each entry is an n-tuple of non-negative integers
    representing one column of the GKZ A-matrix (excluding the homogenising row).

    Parameters
    ----------
    n_vertices : int
        Number of internal vertices.
    edge_pairs : list of (int, int)
        0-indexed internal-vertex pairs for each internal edge (length n).
    leg_to_vertex : dict[int, int]
        Maps external-leg number (1-based) to 0-indexed internal vertex.
    momentum_products : dict[(int,int), Any]
        Symbolic momentum dot-products; pairs (j, k) with j < k.
        A product is skipped when it compares equal to 0.
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

    # F_0 monomials: complement of each separating 2-forest, per momentum pair
    ext_legs = sorted(leg_to_vertex)
    for ji, j in enumerate(ext_legs):
        for k in ext_legs[ji + 1 :]:
            p_jk = momentum_products.get((j, k)) or momentum_products.get((k, j))
            if p_jk is None or p_jk == 0:
                continue
            vj, vk = leg_to_vertex[j], leg_to_vertex[k]
            if vj == vk:
                continue
            for forest in _separating_2forests(n_vertices, edge_pairs, vj, vk):
                forest_set = set(forest)
                _add(tuple(0 if i in forest_set else 1 for i in range(n)))

    # F_mass monomials: a_k * (U monomial for T), for each tree T and massive edge k
    mass_edge_indices = [i for i, m in enumerate(internal_edge_masses) if m != 0]
    if mass_edge_indices:
        for tree in _spanning_trees(n_vertices, edge_pairs):
            tree_set = set(tree)
            u_vec = [0 if i in tree_set else 1 for i in range(n)]
            for k in mass_edge_indices:
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
