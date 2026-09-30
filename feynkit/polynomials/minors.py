"""
Symanzik polynomials of Feynman-graph minors.

A minor (Gamma - D)/C of a graph Gamma deletes the propagators of D and
contracts those of C, as :meth:`Graph.delete` and :meth:`Graph.contract` do:
legs move with their vertices, and a vertex left without propagators is
removed with its legs. The minor keeps the edge indices of Gamma, so its
polynomials are written in the parameters of Gamma, and its legs carry the
momenta they carry in Gamma.

A minor need not be connected. Its polynomials then follow the forest
convention: U is the product of the U_c of its components c, and
F = sum_c F_c prod_{c' != c} U_{c'}, which is what the spanning trees and
spanning 2-forests of the minor as a whole give. For a connected graph they
are the U and F of :attr:`FeynmanIntegral.symanzik`, F carrying the factor
1/mu^2 of the energy scale.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import sympy as sp

from ..core.edge import Edge
from ..core.exceptions import ValidationError
from ..core.graph import Graph
from .spanning_trees import _spanning_2forests, _spanning_trees

__all__ = ["minor_polynomials"]

# A polynomial as its terms: exponent vector, in a fixed order of the edges, to coefficient.
Terms = dict[tuple[int, ...], sp.Expr]


def minor_polynomials(
    graph: Graph,
    momentum_products: Mapping[tuple[int, int], Any],
    *,
    contract: Iterable[int] = (),
    delete: Iterable[int] = (),
    parameters: Mapping[int, sp.Symbol] | None = None,
) -> tuple[sp.Expr, sp.Expr]:
    """
    U and F of the minor (Gamma - D)/C, in the parameters and kinematics of Gamma.

    Parameters
    ----------
    graph
        Gamma.
    momentum_products
        p_j . p_k for the legs j < k of Gamma, keyed by their leg numbers, as
        :attr:`FeynmanIntegral.momentum_products` gives them; a missing pair
        is 0.
    contract, delete
        C and D, disjoint sets of internal edge indices.
    parameters
        The variable of each edge, by index; by default the Schwinger
        parameters a_e of ``graph``.

    Returns
    -------
    tuple of sp.Expr
        U and F of the minor, with the forest convention when it is not
        connected.

    Raises
    ------
    ValidationError
        If an index is not that of an internal edge, or is in both C and D.
    GraphTopologyError
        If no propagator would remain.

    Examples
    --------
    >>> fi = FeynmanIntegral.from_cnickel("12e|22e|e|:nnnn")
    >>> u, f = minor_polynomials(fi.graph, fi.momentum_products, contract=[3, 4])
    """
    order = tuple(e.idx for e in graph.get_internal_edges())
    names = graph.schwinger_parameters if parameters is None else parameters
    variables = [names[idx] for idx in order]
    _, u_terms, f_terms = minor_terms(graph, momentum_products, contract, delete, order)
    return _expression(u_terms, variables), _expression(f_terms, variables)


def minor_terms(
    graph: Graph,
    momentum_products: Mapping[tuple[int, int], Any],
    contract: Iterable[int],
    delete: Iterable[int],
    order: Sequence[int],
) -> tuple[Graph, Terms, Terms]:
    """The minor and the terms of its U and F, with exponent vectors indexed as ``order``.

    ``order`` lists edge indices of ``graph`` and must hold every edge of the
    minor. The coefficients of U are 1, those of F are expanded and carry
    1/mu^2; a 2-forest whose coefficient expands to zero gives no term.
    """
    contracted, deleted = frozenset(contract), frozenset(delete)
    both = sorted(contracted & deleted)
    if both:
        raise ValidationError(
            f"edges {', '.join(map(str, both))} are both contracted and deleted; "
            "C and D must be disjoint"
        )
    minor = graph.delete(deleted).contract(contracted)
    position = {idx: k for k, idx in enumerate(order)}
    n = len(order)

    # The leg number in graph of each leg of the minor, found through its edge index.
    leg_number = {e.idx: e.v2 - graph.internal_vertices for e in graph.get_external_edges()}
    legs_at: dict[int, list[int]] = {}
    for e in minor.get_external_edges():
        legs_at.setdefault(e.v1 - 1, []).append(leg_number[e.idx])

    components: dict[int, list[int]] = {}
    root = minor._components()
    for k, e in enumerate(minor.get_internal_edges()):
        components.setdefault(root[e.v1 - 1], []).append(k)

    # p_j . p_k under both orders of the pair.
    products: dict[tuple[int, int], sp.Expr] = {}
    for (j, k), value in momentum_products.items():
        products[(j, k)] = products[(k, j)] = sp.sympify(value)

    scale = minor.energy_scale**2
    edges = minor.get_internal_edges()
    u_total: Terms = {(0,) * n: sp.Integer(1)}
    f_total: Terms = {}
    for members in components.values():
        u_c, f_c = _component_terms(
            [edges[k] for k in members], legs_at, products, position, n, scale
        )
        f_total = _add(_times(f_total, u_c), _times(u_total, f_c))
        u_total = _times(u_total, u_c)
    return minor, u_total, f_total


def _component_terms(
    edges: list[Edge],
    legs_at: Mapping[int, list[int]],
    products: Mapping[tuple[int, int], sp.Expr],
    position: Mapping[int, int],
    n: int,
    scale: sp.Expr,
) -> tuple[Terms, Terms]:
    """U and F of one connected component, given by its propagators."""
    vertices = sorted({e.v1 - 1 for e in edges} | {e.v2 - 1 for e in edges})
    local = {v: k for k, v in enumerate(vertices)}
    pairs = [(local[e.v1 - 1], local[e.v2 - 1]) for e in edges]
    columns = [position[e.idx] for e in edges]

    def complement(chosen: Iterable[int]) -> list[int]:
        exponent = [0] * n
        for k in set(range(len(edges))) - set(chosen):
            exponent[columns[k]] = 1
        return exponent

    u: Terms = {}
    squares = [
        (k, mass2)
        for k, mass2 in enumerate(sp.expand(sp.sympify(e.get_mass()) ** 2) for e in edges)
        if mass2 != 0
    ]
    f: Terms = {}
    for tree in _spanning_trees(len(vertices), pairs):
        exponent = complement(tree)
        u[tuple(exponent)] = sp.Integer(1)
        for k, mass2 in squares:
            if k not in tree:
                squared = list(exponent)
                squared[columns[k]] += 1
                f[tuple(squared)] = sp.expand(mass2 / scale)

    legs = [(local[v], j) for v, js in legs_at.items() if v in local for j in js]
    for forest, side in _spanning_2forests(len(vertices), pairs):
        coefficient = sp.Integer(0)
        for va, j in legs:
            for vb, k in legs:
                if side[va] and not side[vb]:
                    coefficient += products.get((j, k), sp.Integer(0))
        for (a, b), e in zip(pairs, edges, strict=True):
            if side[a] != side[b]:
                coefficient += sp.sympify(e.get_mass()) ** 2
        coefficient = sp.expand(coefficient)
        if coefficient != 0:
            key = tuple(complement(forest))
            f[key] = sp.expand(f.get(key, sp.Integer(0)) + coefficient / scale)
    return u, f


def _times(p: Terms, q: Terms) -> Terms:
    """The product of two polynomials given by their terms.

    Its callers multiply polynomials in disjoint sets of variables, one of them
    with coefficients 1, so no coefficient needs expanding.
    """
    out: Terms = {}
    for a, x in p.items():
        for b, y in q.items():
            key = tuple(i + j for i, j in zip(a, b, strict=True))
            out[key] = out.get(key, sp.Integer(0)) + x * y
    return {k: v for k, v in out.items() if v != 0}


def _add(p: Terms, q: Terms) -> Terms:
    out = dict(p)
    for k, v in q.items():
        out[k] = sp.expand(out.get(k, sp.Integer(0)) + v)
    return {k: v for k, v in out.items() if v != 0}


def _expression(terms: Terms, variables: Sequence[sp.Symbol]) -> sp.Expr:
    return sp.Add(
        *(c * sp.Mul(*(x**k for x, k in zip(variables, e, strict=True))) for e, c in terms.items())
    )
