"""
Syzygy computation for polynomial relations.

Provides functions to compute and analyse syzygies (polynomial relations)
among generators.
"""

from __future__ import annotations

import sympy as sp


def compute_syzygy_module(
    generators: list[sp.Expr],
    variables: list[sp.Symbol],
) -> list[list[sp.Expr]]:
    """
    Compute generators of the first syzygy module of a set of polynomials.

    A syzygy is a relation ``sum_i h_i · f_i = 0`` where ``f_i`` are the
    generators.  The returned vectors generate the module of all such
    relations over the polynomial ring.

    Parameters
    ----------
    generators : List[sp.Expr]
        Polynomial generators.
    variables : List[sp.Symbol]
        Variables in the polynomial ring.

    Returns
    -------
    List[List[sp.Expr]]
        List of syzygies, where each syzygy is a list of coefficients
        ``[h_1, ..., h_n]``.  Zero rows and repeated rows are dropped.

    Notes
    -----
    Runs Buchberger's algorithm while tracking, for every basis element, its
    expression in terms of the input generators.  By Schreyer's theorem the
    relations obtained from reducing every S-polynomial to zero generate the
    syzygy module of the Gröbner basis; pushing them through the cofactor
    matrix gives generators of the syzygy module of the original ``f_i``.

    References
    ----------
    .. [1] Eisenbud, D. (1995). "Commutative Algebra with a View Toward
            Algebraic Geometry." Springer, Theorem 15.10.
    .. [2] Cox, Little & O'Shea (2005). "Using Algebraic Geometry." Ch. 5.
    """
    order = "grevlex"
    n = len(generators)
    polys = [sp.Poly(sp.expand(f), *variables) for f in generators]

    # Basis elements together with their cofactor rows (g_k = sum_i M[k][i] f_i).
    basis: list[sp.Poly] = []
    cofactors: list[list[sp.Expr]] = []
    syzygies_g: list[list[sp.Expr]] = []  # relations among the basis elements

    for i, f in enumerate(polys):
        if f.is_zero:
            row = [sp.Integer(0)] * n
            row[i] = sp.Integer(1)
            syzygies_g.append(row)  # f_i = 0 is itself a syzygy of the inputs
            continue
        basis.append(f)
        row = [sp.Integer(0)] * n
        row[i] = sp.Integer(1)
        cofactors.append(row)

    zero_rows = list(syzygies_g)
    syzygies_g = []

    def _reduce(poly: sp.Poly) -> tuple[list[sp.Expr], sp.Poly]:
        quotients, remainder = sp.reduced(
            poly.as_expr(), [g.as_expr() for g in basis], *variables, order=order
        )
        return [sp.expand(q) for q in quotients], sp.Poly(remainder, *variables)

    pairs = [(i, j) for i in range(len(basis)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        gi, gj = basis[i], basis[j]
        lt_i = sp.Poly(sp.LT(gi.as_expr(), *variables, order=order), *variables)
        lt_j = sp.Poly(sp.LT(gj.as_expr(), *variables, order=order), *variables)
        lcm = sp.Poly(sp.lcm(lt_i.as_expr(), lt_j.as_expr()), *variables)
        a_i = sp.cancel(lcm.as_expr() / lt_i.as_expr())
        a_j = sp.cancel(lcm.as_expr() / lt_j.as_expr())
        s_poly = sp.Poly(sp.expand(a_i * gi.as_expr() - a_j * gj.as_expr()), *variables)

        quotients, remainder = (
            _reduce(s_poly) if not s_poly.is_zero else ([sp.Integer(0)] * len(basis), s_poly)
        )

        # Relation among basis elements: a_i e_i - a_j e_j - sum q_k e_k - e_new = 0.
        relation = [-q for q in quotients]
        relation[i] += a_i
        relation[j] -= a_j
        if not remainder.is_zero:
            new_index = len(basis)
            basis.append(remainder)
            new_row = [
                sp.expand(
                    a_i * cofactors[i][c]
                    - a_j * cofactors[j][c]
                    - sum(q * cofactors[k][c] for k, q in enumerate(quotients))
                )
                for c in range(n)
            ]
            cofactors.append(new_row)
            pairs.extend((new_index, k) for k in range(new_index))
            relation.append(sp.Integer(-1))
        syzygies_g.append(relation)

    # Push relations among basis elements down to relations among the inputs.
    result: list[list[sp.Expr]] = list(zero_rows)
    seen: set[tuple[sp.Expr, ...]] = set()
    for rel in syzygies_g:
        row = [sp.expand(sum(rel[k] * cofactors[k][c] for k in range(len(rel)))) for c in range(n)]
        if all(c == 0 for c in row):
            continue
        key = tuple(row)
        neg_key = tuple(-c for c in row)
        if key in seen or neg_key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def trivial_syzygy(
    gen_idx1: int,
    gen_idx2: int,
    num_generators: int,
) -> list[sp.Expr]:
    """
    Create a trivial syzygy: f_i · f_j - f_j · f_i = 0.

    Parameters
    ----------
    gen_idx1 : int
        Index of first generator.
    gen_idx2 : int
        Index of second generator.
    num_generators : int
        Total number of generators.

    Returns
    -------
    List[sp.Expr]
        Coefficients of the trivial syzygy.

    Examples
    --------
    >>> # For 3 generators, syzygy between gen 0 and gen 1:
    >>> syzygy = trivial_syzygy(0, 1, 3)
    >>> # Returns: [f_1, -f_0, 0]
    """
    coeffs = [sp.Integer(0)] * num_generators

    # Create symbolic generators
    f = [sp.Symbol(f"f_{i}") for i in range(num_generators)]

    coeffs[gen_idx1] = f[gen_idx2]
    coeffs[gen_idx2] = -f[gen_idx1]

    return coeffs
