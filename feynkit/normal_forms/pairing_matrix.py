"""
Maximal pairing matrix canonicalisation.

maximal_pairing_matrix returns the lexicographically largest matrix that
independent permutations of the rows and of the columns make from a given
matrix, with the permutations. Matrices are compared as
matrix_lexicographic_compare does: row by row, left to right, entry by entry
in SymPy's default_sort_key order. is_canonical tells whether a matrix is the
maximum of its own orbit.

References
----------
 .. [1] Grinis, R., Kasprzyk, A. M. (2013). Normal Forms of Convex Lattice Polytopes.
        arXiv:1301.6641 [math.AG]
"""

from dataclasses import dataclass
from typing import Any

import sympy as sp
from sympy.core.sorting import default_sort_key


@dataclass
class PairingMatrixResult:
    """
    Result of maximal pairing matrix computation.

    Attributes
    ----------
    PM_max : sp.Matrix
        The lexicographically largest matrix in the orbit of the input under
        independent row and column permutations. It is unique up to entries
        that compare as equal, such as 1.0 and 1.
    row_permutation : list[int]
        The input's rows in the order of PM_max: row i of PM_max is row
        row_permutation[i] of the input.
    col_permutation : list[int]
        The input's columns in the order of PM_max, so that
        PM_max[i, j] == PM[row_permutation[i], col_permutation[j]].
    symmetry_vector : list[tuple[int, int]] | None
        The symmetry vector s_i = (r_i, c_i) of the input: the numbers of
        distinct rows and columns left after i steps that each remove a maximal
        row and a column. PM_max does not depend on it. Default None.
    """

    PM_max: sp.Matrix
    row_permutation: list[int]
    col_permutation: list[int]
    symmetry_vector: list[tuple[int, int]] | None = None


def symbolic_compare(expr1: sp.Expr, expr2: sp.Expr) -> int:
    """
    Compare two SymPy expressions symbollicaly.

    Returns
    -------
        -1 if expr1 < expr2
         0 if expr1 == expr2
         1 if expr1 > expr2

    Notes
    -----
    Uses SymPy's default_sort_key for deterministic ordering.
    """
    key1 = default_sort_key(expr1)
    key2 = default_sort_key(expr2)

    if key1 < key2:
        return -1
    elif key1 > key2:
        return 1
    else:
        return 0


def matrix_lexicographic_compare(M1: sp.Matrix, M2: sp.Matrix) -> int:
    """
    Compare two matrices lexicographically (row by row, left to right).

    Returns
    -------
        -1 if M1 < M2
         0 if M1 == M2
         1 if M1 > M2
    """
    if M1.shape != M2.shape:
        raise ValueError("Matrices must have the same shape for comparison")

    rows1 = M1.tolist()
    rows2 = M2.tolist()

    for row1, row2 in zip(rows1, rows2, strict=True):
        for elem1, elem2 in zip(row1, row2, strict=True):
            cmp = symbolic_compare(elem1, elem2)
            if cmp != 0:
                return cmp

    return 0


def _compute_symmetry_vector(PM: sp.Matrix) -> list[tuple[int, int]]:
    """
    Compute the symmetry vector s=(s_0, s_1, ..., s_{m-1})
    where s_i = (r_i, c_i) counts distinct rows and columns at step i.

    Algorithm from Appendix B, Step 1.
    """
    m, n = PM.shape
    s: list[tuple[int, int]] = []

    # Build frequency maps for rows and columns
    row_multiset: dict[tuple[Any, ...], int] = {}
    col_multiset: dict[tuple[Any, ...], int] = {}

    for i in range(m):
        row_key = tuple(PM.row(i))
        row_multiset[row_key] = row_multiset.get(row_key, 0) + 1

    for j in range(n):
        col_key = tuple(PM.col(j))
        col_multiset[col_key] = col_multiset.get(col_key, 0) + 1

    # Initial counts
    r_0 = len(row_multiset)  # distinct rows
    c_0 = len(col_multiset)  # distinct columns
    s.append((r_0, c_0))

    # Iteratively refine by removing maximal elements
    for _step in range(1, m):
        # Find and remove maximal row
        max_row = max(row_multiset.keys(), key=lambda r: [default_sort_key(x) for x in r])
        row_multiset[max_row] -= 1
        if row_multiset[max_row] == 0:
            del row_multiset[max_row]

        # Find index of maximal row in original matrix
        max_row_idx = None
        for i in range(m):
            if tuple(PM.row(i)) == max_row:
                max_row_idx = i
                break

        # Find and remove maximal column (restricted to max row)
        if max_row_idx is not None:
            restricted_cols = {}
            for col_key, _count in col_multiset.items():
                # Column value at max_row_idx
                for j in range(n):
                    if tuple(PM.col(j)) == col_key:
                        restricted_cols[col_key] = PM[max_row_idx, j]
                        break

            if restricted_cols:
                max_col = max(
                    restricted_cols.keys(), key=lambda c: default_sort_key(restricted_cols[c])
                )
                col_multiset[max_col] -= 1
                if col_multiset[max_col] == 0:
                    del col_multiset[max_col]

        r_i = len(row_multiset)
        c_i = len(col_multiset)
        s.append((r_i, c_i))

    return s


def _apply_permutation(PM: sp.Matrix, row_perm: list[int], col_perm: list[int]) -> sp.Matrix:
    """
    Apply row and column permutations to matrix PM.

    Arguments
    ---------
    PM : sp.Matrix
        Input matrix
    row_perm : List[int]
        Permutation to apply to rows (row_perm[i] = original row index for new row i)
    col_perm : List[int]
        Permutation to apply to columns

    Returns
    -------
    sp.Matrix
        Matrix with row and column permuations applied.
    """
    m, n = PM.shape
    PM_new = sp.zeros(m, n)

    for i in range(m):
        for j in range(n):
            PM_new[i, j] = PM[row_perm[i], col_perm[j]]

    return PM_new


# A matrix as the ranks of its entries in the default_sort_key order, row by row.
_Ranks = list[tuple[int, ...]]
# The columns of one block agree on every row placed so far, and the blocks are
# in the order those rows fix.
_Blocks = tuple[tuple[int, ...], ...]


def _ranks(PM: sp.Matrix) -> _Ranks:
    """
    PM with each entry replaced by the rank of its default_sort_key among those of PM.

    Keys that are neither equal nor ordered, such as those of 1.0 and 1, tie
    and share a rank, so the ranks order entries as symbolic_compare does.
    """
    keys = [[default_sort_key(PM[i, j]) for j in range(PM.cols)] for i in range(PM.rows)]
    rank: dict[Any, int] = {}
    k, previous = -1, None
    for key in sorted({key for row in keys for key in row}):
        if previous is None or previous < key:
            k += 1
        rank[key] = k
        previous = key
    return [tuple(rank[key] for key in row) for row in keys]


def _place(row: tuple[int, ...], blocks: _Blocks) -> tuple[tuple[int, ...], _Blocks]:
    """
    The largest the row can be below the rows already placed, and the blocks it leaves.

    The columns of a block can still be permuted freely, so the row is largest
    with its entries in decreasing order within each block. Each block then
    splits into the columns that take each of its values, the largest first.
    """
    key: list[int] = []
    split: list[tuple[int, ...]] = []
    for block in blocks:
        for value in sorted({row[c] for c in block}, reverse=True):
            columns = tuple(c for c in block if row[c] == value)
            key.extend([value] * len(columns))
            split.append(columns)
    return tuple(key), tuple(split)


def _future(ranks: _Ranks, order: tuple[int, ...], blocks: _Blocks) -> object:
    """
    What the rest of the search depends on: the rows not yet placed, and each
    block as the multiset of its columns restricted to those rows.
    """
    left = [r for r in range(len(ranks)) if r not in order]
    return (
        tuple(left),
        tuple(tuple(sorted(tuple(ranks[r][c] for r in left) for c in block)) for block in blocks),
    )


def _maximal_orders(ranks: _Ranks, n_cols: int) -> tuple[list[int], list[int]]:
    """
    Row and column orders that give the lexicographic maximum of a matrix.

    The first k rows of the maximum are the largest k rows that any
    arrangement starts with, so the rows are placed one at a time, keeping
    every arrangement that ties. An arrangement is the rows placed, in order,
    and the column blocks they leave. Arrangements with the same _future end
    alike, so only one of them is kept, and of equal rows only the first is
    tried. After the last row the columns of each block are equal, so any
    order within a block gives the maximum.
    """
    states: dict[object, tuple[tuple[int, ...], _Blocks]] = {None: ((), (tuple(range(n_cols)),))}
    for _ in ranks:
        best: tuple[int, ...] | None = None
        following: dict[object, tuple[tuple[int, ...], _Blocks]] = {}
        for order, blocks in states.values():
            tried: set[tuple[int, ...]] = set()
            for r, row in enumerate(ranks):
                if r in order or row in tried:
                    continue
                tried.add(row)
                key, split = _place(row, blocks)
                if best is None or key > best:
                    best, following = key, {}
                if key == best:
                    state = ((*order, r), split)
                    future = _future(ranks, *state)
                    if future not in following or state < following[future]:
                        following[future] = state
        states = following
    order, blocks = min(states.values())
    return list(order), [c for block in blocks for c in block]


def maximal_pairing_matrix(PM: sp.Matrix) -> PairingMatrixResult:
    """
    The lexicographic maximum of PM under independent row and column permutations.

    Matrices are compared as matrix_lexicographic_compare does, row by row,
    and entries that compare as equal, such as 1.0 and 1, tie. The search is
    exact: it keeps every arrangement that ties at each row, so its cost grows
    with the symmetries of PM.

    Arguments
    ---------
    PM : sp.Matrix
        A matrix with numeric or symbolic entries.

    Returns
    -------
    PairingMatrixResult
        PM_max, the permutations with
        PM_max[i, j] == PM[row_permutation[i], col_permutation[j]], and the
        symmetry vector of PM.

    Example
    -------
    >>> from sympy import Matrix
    >>> maximal_pairing_matrix(Matrix([[0, 1], [1, 0]])).PM_max
    Matrix([
    [1, 0],
    [0, 1]])
    """
    row_permutation, col_permutation = _maximal_orders(_ranks(PM), PM.cols)
    return PairingMatrixResult(
        PM_max=_apply_permutation(PM, row_permutation, col_permutation),
        row_permutation=row_permutation,
        col_permutation=col_permutation,
        symmetry_vector=_compute_symmetry_vector(PM),
    )


def is_canonical(PM: sp.Matrix) -> bool:
    """
    Whether PM is the lexicographic maximum of its orbit under independent row
    and column permutations, that is, whether matrix_lexicographic_compare
    finds PM equal to maximal_pairing_matrix(PM).PM_max.

    Matrices in one orbit have the same maximum, so each orbit holds one
    canonical matrix, up to entries that compare as equal: [[1.0, 1]] and
    [[1, 1.0]] are both canonical. Every PM_max is canonical.

    Arguments
    ---------
    PM : sp.Matrix
        A matrix with numeric or symbolic entries.

    Returns
    -------
    bool
        True if PM is its own maximum, False otherwise.
    """
    # The search alone, without the symmetry vector maximal_pairing_matrix also
    # computes.
    rows, cols = _maximal_orders(_ranks(PM), PM.cols)
    return matrix_lexicographic_compare(PM, PM.extract(rows, cols)) == 0
