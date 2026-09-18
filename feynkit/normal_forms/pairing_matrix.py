"""
Maximal Pairing Matrix Canonicalisation

This module provides symbolic canonicalisation of pairing matrices under
independent row and column permutation, returning the lexicographically
maximal representative.

References
----------
 .. [1] Grinis, A., Kaspryzyk, A. (2013). Normal Forms of Convex Lattice Polytopes."
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
        The lexicographically maximal pairing matrix
    row_permutation : List[int]
        Permutation applied to rows (as list of indices)
    col_permutation : List[int]
        Permutation applied to columns (as list of indices)
    symmetry_vector : Optional[List[Tuple[int, int]]]
        The symmetry vector s computed during canonicalisation, default None
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

    for row1, row2 in zip(rows1, rows2):
        for elem1, elem2 in zip(row1, row2):
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


def _find_maximal_row(PM: sp.Matrix) -> int:
    """
    Find the index of the lexicographically maximal row.

    Returns
    -------
    int
        Index of the lexicographically maximal row.
    """
    m = PM.shape[0]
    max_idx = 0
    max_row = PM.row(0)

    for i in range(1, m):
        row = PM.row(i)
        if matrix_lexicographic_compare(sp.Matrix([row]), sp.Matrix([max_row])) > 0:
            max_idx = i
            max_row = row

    return max_idx


def _find_distinct_row_representatives(PM: sp.Matrix, max_row_idx: int) -> list[int]:
    """
    Find all rows equivalent to the maximal row (same entries up to permutation).

    Returns
    -------
    List[int]
        Indices of representative rows.
    """
    m = PM.shape[0]
    max_row = set(PM.row(max_row_idx))

    equivalent_indices = []
    for i in range(m):
        if set(PM.row(i)) == max_row:
            equivalent_indices.append(i)

    return equivalent_indices


def _find_maximal_element_in_row(PM: sp.Matrix, row_idx: int) -> int:
    """
    Find the index of the maximal element in the given row.

    Returns
    -------
    int
        Index of maximal element in given row.
    """
    row = PM.row(row_idx)
    max_idx = 0
    max_val = row[0]
    for j in range(1, len(row)):
        if symbolic_compare(row[j], max_val) > 0:
            max_idx = j
            max_val = row[j]

    return max_idx


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


def _swap_rows(PM: sp.Matrix, i: int, j: int) -> sp.Matrix:
    """Swap rows i and j in matrix PM."""
    if i == j:
        return PM
    PM_new = PM.copy()
    PM_new[i, :], PM_new[j, :] = PM[j, :].copy(), PM[i, :].copy()
    return PM_new


def _swap_cols(PM: sp.Matrix, i: int, j: int) -> sp.Matrix:
    """Swap colums i and j in matrix PM."""
    if i == j:
        return PM
    PM_new = PM.copy()
    PM_new[:, i], PM_new[:, j] = PM[:, j].copy(), PM[:, i].copy()
    return PM_new


def _maximal_pairing_recursive(
    PM: sp.Matrix, current_row_perm: list[int], current_col_perm: list[int], fixed_rows: int
) -> tuple[sp.Matrix, list[int], list[int]]:
    """
    Recursive core of the maximal pairing matrix algorithm.

    Arguments
    ---------
    PM : sp.Matrix
        Current working matrix
    current_row_perm : List[int]
        Row permutation applied so far
    current_col_perm : List[int]
        Column permutation applied so far
    fixed_rows : int
        Number of rows already fixed from the top

    Returns
    -------
    Tuple[sp.Matrix, List[int], List[int]]
        Tuple of (maximal matrix, final row permutation, final column permutation)
    """
    m, n = PM.shape

    # Base case: all rows fixed OR we've exceeded column count
    if fixed_rows >= m or fixed_rows >= n:
        return PM, current_row_perm[:], current_col_perm[:]

    # Find maximal row among unfixed rows
    unfixed_start = fixed_rows
    max_idx = unfixed_start
    max_row = PM.row(unfixed_start)

    for i in range(unfixed_start + 1, m):
        row = PM.row(i)
        if matrix_lexicographic_compare(sp.Matrix([row]), sp.Matrix([max_row])) > 0:
            max_idx = i
            max_row = row

    # Swap maximal row to position fixed_rows
    PM_working = PM
    row_perm_working = current_row_perm[:]

    if max_idx != fixed_rows:
        PM_working = _swap_rows(PM_working, fixed_rows, max_idx)
        row_perm_working[fixed_rows], row_perm_working[max_idx] = (
            row_perm_working[max_idx],
            row_perm_working[fixed_rows],
        )

    # Find maximal element in the fixed row (only consider unfixed columns)
    max_col_val = None
    max_col_idx = None

    for j in range(fixed_rows, n):
        val = PM_working[fixed_rows, j]
        if max_col_val is None or symbolic_compare(val, max_col_val) > 0:
            max_col_val = val
            max_col_idx = j

    if max_col_idx is None:
        # No columns to process (shouldn't happen in valid input)
        return PM_working, row_perm_working, current_col_perm[:]

    # Find all unfixed columns with the same maximal value
    equivalent_cols = []
    for j in range(fixed_rows, n):
        if symbolic_compare(PM_working[fixed_rows, j], max_col_val) == 0:
            equivalent_cols.append(j)

    # Try all equivalent column placements
    best_matrix = None
    best_row_perm = None
    best_col_perm = None

    for col_idx in equivalent_cols:
        # Create trial configuration with this column choice
        trial_PM = PM_working
        trial_col_perm = current_col_perm[:]

        if col_idx != fixed_rows:
            trial_PM = _swap_cols(trial_PM, fixed_rows, col_idx)
            trial_col_perm[fixed_rows], trial_col_perm[col_idx] = (
                trial_col_perm[col_idx],
                trial_col_perm[fixed_rows],
            )

        # Recurse with one more row fixed
        result_matrix, result_row_perm, result_col_perm = _maximal_pairing_recursive(
            trial_PM, current_row_perm[:], trial_col_perm[:], fixed_rows + 1
        )

    # Keep lexicographically maximal result
    if best_matrix is None or matrix_lexicographic_compare(result_matrix, best_matrix) > 0:
        best_matrix = result_matrix
        best_row_perm = result_row_perm[:]
        best_col_perm = result_col_perm[:]

    return best_matrix, best_row_perm, best_col_perm


def maximal_pairing_matrix(PM: sp.Matrix) -> PairingMatrixResult:
    """
    Compute the lexicographically maximal representative of a pairing matrix
    under independent row and column permutations.

    Arguments
    ---------
    PM : sp.Matrix
        A matrix with symbolic entries.

    Returns
    -------
    PairingMatrixResult
        - PM_max
        - row_permutation
        - col_permutation
        - symmetry_vector

    Example
    -------
    >>> from sympy import Matrix, symbols
    >>> a, b = symbols('a b')
    >>> PM = Matrix([[a, b], [b, a]])
    >>> result = maximal_pairing_matrix(PM)
    >>> result.PM_max
    """
    m, n = PM.shape

    # Step 1: Compute symmetry vector
    s = _compute_symmetry_vector(PM)

    # Step 2: Initialise permutations
    initial_row_perm = list(range(m))
    initial_col_perm = list(range(n))

    # Step 3: Run recursive maximisation
    PM_max, row_perm, col_perm = _maximal_pairing_recursive(
        PM, initial_row_perm, initial_col_perm, fixed_rows=0
    )

    return PairingMatrixResult(
        PM_max=PM_max, row_permutation=row_perm, col_permutation=col_perm, symmetry_vector=s
    )


def is_canonical(PM: sp.Matrix) -> bool:
    """
    Check if a pairing matrix is already in canonical (maximal) form.

    Arguments
    ---------
    PM : sp.Matrix
        A matrix with symbolic entries.

    Returns
    -------
    bool
        True if PM is already maximal, False otherwise
    """
    result = maximal_pairing_matrix(PM)
    return matrix_lexicographic_compare(PM, result.PM_max) == 0
