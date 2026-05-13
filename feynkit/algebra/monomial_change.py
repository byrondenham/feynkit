"""Monomial changes of variables induced by affine maps on exponent vectors."""

from __future__ import annotations

import sympy as sp

from ..core.exceptions import ValidationError


def _as_column_vector(vec: sp.Matrix, name: str) -> sp.Matrix:
    """Validate and normalise an input SymPy matrix as a column vector."""
    if not isinstance(vec, sp.MatrixBase):
        raise ValidationError(f"{name} must be a SymPy Matrix.")

    if vec.cols == 1:
        return sp.Matrix(vec)

    if vec.rows == 1:
        return sp.Matrix(vec).T

    raise ValidationError(f"{name} must be a row or column vector.")


def monomial_substitution_from_affine(
    M: sp.Matrix,
    source_vars: list[sp.Symbol],
    target_vars: list[sp.Symbol],
) -> dict[sp.Symbol, sp.Expr]:
    """Build the monomial substitution ``x_i -> prod_r y_r**M[r, i]``."""
    if not isinstance(M, sp.MatrixBase):
        raise ValidationError("M must be a SymPy Matrix.")

    if M.rows != M.cols:
        raise ValidationError("M must be square.")

    if len(source_vars) != M.cols:
        raise ValidationError("len(source_vars) must equal the number of columns of M.")

    if len(target_vars) != M.rows:
        raise ValidationError("len(target_vars) must equal the number of rows of M.")

    subs: dict[sp.Symbol, sp.Expr] = {}
    for i, x_i in enumerate(source_vars):
        monomial = sp.Integer(1)
        for r, y_r in enumerate(target_vars):
            exponent = sp.nsimplify(M[r, i])
            monomial *= y_r**exponent
        subs[x_i] = monomial

    return subs


def overall_monomial_factor(c: sp.Matrix, target_vars: list[sp.Symbol]) -> sp.Expr:
    """Return the overall monomial factor ``y**c`` as an exact SymPy expression."""
    c_vec = _as_column_vector(c, "c")

    if c_vec.rows != len(target_vars):
        raise ValidationError("len(target_vars) must equal the length of c.")

    factor = sp.Integer(1)
    for idx, y_i in enumerate(target_vars):
        factor *= y_i ** sp.nsimplify(c_vec[idx, 0])

    return factor


def apply_monomial_change(
    polynomial: sp.Expr,
    M: sp.Matrix,
    c: sp.Matrix,
    source_vars: list[sp.Symbol],
    target_vars: list[sp.Symbol],
    include_factor: bool = True,
) -> sp.Expr:
    """Apply the monomial substitution induced by ``a -> c + M a`` to ``polynomial``."""
    subs = monomial_substitution_from_affine(M, source_vars, target_vars)
    transformed = polynomial.subs(subs)

    if include_factor:
        transformed *= overall_monomial_factor(c, target_vars)

    return sp.expand(transformed)


def transform_exponent_vector(
    a: tuple[int, ...] | sp.Matrix,
    M: sp.Matrix,
    c: sp.Matrix,
) -> tuple[sp.Expr, ...]:
    """Transform an exponent vector via ``a -> c + M a``."""
    if not isinstance(M, sp.MatrixBase):
        raise ValidationError("M must be a SymPy Matrix.")

    c_vec = _as_column_vector(c, "c")

    if isinstance(a, tuple):
        a_vec = sp.Matrix(a)
    elif isinstance(a, sp.MatrixBase):
        a_vec = _as_column_vector(a, "a")
    else:
        raise ValidationError("a must be a tuple or a SymPy Matrix.")

    if M.cols != a_vec.rows:
        raise ValidationError("Dimensions of M and a are incompatible.")

    if M.rows != c_vec.rows:
        raise ValidationError("Dimensions of M and c are incompatible.")

    transformed = c_vec + M * a_vec
    return tuple(sp.nsimplify(transformed[i, 0]) for i in range(transformed.rows))


def transform_support(
    support: list[tuple[tuple[int, ...], sp.Expr]],
    M: sp.Matrix,
    c: sp.Matrix,
) -> list[tuple[tuple[sp.Expr, ...], sp.Expr]]:
    """Apply the affine map ``a -> c + M a`` to each exponent vector in support."""
    return [(transform_exponent_vector(a, M, c), coeff) for a, coeff in support]
