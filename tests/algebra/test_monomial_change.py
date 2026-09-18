"""Tests for affine-induced monomial changes of variables."""

import sympy as sp

from feynkit.algebra import (
    apply_monomial_change,
    monomial_substitution_from_affine,
    overall_monomial_factor,
    transform_exponent_vector,
    transform_support,
)


def _support_set(expr: sp.Expr, vars_: list[sp.Symbol]) -> set[tuple[sp.Expr, ...]]:
    support: set[tuple[sp.Expr, ...]] = set()
    for term in sp.Add.make_args(sp.expand(expr)):
        powers = term.as_powers_dict()
        support.add(tuple(sp.nsimplify(powers.get(v, 0)) for v in vars_))
    return support


def test_identity_map() -> None:
    x1, x2 = sp.symbols("x1 x2")
    y1, y2 = sp.symbols("y1 y2")
    M = sp.eye(2)
    c = sp.Matrix([0, 0])

    subs = monomial_substitution_from_affine(M, [x1, x2], [y1, y2])
    assert subs == {x1: y1, x2: y2}

    poly = x1**2 + 3 * x1 * x2 + x2
    transformed = apply_monomial_change(poly, M, c, [x1, x2], [y1, y2])
    assert sp.expand(transformed - (y1**2 + 3 * y1 * y2 + y2)) == 0


def test_pure_translation() -> None:
    x1, x2 = sp.symbols("x1 x2")
    y1, y2 = sp.symbols("y1 y2")
    M = sp.eye(2)
    c = sp.Matrix([1, 2])

    poly = x1 + x2
    transformed = apply_monomial_change(poly, M, c, [x1, x2], [y1, y2], include_factor=True)
    assert sp.expand(transformed - (y1 * y2**2 * (y1 + y2))) == 0


def test_triangle_triplek_substitution_and_factor() -> None:
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    y1, y2, y3 = sp.symbols("y1 y2 y3")

    M = sp.Matrix(
        [
            [0, -1, -1],
            [-1, 0, -1],
            [-1, -1, 0],
        ]
    )
    c = sp.Matrix([1, 1, 1])

    subs = monomial_substitution_from_affine(M, [x1, x2, x3], [y1, y2, y3])
    assert subs == {
        x1: 1 / (y2 * y3),
        x2: 1 / (y1 * y3),
        x3: 1 / (y1 * y2),
    }

    factor = overall_monomial_factor(c, [y1, y2, y3])
    assert factor == y1 * y2 * y3


def test_triangle_exponent_vector_transforms() -> None:
    M = sp.Matrix(
        [
            [0, -1, -1],
            [-1, 0, -1],
            [-1, -1, 0],
        ]
    )
    c = sp.Matrix([1, 1, 1])

    input_exponents = [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]
    expected = {
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    }

    transformed = {transform_exponent_vector(a, M, c) for a in input_exponents}
    assert transformed == expected


def test_triangle_polynomial_support_level_check() -> None:
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    y1, y2, y3 = sp.symbols("y1 y2 y3")

    P_triangle = x1 * x2 + x1 * x3 + x2 * x3 + x1 + x2 + x3

    M = sp.Matrix(
        [
            [0, -1, -1],
            [-1, 0, -1],
            [-1, -1, 0],
        ]
    )
    c = sp.Matrix([1, 1, 1])

    transformed = apply_monomial_change(
        P_triangle,
        M,
        c,
        [x1, x2, x3],
        [y1, y2, y3],
        include_factor=True,
    )

    support = _support_set(transformed, [y1, y2, y3])
    expected_support = {
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    }

    assert support == expected_support


def test_transform_support() -> None:
    support = [((1, 0), sp.Integer(2)), ((0, 1), sp.Integer(3))]
    M = sp.Matrix([[1, 0], [0, sp.Rational(1, 2)]])
    c = sp.Matrix([0, -1])

    transformed = transform_support(support, M, c)
    assert transformed == [((1, -1), sp.Integer(2)), ((0, sp.Rational(-1, 2)), sp.Integer(3))]
