"""Tests for GKZ hypergeometric system construction.

The lower-level utilities (extract_monomial_support, construct_gkz_matrix,
create_euler_equations) are exercised directly because they are
general-purpose helpers; the complete GKZ system is exercised through the
FeynmanIntegral façade.
"""

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.systems import (
    construct_gkz_matrix,
    create_euler_equations,
    extract_monomial_support,
)


@pytest.fixture  # type: ignore[misc]
def simple_polynomial() -> Generator[tuple[sp.Expr, list[sp.Symbol]], None, None]:
    """A small polynomial used to exercise the low-level GKZ helpers."""
    u1, u2 = sp.symbols("u1 u2", nonnegative=True)
    poly = u1**2 + u1 * u2 + u2**2
    yield poly, [u1, u2]


@pytest.fixture  # type: ignore[misc]
def bubble_integral() -> Generator[FeynmanIntegral, None, None]:
    """Bubble-diagram FeynmanIntegral whose .gkz is the system under test."""
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

    nu1, nu2 = sp.symbols("nu1 nu2", positive=True)
    yield FeynmanIntegral(graph, propagator_exponents={1: nu1, 2: nu2})


class TestMonomialSupport:
    def test_extract_support_simple_polynomial(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        poly, variables = simple_polynomial
        support = extract_monomial_support(poly, variables)
        assert len(support) == 3

        exponents = [exp_vec for exp_vec, _ in support]
        assert (2, 0) in exponents
        assert (1, 1) in exponents
        assert (0, 2) in exponents

    def test_support_coefficients(self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]) -> None:
        poly, variables = simple_polynomial
        support = extract_monomial_support(poly, variables)
        for _, coeff in support:
            assert coeff == 1


class TestGKZMatrix:
    def test_construct_matrix_simple(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        poly, variables = simple_polynomial
        A = construct_gkz_matrix(poly, variables)

        # 2 variables + 1 homogenising row, 3 monomials.
        assert A.shape == (3, 3)
        for j in range(3):
            assert A[0, j] == 1

    def test_matrix_exponents_match_support(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        poly, variables = simple_polynomial
        A = construct_gkz_matrix(poly, variables)
        support = extract_monomial_support(poly, variables)

        for j, (exp_vec, _) in enumerate(support):
            assert A[1, j] == exp_vec[0]
            assert A[2, j] == exp_vec[1]


class TestEulerEquations:
    def test_create_euler_equations(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        poly, variables = simple_polynomial
        A = construct_gkz_matrix(poly, variables)

        beta = [sp.Symbol(f"beta_{i}") for i in range(3)]
        z_vars = [sp.Symbol(f"z_{i}") for i in range(3)]

        equations = create_euler_equations(A, beta, z_vars)
        assert len(equations) == 3
        for eq in equations:
            assert isinstance(eq, sp.Equality)


class TestCompleteGKZSystem:
    def test_create_gkz_system_bubble(self, bubble_integral: FeynmanIntegral) -> None:
        gkz = bubble_integral.gkz
        u_vars = bubble_integral.symanzik.lp_parameters

        assert gkz.a_matrix is not None
        assert len(gkz.z_variables) > 0
        assert len(gkz.support) > 0
        assert len(gkz.beta_parameters) == len(u_vars) + 1
        assert len(gkz.euler_equations) == len(u_vars) + 1

    def test_gkz_beta_parameters(self, bubble_integral: FeynmanIntegral) -> None:
        gkz = bubble_integral.gkz
        D = bubble_integral.dimension
        nus = list(bubble_integral.propagator_exponents.values())

        expected_beta_0 = sum(nus) - D / 2
        assert sp.simplify(gkz.beta_parameters[0] - expected_beta_0) == 0
        for i, nu in enumerate(nus, start=1):
            assert sp.simplify(gkz.beta_parameters[i] - nu) == 0

    def test_gkz_system_str(self, bubble_integral: FeynmanIntegral) -> None:
        gkz_str = str(bubble_integral.gkz)
        assert "GKZ Hypergeometric System" in gkz_str
        assert "A-matrix" in gkz_str
        assert "Euler equations" in gkz_str
