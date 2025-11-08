"""Tests for GKZ hypergeometric system construction."""

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import Edge, Graph, create_momentum_products, create_parametrisations
from feynkit.systems import (
    construct_gkz_matrix,
    create_euler_equations,
    create_gkz_system,
    extract_monomial_support,
)


@pytest.fixture  # type: ignore
def simple_polynomial() -> Generator[tuple[sp.Expr, list[sp.Symbol]], None, None]:
    """Create a simple polynomial for testing."""
    u1, u2 = sp.symbols("u1 u2", nonnegative=True)
    # poly = u1^2 + u1*u2 + u2^2
    poly = u1**2 + u1 * u2 + u2**2
    yield poly, [u1, u2]


@pytest.fixture  # type: ignore
def bubble_gkz_setup() -> (
    Generator[tuple[sp.Expr, list[sp.Symbol], sp.Symbol, list[sp.Symbol]], None, None]
):
    """Create bubble diagram and GKZ system."""
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

    D = sp.Symbol("D", positive=True)
    nu1, nu2 = sp.symbols("nu1 nu2", positive=True)
    nus = {1: nu1, 2: nu2}
    p_dot = create_momentum_products(n_external=2, use_mandelstam=True)

    all_param = create_parametrisations(graph, D, 1, nus, p_dot)
    G = all_param.lee_pomeransky.get_g_polynomial()
    u_vars = all_param.lee_pomeransky.compute().parameters

    yield G, u_vars, D, [nu1, nu2]


class TestMonomialSupport:
    """Test monomial support extraction."""

    def test_extract_support_simple_polynomial(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Test extracting support from simple polynomial."""
        poly, variables = simple_polynomial
        support = extract_monomial_support(poly, variables)

        # Should have 3 monomials
        assert len(support) == 3

        # Check exponent vectors are present
        exponents = [exp_vec for exp_vec, _ in support]
        assert (2, 0) in exponents
        assert (1, 1) in exponents
        assert (0, 2) in exponents

    def test_support_coefficients(self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]) -> None:
        """Test that coefficients are extracted correctly."""
        poly, variables = simple_polynomial
        support = extract_monomial_support(poly, variables)

        # All coefficients should be 1
        for _, coeff in support:
            assert coeff == 1


class TestGKZMatrix:
    """Test GKZ A-matrix construction."""

    def test_construct_matrix_simple(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Test A-matrix construction from simple polynomial."""
        poly, variables = simple_polynomial
        A = construct_gkz_matrix(poly, variables)

        # Should be 3x3 matrix (2 variables + 1 homogenising row, 3 monomials)
        assert A.shape == (3, 3)

        # First row should be all ones
        for j in range(3):
            assert A[0, j] == 1

    def test_matrix_exponents_match_support(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Test that matrix exponents match monomial support."""
        poly, variables = simple_polynomial
        A = construct_gkz_matrix(poly, variables)
        support = extract_monomial_support(poly, variables)

        # Check each column corresponds to a monomial
        for j, (exp_vec, _) in enumerate(support):
            # Check exponents match (rows 1 and 2 correspond to u1 and u2)
            assert A[1, j] == exp_vec[0]  # u1 exponent
            assert A[2, j] == exp_vec[1]  # u2 exponent


class TestEulerEquations:
    """Test Euler equation generation."""

    def test_create_euler_equations(
        self, simple_polynomial: tuple[sp.Expr, list[sp.Symbol]]
    ) -> None:
        """Test creating Euler equations."""
        poly, variables = simple_polynomial
        A = construct_gkz_matrix(poly, variables)

        # Create beta parameters
        beta = [sp.Symbol(f"beta_{i}") for i in range(3)]

        # Create z variables
        z_vars = [sp.Symbol(f"z_{i}") for i in range(3)]

        equations = create_euler_equations(A, beta, z_vars)

        # Should have 3 equations (one per row of A)
        assert len(equations) == 3

        # Each equation should be an Equality
        for eq in equations:
            assert isinstance(eq, sp.Equality)


class TestCompleteGKZSystem:
    """Test complete GKZ system construction."""

    def test_create_gkz_system_bubble(
        self, bubble_gkz_setup: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, list[sp.Symbol]]
    ) -> None:
        """Test creating complete GKZ system for bubble."""
        G, u_vars, D, nus = bubble_gkz_setup

        gkz = create_gkz_system(G, u_vars, D, nus)

        assert gkz.a_matrix is not None
        assert len(gkz.z_variables) > 0
        assert len(gkz.support) > 0
        assert len(gkz.beta_parameters) == len(u_vars) + 1
        assert len(gkz.euler_equations) == len(u_vars) + 1

    def test_gkz_beta_parameters(
        self, bubble_gkz_setup: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, list[sp.Symbol]]
    ) -> None:
        """Test beta parameter construction."""
        G, u_vars, D, nus = bubble_gkz_setup

        gkz = create_gkz_system(G, u_vars, D, nus)

        # First beta should be sum(nus) - D/2
        nu1, nu2 = nus
        expected_beta_0 = nu1 + nu2 - D / 2
        assert sp.simplify(gkz.beta_parameters[0] - expected_beta_0) == 0

        # Remaining betas should be the nus
        assert sp.simplify(gkz.beta_parameters[1] - nu1) == 0
        assert sp.simplify(gkz.beta_parameters[2] - nu2) == 0

    def test_gkz_system_str(
        self, bubble_gkz_setup: tuple[sp.Expr, list[sp.Symbol], sp.Symbol, list[sp.Symbol]]
    ) -> None:
        """Test string representation of GKZ system."""
        G, u_vars, D, nus = bubble_gkz_setup

        gkz = create_gkz_system(G, u_vars, D, nus)
        gkz_str = str(gkz)

        assert "GKZ Hypergeometric System" in gkz_str
        assert "A-matrix" in gkz_str
        assert "Euler equations" in gkz_str
