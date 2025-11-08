"""Tests for parametric representations."""

from collections.abc import Generator
from typing import Any, cast

import pytest
import sympy as sp

from feynkit import Edge, Graph, create_momentum_products
from feynkit.parametrisations import create_parametrisations
from feynkit.parametrisations.base import ParametrisationResult


@pytest.fixture  # type: ignore
def bubble_setup() -> (
    Generator[tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]], None, None]
):
    """Create bubble diagram setup for testing."""
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

    D = sp.Symbol("D", positive=True)
    nu1, nu2 = sp.symbols("nu1 nu2", positive=True)
    nus = {1: nu1, 2: nu2}
    p_dot = create_momentum_products(n_external=2, use_mandelstam=True)

    yield graph, D, nus, p_dot


class TestCreateParametrisations:
    """Test parametrisation factory function."""

    def test_create_all_parametrisations(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test creating all parametrisations for bubble diagram."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)

        assert all_param.schwinger is not None
        assert all_param.feynman is not None
        assert all_param.lee_pomeransky is not None
        assert all_param.u_polynomial is not None
        assert all_param.f_polynomial is not None

    def test_symanzik_polynomials_computed(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test that Symanzik polynomials are computed."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)

        # U and F should be non-zero expressions
        assert all_param.u_polynomial != 0
        assert all_param.f_polynomial != 0


class TestSchwingerParametrisation:
    """Test Schwinger parametrisation."""

    def test_schwinger_compute(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test computing Schwinger parametrisation."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.schwinger.compute()

        assert result.name == "Schwinger"
        assert len(result.parameters) == 2  # Two alpha parameters
        assert result.prefactor is not None
        assert result.measure is not None
        assert result.integrand is not None

    def test_schwinger_parameters_named_correctly(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test that Schwinger parameters have correct names."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.schwinger.compute()

        param_names = [str(p) for p in result.parameters]
        assert "alpha_1" in param_names
        assert "alpha_2" in param_names

    def test_schwinger_no_constraints(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test that Schwinger has no explicit constraints."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.schwinger.compute()

        # Schwinger has implicit positivity, no explicit constraints
        assert len(result.constraints) == 0


class TestFeynmanParametrisation:
    """Test Feynman parametrisation."""

    def test_feynman_compute(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test computing Feynman parametrisation."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.feynman.compute()

        assert result.name == "Feynman"
        assert len(result.parameters) == 2  # Two a parameters
        assert result.prefactor is not None
        assert result.measure is not None
        assert result.integrand is not None

    def test_feynman_simplex_constraint(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test that Feynman has simplex constraint."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.feynman.compute()

        # Should have one constraint: sum(a_i) = 1
        assert len(result.constraints) == 1
        constraint = result.constraints[0]
        assert isinstance(constraint, sp.Equality)


class TestLeePomeranskyParametrisation:
    """Test Lee-Pomeransky parametrisation."""

    def test_lee_pomeransky_compute(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test computing Lee-Pomeransky parametrisation."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.lee_pomeransky.compute()

        assert result.name == "Lee-Pomeransky"
        assert len(result.parameters) == 2  # Two u parameters
        assert result.prefactor is not None
        assert result.measure is not None
        assert result.integrand is not None

    def test_lee_pomeransky_parameters_named_correctly(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test that Lee-Pomeransky parameters have correct names."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.lee_pomeransky.compute()

        param_names = [str(p) for p in result.parameters]
        assert "u_1" in param_names
        assert "u_2" in param_names

    def test_lee_pomeransky_g_polynomial(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test getting G polynomial from Lee-Pomeransky."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        G = all_param.lee_pomeransky.get_g_polynomial()

        # G should be non-zero
        assert G != 0


class TestComputeAll:
    """Test computing all parametrisations at once."""

    def test_compute_all_returns_three_results(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test that compute_all returns all three results."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        schwinger, feynman, lee_pom = cast(
            tuple[ParametrisationResult, ParametrisationResult, ParametrisationResult],
            all_param.compute_all(),
        )

        assert schwinger.name == "Schwinger"
        assert feynman.name == "Feynman"
        assert lee_pom.name == "Lee-Pomeransky"


class TestParametrisationResult:
    """Test ParametrisationResult string representation."""

    def test_parametrisation_result_str(
        self, bubble_setup: tuple[Graph, sp.Symbol, dict[int, sp.Symbol], dict[Any, sp.Expr]]
    ) -> None:
        """Test string representation of result."""
        graph, D, nus, p_dot = bubble_setup

        all_param = create_parametrisations(graph, D, 1, nus, p_dot)
        result = all_param.schwinger.compute()

        result_str = str(result)
        assert "Schwinger" in result_str
        assert "Parameters:" in result_str
        assert "Prefactor:" in result_str
        assert "Measure:" in result_str
        assert "Integrand:" in result_str
