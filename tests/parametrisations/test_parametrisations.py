"""Tests for parametric representations, exercised through the FeynmanIntegral façade."""

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph


@pytest.fixture  # type: ignore[misc]
def bubble_integral() -> Generator[FeynmanIntegral, None, None]:
    """Build a bubble-diagram FeynmanIntegral with the standard symbolic defaults."""
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    nu1, nu2 = sp.symbols("nu1 nu2", positive=True)
    yield FeynmanIntegral(graph, propagator_exponents={1: nu1, 2: nu2})


class TestParametrisationsAvailable:
    """All three parametrisations and the Symanzik polynomials are accessible."""

    def test_all_three_parametrisations_present(self, bubble_integral: FeynmanIntegral) -> None:
        assert bubble_integral.schwinger is not None
        assert bubble_integral.feynman is not None
        assert bubble_integral.lee_pomeransky is not None

    def test_symanzik_polynomials_nonzero(self, bubble_integral: FeynmanIntegral) -> None:
        assert bubble_integral.symanzik.u != 0
        assert bubble_integral.symanzik.f != 0

    def test_all_three_have_correct_names(self, bubble_integral: FeynmanIntegral) -> None:
        assert bubble_integral.schwinger.name == "Schwinger"
        assert bubble_integral.feynman.name == "Feynman"
        assert bubble_integral.lee_pomeransky.name == "Lee-Pomeransky"


class TestSchwingerParametrisation:
    def test_schwinger_compute(self, bubble_integral: FeynmanIntegral) -> None:
        result = bubble_integral.schwinger
        assert result.name == "Schwinger"
        assert len(result.parameters) == 2
        assert result.prefactor is not None
        assert result.measure is not None
        assert result.integrand is not None

    def test_schwinger_parameters_named_correctly(self, bubble_integral: FeynmanIntegral) -> None:
        param_names = [str(p) for p in bubble_integral.schwinger.parameters]
        assert "alpha_1" in param_names
        assert "alpha_2" in param_names

    def test_schwinger_no_constraints(self, bubble_integral: FeynmanIntegral) -> None:
        # Schwinger has implicit positivity, no explicit constraints.
        assert len(bubble_integral.schwinger.constraints) == 0


class TestFeynmanParametrisation:
    def test_feynman_compute(self, bubble_integral: FeynmanIntegral) -> None:
        result = bubble_integral.feynman
        assert result.name == "Feynman"
        assert len(result.parameters) == 2
        assert result.prefactor is not None
        assert result.measure is not None
        assert result.integrand is not None

    def test_feynman_simplex_constraint(self, bubble_integral: FeynmanIntegral) -> None:
        result = bubble_integral.feynman
        # One constraint: sum(a_i) = 1.
        assert len(result.constraints) == 1
        assert isinstance(result.constraints[0], sp.Equality)


class TestLeePomeranskyParametrisation:
    def test_lee_pomeransky_compute(self, bubble_integral: FeynmanIntegral) -> None:
        result = bubble_integral.lee_pomeransky
        assert result.name == "Lee-Pomeransky"
        assert len(result.parameters) == 2
        assert result.prefactor is not None
        assert result.measure is not None
        assert result.integrand is not None

    def test_lee_pomeransky_parameters_named_correctly(
        self, bubble_integral: FeynmanIntegral
    ) -> None:
        param_names = [str(p) for p in bubble_integral.lee_pomeransky.parameters]
        assert "u_1" in param_names
        assert "u_2" in param_names

    def test_g_polynomial_present(self, bubble_integral: FeynmanIntegral) -> None:
        assert bubble_integral.symanzik.g != 0


class TestParametrisationResultRendering:
    def test_str_contains_expected_sections(self, bubble_integral: FeynmanIntegral) -> None:
        result_str = str(bubble_integral.schwinger)
        assert "Schwinger" in result_str
        assert "Parameters:" in result_str
        assert "Prefactor:" in result_str
        assert "Measure:" in result_str
        assert "Integrand:" in result_str
