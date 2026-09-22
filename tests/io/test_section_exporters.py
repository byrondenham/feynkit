"""Tests for the per-section LaTeX and plain-text exporters."""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.io.latex import (
    gkz_system_to_latex,
    parametrisation_to_latex,
    to_latex,
    toric_ideal_to_latex,
)
from feynkit.io.text import (
    gkz_system_to_text,
    parametrisation_to_text,
    toric_ideal_to_text,
)


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


class TestParametrisationExporters:
    @pytest.mark.parametrize("name", ["schwinger", "feynman", "lee_pomeransky"])
    def test_latex_mentions_name_and_parameters(self, triangle: FeynmanIntegral, name: str) -> None:
        result = getattr(triangle, name)
        out = parametrisation_to_latex(result)
        assert result.name in out
        assert "\\begin{" in out
        assert sp.latex(result.parameters[0]) in out

    @pytest.mark.parametrize("name", ["schwinger", "feynman", "lee_pomeransky"])
    def test_text_mentions_name_and_parameters(self, triangle: FeynmanIntegral, name: str) -> None:
        result = getattr(triangle, name)
        out = parametrisation_to_text(result)
        assert result.name in out
        assert str(result.parameters[0]) in out

    def test_text_includes_constraints_when_present(self, triangle: FeynmanIntegral) -> None:
        out = parametrisation_to_text(triangle.feynman)
        # The Feynman representation carries the simplex delta constraint.
        assert triangle.feynman.constraints
        assert str(triangle.feynman.constraints[0]) in out


class TestGkzExporters:
    def test_latex_contains_a_matrix_and_euler_equations(self, triangle: FeynmanIntegral) -> None:
        out = gkz_system_to_latex(triangle.gkz)
        assert "matrix" in out
        assert "\\Phi" in out
        assert "\\beta" in out

    def test_latex_embeds_polytope_tikz_when_given(self, triangle: FeynmanIntegral) -> None:
        marker = "%% POLYTOPE-TIKZ-MARKER"
        out = gkz_system_to_latex(triangle.gkz, include_polytope_tikz=marker)
        assert marker in out
        assert marker not in gkz_system_to_latex(triangle.gkz)

    def test_text_lists_dimensions_and_equations(self, triangle: FeynmanIntegral) -> None:
        gkz = triangle.gkz
        out = gkz_system_to_text(gkz)
        assert f"A-matrix shape:        {gkz.a_matrix.rows} x {gkz.a_matrix.cols}" in out
        assert f"Euler equations ({len(gkz.euler_equations)})" in out
        assert "beta" in out


class TestToricExporters:
    def test_latex_lists_every_generator(self, triangle: FeynmanIntegral) -> None:
        gens = triangle.toric_ideal.generators
        out = toric_ideal_to_latex(gens)
        assert str(len(gens)) in out
        for g in gens:
            assert to_latex(g) in out

    def test_text_lists_every_generator(self, triangle: FeynmanIntegral) -> None:
        gens = triangle.toric_ideal.generators
        out = toric_ideal_to_text(gens)
        for g in gens:
            assert str(g) in out

    def test_empty_generator_list_is_handled(self) -> None:
        assert "0" in toric_ideal_to_text([]) or "no" in toric_ideal_to_text([]).lower()
        assert isinstance(toric_ideal_to_latex([]), str)
