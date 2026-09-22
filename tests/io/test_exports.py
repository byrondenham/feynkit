"""Tests for I/O export functions."""

from collections.abc import Generator
from pathlib import Path

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.io import (
    edges_to_latex_table,
    edges_to_text,
    factor_energy_scale,
    graph_to_latex_table,
    graph_to_text,
    to_latex,
)
from feynkit.io.latex import (
    _create_analysis_document,
    euler_equation_to_latex,
    to_latex_split,
)
from feynkit.io.text import _create_analysis_report


@pytest.fixture  # type: ignore
def simple_graph() -> Generator[Graph, None, None]:
    """Create a simple graph for testing."""
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    yield Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle() -> FeynmanIntegral:
    """Massless triangle, used by the LaTeX helper tests."""
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


class TestLatexExport:
    """Test LaTeX export functions."""

    def test_to_latex_simple(self) -> None:
        """Test converting simple expression to LaTeX."""
        x, y = sp.symbols("x y")
        expr = x**2 + y
        latex = to_latex(expr)

        assert "x^{2}" in latex
        assert "y" in latex

    def test_graph_to_latex_table(self, simple_graph: Graph) -> None:
        """Test generating LaTeX table for graph."""
        table = graph_to_latex_table(simple_graph)

        assert "\\begin{tabular}" in table
        assert "\\end{tabular}" in table
        assert "Internal vertices" in table
        assert "2" in table

    def test_edges_to_latex_table(self, simple_graph: Graph) -> None:
        """Test generating LaTeX table for edges."""
        table = edges_to_latex_table(simple_graph)

        assert "\\begin{tabular}" in table
        assert "Internal" in table
        assert "External" in table

    def test_euler_equation_latex_collapses_phi_arguments(self, triangle: FeynmanIntegral) -> None:
        eq = triangle.gkz.euler_equations[0]
        out = euler_equation_to_latex(eq)
        assert "\\Phi" in out
        assert "\\Phi{\\left(" not in out
        assert "=" in out

    def test_create_analysis_document(self, simple_graph: Graph) -> None:
        """Test creating complete LaTeX document."""
        U = sp.Symbol("U")
        F = sp.Symbol("F")

        doc = _create_analysis_document(
            graph=simple_graph,
            u_polynomial=U,
            f_polynomial=F,
            title="Test Analysis",
        )

        assert "\\documentclass" in doc
        assert "\\begin{document}" in doc
        assert "\\end{document}" in doc
        assert "Test Analysis" in doc


class TestTextExport:
    """Test text export functions."""

    def test_graph_to_text(self, simple_graph: Graph) -> None:
        """Test generating text summary of graph."""
        text = graph_to_text(simple_graph)

        assert "FEYNMAN GRAPH" in text
        assert "Internal vertices" in text
        assert "2" in text

    def test_edges_to_text(self, simple_graph: Graph) -> None:
        """Test generating text list of edges."""
        text = edges_to_text(simple_graph)

        assert "EDGE LIST" in text
        assert "Internal" in text
        assert "External" in text

    def test_create_analysis_report(self, simple_graph: Graph) -> None:
        """Test creating complete text report."""
        U = sp.Symbol("U")
        F = sp.Symbol("F")

        report = _create_analysis_report(
            graph=simple_graph,
            u_polynomial=U,
            f_polynomial=F,
            title="Test Report",
        )

        assert "Test Report" in report
        assert "SYMANZIK POLYNOMIALS" in report
        assert "U =" in report
        assert "F =" in report


class TestFileSaving:
    """Test file saving functions."""

    def test_save_latex_document(self, simple_graph: Graph, tmp_path: Path) -> None:
        """Test saving LaTeX document to file."""
        from feynkit.io import save_latex_document

        U = sp.Symbol("U")
        F = sp.Symbol("F")

        doc = _create_analysis_document(
            graph=simple_graph,
            u_polynomial=U,
            f_polynomial=F,
        )

        output_file = tmp_path / "test.tex"
        _result = save_latex_document(doc, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "\\documentclass" in content

    def test_save_text_report(self, simple_graph: Graph, tmp_path: Path) -> None:
        """Test saving text report to file."""
        from feynkit.io import save_text_report

        U = sp.Symbol("U")
        F = sp.Symbol("F")

        report = _create_analysis_report(
            graph=simple_graph,
            u_polynomial=U,
            f_polynomial=F,
        )

        output_file = tmp_path / "test.txt"
        _result = save_text_report(report, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "FEYNMAN GRAPH" in content


class TestLatexHelpers:
    def test_products_are_juxtaposed_by_default(self) -> None:
        x, y = sp.symbols("x y")
        assert to_latex(2 * x * y) == "2 x y"

    def test_factor_energy_scale_pulls_out_mu(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        mu = fi.graph.energy_scale
        numerator, power = factor_energy_scale(fi.symanzik.f, mu)
        assert power == 2
        assert mu not in numerator.free_symbols
        assert sp.expand(numerator / mu**2 - fi.symanzik.f) == 0

    def test_factor_energy_scale_leaves_u_alone(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        numerator, power = factor_energy_scale(fi.symanzik.u, fi.graph.energy_scale)
        assert power == 0 and numerator == fi.symanzik.u


class TestLatexSplit:
    def test_short_expression_is_unchanged(self) -> None:
        x = sp.Symbol("x")
        assert to_latex_split(x**2 + 1) == sp.latex(x**2 + 1)

    def test_long_expression_gets_split_environment(self) -> None:
        xs = sp.symbols("x0:40")
        expr = sum(x**2 for x in xs)
        out = to_latex_split(expr, max_length=40)
        assert "\\begin{split}" in out
        assert "\\\\" in out
        # Every term survives the split.
        assert out.replace("\\\\", "").replace("&", "").count("x_{") == 40
