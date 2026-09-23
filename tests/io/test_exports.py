"""Tests for I/O export functions."""

from collections.abc import Generator
from pathlib import Path

import pytest
import sympy as sp

import feynkit
import feynkit.io
from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.io import (
    AnalysisReport,
    edges_to_latex_table,
    edges_to_text,
    factor_energy_scale,
    graph_to_latex_table,
    graph_to_text,
    render_latex,
    render_text,
    to_latex,
)
from feynkit.io.latex import (
    euler_equation_to_latex,
    to_latex_lines,
    to_latex_split,
)


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


@pytest.fixture(scope="module")  # type: ignore[misc]
def bubble() -> FeynmanIntegral:
    """Massless bubble, whose analysis documents the facade tests render."""
    return FeynmanIntegral.from_cnickel("11e|e|:zz")


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

    def test_create_analysis_document(self, bubble: FeynmanIntegral) -> None:
        """The facade renders the analysis report as a LaTeX document."""
        doc = bubble.to_latex()

        assert doc.startswith("\\documentclass")
        assert "\\begin{document}" in doc
        assert doc.rstrip().endswith("\\end{document}")
        assert f"\\title{{Feynman integral \\texttt{{{bubble.cnickel}}}}}" in doc


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

    def test_create_analysis_report(self, bubble: FeynmanIntegral) -> None:
        """The facade renders the analysis report as plain text."""
        report = bubble.to_text()

        title = f"Feynman integral {bubble.cnickel}"
        assert report.startswith(f"{title}\n{'=' * len(title)}\n")
        assert "    U = a_1 + a_2\n" in report
        assert "\nReferences\n----------\n" in report


class TestFileSaving:
    """Test file saving functions."""

    def test_save_latex_document(self, bubble: FeynmanIntegral, tmp_path: Path) -> None:
        """The LaTeX document is written to the file named, unchanged."""
        from feynkit.io import save_latex_document

        doc = bubble.to_latex()
        output_file = tmp_path / "test.tex"

        assert save_latex_document(doc, str(output_file)) == str(output_file)
        content = output_file.read_text(encoding="utf-8")
        assert content == doc
        assert content.startswith("\\documentclass") and bubble.cnickel in content

    def test_save_text_report(self, bubble: FeynmanIntegral, tmp_path: Path) -> None:
        """The text report is written to the file named, unchanged."""
        from feynkit.io import save_text_report

        report = bubble.to_text()
        output_file = tmp_path / "test.txt"

        assert save_text_report(report, str(output_file)) == str(output_file)
        content = output_file.read_text(encoding="utf-8")
        assert content == report
        assert content.startswith(f"Feynman integral {bubble.cnickel}\n")


class TestAnalysisFacade:
    """FeynmanIntegral.to_latex and to_text build the report and render it."""

    def test_documents_render_the_full_report(self, bubble: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(bubble)
        assert bubble.to_latex() == render_latex(report)
        assert bubble.to_text() == render_text(report)

    def test_sections_and_title_pass_through(self, bubble: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(bubble, ["gkz"])
        latex = bubble.to_latex(["gkz"], title="Bubble")
        text = bubble.to_text(["gkz"], title="Bubble")
        assert latex == render_latex(report, title="Bubble")
        assert text == render_text(report, title="Bubble")
        assert "\\section{GKZ system}" in latex and "\\section{Landau surfaces}" not in latex
        assert text.startswith("Bubble\n======\n") and "\nLandau surfaces\n" not in text

    def test_max_face_points_passes_through(self) -> None:
        # The massless bubble's polytope is a simplex, whose faces need no
        # elimination; the massive bubble's two largest faces exceed two points.
        massive_bubble = FeynmanIntegral.from_cnickel("11e|e|:nn")
        for document in (
            massive_bubble.to_latex(["landau"], max_face_points=2),
            massive_bubble.to_text(["landau"], max_face_points=2),
        ):
            assert "2 faces were skipped as too large to eliminate" in " ".join(document.split())
        assert "skipped" not in massive_bubble.to_text(["landau"])

    def test_old_keyword_arguments_are_rejected(self, bubble: FeynmanIntegral) -> None:
        with pytest.raises(TypeError):
            bubble.to_latex(author="someone")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            bubble.to_text(author="someone")  # type: ignore[call-arg]

    def test_report_api_is_exported(self) -> None:
        from feynkit.io.report import AnalysisReport as defined
        from feynkit.io.report_latex import render_latex as latex_renderer
        from feynkit.io.report_text import render_text as text_renderer

        assert feynkit.AnalysisReport is defined and "AnalysisReport" in feynkit.__all__
        assert feynkit.io.AnalysisReport is defined
        assert feynkit.io.render_latex is latex_renderer
        assert feynkit.io.render_text is text_renderer
        for name in ("AnalysisReport", "render_latex", "render_text"):
            assert name in feynkit.io.__all__

    def test_old_document_generators_are_gone(self) -> None:
        import feynkit.io.latex
        import feynkit.io.text

        for module, suffix, document in (
            (feynkit.io.latex, "latex", "_create_analysis_document"),
            (feynkit.io.text, "text", "_create_analysis_report"),
        ):
            assert not hasattr(module, document)
            for part in ("parametrisation", "gkz_system", "toric_ideal"):
                name = f"{part}_to_{suffix}"
                assert not hasattr(module, name) and not hasattr(feynkit.io, name), name


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

    def test_long_products_split_with_balanced_delimiters(self) -> None:
        # The sums inside \left( ... \right) sit at brace depth zero, so a
        # splitter that counts only braces would break inside them and leave
        # a \left without its \right on the line.
        xs = sp.symbols("x0:12")
        ys = sp.symbols("y0:12")
        s, t = sp.symbols("s t")
        expr = s * sp.Add(*xs) * sp.Add(*ys) + t * sp.Add(*xs[:6]) * sp.Add(*ys[6:])
        out = to_latex_split(expr, max_length=40)
        assert "\\begin{split}" in out
        body = out.removeprefix("\\begin{split}\n").removesuffix("\n\\end{split}")
        lines = body.split(" \\\\\n")
        assert len(lines) == 2
        for line in lines:
            assert line.count("\\left") == line.count("\\right") > 0

    def test_long_product_is_never_broken_inside_its_factors(self) -> None:
        xs = sp.symbols("x0:20")
        expr = sp.Symbol("s") * sp.Add(*xs) * (sp.Add(*xs[:10]) - 1)
        for line in to_latex_lines(expr, max_length=40):
            assert line.count("\\left") == line.count("\\right")


class TestLatexLines:
    def test_short_expression_is_one_line(self) -> None:
        x = sp.Symbol("x")
        assert to_latex_lines(x**2 + 1) == [sp.latex(x**2 + 1)]

    def test_lines_are_the_rows_of_the_split(self) -> None:
        xs = sp.symbols("x0:40")
        expr = sum(x**2 for x in xs)
        lines = to_latex_lines(expr, max_length=40)
        assert len(lines) > 1
        assert all(line.startswith(("+ ", "- ")) for line in lines[1:])
        body = " \\\\\n& ".join(lines)
        assert to_latex_split(expr, max_length=40) == "\\begin{split}\n" + body + "\n\\end{split}"
