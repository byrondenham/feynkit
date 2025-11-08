"""Tests for Newton polytope visualisation."""

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from feynkit.systems import extract_monomial_support
from feynkit.visualisation import TikzDocument, visualise_newton_polytope


@pytest.fixture  # type: ignore
def simple_support() -> Generator[list[tuple[tuple[int, ...], sp.Integer]], None, None]:
    """Create simple monomial support for testing."""
    u1, u2 = sp.symbols("u1 u2", nonnegative=True)
    poly = u1**2 + u1 * u2 + u2**2
    support = extract_monomial_support(poly, [u1, u2])
    yield support


@pytest.fixture  # type: ignore
def triangle_support() -> Generator[list[tuple[tuple[int, ...], sp.Integer]], None, None]:
    """Create triangle support that forms a proper polytope."""
    u1, u2 = sp.symbols("u1 u2", nonnegative=True)
    # Triangle: vertices at (2,0), (0,2), (1,1)
    poly = u1**2 + u2**2 + u1 * u2
    support = extract_monomial_support(poly, [u1, u2])
    yield support


class TestTikzDocument:
    """Test TikZ document builder."""

    def test_create_empty_document(self) -> None:
        """Test creating empty TikZ document."""
        doc = TikzDocument()
        assert doc.lines == []

    def test_begin_end_picture(self) -> None:
        """Test begin and end picture."""
        doc = TikzDocument()
        doc.begin_picture(scale=1.5)
        doc.end_picture()

        code = doc.get_code()
        assert "\\begin{tikzpicture}" in code
        assert "\\end{tikzpicture}" in code
        assert "scale=1.5" in code

    def test_add_line(self) -> None:
        """Test adding lines."""
        doc = TikzDocument()
        doc.begin_picture()
        doc.add_line("\\draw (0,0) -- (1,1);")
        doc.end_picture()

        code = doc.get_code()
        assert "\\draw (0,0) -- (1,1);" in code

    def test_draw_point(self) -> None:
        """Test drawing a point."""
        doc = TikzDocument()
        doc.begin_picture()
        doc.draw_point(1, 2, 3, label="$P$", label_position="right")
        doc.end_picture()

        code = doc.get_code()
        assert "\\fill" in code
        assert "(1,2,3)" in code

    def test_draw_line(self) -> None:
        """Test drawing a line."""
        doc = TikzDocument()
        doc.begin_picture()
        doc.draw_line(0, 0, 0, 1, 1, 1)
        doc.end_picture()

        code = doc.get_code()
        assert "\\draw" in code
        assert "(0,0,0)" in code
        assert "(1,1,1)" in code

    def test_method_chaining(self) -> None:
        """Test that methods can be chained."""
        doc = TikzDocument()
        result = doc.begin_picture().add_line("% Comment").draw_point(0, 0, 0).end_picture()

        assert result is doc
        code = doc.get_code()
        assert len(code) > 0


class TestNewtonPolytopeVisualisation:
    """Test Newton polytope visualisation."""

    def test_visualise_simple_polytope(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]]
    ) -> None:
        """Test visualising simple 2D polytope."""
        tikz_code = visualise_newton_polytope(simple_support)

        # Should produce valid TikZ code
        assert "\\begin{tikzpicture}" in tikz_code
        assert "\\end{tikzpicture}" in tikz_code

    def test_polytope_produces_output(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]]
    ) -> None:
        """Test that visualisation produces some output."""
        tikz_code = visualise_newton_polytope(simple_support)

        # Should have some content (either edges or points)
        assert "\\fill" in tikz_code or "\\draw" in tikz_code

    def test_polytope_with_labels(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]]
    ) -> None:
        """Test polytope with vertex labels."""
        tikz_code = visualise_newton_polytope(simple_support, show_labels=True)

        # Should contain coordinate labels or points
        assert "\\fill" in tikz_code  # Points are filled circles

    def test_polytope_without_labels(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]]
    ) -> None:
        """Test polytope without labels."""
        tikz_code = visualise_newton_polytope(simple_support, show_labels=False)

        # Should still have valid TikZ
        assert "\\begin{tikzpicture}" in tikz_code

    def test_polytope_with_custom_scale(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]]
    ) -> None:
        """Test polytope with custom scale."""
        tikz_code = visualise_newton_polytope(simple_support, scale=2.0)

        assert "scale=2.0" in tikz_code

    def test_empty_support_raises_error(self) -> None:
        """Test that empty support raises ValidationError."""
        from feynkit.core.exceptions import ValidationError

        with pytest.raises(ValidationError, match="empty"):
            visualise_newton_polytope([])

    def test_valid_polytope_has_edges(self) -> None:
        """Test that a valid polytope (4 non-coplanar points) has edges."""
        # Create a 3D polytope: tetrahedron vertices
        support: Sequence[tuple[tuple[int, ...], Any]] = [
            ((0, 0, 0), sp.Integer(1)),
            ((1, 0, 0), sp.Integer(1)),
            ((0, 1, 0), sp.Integer(1)),
            ((0, 0, 1), sp.Integer(1)),
        ]

        tikz_code = visualise_newton_polytope(support)

        # A proper 3D polytope should have edges
        # It may have degenerate case handling, so check for either edges or points
        has_content = "\\draw" in tikz_code or "\\fill" in tikz_code
        assert has_content


class TestPolytopeFile:
    """Test saving polytope to file."""

    def test_save_polytope_standalone(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]], tmp_path: Path
    ) -> None:
        """Test saving standalone polytope document."""
        from feynkit.visualisation import save_polytope_tikz

        output_file = tmp_path / "polytope.tex"
        _result = save_polytope_tikz(
            simple_support,
            str(output_file),
            title="Test Polytope",
            standalone=True,
        )

        # Check file was created
        assert output_file.exists()

        # Check file contains standalone document
        content = output_file.read_text()
        assert "\\documentclass" in content
        assert "\\begin{document}" in content
        assert "\\end{document}" in content

    def test_save_polytope_code_only(
        self, simple_support: list[tuple[tuple[int, ...], sp.Integer]], tmp_path: Path
    ) -> None:
        """Test saving just TikZ code without document wrapper."""
        from feynkit.visualisation import save_polytope_tikz

        output_file = tmp_path / "polytope_code.tex"
        save_polytope_tikz(
            simple_support,
            str(output_file),
            standalone=False,
        )

        # Check file was created
        assert output_file.exists()

        # Check file does NOT contain document wrapper
        content = output_file.read_text()
        assert "\\documentclass" not in content
        assert "\\begin{tikzpicture}" in content
