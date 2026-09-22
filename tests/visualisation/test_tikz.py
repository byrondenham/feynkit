"""Tests for the TikZ output of feynkit.visualisation.tikz and the polytope writer.

graph_to_tikz lays the internal vertices out on a line for a two-vertex graph,
on a triangle for a three-vertex graph and on a circle otherwise, in the order of
a walk over the internal edges, then draws one
propagator per internal edge, bending parallel propagators apart, and one dashed
leg per external edge, pointing away from the centre of the diagram.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral, Graph
from feynkit.visualisation import graph_to_tikz, save_polytope_tikz


def count_vertices(code: str) -> int:
    """Number of internal vertex nodes in the TikZ code."""
    return code.count("\\node[vertex]")


def count_propagators(code: str) -> int:
    """Number of internal propagator lines in the TikZ code."""
    return code.count("\\draw[propagator]")


def count_external_nodes(code: str) -> int:
    """Number of external leg endpoints in the TikZ code."""
    return code.count("\\node[external]")


@pytest.fixture  # type: ignore[misc]
def bubble_graph() -> Generator[Graph, None, None]:
    """Massless bubble: two vertices joined by two parallel propagators."""
    yield Graph.from_cnickel("11e|e|:zz")


@pytest.fixture  # type: ignore[misc]
def triangle_graph() -> Generator[Graph, None, None]:
    """Massless triangle: three vertices, three propagators, three legs."""
    yield Graph.from_cnickel("12e|2e|e|:zzz")


@pytest.fixture  # type: ignore[misc]
def box_graph() -> Generator[Graph, None, None]:
    """Massless box: four vertices, four propagators, four legs."""
    yield Graph.from_cnickel("13e|2e|3e|e|:zzzz")


class TestFigureScaffolding:
    @pytest.mark.parametrize(  # type: ignore[misc]
        "cnickel", ["11e|e|:zz", "12e|2e|e|:zzz", "13e|2e|3e|e|:zzzz"]
    )
    def test_figure_environment_and_label_are_present(self, cnickel: str) -> None:
        code = graph_to_tikz(Graph.from_cnickel(cnickel))

        assert code.startswith("\\begin{figure}[htbp]")
        assert code.rstrip().endswith("\\end{figure}")
        assert "\\centering" in code
        assert "\\begin{tikzpicture}[" in code
        assert "\\end{tikzpicture}" in code
        assert "\\caption{" in code
        assert "\\label{fig:feynman_graph}" in code

    def test_style_definitions_are_declared_once(self, triangle_graph: Graph) -> None:
        code = graph_to_tikz(triangle_graph)

        assert code.count("vertex/.style=") == 1
        assert code.count("external/.style=") == 1
        assert code.count("propagator/.style=") == 1
        assert code.count("external_leg/.style=") == 1


class TestTwoVertexGraph:
    def test_two_vertices_are_placed_on_a_horizontal_line(self, bubble_graph: Graph) -> None:
        code = graph_to_tikz(bubble_graph)

        assert count_vertices(code) == 2
        assert "\\node[vertex] (v1) at (0, 0) {};" in code
        assert "\\node[vertex] (v2) at (3, 0) {};" in code

    def test_both_parallel_propagators_should_be_drawn(self, bubble_graph: Graph) -> None:
        internal_edges = [edge for edge in bubble_graph.edges if edge.is_internal]
        assert len(internal_edges) == 2

        code = graph_to_tikz(bubble_graph)

        assert count_propagators(code) == 2

    def test_banana_draws_three_distinct_propagators(self) -> None:
        graph = Graph.from_cnickel("111e|e|:nnn")
        code = graph_to_tikz(graph)
        assert count_propagators(code) == 3
        assert code.count("bend") == 2  # one straight, two bent

    def test_two_external_legs_are_attached(self, bubble_graph: Graph) -> None:
        code = graph_to_tikz(bubble_graph)

        assert count_external_nodes(code) == 2
        assert "\\draw[external_leg] (v1) -- (e1);" in code
        assert "\\draw[external_leg] (v2) -- (e2);" in code


class TestThreeVertexGraph:
    def test_three_vertices_are_placed_on_a_triangle(self, triangle_graph: Graph) -> None:
        """The fixed layout puts the vertices at 90, 210 and 330 degrees."""
        code = graph_to_tikz(triangle_graph)

        assert count_vertices(code) == 3
        assert "\\node[vertex] (v1) at (90:2cm) {};" in code
        assert "\\node[vertex] (v2) at (210:2cm) {};" in code
        assert "\\node[vertex] (v3) at (330:2cm) {};" in code

    def test_every_internal_edge_is_drawn(self, triangle_graph: Graph) -> None:
        """The three propagators 1-2, 1-3 and 2-3 each appear exactly once."""
        code = graph_to_tikz(triangle_graph)
        internal_edges = [edge for edge in triangle_graph.edges if edge.is_internal]

        assert count_propagators(code) == len(internal_edges) == 3
        for edge in internal_edges:
            assert f"\\draw[propagator] (v{edge.v1}) -- (v{edge.v2});" in code

    def test_three_external_legs_are_attached(self, triangle_graph: Graph) -> None:
        code = graph_to_tikz(triangle_graph)

        assert count_external_nodes(code) == 3
        for vertex in (1, 2, 3):
            assert f"\\draw[external_leg] (v{vertex}) -- (e{vertex});" in code

    def test_external_legs_attach_at_their_own_vertex(self) -> None:
        graph = Graph.from_cnickel("12e|2e|e|:zzz")
        code = graph_to_tikz(graph)
        for line in code.splitlines():
            if "\\node[external]" in line:
                # every external node is placed relative to a vertex: "at ($(vK)+(angle:1.5cm)$)"
                assert "$(v" in line


class TestFourVertexGraph:
    def test_four_vertices_are_placed_on_a_circle(self, box_graph: Graph) -> None:
        """Beyond three vertices the layout is generic: 360/4 = 90 degrees apart,
        starting at 90 degrees, and the node names carry the real vertex labels."""
        code = graph_to_tikz(box_graph)

        assert count_vertices(code) == 4
        for index, vertex in enumerate((1, 2, 3, 4)):
            angle = 90 + index * 90.0
            assert f"\\node[vertex] (v{vertex}) at ({angle}:2cm) {{}};" in code

    def test_every_internal_edge_is_drawn(self, box_graph: Graph) -> None:
        """The box has four distinct propagators 1-2, 1-4, 2-3 and 3-4."""
        code = graph_to_tikz(box_graph)
        internal_edges = [edge for edge in box_graph.edges if edge.is_internal]

        assert count_propagators(code) == len(internal_edges) == 4
        for edge in internal_edges:
            assert f"\\draw[propagator] (v{edge.v1}) -- (v{edge.v2});" in code

    def test_box_edges_join_angularly_adjacent_vertices(self) -> None:
        """The circle is walked along the internal edges, so the box whose
        cnickel lists the edges out of cyclic order still draws as a square
        rather than as two diameters crossing at the centre."""
        graph = Graph.from_cnickel("12e|3e|3e|e|:zzzz")
        code = graph_to_tikz(graph)

        angles: dict[int, float] = {}
        for line in code.splitlines():
            found = re.search(r"\\node\[vertex\] \(v(\d+)\) at \(([-\d.]+):2cm\)", line)
            if found:
                angles[int(found.group(1))] = float(found.group(2))

        assert len(angles) == 4
        for edge in graph.get_internal_edges():
            gap = abs(angles[edge.v1] - angles[edge.v2]) % 360
            assert min(gap, 360 - gap) == pytest.approx(90.0)

    def test_four_external_legs_are_attached(self, box_graph: Graph) -> None:
        code = graph_to_tikz(box_graph)

        assert count_external_nodes(code) == 4
        for vertex in (1, 2, 3, 4):
            assert f"\\draw[external_leg] (v{vertex}) -- (e{vertex});" in code


class TestSavePolytopeTikz:
    """The support of the massless sunrise is the tetrahedron with vertices
    (1,1,0), (1,0,1), (0,1,1) and (1,1,1), since

        G = u_1 u_2 + u_1 u_3 + u_2 u_3 - s u_1 u_2 u_3 / mu^2.
    """

    @pytest.fixture  # type: ignore[misc]
    def sunrise_support(self) -> Generator[list[tuple[tuple[int, ...], object]], None, None]:
        integral = FeynmanIntegral.from_cnickel("111e|e|:zzz")
        yield list(integral.newton_polytope.support)

    def test_support_is_the_expected_tetrahedron(
        self, sunrise_support: list[tuple[tuple[int, ...], object]]
    ) -> None:
        assert {exponents for exponents, _ in sunrise_support} == {
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        }

    def test_standalone_document_is_written(
        self,
        sunrise_support: list[tuple[tuple[int, ...], object]],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "sunrise_polytope.tex"

        returned = save_polytope_tikz(
            sunrise_support,
            str(target),
            title="Sunrise Newton Polytope",
            standalone=True,
            show_labels=True,
            show_fill=False,
        )

        assert returned == str(target)
        assert target.exists()

        content = target.read_text(encoding="utf-8")
        assert content.startswith("\\documentclass[tikz,border=5pt]{standalone}")
        assert "\\usepackage{tikz}" in content
        assert "\\usepackage{tikz-3dplot}" in content
        assert "\\begin{document}" in content
        assert content.rstrip().endswith("\\end{document}")
        assert "% Sunrise Newton Polytope" in content

    def test_every_support_point_is_labelled(
        self,
        sunrise_support: list[tuple[tuple[int, ...], object]],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "sunrise_labels.tex"
        save_polytope_tikz(sunrise_support, str(target), show_labels=True)

        content = target.read_text(encoding="utf-8")

        assert content.count("circle (2pt);") == len(sunrise_support) == 4
        for exponents, _ in sunrise_support:
            label = ",".join(str(component) for component in exponents)
            assert f"$({label})$" in content

    def test_non_standalone_output_omits_the_preamble(
        self,
        sunrise_support: list[tuple[tuple[int, ...], object]],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "sunrise_code.tex"

        save_polytope_tikz(sunrise_support, str(target), standalone=False)

        content = target.read_text(encoding="utf-8")
        assert content.startswith("\\begin{tikzpicture}")
        assert "\\documentclass" not in content
        assert "\\begin{document}" not in content
