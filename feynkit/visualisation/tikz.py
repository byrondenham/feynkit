"""
TikZ code generation for mathematical diagrams.

Provides functions to generate TikZ/LaTeX code for visualising geometric
structures associated with Feynman integrals.
"""

from typing import Optional

from ..core.exceptions import ValidationError
from ..core.graph import Graph


class TikzDocument:
    """
    Builder class for TikZ documents.

    Provides a fluent interface for constructing TikZ pictures with proper
    formatting and structure.

    Examples
    --------
    >>> doc = TikzDocument()
    >>> doc.begin_picture(scale=1.5)
    >>> doc.add_line("\\draw (0,0) -- (1,1);")
    >>> doc.end_picture()
    >>> print(doc.get_code())
    """

    def __init__(self) -> None:
        """Initialise an empty TikZ document."""
        self.lines: list[str] = []
        self._in_picture = False

    def begin_picture(
        self,
        scale: float = 1.0,
        options: Optional[str] = None,
    ) -> "TikzDocument":
        """
        Begin a TikZ picture environment.

        Parameters
        ----------
        scale : float, default 1.0
            Scale factor for the picture.
        options : Optional[str], default None
            Additional TikZ options.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        if self._in_picture:
            raise ValidationError("Already in a tikzpicture environment")

        opts = [f"scale={scale}"]
        if options:
            opts.append(options)

        self.lines.append(f"\\begin{{tikzpicture}}[{', '.join(opts)}]")
        self._in_picture = True
        return self

    def end_picture(self) -> "TikzDocument":
        """
        End the TikZ picture environment.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        if not self._in_picture:
            raise ValidationError("Not in a tikzpicture environment")

        self.lines.append("\\end{tikzpicture}")
        self._in_picture = False
        return self

    def add_line(self, line: str, indent: int = 1) -> "TikzDocument":
        """
        Add a line of TikZ code.

        Parameters
        ----------
        line : str
            Line of TikZ code to add.
        indent : int, default 1
            Indentation level (number of spaces per level = 4).

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        indent_str = "    " * indent
        self.lines.append(f"{indent_str}{line}")
        return self

    def add_comment(self, comment: str, indent: int = 1) -> "TikzDocument":
        """
        Add a comment line.

        Parameters
        ----------
        comment : str
            Comment text (without % prefix).
        indent : int, default 1
            Indentation level.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        return self.add_line(f"% {comment}", indent=indent)

    def draw_point(
        self,
        x: float,
        y: float,
        z: float = 0,
        label: Optional[str] = None,
        label_position: str = "right",
    ) -> "TikzDocument":
        """
        Draw a point with optional label.

        Parameters
        ----------
        x, y, z : float
            3D coordinates of the point.
        label : Optional[str], default None
            Label text for the point.
        label_position : str, default "right"
            Position of label relative to point.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        # Draw point
        self.add_line(f"\\fill ({x},{y},{z}) circle (2pt);")

        # Add label if provided
        if label:
            self.add_line(f"\\node[{label_position}] at ({x},{y},{z}) {{{label}}};")

        return self

    def draw_line(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        style: str = "thick",
        color: str = "black",
    ) -> "TikzDocument":
        """
        Draw a line between two 3D points.

        Parameters
        ----------
        x1, y1, z1 : float
            Coordinates of first point.
        x2, y2, z2 : float
            Coordinates of second point.
        style : str, default "thick"
            Line style (thick, thin, dashed, etc.).
        color : str, default "black"
            Line color.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        self.add_line(f"\\draw[{style},{color}] ({x1},{y1},{z1}) -- ({x2},{y2},{z2});")
        return self

    def draw_dashed_line(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        color: str = "gray",
    ) -> "TikzDocument":
        """
        Draw a dashed line (for hidden edges).

        Parameters
        ----------
        x1, y1, z1 : float
            Coordinates of first point.
        x2, y2, z2 : float
            Coordinates of second point.
        color : str, default "gray"
            Line color.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        self.add_line(f"\\draw[dashed,{color}] ({x1},{y1},{z1}) -- ({x2},{y2},{z2});")
        return self

    def draw_filled_polygon(
        self,
        points: list[tuple[float, float, float]],
        fill_color: str = "blue!20",
        edge_color: str = "black",
    ) -> "TikzDocument":
        """
        Draw a filled polygon.

        Parameters
        ----------
        points : List[Tuple[float, float, float]]
            List of 3D coordinates defining the polygon.
        fill_color : str, default "blue!20"
            Fill color (with opacity).
        edge_color : str, default "black"
            Edge color. Use "none" for no edge.

        Returns
        -------
        TikzDocument
            Self for method chaining.
        """
        coords = " -- ".join([f"({x},{y},{z})" for x, y, z in points])

        if edge_color == "none":
            self.add_line(f"\\fill[{fill_color}] {coords} -- cycle;")
        else:
            self.add_line(f"\\fill[{fill_color}] {coords} -- cycle;")
            self.add_line(f"\\draw[thick,{edge_color}] {coords} -- cycle;")
        return self

    def get_code(self) -> str:
        """
        Get the complete TikZ code.

        Returns
        -------
        str
            Complete TikZ code as a string.

        Raises
        ------
        ValidationError
            If picture environment is still open.
        """
        if self._in_picture:
            raise ValidationError("Picture environment not closed")

        return "\n".join(self.lines)

    def __str__(self) -> str:
        """Return the TikZ code."""
        return self.get_code()


def graph_to_tikz(graph: Graph) -> str:
    """
    Generate a TikZ diagram of a Feynman graph.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        TikZ code wrapped in a figure environment.
    """
    internal_vertices = set()
    for edge in graph.edges:
        if edge.is_internal:
            internal_vertices.add(edge.v1)
            internal_vertices.add(edge.v2)

    num_internal = len(internal_vertices)

    lines = [
        "\\begin{figure}[htbp]",
        "\\centering",
        "\\begin{tikzpicture}[",
        "  vertex/.style={circle, fill=black, inner sep=2pt},",
        "  external/.style={circle, draw=black, inner sep=1.5pt},",
        "  propagator/.style={thick},",
        "  external_leg/.style={dashed}",
        "]",
        "",
    ]

    if num_internal == 2:
        lines.extend(
            [
                "  % Internal vertices",
                "  \\node[vertex] (v1) at (0, 0) {};",
                "  \\node[vertex] (v2) at (3, 0) {};",
            ]
        )
    elif num_internal == 3:
        lines.extend(
            [
                "  % Internal vertices",
                "  \\node[vertex] (v1) at (90:2cm) {};",
                "  \\node[vertex] (v2) at (210:2cm) {};",
                "  \\node[vertex] (v3) at (330:2cm) {};",
            ]
        )
    else:
        angle_step = 360 / num_internal
        lines.append("  % Internal vertices")
        for i, v in enumerate(sorted(internal_vertices)):
            angle = 90 + i * angle_step
            lines.append(f"  \\node[vertex] (v{v}) at ({angle}:2cm) {{}};")

    lines.append("")
    lines.append("  % Internal propagators")
    drawn_edges: set[tuple[int, int]] = set()
    for edge in graph.edges:
        if edge.is_internal:
            edge_key = tuple(sorted([edge.v1, edge.v2]))
            if edge_key not in drawn_edges:
                lines.append(f"  \\draw[propagator] (v{edge.v1}) -- (v{edge.v2});")
                drawn_edges.add(edge_key)

    lines.append("")
    lines.append("  % External legs")
    external_count = 0
    for edge in graph.edges:
        if not edge.is_internal:
            internal_v = edge.v1 if edge.v1 in internal_vertices else edge.v2
            external_count += 1
            angle = 90 + (external_count - 1) * (360 / graph.external_legs)
            lines.append(f"  \\node[external] (e{external_count}) at ({angle}:3.5cm) {{}};")
            lines.append(f"  \\draw[external_leg] (v{internal_v}) -- (e{external_count});")

    lines.extend(
        [
            "",
            "\\end{tikzpicture}",
            "\\caption{Feynman diagram topology. Internal vertices are shown as filled circles, "
            "internal propagators as solid lines, and external legs as dashed lines.}",
            "\\label{fig:feynman_graph}",
            "\\end{figure}",
        ]
    )

    return "\n".join(lines)
