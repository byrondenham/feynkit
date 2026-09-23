"""
TikZ code generation for mathematical diagrams.

Provides functions to generate TikZ/LaTeX code for visualising geometric
structures associated with Feynman integrals.
"""

import math

from ..core.edge import Edge
from ..core.exceptions import ValidationError
from ..core.graph import Graph

# Parallel propagators are bent apart by at most this angle, in degrees.
MAX_BEND = 30.0
# Two parallel propagators are bent apart by this angle instead, in degrees.
PAIR_BEND = 20.0
# Legs meeting at one vertex are spread by this angle, in degrees.
LEG_SPREAD = 40.0
# Successive self-loops at one vertex take these sides in turn.
LOOP_SIDES = ("above", "below", "left", "right")
# Distance from a vertex to the endpoint of its external legs.
LEG_LENGTH = "1.5cm"


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
        options: str | None = None,
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
        # A blank line gets no indent, so the output has no trailing whitespace.
        indent_str = "    " * indent if line else ""
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
        label: str | None = None,
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
            Line colour.

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
            Line colour.

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
            Fill colour (with opacity).
        edge_color : str, default "black"
            Edge colour. Use "none" for no edge.

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


def _format_angle(angle: float) -> str:
    """Format an angle in degrees for a TikZ polar coordinate."""
    return f"{round(angle, 1) + 0.0:g}"


def _polar(angle: float, radius: float) -> tuple[float, float]:
    """Cartesian coordinates of a point given in polar form, angle in degrees."""
    return radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))


def _bend_angles(count: int) -> list[float]:
    """
    Bend angles for a bundle of parallel propagators.

    A single propagator is straight; two are bent apart symmetrically; more are
    spread evenly over the full range, so that the middle one stays straight
    when the count is odd.
    """
    if count == 1:
        return [0.0]
    if count == 2:
        return [-PAIR_BEND, PAIR_BEND]
    step = 2 * MAX_BEND / (count - 1)
    return [-MAX_BEND + index * step for index in range(count)]


def _walk_order(vertices: list[int], edges: list[Edge]) -> list[int]:
    """
    Order the vertices along a walk over the internal edges.

    A depth-first walk from the lowest-numbered vertex, taking the lowest
    neighbour first, appends each vertex as it is reached. Laying the vertices
    out in this order draws a one-loop graph as a polygon, where the label order
    would run the propagators across the middle of the diagram. Vertices the
    walk does not reach keep their label order at the end.
    """
    neighbours: dict[int, list[int]] = {vertex: [] for vertex in vertices}
    for edge in edges:
        if edge.v1 == edge.v2:
            continue
        neighbours[edge.v1].append(edge.v2)
        neighbours[edge.v2].append(edge.v1)

    order: list[int] = []
    seen: set[int] = set()
    for start in vertices:
        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex in seen:
                continue
            seen.add(vertex)
            order.append(vertex)
            stack.extend(sorted(neighbours[vertex], reverse=True))
    return order


def _vertex_layout(
    vertices: list[int],
    edges: list[Edge],
) -> tuple[list[str], dict[int, tuple[float, float]]]:
    """
    Place the internal vertices and return the node lines and their coordinates.

    Two vertices sit on a horizontal line, three on a triangle and more on a
    circle, taken in the order of a walk over the internal edges so that
    neighbouring vertices are placed next to one another. The coordinates are
    kept alongside the TikZ strings so that the external legs can be pointed
    away from the centre of the diagram.
    """
    if len(vertices) == 2:
        places = ["(0, 0)", "(3, 0)"]
        points = [(0.0, 0.0), (3.0, 0.0)]
    elif len(vertices) == 3:
        places = ["(90:2cm)", "(210:2cm)", "(330:2cm)"]
        points = [_polar(90.0, 2.0), _polar(210.0, 2.0), _polar(330.0, 2.0)]
    else:
        step = 360 / len(vertices)
        angles = [90 + index * step for index in range(len(vertices))]
        places = [f"({angle}:2cm)" for angle in angles]
        points = [_polar(angle, 2.0) for angle in angles]

    lines = ["  % Internal vertices"]
    positions: dict[int, tuple[float, float]] = {}
    for vertex, place, point in zip(_walk_order(vertices, edges), places, points, strict=True):
        lines.append(f"  \\node[vertex] (v{vertex}) at {place} {{}};")
        positions[vertex] = point
    return lines, positions


def _outward_angle(point: tuple[float, float], centre: tuple[float, float]) -> float:
    """Direction in degrees pointing from the centre of the diagram to a vertex."""
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    if math.hypot(dx, dy) < 1e-9:
        # A vertex at the centre has no outward direction; point downwards, away
        # from the self-loops, which are drawn above.
        return -90.0
    return math.degrees(math.atan2(dy, dx))


def _edge_label(edge: Edge) -> str:
    """A node that labels a propagator with its edge index, halfway along it."""
    return f"node[midway, auto, font=\\scriptsize] {{${edge.idx}$}}"


def _propagator_lines(edges: list[Edge]) -> list[str]:
    """
    Draw every internal edge, bending parallel propagators apart.

    Edges are grouped by the unordered pair of vertices they join, in order of
    their index, so that a bundle of parallel propagators is drawn as a fan
    rather than collapsed onto one line. Self-loops are drawn as loops around
    their vertex, taking a fresh side for each one. Each propagator carries its
    edge index as a label, the index of its parameters a_e and u_e, inside its
    own draw command.
    """
    bundles: dict[tuple[int, int], list[Edge]] = {}
    for edge in edges:
        pair = (min(edge.v1, edge.v2), max(edge.v1, edge.v2))
        bundles.setdefault(pair, []).append(edge)

    lines = ["  % Internal propagators"]
    for (first, second), bundle in bundles.items():
        if first == second:
            for index, edge in enumerate(bundle):
                side = LOOP_SIDES[index % len(LOOP_SIDES)]
                lines.append(
                    f"  \\draw[propagator] (v{first}) to[loop {side}] {_edge_label(edge)} "
                    f"(v{first});"
                )
            continue
        for edge, bend in zip(bundle, _bend_angles(len(bundle)), strict=True):
            label = _edge_label(edge)
            if bend == 0.0:
                lines.append(f"  \\draw[propagator] (v{first}) -- {label} (v{second});")
                continue
            side = "left" if bend > 0 else "right"
            option = f"bend {side}={_format_angle(abs(bend))}"
            lines.append(f"  \\draw[propagator] (v{first}) to[{option}] {label} (v{second});")
    return lines


def _external_leg_lines(
    edges: list[Edge],
    positions: dict[int, tuple[float, float]],
) -> list[str]:
    """
    Attach each external leg to its own vertex, pointing away from the centre.

    Legs meeting at the same vertex are spread symmetrically about that
    direction. The endpoints are placed with the ``calc`` library, so that the
    leg is drawn relative to the vertex wherever the vertex happens to sit.
    """
    centre = (
        sum(x for x, _ in positions.values()) / len(positions),
        sum(y for _, y in positions.values()) / len(positions),
    )
    legs = [
        (number, edge.v1 if edge.v1 in positions else edge.v2)
        for number, edge in enumerate(edges, start=1)
    ]
    at_vertex: dict[int, list[int]] = {}
    for number, vertex in legs:
        at_vertex.setdefault(vertex, []).append(number)

    lines = ["  % External legs"]
    for number, vertex in legs:
        siblings = at_vertex[vertex]
        offset = (siblings.index(number) - (len(siblings) - 1) / 2) * LEG_SPREAD
        angle = _format_angle(_outward_angle(positions[vertex], centre) + offset)
        lines.append(
            f"  \\node[external] (e{number}) at ($(v{vertex})+({angle}:{LEG_LENGTH})$) {{}};"
        )
        lines.append(f"  \\draw[external_leg] (v{vertex}) -- (e{number});")
    return lines


def graph_to_tikz(graph: Graph) -> str:
    """
    Generate a TikZ diagram of a Feynman graph.

    Every internal edge is drawn, with parallel propagators bent apart, and
    every external leg points away from the centre of the diagram. The figure
    needs the ``calc`` TikZ library, as its comment header records.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        TikZ code wrapped in a figure environment.
    """
    internal_edges = graph.get_internal_edges()
    vertices = sorted({vertex for edge in internal_edges for vertex in (edge.v1, edge.v2)})

    lines = [
        "\\begin{figure}[htbp]",
        "\\centering",
        "\\begin{tikzpicture}[",
        "  vertex/.style={circle, fill=black, inner sep=2pt},",
        "  external/.style={circle, draw=black, inner sep=1.5pt},",
        "  propagator/.style={thick},",
        "  external_leg/.style={dashed}",
        "]",
        "  % needs \\usetikzlibrary{calc}",
        "",
    ]

    vertex_lines, positions = _vertex_layout(vertices, internal_edges)
    lines.extend(vertex_lines)
    lines.append("")
    lines.extend(_propagator_lines(internal_edges))
    lines.append("")
    lines.extend(_external_leg_lines(graph.get_external_edges(), positions))

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
