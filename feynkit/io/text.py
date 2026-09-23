"""
Plain text export utilities for Feynman integral analysis.

Provides text summaries of a graph and its edges, and a helper that saves a
report. The analysis report itself is rendered by
:mod:`feynkit.io.report_text`.
"""

from ..core.graph import Graph


def graph_to_text(graph: Graph) -> str:
    """
    Generate a text summary of graph properties.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        Text summary.

    Examples
    --------
    >>> print(graph_to_text(my_graph))
    """
    lines = [
        "=" * 70,
        "FEYNMAN GRAPH PROPERTIES",
        "=" * 70,
        f"Internal vertices:     {graph.internal_vertices}",
        f"External legs:         {graph.external_legs}",
        f"Internal propagators:  {len(graph.get_internal_edges())}",
        f"Loop number:           {graph.get_loop_count()}",
        "",
    ]
    return "\n".join(lines)


def edges_to_text(graph: Graph) -> str:
    """
    Generate a text list of all edges.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        Text edge list.
    """
    lines = [
        "=" * 70,
        "EDGE LIST",
        "=" * 70,
        f"{'Idx':<5} {'Type':<10} {'From':<5} {'To':<5} {'Mass':<20} {'Name':<15}",
        "-" * 70,
    ]

    for edge in graph.edges:
        edge_type = "Internal" if edge.is_internal else "External"
        mass_str = str(edge.get_mass()) if edge.mass else "---"
        name_str = edge.name if edge.name else "---"
        lines.append(
            f"{edge.idx:<5} {edge_type:<10} {edge.v1:<5} {edge.v2:<5} {mass_str:<20} {name_str:<15}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def save_text_report(content: str, filename: str) -> str:
    """
    Save text report to a file.

    Parameters
    ----------
    content : str
        Report content.
    filename : str
        Output filename.

    Returns
    -------
    str
        The filename where content was saved.

    Examples
    --------
    >>> save_text_report(integral.to_text(), "analysis.txt")
    'analysis.txt'
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename
