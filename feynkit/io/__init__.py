"""
Input/Output module for Feynman integral analysis.

The analysis report of an integral is available as
:meth:`feynkit.FeynmanIntegral.to_latex` and
:meth:`feynkit.FeynmanIntegral.to_text`. This module exposes the report, its
two renderers, lower-level formatters and the file-saving helpers.

Classes
-------
AnalysisReport
    Everything feynkit computes for one integral, section by section.

Functions
---------
render_latex
    Render an analysis report as a LaTeX document.
render_text
    Render an analysis report as plain text.
to_latex
    Convert SymPy expression to LaTeX.
factor_energy_scale
    Write an expression as a numerator over a power of the energy scale.
graph_to_latex_table
    Generate LaTeX table of graph properties.
edges_to_latex_table
    Generate LaTeX table of edges.
save_latex_document
    Save LaTeX content to file.
graph_to_text
    Generate text summary of graph.
edges_to_text
    Generate text list of edges.
save_text_report
    Save text report to file.
"""

from .latex import (
    edges_to_latex_table,
    factor_energy_scale,
    graph_to_latex_table,
    save_latex_document,
    to_latex,
)
from .report import AnalysisReport
from .report_latex import render_latex
from .report_text import render_text
from .text import (
    edges_to_text,
    graph_to_text,
    save_text_report,
)

__all__ = [
    # Analysis report
    "AnalysisReport",
    "render_latex",
    "render_text",
    # LaTeX export
    "to_latex",
    "factor_energy_scale",
    "graph_to_latex_table",
    "edges_to_latex_table",
    "save_latex_document",
    # Text export
    "graph_to_text",
    "edges_to_text",
    "save_text_report",
]
