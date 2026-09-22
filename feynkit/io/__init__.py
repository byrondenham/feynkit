"""
Input/Output module for Feynman integral analysis.

Top-level LaTeX export for an integral is available as
:meth:`feynkit.FeynmanIntegral.to_latex`. This module exposes lower-level
formatters and the file-saving helpers.

Functions
---------
to_latex
    Convert SymPy expression to LaTeX.
factor_energy_scale
    Write an expression as a numerator over a power of the energy scale.
graph_to_latex_table
    Generate LaTeX table of graph properties.
edges_to_latex_table
    Generate LaTeX table of edges.
parametrisation_to_latex
    Generate LaTeX for parametrisation result.
gkz_system_to_latex
    Generate LaTeX for GKZ system.
toric_ideal_to_latex
    Generate LaTeX for toric ideal generators.
save_latex_document
    Save LaTeX content to file.
graph_to_text
    Generate text summary of graph.
edges_to_text
    Generate text list of edges.
parametrisation_to_text
    Generate text for parametrisation result.
gkz_system_to_text
    Generate text for GKZ system.
toric_ideal_to_text
    Generate text for toric ideal generators.
save_text_report
    Save text report to file.
"""

from .latex import (
    edges_to_latex_table,
    factor_energy_scale,
    gkz_system_to_latex,
    graph_to_latex_table,
    parametrisation_to_latex,
    save_latex_document,
    to_latex,
    toric_ideal_to_latex,
)
from .text import (
    edges_to_text,
    gkz_system_to_text,
    graph_to_text,
    parametrisation_to_text,
    save_text_report,
    toric_ideal_to_text,
)

__all__ = [
    # LaTeX export
    "to_latex",
    "factor_energy_scale",
    "graph_to_latex_table",
    "edges_to_latex_table",
    "parametrisation_to_latex",
    "gkz_system_to_latex",
    "toric_ideal_to_latex",
    "save_latex_document",
    # Text export
    "graph_to_text",
    "edges_to_text",
    "parametrisation_to_text",
    "gkz_system_to_text",
    "toric_ideal_to_text",
    "save_text_report",
]
