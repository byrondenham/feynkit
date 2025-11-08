"""
Input/Output module for Feynman integral analysis.

Provides utilities for exporting analysis results to various formats including
LaTeX documents and plain text reports.

Functions
---------
to_latex
    Convert SymPy expression to LaTeX.
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
create_analysis_document
    Create complete LaTeX analysis document.
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
create_analysis_report
    Create complete text analysis report.
save_text_report
    Save text report to file.

Examples
--------
>>> from feynkit.io import create_analysis_document, create_analysis_report
>>> from feynkit.algebra import compute_toric_ideal_generators
>>>
>>> # Create LaTeX document with toric ideal
>>> toric_gens = compute_toric_ideal_generators(gkz.a_matrix)
>>> latex_doc = create_analysis_document(
...     graph=my_graph,
...     u_polynomial=U,
...     f_polynomial=F,
...     gkz_system=gkz,
...     toric_generators=toric_gens,
...     title="My Analysis"
... )
>>> with open("analysis.tex", "w") as f:
...     f.write(latex_doc)
>>>
>>> # Create text report
>>> text_report = create_analysis_report(
...     graph=my_graph,
...     u_polynomial=U,
...     f_polynomial=F,
...     toric_generators=toric_gens,
... )
>>> print(text_report)
"""

from .latex import (
    create_analysis_document,
    edges_to_latex_table,
    gkz_system_to_latex,
    graph_to_latex_table,
    parametrisation_to_latex,
    save_latex_document,
    to_latex,
    toric_ideal_to_latex,
)
from .text import (
    create_analysis_report,
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
    "graph_to_latex_table",
    "edges_to_latex_table",
    "parametrisation_to_latex",
    "gkz_system_to_latex",
    "toric_ideal_to_latex",
    "create_analysis_document",
    "save_latex_document",
    # Text export
    "graph_to_text",
    "edges_to_text",
    "parametrisation_to_text",
    "gkz_system_to_text",
    "toric_ideal_to_text",
    "create_analysis_report",
    "save_text_report",
]
