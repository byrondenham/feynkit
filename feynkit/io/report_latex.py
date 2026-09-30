"""
LaTeX rendering of an analysis report.

:func:`render_latex` turns an :class:`AnalysisReport` into a complete
``article`` document: a summary table, the graph, the conventions, the
Symanzik polynomials, the parametric representations with the convergence
region, the Newton polytope with the rank statement, the candidate Euler
characteristic from finite-field point counts, the GKZ system, the symmetries,
the Landau surfaces, the Schwinger-representation system and the references.
Each section is rendered by its own module of :mod:`feynkit.io.sections` and
omitted when the report does not carry it. Citations and the width of the widest matrix are
collected while the sections are rendered, so the bibliography lists exactly
the works the text cites, in the order of first citation, and the preamble
allows exactly as many matrix columns as the document needs. The document
skeleton and the summary table are here; the helpers for displays and escaping
are in ``_latex_kit``, and what the text renderer must state the same way, the
bibliography and the integrand templates, comes from ``_report_shared``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ._latex_kit import LatexDocument, _escape
from .sections import render_sections

if TYPE_CHECKING:
    from .report import AnalysisReport

from ._report_shared import CITATIONS

__all__ = ["CITATIONS", "render_latex"]


def _summary(report: AnalysisReport) -> str:
    rows = []
    for label, value in report.summary():
        text = re.sub(r"\b([FG])\b", r"$\1$", _escape(label))
        rows.append(f"{text} & {_escape(value)} \\\\")
    return "\n".join(
        [
            "\\begin{center}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
        ]
    )


def render_latex(report: AnalysisReport, *, title: str | None = None) -> str:
    """Render an analysis report as a complete LaTeX ``article``.

    Parameters
    ----------
    report
        The report to render; sections it does not carry are omitted.
    title
        Document title, escaped for LaTeX; by default "Feynman integral"
        followed by the CNickel string.

    Returns
    -------
    str
        The document source, ASCII only, compilable with pdflatex.
    """
    doc = LatexDocument()
    if title is None:
        heading = f"Feynman integral \\texttt{{{_escape(report.identity.cnickel)}}}"
    else:
        heading = _escape(title)

    sections = render_sections(
        report,
        summary=lambda: _summary(report),
        render=lambda section, data: section.latex(report, data, doc),
    )

    document = [
        doc.preamble(),
        "",
        f"\\title{{{heading}}}",
        "\\date{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\n\n".join(f"\\section{{{name}}}\n{body}" for name, body in sections),
        "",
        doc.bibliography(),
        "",
        "\\end{document}",
        "",
    ]
    return "\n".join(document)
