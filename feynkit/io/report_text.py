"""
Plain-text rendering of an analysis report.

:func:`render_text` states the facts of :func:`~feynkit.io.report_latex.render_latex`
in the same order, with the same sentences and citations, as plain ASCII:
maths is written out in plain names (``beta = (-D/2, -nu_1, ..., -nu_N)``),
computed expressions are printed like ``sympy.sstr`` but with ``^`` for
powers, as the prose writes them, matrices by ``sympy.pretty`` in ASCII, and
citations as ``[key]``, listed at the end with their entries stripped of
LaTeX. The text has no figures, so the sentences
that point at a figure are left out and a cross-reference names the section
it refers to. Prose is wrapped at 79 columns without breaking inline maths;
displays are indented and broken to fit the same width wherever a sum, a
product, a quotient or a vector allows it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._report_shared import join_words
from ._text_kit import TextDocument, _heading, _table
from .sections import render_sections

if TYPE_CHECKING:
    from .report import AnalysisReport

__all__ = ["render_text"]


def _summary(report: AnalysisReport) -> str:
    return _table(None, report.summary())


def render_text(
    report: AnalysisReport, *, title: str | None = None, substitutions: Sequence[str] = ()
) -> str:
    """Render an analysis report as plain text.

    Parameters
    ----------
    report
        The report to render; sections it does not carry are omitted.
    title
        Title line, used as given; by default "Feynman integral" followed by
        the CNickel string.
    substitutions
        The kinematic substitutions applied to the integral, such as
        ``"p4^2 = 0"``, stated in a line under the title; none by default.

    Returns
    -------
    str
        The report, ASCII only: the title underlined with ``=``, each section
        heading with ``-``, and the references last.
    """
    doc = TextDocument()
    heading = f"Feynman integral {report.identity.cnickel}" if title is None else title
    sections = render_sections(
        report,
        summary=lambda: _summary(report),
        render=lambda section, data: section.text(report, data, doc),
    )
    blocks = [
        _heading(heading, "="),
        *([f"With {join_words(substitutions)}."] if substitutions else []),
        *(f"{_heading(name, '-')}\n\n{body}" for name, body in sections),
        f"{_heading('References', '-')}\n\n{doc.references()}",
    ]
    return "\n\n".join(blocks) + "\n"
