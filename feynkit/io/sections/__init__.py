"""
The sections of the analysis report, one module each.

:data:`SECTIONS` lists them in the order of the document. A new section is a
module with a ``SECTION`` and one line here; the field that holds its data in
:class:`~feynkit.io.report.AnalysisReport` is the only other place it appears.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import (
    conventions,
    degeneracy,
    face_lattice,
    faces,
    gkz,
    hasse,
    identity,
    landau,
    polynomials,
    polytope,
    representations,
    resonance,
    schwinger,
    sectors,
    symmetries,
    torus,
)
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport

__all__ = ["SECTIONS", "Section", "SummaryPart", "render_sections", "summary_rows"]

SECTIONS: tuple[Section, ...] = (
    identity.SECTION,
    conventions.SECTION,
    polynomials.SECTION,
    representations.SECTION,
    polytope.SECTION,
    torus.SECTION,
    gkz.SECTION,
    resonance.SECTION,
    faces.SECTION,
    face_lattice.SECTION,
    hasse.SECTION,
    sectors.SECTION,
    symmetries.SECTION,
    landau.SECTION,
    degeneracy.SECTION,
    schwinger.SECTION,
)


def render_sections(
    report: AnalysisReport,
    *,
    summary: Callable[[], str],
    render: Callable[[Section, Any], str],
) -> list[tuple[str, str]]:
    """The heading and body of each section the report carries, in document order.

    The summary comes first, rendered by ``summary``; ``render`` renders the
    body of each other section from its data. The first three sections are
    always present; each of the others is rendered only when the report
    carries it. The sections are rendered in document order, so citations made
    while rendering are recorded in order of first use.
    """
    sections = [("Summary", summary())]
    for section in SECTIONS:
        data = getattr(report, section.name)
        if data is not None:
            sections.append((section.heading, render(section, data)))
    return sections


def summary_rows(report: AnalysisReport) -> tuple[tuple[str, str], ...]:
    """The report's numbers as (label, value) rows, sections absent omitted."""
    parts = [
        (part.rank, part.rows(data))
        for section in SECTIONS
        if (data := getattr(report, section.name)) is not None
        for part in section.summary
    ]
    return tuple(row for _, rows in sorted(parts, key=lambda p: p[0]) for row in rows)
