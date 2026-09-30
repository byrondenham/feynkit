"""
What a report section is, and how one registers itself.

A section is a module of :mod:`feynkit.io.sections` that holds everything the
report says about one topic: the frozen data type, the function that builds it
from an integral, its plain-text and LaTeX renderings, its rows of the summary
and, through the summary, its fields of the JSON that ``fk analyse --json``
prints. It ends with a :class:`Section` named ``SECTION``, which the tuple in
``feynkit/io/sections/__init__.py`` lists in document order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._latex_kit import LatexDocument
    from .._text_kit import TextDocument
    from ..report import AnalysisReport, BuildContext

__all__ = ["Section", "SummaryPart"]


@dataclass(frozen=True)
class SummaryPart:
    """Rows a section adds to the summary, and where they sit among the others.

    Attributes
    ----------
    rank
        Parts are ordered by rank, and by position in the registry when ranks
        are equal. The ranks are spaced by ten so that a new part fits between
        two others without renumbering.
    rows
        Called with the section's data; returns (label, value) rows. The JSON
        keys of ``fk analyse --json`` are the lower-case labels with underscores.
    """

    rank: int
    rows: Callable[[Any], Sequence[tuple[str, str]]]


@dataclass(frozen=True)
class Section:
    """One section of the analysis report.

    Attributes
    ----------
    name
        The name of the section in ``SECTION_NAMES``, and the field of
        :class:`~feynkit.io.report.AnalysisReport` that holds its data.
    heading
        The heading of the section in the rendered documents.
    build
        Builds the data from a :class:`~feynkit.io.report.BuildContext`.
    text, latex
        Render the section's data as the body of the plain-text and LaTeX
        documents; called with the report, the section's data and the document
        that collects citations.
    summary
        The section's rows of the summary.
    always
        True for the sections built whatever the caller asks for.
    """

    name: str
    heading: str
    build: Callable[[BuildContext], Any]
    text: Callable[[AnalysisReport, Any, TextDocument], str]
    latex: Callable[[AnalysisReport, Any, LatexDocument], str]
    summary: tuple[SummaryPart, ...] = ()
    always: bool = False
