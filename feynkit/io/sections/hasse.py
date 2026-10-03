"""
The Hasse diagram of the face lattice of the Newton polytope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...face_lattice import DecoratedFaceLattice
from ...visualisation.hasse import MAX_FACES, hasse_faces, hasse_tikz
from .._latex_kit import LatexDocument
from .._text_kit import TextDocument, _blocks, _paragraph
from ._base import Section
from .faces import FACE_CODIMENSION
from .resonance import integer_powers

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Hasse:
    """The Hasse diagram of the faces of P up to a codimension.

    Attributes
    ----------
    lattice
        The decorated face lattice, as in the face-lattice section.
    codimension
        The largest codimension drawn: FACE_CODIMENSION, or less when the
        diagram would have more than ``feynkit.visualisation.hasse.MAX_FACES``
        faces.
    faces
        The number of faces drawn.
    tikz
        The diagram as a TikZ picture.
    """

    lattice: DecoratedFaceLattice
    codimension: int
    faces: int
    tikz: str


def build(ctx: BuildContext) -> Hasse:
    fi = ctx.integral
    powers, _ = integer_powers(fi)
    edges = [e.idx for e in fi.graph.get_internal_edges()]
    lattice = fi.face_lattice(
        ctx.d0,
        nu=dict(zip(edges, powers, strict=True)),
        identify_codimension=FACE_CODIMENSION,
    )
    k = FACE_CODIMENSION
    while len(hasse_faces(lattice, "codim", k, max_faces=10**9)) > MAX_FACES and k > 0:
        k -= 1
    return Hasse(
        lattice=lattice,
        codimension=k,
        faces=len(hasse_faces(lattice, "codim", k, max_faces=10**9)),
        tikz=hasse_tikz(lattice, "codim", k),
    )


def _opening(section: Hasse) -> str:
    where = (
        f"the faces of codimension at most {section.codimension}"
        if section.codimension
        else "the polytope itself"
    )
    count = f"{section.faces} face" + ("" if section.faces == 1 else "s")
    tail = (
        ""
        if section.codimension == FACE_CODIMENSION
        else (f" A diagram of codimension {FACE_CODIMENSION} would be too large to read.")
    )
    return (
        f"The Hasse diagram of the face lattice shows {where}, {count} in all, one row to a "
        "dimension, with an edge from each face to the faces of the next dimension that "
        "contain it. A face is coloured by when it is resonant: for every epsilon, at a "
        "progression of epsilon, or never. The diagram itself is a figure of the LaTeX report."
        + tail
    )


def text(_report: AnalysisReport, section: Hasse, _doc: TextDocument) -> str:
    return _blocks(_paragraph(_opening(section)))


def latex(_report: AnalysisReport, section: Hasse, _doc: LatexDocument) -> str:
    return "\n\n".join(
        [
            _opening(section).replace("epsilon", "$\\varepsilon$"),
            "\\begin{figure}[htbp]",
            "\\centering",
            "\\resizebox{\\textwidth}{!}{%",
            section.tikz + "}",
            "\\caption{The Hasse diagram of the face lattice of $P$.}",
            "\\label{fig:hasse}",
            "\\end{figure}",
        ]
    )


SECTION = Section(
    name="hasse",
    heading="Hasse diagram of the face lattice",
    build=build,
    text=text,
    latex=latex,
)
