"""
The graphs of the faces of the Newton polytope, up to a fixed codimension.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ...face_identification import FACE_KINDS, FaceIdentification, FlagLevel, identify_faces
from .._latex_kit import LatexDocument, _escape, _latex_equation, _longtable
from .._text_kit import (
    _INDENT,
    _WIDTH,
    TextDocument,
    _blocks,
    _lines,
    _paragraph,
    _room,
    _str,
    _table,
    _text_equation,
)
from ..latex import to_latex, to_latex_lines
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


# The faces section identifies the faces of P up to this codimension.
FACE_CODIMENSION = 2


@dataclass(frozen=True)
class Faces:
    """The graphs of the faces of P, up to codimension FACE_CODIMENSION.

    Attributes
    ----------
    faces
        One record per face, from :func:`feynkit.face_identification.identify_faces`:
        P itself, then the facets in the order of ``PolytopeData.facets``, then
        the faces of codimension 2.
    max_codimension
        The largest codimension included.
    full_dimensional
        Whether P is full-dimensional; below full dimension no face is
        identified.
    """

    faces: tuple[FaceIdentification, ...]
    max_codimension: int
    full_dimensional: bool


def build(ctx: BuildContext) -> Faces:
    fi = ctx.integral
    data = ctx.polytope_data()
    return Faces(
        faces=identify_faces(fi, max_codimension=FACE_CODIMENSION, data=data),
        max_codimension=FACE_CODIMENSION,
        full_dimensional=data.is_full_dimensional,
    )


# The faces section writes out at most this many unidentified faces.
MAX_UNIDENTIFIED_SHOWN = 10


# The name of each class of face in the faces section.
FACE_CLASSES = {
    "whole": "whole graph",
    "contraction": "contraction",
    "product_uv": "UV product",
    "product_ir": "IR product",
    "u_layer": "U layer",
    "f_layer": "F layer",
    "support_product": "support product",
    "unidentified": "unidentified",
}


def _level_name(level: FlagLevel, latex: bool) -> str:
    if not latex:
        return level.name()
    minor = level.minor().replace("Gamma", r"\Gamma").replace("{", r"\{").replace("}", r"\}")
    return rf"\mathcal{{{level.kind}}}_{{{minor}}}"


def face_name(face: FaceIdentification, *, latex: bool) -> str:
    """The flag's product of minor polynomials, as FaceIdentification.name, or in LaTeX
    maths without the dollars."""
    factors = [_level_name(level, latex) for level in face.levels if not level.trivial]
    if not face.levels:
        return "none"
    return (r"\," if latex else " ").join(factors) or "1"


def facet_graphs(
    report: AnalysisReport, section: Faces
) -> list[tuple[int, sp.Expr, int, FaceIdentification]]:
    """Per facet: its number k in F_k, the left side m . x and right side b of its
    inequality, and its identification; the coordinates x_e are named by the edge
    indices, as in the resonance section."""
    x = [sp.Symbol(f"x_{e}") for e in report.identity.edge_indices]
    rows: list[tuple[int, sp.Expr, int, FaceIdentification]] = []
    for face in section.faces:
        if face.facet is not None:
            lhs = sp.Add(*(m * v for m, v in zip(face.facet.normal, x, strict=True)))
            rows.append((len(rows) + 1, lhs, face.facet.offset, face))
    return rows


def facet_graph(face: FaceIdentification, *, latex: bool) -> str:
    """The graph column of a facet: its name, marked "(support)" for a support product, or a
    dash when it is not identified."""
    name = f"${face_name(face, latex=True)}$" if latex else face_name(face, latex=False)
    if face.verified:
        return name
    if face.kind == "support_product":
        return f"{name} (support)"
    return "--" if latex else "-"


def unverified_label(face: FaceIdentification, label: str, *, latex: bool) -> str:
    """The line before G|_F and the prediction of a face that is not identified."""
    name = f"${face_name(face, latex=True)}$" if latex else face_name(face, latex=False)
    if face.kind == "support_product":
        return f"{label}, with the exponents of {name}:"
    return f"{label}, predicted as {name}:"


def face_counts(section: Faces) -> list[tuple[str, list[int]]]:
    """Per class that occurs: its name and the number of its faces of each codimension from
    0 to the section's largest."""
    rows = []
    for kind in FACE_KINDS:
        counts = [
            sum(face.kind == kind and face.codimension == c for face in section.faces)
            for c in range(section.max_codimension + 1)
        ]
        if any(counts):
            rows.append((FACE_CLASSES[kind], counts))
    return rows


def more_faces(left: int) -> str:
    """The sentence after the unidentified faces shown, when some are not."""
    return "One more face is not shown." if left == 1 else f"{left} more faces are not shown."


# The maths of the faces section, as the text prints it and in LaTeX.
_FACE_MATHS = {
    "F": ("F", "F"),
    "P": ("P", "P"),
    "G": ("G", "G"),
    "G_F": ("G|_F", r"G|_F"),
    "w": ("w", "w"),
    "sum": ("w = -sum_j m_j", r"w = -\sum_j m_j"),
    "facet": ("m_j . x <= b_j", r"m_j \cdot x \le b_j"),
    "t": ("t_1 > ... > t_k", r"t_1 > \dots > t_k"),
    "w_e": ("w_e", "w_e"),
    "H_j": ("H_j", "H_j"),
    "t_j": ("t_j", "t_j"),
    "U(H_j)": ("U(H_j)", r"\mathcal{U}_{H_j}"),
    "F(H_j)": ("F(H_j)", r"\mathcal{F}_{H_j}"),
    "G(H_j)": ("G(H_j)", r"\mathcal{G}_{H_j}"),
    "zero": ("t_j = 0", "t_j = 0"),
    "negative": ("t_j < 0", "t_j < 0"),
    "minor": ("sigma/tau", r"\sigma/\tau"),
    "sigma": ("sigma", r"\sigma"),
    "tau": ("tau", r"\tau"),
    "Gamma": ("Gamma", r"\Gamma"),
    "gamma": ("gamma", r"\gamma"),
    "uv": ("U(gamma) G(Gamma/gamma)", r"\mathcal{U}_\gamma \mathcal{G}_{\Gamma/\gamma}"),
    "u_e": ("u_e", "u_e"),
    "edge": ("G(Gamma/{e})", r"\mathcal{G}_{\Gamma/\{e\}}"),
    "F_quotient": ("F(Gamma/gamma)", r"\mathcal{F}_{\Gamma/\gamma}"),
}


def face_paragraphs(
    section: Faces, cite: Callable[..., str], *, latex: bool
) -> tuple[list[str], str, str]:
    """The paragraphs of the faces section before its tables, and the sentences before the
    table of classes and before the unidentified faces."""

    def m(text: str, tex: str) -> str:
        return f"${tex}$" if latex else text

    x = {key: m(text, tex) for key, (text, tex) in _FACE_MATHS.items()}
    u, f, g = m("U", r"\mathcal{U}"), m("F", r"\mathcal{F}"), m("G", r"\mathcal{G}")
    opening = (
        f"For a face {x['F']} of {x['P']}, {x['G_F']} is the sum of the terms of {x['G']} "
        f"whose exponents lie on {x['F']}: the initial form of {x['G']} for any weight "
        f"{x['w']} in the relative interior of the normal cone of {x['F']}, the terms of least "
        f"{x['w']}-degree{cite('fmt2024')}. The report takes {x['sum']} over the facets "
        f"{x['facet']} containing {x['F']}, and the distinct values {x['t']} of the {x['w_e']} "
        f"give a flag of minors of the graph: {x['H_j']} keeps the edges of weight {x['t_j']}, "
        "contracts those of greater weight and deletes the others, legs moving with their "
        "vertices and a vertex left without propagators dropped with its legs. The terms of "
        f"{u} of least {x['w']}-degree are the product of the {x['U(H_j)']}. For the last level "
        f"whose {x['F(H_j)']} is not zero, the prediction for {x['G_F']} replaces "
        f"{x['U(H_j)']} in that product by {x['G(H_j)']} when {x['zero']} and by "
        f"{x['F(H_j)']} when {x['negative']}; a face is identified only when {x['G_F']} equals "
        f"the prediction exactly. A minor is written {x['minor']}, the subgraph on the edges "
        f"{x['sigma']} with those of {x['tau']} contracted, and {x['Gamma']} is the whole "
        "graph."
    )
    known = (
        f"For the weight that is 1 on the edges of a connected subgraph {x['gamma']} and 0 "
        f"elsewhere, with every mass non-zero, the initial form is {x['uv']}"
        f"{cite('fmt2024')}, and the face of the terms free of {x['u_e']} gives "
        f"{x['edge']}{cite('britto2026')}. "
        f"Arkani-Hamed, Hillman and Mizera label the facets of the Feynman polytope by "
        f"subgraphs {x['gamma']}, ultraviolet when {x['F_quotient']} is not zero and infrared "
        f"when it is{cite('ahm2022')}."
    )
    paragraphs = [opening, known]
    if not section.full_dimensional:
        paragraphs.append(
            f"{x['P']} is not full-dimensional, so its faces are not identified: the normal of "
            f"a facet relative to the affine hull of {x['P']} is fixed only modulo the "
            "equations of the hull, and so is the flag."
        )
    classes = (
        f"A contraction is the {g} of a quotient alone, a UV product has its {g} factor on a "
        f"quotient and an IR product on a minor with edges deleted; the {u} layer holds the "
        f"products of {u}'s alone and the {f} layer those with an {f} factor. A support "
        f"product is a face whose {x['G_F']} is not the prediction but has exactly its "
        "exponents, the sums of one exponent of each factor. The faces of each codimension "
        "fall into the classes"
    )
    unidentified = (
        f"On these faces {x['G_F']} differs from the prediction of its flag, written below it:"
    )
    return paragraphs, classes, unidentified


def _facet_table(rows: Sequence[tuple[str, str, str]]) -> str:
    """The facets with their inequalities and graphs.

    A graph too long for the rest of its line goes on the lines below, under
    the inequality, broken between its factors.
    """
    header = ("Facet", "Inequality", "Graph")
    first = max(len(row[0]) for row in [header, *rows])
    second = max(len(row[1]) for row in [header, *rows])
    lead = len(_INDENT) + first + 2
    lines = []
    for k, (facet, inequality, graph) in enumerate([header, *rows]):
        start = f"{_INDENT}{facet.ljust(first)}  {inequality.ljust(second)}  "
        if len(start) + len(graph) <= _WIDTH:
            lines.append((start + graph).rstrip())
        else:
            lines.append(start.rstrip())
            below = [""]
            for factor in graph.split(" "):
                if below[-1] and lead + len(below[-1]) + 1 + len(factor) > _WIDTH:
                    below.append("")
                below[-1] = f"{below[-1]} {factor}".strip()
            lines.extend(" " * lead + part for part in below)
        if k == 0:
            rules = ("-" * first, "-" * second, "-" * max(len(row[2]) for row in [header, *rows]))
            lines.append(f"{_INDENT}{rules[0]}  {rules[1]}  {rules[2]}"[:_WIDTH])
    return "\n".join(lines)


def text(report: AnalysisReport, section: Faces, doc: TextDocument) -> str:
    paragraphs, classes, intro = face_paragraphs(section, doc.cite, latex=False)
    blocks = [_paragraph(p) for p in paragraphs]
    if not section.full_dimensional:
        return _blocks(*blocks)
    graphs = facet_graphs(report, section)
    rows = [
        (f"F_{k}", f"{_str(lhs)} <= {b}", facet_graph(face, latex=False))
        for k, lhs, b, face in graphs
    ]
    counts = face_counts(section)
    header = ("Class", *(f"Codim {c}" for c in range(section.max_codimension + 1)))
    blocks += [
        _paragraph(
            "The graph of each facet, a support product marked (support) and a facet that is "
            "not identified by a dash:"
        ),
        _facet_table(rows),
        _paragraph(f"{classes}:"),
        _table(header, [(name, *map(str, row)) for name, row in counts]),
    ]
    unidentified = [face for face in section.faces if not face.verified]
    if unidentified:
        number = {id(face): k for k, _, _, face in graphs}
        blocks.append(_paragraph(intro))
        for face in unidentified[:MAX_UNIDENTIFIED_SHOWN]:
            k = number.get(id(face))
            label = f"F_{k}" if k is not None else f"A face of dimension {face.dimension}"
            assert face.prediction is not None
            blocks += [
                _paragraph(unverified_label(face, label, latex=False)),
                _text_equation("G|_F", _lines(face.polynomial, _room("G|_F"))),
                _text_equation("prediction", _lines(face.prediction, _room("prediction"))),
            ]
        left = len(unidentified) - MAX_UNIDENTIFIED_SHOWN
        if left > 0:
            blocks.append(_paragraph(more_faces(left)))
    return _blocks(*blocks)


def latex(report: AnalysisReport, section: Faces, doc: LatexDocument) -> str:
    paragraphs, classes, intro = face_paragraphs(section, doc.cite, latex=True)
    parts = list(paragraphs)
    if not section.full_dimensional:
        return "\n\n".join(parts)
    graphs = facet_graphs(report, section)
    rows = [
        (
            f"$F_{{{k}}}$",
            f"${to_latex(lhs)} \\le {b}$",
            facet_graph(face, latex=True),
        )
        for k, lhs, b, face in graphs
    ]
    counts = face_counts(section)
    header = ("Class", *(f"Codim.\\ {c}" for c in range(section.max_codimension + 1)))
    parts += [
        "The graph of each facet, a support product marked (support) and a facet that is not "
        "identified by a dash:",
        _longtable("lll", ("Facet", "Inequality", "Graph"), rows),
        f"{classes}:",
        _longtable(
            "l" + "r" * (section.max_codimension + 1),
            header,
            [(_escape(name), *map(str, row)) for name, row in counts],
        ),
    ]
    unidentified = [face for face in section.faces if not face.verified]
    if unidentified:
        number = {id(face): k for k, _, _, face in graphs}
        parts.append(intro)
        for face in unidentified[:MAX_UNIDENTIFIED_SHOWN]:
            k = number.get(id(face))
            label = f"$F_{{{k}}}$" if k is not None else f"A face of dimension {face.dimension}"
            assert face.prediction is not None
            parts += [
                unverified_label(face, label, latex=True),
                _latex_equation("G|_F", to_latex_lines(face.polynomial)),
                _latex_equation("\\text{prediction}", to_latex_lines(face.prediction)),
            ]
        left = len(unidentified) - MAX_UNIDENTIFIED_SHOWN
        if left > 0:
            parts.append(more_faces(left))
    return "\n\n".join(parts)


def summary(faces: Faces) -> list[tuple[str, str]]:
    unidentified = sum(face.kind == "unidentified" for face in faces.faces)
    return [("Unidentified faces", str(unidentified))]


SECTION = Section(
    name="faces",
    heading="Faces as graphs",
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(70, summary),),
)
