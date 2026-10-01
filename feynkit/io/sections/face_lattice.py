"""
The resonance of every face of the Newton polytope, the resonance centres and reducibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...face_lattice import DecoratedFace, DecoratedFaceLattice, Epsilon, SchwingerCheck
from ...resonance import D0Source
from .._latex_kit import LatexDocument, _longtable
from .._text_kit import TextDocument, _blocks, _paragraph, _table
from ._base import Section, SummaryPart
from .faces import FACE_CODIMENSION, facet_graph
from .resonance import integer_powers

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class FaceLattice:
    """Every face of P with its resonance, and the resonance centres it gives.

    Attributes
    ----------
    lattice
        The decorated face lattice at the D_0 and powers of the resonance
        section; see :meth:`feynkit.FeynmanIntegral.face_lattice`. The faces up
        to codimension FACE_CODIMENSION carry their identification.
    d0_source
        Where D_0 comes from, as in the resonance section.
    unit_powers
        Whether the powers are all 1 because the exponents of the integral are
        not all integers.
    schwinger
        The face-by-face comparison with the Cayley configuration of the
        Schwinger representation, when it was asked for; None otherwise.
    """

    lattice: DecoratedFaceLattice
    d0_source: D0Source
    unit_powers: bool
    schwinger: SchwingerCheck | None


def build(ctx: BuildContext) -> FaceLattice:
    fi = ctx.integral
    powers, unit = integer_powers(fi)
    edges = [e.idx for e in fi.graph.get_internal_edges()]
    lattice = fi.face_lattice(
        ctx.d0,
        nu=dict(zip(edges, powers, strict=True)),
        identify_codimension=FACE_CODIMENSION,
    )
    return FaceLattice(
        lattice=lattice,
        d0_source=ctx.d0_source,
        unit_powers=unit,
        schwinger=lattice.check_schwinger() if ctx.check_schwinger else None,
    )


# The kinds of resonant set, in the columns of the count table.
_KINDS = ("all", "progression", "point", "never")


def kind_counts(lattice: DecoratedFaceLattice) -> list[tuple[int, list[int]]]:
    """Per dimension from -1 up: the number of faces resonant for every eps, on a progression,
    at one value and for none."""
    top = lattice.faces[-1].dimension
    return [
        (
            k,
            [
                sum(f.dimension == k and f.resonant.kind == kind for f in lattice.faces)
                for kind in _KINDS
            ],
        )
        for k in range(-1, top + 1)
    ]


def facet_names(face: DecoratedFace, *, latex: bool) -> str:
    """The facets containing a face, F_k numbered as in the resonance section; "every facet"
    for the empty face and "none" for P."""
    if not face.point_indices:
        return "every facet"
    if not face.facets:
        return "none"
    if latex:
        return "$" + ", ".join(f"F_{{{k + 1}}}" for k in face.facets) + "$"
    return ", ".join(f"F_{k + 1}" for k in face.facets)


Rows = list[tuple[str, ...]]


def _dimension(face: DecoratedFace, latex: bool) -> str:
    return f"${face.dimension}$" if latex else str(face.dimension)


def face_rows(faces: Sequence[DecoratedFace], *, latex: bool) -> Rows:
    """Per face: its dimension, the facets containing it, and its graph; a dash for the graph
    beyond the codimension identified, for a face not identified and for the empty face."""
    rows: Rows = []
    for face in faces:
        found = face.identification
        graph = facet_graph(found, latex=latex) if found is not None else ("--" if latex else "-")
        rows.append((_dimension(face, latex), facet_names(face, latex=latex), graph))
    return rows


def verdict(lattice: DecoratedFaceLattice, eps: Epsilon) -> str:
    """What the resonance centres at eps say about reducibility, as the end of a sentence."""
    if not lattice.full_rank:
        return "A does not have full rank, so the report does not decide reducibility"
    if lattice.reducible(eps):
        which = "it" if len(lattice.centres(eps)) == 1 else "any of them"
        return f"A is not a pyramid over {which}, so the GKZ system has reducible monodromy"
    return "A is a pyramid over it, so the GKZ system has irreducible monodromy"


def centres_part(
    lattice: DecoratedFaceLattice, eps: Epsilon, where: str, *, latex: bool
) -> tuple[str, Rows | None]:
    """The sentence on the resonance centres at eps, where saying which eps, as in "At generic
    epsilon", and the rows of the table of centres after it; None when there is no table, for
    no centre or the empty face alone."""
    centres = lattice.centres(eps)
    if not centres:
        return (
            f"{where} no face is resonant: beta lies outside the span of A, where the GKZ "
            "system has no non-zero solutions.",
            None,
        )
    if not centres[0].point_indices:
        return f"{where} the resonance centre is the empty face: {verdict(lattice, eps)}.", None
    lead = "centre is the face" if len(centres) == 1 else "centres are the faces"
    return (
        f"{where} the resonance {lead} below; {verdict(lattice, eps)}:",
        face_rows(centres, latex=latex),
    )


def cayley_line(check: SchwingerCheck) -> str:
    """The sentence on the comparison with the Cayley configuration."""
    if check.agrees:
        return (
            "The Cayley configuration of the Schwinger representation, decorated from its own "
            "columns, agrees with this lattice face by face under the matrix T of the "
            "Schwinger-representation section."
        )
    if not check.bijective:
        return (
            "The matrix T of the Schwinger-representation section does not map these columns one "
            "to one onto those of the Cayley matrix."
        )
    return (
        "The Cayley configuration of the Schwinger representation, decorated from its own "
        f"columns, differs from this lattice on {len(check.mismatches)} faces."
    )


_FACES = ("Dimension", "In facets", "Graph")


# The maths of the section, as the text prints it and in LaTeX.
_MATHS = {
    "eps": ("epsilon", r"\varepsilon"),
    "D": ("D = D_0 - 2 epsilon", r"D = D_0 - 2\varepsilon"),
    "nu_e": ("nu_e = 1", r"\nu_e = 1"),
    "G": ("G", "G"),
    "P": ("P", "P"),
    "A": ("A", "A"),
    "beta": ("beta", r"\beta"),
    "resonant": ("span(A_G) + ZA", r"\operatorname{span}_{\mathbb{C}} A_G + \mathbb{Z}A"),
    "span": ("span(A_G)", r"\operatorname{span}_{\mathbb{C}} A_G"),
    "N_G": ("N_G", "N_G"),
    "ZA": ("ZA", r"\mathbb{Z}A"),
    "A_G": ("A_G", "A_G"),
    "T_G": ("|T_G|", r"|T_G|"),
    "one": ("|T_G| = 1", r"|T_G| = 1"),
    "F_k": ("F_k", "F_k"),
    "centre": ("ZA + span(A_G)", r"\mathbb{Z}A + \operatorname{span}_{\mathbb{C}} A_G"),
    "M_A": ("M_A(beta)", r"M_A(\beta)"),
    "zero": ("epsilon = 0", r"\varepsilon = 0"),
}


def _maths(latex: bool) -> dict[str, str]:
    return {key: (f"${tex}$" if latex else text) for key, (text, tex) in _MATHS.items()}


def opening(section: FaceLattice, cite: Callable[..., str], *, latex: bool) -> list[str]:
    """The paragraphs before the count table."""
    x = _maths(latex)
    lattice = section.lattice
    d0_is = f"$D_0 = {lattice.d0}$" if latex else f"D_0 = {lattice.d0}"
    clause = {
        "given": f"{d0_is} as given",
        "dimension": f"{d0_is}, read from the dimension of the integral",
        "default": f"{d0_is} by default",
    }[section.d0_source]
    if section.unit_powers:
        powers = (
            f"{x['nu_e']} on every edge, since the exponents of the integral are not all "
            "integers"
        )
    else:
        vector = "(" + ", ".join(map(str, lattice.nu)) + ")"
        powers = f"the powers $\\nu = {vector}$" if latex else f"the powers nu = {vector}"
    first = (
        f"Let {x['D']} with {clause}, and {powers}. A face {x['G']} of {x['P']}, the empty "
        f"face included, is resonant when {x['beta']} lies in {x['resonant']} and admissible "
        f"when it lies in {x['span']}{cite('britto2026')}. Let {x['N_G']} be the linear forms "
        f"that take integer values on {x['ZA']} and vanish on {x['A_G']}. {x['G']} is resonant "
        f"exactly when {x['beta']} lies in the span of {x['A']} and every form of a basis of "
        f"{x['N_G']} is an integer on it, and admissible exactly when every such form vanishes "
        "on it; for a facet this is the test of the resonance section. The forms of the facets "
        f"containing {x['G']} span a subgroup of {x['N_G']} of index {x['T_G']}, the lattice "
        f"defect, and {x['G']} is resonant wherever all those facets are, for every "
        f"{x['beta']}, exactly when {x['one']}. A face is named below by the facets {x['F_k']} "
        "containing it, numbered as in the resonance section."
    )
    second = (
        f"A resonance centre is a minimal face {x['G']} with {x['beta']} in {x['centre']}. "
        f"{x['M_A']} has reducible monodromy when {x['A']} is not a pyramid over a centre, and "
        f"irreducible monodromy when {x['A']} is a pyramid over one, which is then the only "
        f"centre{cite('schulze2012')}. Both results assume that {x['A']} has full rank."
    )
    return [first, second]


Part = tuple[str, tuple[str, ...], Rows | None]


def parts(section: FaceLattice, *, latex: bool) -> list[Part]:
    """The sentences after the count table, each with the header and rows of the table that
    follows it, or None without one: the centres at generic eps and at eps = 0, the faces over
    which A is a pyramid besides P and those with a non-trivial lattice defect, when there are
    any, and the comparison with the Cayley side, when it was made."""
    x = _maths(latex)
    lattice = section.lattice
    out: list[Part] = []
    moments: tuple[tuple[Epsilon, str], ...] = (
        ("generic", f"At generic {x['eps']}"),
        (0, f"At {x['zero']}"),
    )
    for eps, where in moments:
        sentence, rows = centres_part(lattice, eps, where, latex=latex)
        out.append((sentence, _FACES, rows))
    pyramids = [f for f in lattice.faces if f.pyramid and f.codimension > 0]
    if pyramids:
        out.append(
            (
                f"Besides {x['P']}, {x['A']} is a pyramid over these faces:",
                _FACES,
                face_rows(pyramids, latex=latex),
            )
        )
    defects = [f for f in lattice.faces if f.lattice_defect != 1]
    if defects:
        out.append(
            (
                f"These faces have a non-trivial lattice defect {x['T_G']}:",
                ("Dimension", "In facets", x["T_G"]),
                [
                    (
                        _dimension(f, latex),
                        facet_names(f, latex=latex),
                        str(f.lattice_defect),
                    )
                    for f in defects
                ],
            )
        )
    if section.schwinger is not None:
        out.append((cayley_line(section.schwinger), (), None))
    return out


_COUNT_CAPTION = (
    "The faces of each dimension, the empty face having dimension {low}, by where they are "
    "resonant:"
)


def _count_rows(lattice: DecoratedFaceLattice, *, latex: bool) -> Rows:
    return [
        (f"${k}$" if latex else str(k), str(sum(row)), *map(str, row))
        for k, row in kind_counts(lattice)
    ]


def text(_report: AnalysisReport, section: FaceLattice, doc: TextDocument) -> str:
    blocks = [_paragraph(p) for p in opening(section, doc.cite, latex=False)]
    header = ("Dimension", "Faces", "Every epsilon", "Progression", "One value", "None")
    blocks += [
        _paragraph(_COUNT_CAPTION.format(low="-1")),
        _table(header, _count_rows(section.lattice, latex=False)),
    ]
    for sentence, table_header, rows in parts(section, latex=False):
        blocks.append(_paragraph(sentence))
        if rows is not None:
            blocks.append(_table(table_header, rows))
    return _blocks(*blocks)


def latex(_report: AnalysisReport, section: FaceLattice, doc: LatexDocument) -> str:
    out = list(opening(section, doc.cite, latex=True))
    header = ("Dimension", "Faces", "Every $\\varepsilon$", "Progression", "One value", "None")
    out += [
        _COUNT_CAPTION.format(low="$-1$"),
        _longtable("rrrrrr", header, _count_rows(section.lattice, latex=True)),
    ]
    for sentence, table_header, rows in parts(section, latex=True):
        out.append(sentence)
        if rows is not None:
            out.append(_longtable("rl" + "l" * (len(table_header) - 2), table_header, rows))
    return "\n\n".join(out)


def summary(section: FaceLattice) -> list[tuple[str, str]]:
    lattice = section.lattice
    reducible = lattice.reducible("generic")
    value = "not decided" if reducible is None else "yes" if reducible else "no"
    return [
        ("Resonance centres at generic D", str(len(lattice.centres("generic")))),
        ("Reducible at generic D", value),
    ]


SECTION = Section(
    name="face_lattice",
    heading="Face resonance and reducibility",
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(75, summary),),
)
