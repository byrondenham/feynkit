"""
The degenerate faces of the Newton polytope of G, at the point of the torus counts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

import sympy as sp

from ...degeneracy import DegeneracyAnalysis
from ...face_identification import FaceIdentification
from ...landau import LandauAnalysis
from ...point_count import kinematic_point
from .._latex_kit import LatexDocument, _longtable
from .._report_shared import count_noun, join_words
from .._text_kit import TextDocument, _blocks, _paragraph, _str, _table
from ..latex import to_latex
from ._base import Section, SummaryPart
from .faces import FACE_CODIMENSION, facet_graph

if TYPE_CHECKING:
    from ...integral import FeynmanIntegral
    from ..report import AnalysisReport, BuildContext

DegeneracyMode = Literal["point", "generic"]

# How the report decides the faces when the caller does not say.
DEFAULT_DEGENERACY_MODE: DegeneracyMode = "point"


@dataclass(frozen=True)
class Degeneracy:
    """Which faces of the Newton polytope of G are degenerate.

    Attributes
    ----------
    analysis
        From :meth:`~feynkit.integral.FeynmanIntegral.face_degeneracy`, at the
        kinematic point the torus counts draw with the report's seed, or at
        the generic point of the kinematics; None when a face needed Singular
        and it was not found.
    seed
        The seed the point was drawn with; None at the generic point.
    identifications
        For each face of ``analysis.degenerate_faces``, in order, its graph from
        :meth:`~feynkit.integral.FeynmanIntegral.face_identification` when the
        face lies within codimension FACE_CODIMENSION of the Newton polytope of
        the family; None otherwise.
    """

    analysis: DegeneracyAnalysis | None
    seed: int | None
    identifications: tuple[FaceIdentification | None, ...] = ()


def torus_point(
    integral: FeynmanIntegral, seed: int, landau: LandauAnalysis | None = None
) -> dict[sp.Expr, Fraction]:
    """The kinematic point :meth:`~feynkit.integral.FeynmanIntegral.torus_count` draws with
    this seed, keyed as its ``point``; ``landau`` is the integral's Landau analysis, computed
    when not given."""
    sym = integral.symanzik
    return dict(
        kinematic_point(
            sym.g,
            sym.lp_parameters,
            scale=integral.graph.energy_scale,
            seed=seed,
            landau=landau,
        )
    )


def _identifications(
    integral: FeynmanIntegral, analysis: DegeneracyAnalysis
) -> tuple[FaceIdentification | None, ...]:
    points = tuple(tuple(int(x) for x in p) for p in integral.newton_polytope.points)
    found = {
        frozenset(points[i] for i in face.point_indices): face
        for face in integral.face_identification(FACE_CODIMENSION)
    }
    return tuple(
        found.get(frozenset(analysis.points[i] for i in face.point_indices))
        for face in analysis.degenerate_faces
    )


def build(ctx: BuildContext) -> Degeneracy:
    fi = ctx.integral
    generic = ctx.degeneracy_mode == "generic"
    seed = None if generic else ctx.torus_seed
    try:
        if generic:
            analysis = fi.face_degeneracy()
        else:
            analysis = fi.face_degeneracy(torus_point(fi, ctx.torus_seed, ctx.analysis()))
    except RuntimeError:
        # Raised only when a face needs Singular and it is not installed.
        return Degeneracy(analysis=None, seed=seed)
    return Degeneracy(analysis=analysis, seed=seed, identifications=_identifications(fi, analysis))


# --- the prose ---------------------------------------------------------------------


# The maths of the section, as the text prints it and in LaTeX.
_MATHS = {
    "F": ("F", "F"),
    "G": ("G", "G"),
    "G_F": ("G|_F", "G|_F"),
    "P_z": ("P_z", "P_z"),
    "z": ("z", "z"),
    "system": (
        "G|_F = u_1 dG|_F/du_1 = ... = u_N dG|_F/du_N = 0",
        r"G|_F = u_1 \partial_1 G|_F = \dots = u_N \partial_N G|_F = 0",
    ),
    "torus": ("(C^*)^N", r"(\mathbb{C}^*)^N"),
    "lattice": ("G|_F = u^a g_F(t)", r"G|_F = u^a g_F(t)"),
    "t": ("t", "t"),
    "d": ("d", "d"),
    "g_F": ("g_F", "g_F"),
    "torus_d": ("(C^*)^d", r"(\mathbb{C}^*)^d"),
    "saturation": (
        "(g_F, dg_F/dt_1, ..., dg_F/dt_d) : (t_1 ... t_d)^infinity",
        r"(g_F, \partial_1 g_F, \dots, \partial_d g_F) : (t_1 \cdots t_d)^\infty",
    ),
    "mu": ("mu = 1", r"\mu = 1"),
    "E_A": ("E_A", "E_A"),
    "X_A": ("X_A", "X_A"),
    "X": ("X", "X"),
    "derivatives": ("u_1 dG/du_1, ..., u_N dG/du_N", r"u_1 \partial_1 G, \dots, u_N \partial_N G"),
    "tau": ("tau", r"\tau"),
    "F_p": ("F_p", r"\mathbb{F}_p"),
    "Q(E)": ("Q(E)", r"\mathbb{Q}(\mathcal{E})"),
    "V": ("{G = 0}", r"\{G = 0\}"),
}


def _value(value: Fraction, latex: bool) -> str:
    q = sp.Rational(value.numerator, value.denominator)
    return to_latex(q) if latex else str(_str(q))


def _where(analysis: DegeneracyAnalysis, latex: bool) -> str:
    assert analysis.point is not None
    if latex:
        return join_words([f"${to_latex(key)} = {_value(v, True)}$" for key, v in analysis.point])
    return join_words([f"{_str(key)} = {_value(v, False)}" for key, v in analysis.point])


def paragraphs(
    report: AnalysisReport, section: Degeneracy, cite: Callable[..., str], *, latex: bool
) -> tuple[list[str], str | None, list[str]]:
    """The paragraphs before the table of degenerate faces, the sentence that introduces it
    (None without one) and the paragraphs after it."""

    def m(text: str, tex: str) -> str:
        return f"${tex}$" if latex else text

    x = {key: m(text, tex) for key, (text, tex) in _MATHS.items()}
    opening = (
        f"A face {x['F']} of the Newton polytope {x['P_z']} of {x['G']} at a kinematic point "
        f"{x['z']}, the convex hull of the exponents whose coefficients do not vanish there, "
        f"is degenerate when {x['G_F']}, the sum of the terms of {x['G']} on {x['F']}, has a "
        f"singular point in the torus: {x['system']} has a solution in {x['torus']}. "
        f"{x['P_z']} itself counts as a face. Write {x['lattice']}, with {x['g_F']} in the "
        f"lattice coordinates {x['t']} of {x['F']} and {x['d']} its dimension; {x['F']} is "
        f"degenerate exactly when {x['g_F']} has a singular point in {x['torus_d']}. Vertices "
        "and faces whose points are affinely independent never are. An edge is decided by the "
        f"discriminant of {x['g_F']}, and any other face by whether {x['saturation']} is the "
        "unit ideal, with Groebner bases in Singular."
    )
    before = [opening]
    analysis = section.analysis
    if analysis is None:
        before.append(
            "Faces of dimension 2 or more whose points are not affinely independent need "
            "Singular, which was not found, so the faces are not decided."
        )
        return before, None, []

    if analysis.mode == "generic":
        origin = (
            "The faces are decided at the generic point of the kinematics, over the field "
            f"{x['Q(E)']} of rational functions in the kinematic symbols: a face is degenerate "
            "there exactly when it is degenerate on a dense set of kinematic points."
        )
    elif not analysis.point:
        origin = (
            f"{x['G']} has no kinematic symbols, so the faces need no kinematic point; they "
            f"are decided with {x['mu']}."
        )
    else:
        origin = (
            f"The faces are decided at the kinematic point {_where(analysis, latex)}, with "
            f"{x['mu']}, where the point counts are taken: drawn with seed {section.seed} so "
            f"that every coefficient of {x['G']} and every face discriminant is non-zero."
        )
    if analysis.mode == "point":
        call = (
            "\\texttt{FeynmanIntegral.face\\_degeneracy}"
            if latex
            else "FeynmanIntegral.face_degeneracy"
        )
        origin += (
            " Each face can also be decided at the generic point of the kinematics, with "
            f"{call} without a point, which can take minutes for larger graphs."
        )
    if analysis.support_loss:
        origin += (
            f" Coefficients vanish at the point, and {x['P_z']} is smaller than the Newton "
            f"polytope of the family: its normalised volume is {analysis.support_loss} less."
        )
    before.append(origin)

    degenerate = analysis.degenerate_faces
    undecided = analysis.undecided_faces
    full = analysis.volume > 0
    after: list[str] = []
    intro = None
    if degenerate:
        intro = (
            f"{'One face is' if len(degenerate) == 1 else f'{len(degenerate)} faces are'} "
            "degenerate. The table gives the dimension of the singular locus of each in the "
            f"torus and, when that locus is finite, its Tjurina number {x['tau']}, both in the "
            f"lattice coordinates of the face; a face beyond codimension {FACE_CODIMENSION} or "
            "not identified has a dash for its graph:"
        )
    else:
        verdict = "No face is degenerate" if not undecided else "No face decided is degenerate"
        if full and not undecided:
            at = "for generic kinematics" if analysis.mode == "generic" else "at the point"
            verdict += (
                f". Then the principal A-determinant {x['E_A']} of {x['G']} does not vanish "
                f"{at}: it is the A-resultant of {x['derivatives']} and {x['G']}"
                f"{cite('gkz1994')} (Ch. 10, (1.1), p. 297), which vanishes only where "
                f"they have a common zero on the toric variety {x['X_A']} of {x['P_z']} "
                "(Ch. 8, Prop.-Def. 1.1, p. 252), and a common zero on the orbit of a face "
                f"{x['F']} is a singular point of {x['G_F']} in the torus (Ch. 5, Prop. 1.9, "
                f"p. 171). So the complement {x['X']} of {x['V']} in the torus has "
                f"{m(_chi(analysis.volume, False), _chi(analysis.volume, True))} by Theorem "
                f"2.3 of{cite('fmt2024')}"
            )
        before.append(verdict + ".")
    if undecided:
        after.append(
            f"{'One face is' if len(undecided) == 1 else f'{len(undecided)} faces are'} "
            "undecided: Singular ran past its time limit on "
            f"{'it' if len(undecided) == 1 else 'them'}, and nothing is assumed about "
            f"{'it' if len(undecided) == 1 else 'them'}."
        )
    if analysis.prime is not None:
        checked = [f for f in analysis.faces if f.modular is not None and f.degenerate is not None]
        differ = sum(f.modular != f.degenerate for f in checked)
        agreement = (
            "it agrees on every face"
            if not differ
            else f"it differs on {count_noun(differ, 'face')}, where the exact verdict stands"
        )
        after.append(
            "Each face decided in Singular is decided a second time over "
            f"{x['F_p']} at {m(f'p = {analysis.prime}', f'p = {analysis.prime}')}, a "
            f"probabilistic check that never decides; {agreement}."
        )
    torus = report.torus
    if (
        analysis.mode == "point"
        and torus is not None
        and torus.point == analysis.point
        and not torus.on_shell
        and torus.candidate_euler_characteristic is not None
        and full
    ):
        chi = abs(torus.candidate_euler_characteristic)
        gap = analysis.volume - chi
        after.append(
            f"The point counts give the candidate {m(f'|chi(X)| = {chi}', _candidate(chi))}, "
            f"and {m(f'N! Vol(P_z) - |chi(X)| = {gap}', _gap(gap))}."
        )
    return before, intro, after


def _chi(volume: int, latex: bool) -> str:
    if latex:
        return rf"|\chi(X)| = N!\,\mathrm{{Vol}}(P_z) = {volume}"
    return f"|chi(X)| = N! Vol(P_z) = {volume}"


def _candidate(chi: int) -> str:
    return rf"|\chi(X)| = {chi}"


def _gap(gap: int) -> str:
    return rf"N!\,\mathrm{{Vol}}(P_z) - |\chi(X)| = {gap}"


def _rows(section: Degeneracy, *, latex: bool) -> list[tuple[str, ...]]:
    analysis = section.analysis
    assert analysis is not None
    dash = "--" if latex else "-"
    rows: list[tuple[str, ...]] = []
    for face, found in zip(analysis.degenerate_faces, section.identifications, strict=True):
        graph = facet_graph(found, latex=latex) if found is not None else dash
        rows.append(
            (
                str(face.dimension),
                str(len(face.point_indices)),
                graph,
                str(face.singular_dimension),
                dash if face.tjurina is None else str(face.tjurina),
            )
        )
    return rows


def _face_header(latex: bool) -> tuple[str, ...]:
    if latex:
        return ("Dimension", "Points", "Graph", r"$\dim \mathrm{Sing}$", r"$\tau$")
    return ("Dimension", "Points", "Graph", "dim Sing", "tau")


def text(report: AnalysisReport, section: Degeneracy, doc: TextDocument) -> str:
    before, intro, after = paragraphs(report, section, doc.cite, latex=False)
    blocks = [_paragraph(p) for p in before]
    if intro is not None:
        blocks += [_paragraph(intro), _table(_face_header(False), _rows(section, latex=False))]
    blocks += [_paragraph(p) for p in after]
    return _blocks(*blocks)


def latex(report: AnalysisReport, section: Degeneracy, doc: LatexDocument) -> str:
    before, intro, after = paragraphs(report, section, doc.cite, latex=True)
    parts = list(before)
    if intro is not None:
        parts += [intro, _longtable("rrlrr", _face_header(True), _rows(section, latex=True))]
    parts += after
    return "\n\n".join(parts)


def _count(analysis: DegeneracyAnalysis | None) -> str:
    if analysis is None:
        return "not decided"
    found = len(analysis.degenerate_faces)
    if analysis.undecided_faces:
        return f"at least {found}" if found else "not decided"
    return str(found)


def summary(section: Degeneracy) -> list[tuple[str, str]]:
    return [("Degenerate faces", _count(section.analysis))]


SECTION = Section(
    name="degeneracy",
    heading="Degenerate faces",
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(110, summary),),
)
