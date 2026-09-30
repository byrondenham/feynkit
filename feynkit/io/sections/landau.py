"""
The Landau surfaces: discriminants of G on the faces of its Newton polytope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import sympy as sp

from ... import _exact
from ...landau import (
    LandauAnalysis,
    LimitSurface,
    one_loop_bridge_poles,
    one_loop_landau_surfaces_by_type,
)
from .._latex_kit import LatexDocument, _escape
from .._report_shared import count_noun, join_words
from .._text_kit import TextDocument, _blocks, _expressions, _paragraph
from ..latex import to_latex_lines
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Landau:
    """The reduced principal A-determinant and the surfaces it factors into.

    Attributes
    ----------
    analysis
        The face-by-face computation.
    by_dimension
        The non-trivial face discriminants grouped by face dimension,
        ascending, each group free of repeats.
    first_type, second_type
        The Cayley and Gram factors of the one-loop closed form; both empty
        for more than one loop, where no closed form is available. For a
        graph with bridges they are those of its cycle.
    skipped
        One (dimension, number of points, whether it is P itself) triple per
        face left out as too large to eliminate, in the order of
        ``analysis.skipped_faces``.
    bridge_poles
        The factors of the poles m_b^2 = q_b^2 of the bridges of a one-loop
        graph, from :func:`~feynkit.landau.one_loop_bridge_poles`; empty
        without bridges or for more than one loop.
    timed_out
        For each entry of ``skipped``, whether the face had few enough points
        to be eliminated and was skipped because its elimination ran past the
        time limit.
    """

    analysis: LandauAnalysis
    by_dimension: tuple[tuple[int, tuple[sp.Expr, ...]], ...]
    first_type: tuple[sp.Expr, ...]
    second_type: tuple[sp.Expr, ...]
    skipped: tuple[tuple[int, int, bool], ...] = ()
    bridge_poles: tuple[sp.Expr, ...] = ()
    timed_out: tuple[bool, ...] = ()


def build(ctx: BuildContext) -> Landau:
    fi = ctx.integral
    analysis = ctx.analysis()
    max_face_points = ctx.max_face_points
    grouped: dict[int, list[sp.Expr]] = {}
    for face in analysis.face_discriminants:
        if face.discriminant == 1:
            continue
        distinct = grouped.setdefault(face.dimension, [])
        if face.discriminant not in distinct:
            distinct.append(face.discriminant)
    first: tuple[sp.Expr, ...] = ()
    second: tuple[sp.Expr, ...] = ()
    poles: tuple[sp.Expr, ...] = ()
    if fi.loop_count == 1:
        first, second = one_loop_landau_surfaces_by_type(fi)
        poles = one_loop_bridge_poles(fi)
    dimensions = [_affine_dimension(face) for face in analysis.skipped_faces]
    # Only P itself has the top dimension among the faces.
    top = max([face.dimension for face in analysis.face_discriminants] + dimensions, default=0)
    return Landau(
        analysis=analysis,
        by_dimension=tuple((d, tuple(faces)) for d, faces in sorted(grouped.items())),
        first_type=first,
        second_type=second,
        skipped=tuple(
            (d, len(face), d == top)
            for d, face in zip(dimensions, analysis.skipped_faces, strict=True)
        ),
        bridge_poles=poles,
        timed_out=tuple(len(face) <= max_face_points for face in analysis.skipped_faces),
    )


def _affine_dimension(points: tuple[tuple[int, ...], ...]) -> int:
    """The dimension of the affine hull of the points."""
    return _exact.affine_rank(points)


def display_factors(expr: sp.Expr, scale: sp.Symbol) -> list[sp.Expr]:
    """The factors of a face discriminant that the document lists, one per line.

    This is a display rule; the discriminants stored in ``Landau.by_dimension``
    stay exact. The product is split into its sums, or powers of sums, and a
    monomial. A factor free of every symbol but the energy scale mu, that is a
    number or a power of mu, is dropped: it only reflects the normalisation of
    the coefficients z_j = c / mu^k and vanishes nowhere. The monomial
    therefore loses its numerical coefficient and its power of mu, so that
    -p_1^2 / mu^2 is listed as p_1^2 and mu s as s. A sum is never altered,
    even one that contains mu alongside kinematic symbols.
    """
    sums: list[sp.Expr] = []
    monomial: list[sp.Expr] = []
    for arg in sp.Mul.make_args(expr):
        if arg.free_symbols <= {scale}:
            continue
        base = arg.base if isinstance(arg, sp.Pow) else arg
        (sums if isinstance(base, sp.Add) else monomial).append(arg)
    if monomial:
        return [sp.Mul(*monomial), *sums]
    return sums


def sorted_factors(factors: Sequence[sp.Expr]) -> list[sp.Expr]:
    """Distinct factors in a deterministic order."""
    return sorted(dict.fromkeys(factors), key=sp.default_sort_key)


class LandauFactors(NamedTuple):
    """The factors the Landau section lists.

    Attributes
    ----------
    by_dimension
        The distinct factors of the face discriminants of each dimension,
        ascending, for the dimensions that have any.
    closed_form
        Whether the one-loop closed form gave first-type or second-type
        factors or bridge poles, so that the section compares against it.
    first_type, second_type
        The distinct Cayley and Gram factors.
    in_both
        How many factors appear in both lists.
    bridge_poles
        The distinct factors of the poles of the bridges.
    """

    by_dimension: list[tuple[int, list[sp.Expr]]]
    closed_form: bool
    first_type: list[sp.Expr]
    second_type: list[sp.Expr]
    in_both: int
    bridge_poles: list[sp.Expr]


def landau_factors(landau: Landau, scale: sp.Symbol) -> LandauFactors:
    """The factors the Landau section lists, through :func:`display_factors`."""

    def factors(expressions: Sequence[sp.Expr]) -> list[sp.Expr]:
        return sorted_factors([f for x in expressions for f in display_factors(x, scale)])

    by_dimension = [
        (dimension, factors(discriminants)) for dimension, discriminants in landau.by_dimension
    ]
    first = factors(landau.first_type)
    second = factors(landau.second_type)
    return LandauFactors(
        by_dimension=[(dimension, listed) for dimension, listed in by_dimension if listed],
        closed_form=bool(landau.first_type or landau.second_type or landau.bridge_poles),
        first_type=first,
        second_type=second,
        in_both=len(set(first) & set(second)),
        bridge_poles=factors(landau.bridge_poles),
    )


class LimitFactors(NamedTuple):
    """The factors of the parent family's surfaces, restricted, that the Landau section
    lists apart from the principal Landau determinant.

    Attributes
    ----------
    surfaces, candidates
        The limit surfaces and the candidates, each sorted by its factor.
    parent_skipped
        How many faces the analysis of the parent family skipped.
    """

    surfaces: list[LimitSurface]
    candidates: list[LimitSurface]
    parent_skipped: int


def limit_factors(landau: Landau) -> LimitFactors | None:
    """The limit surfaces and candidates of the section, or None without a parent family."""
    analysis = landau.analysis
    if analysis.parent is None:
        return None

    def ordered(records: Sequence[LimitSurface]) -> list[LimitSurface]:
        return sorted(records, key=lambda record: sp.default_sort_key(record.surface))

    return LimitFactors(
        surfaces=ordered(analysis.limit_surfaces),
        candidates=ordered(analysis.limit_candidates),
        parent_skipped=len(analysis.parent.skipped_faces),
    )


class LimitSentences(NamedTuple):
    """The prose around the lists of limit surfaces and candidates, with ``{chi}`` and
    ``{function}`` and ``{complement}`` for the renderer's maths, and ``{cite}``."""

    intro: str
    surfaces: str | None
    counts: str | None
    candidates: str | None
    reasons: str | None
    closing: str | None


def limit_sentences(limits: LimitFactors) -> LimitSentences:
    """The sentences of the limit-surface paragraphs, with placeholders for the maths."""
    intro = (
        "These kinematics restrict those of the same graph and masses with generic external "
        "momenta. The factors above form the principal Landau determinant of the family as "
        "given{cite}, which can leave out a component whose singular points leave the torus as the "
        "kinematics specialise; the surfaces of the generic family, restricted here, keep such "
        "limits."
    )
    surfaces = counts = candidates = reasons = closing = None
    if limits.surfaces:
        surfaces = (
            "The following factors of those restrictions are not among the factors above, and "
            "each is a limit surface: the number of critical points of {function} on {complement}, "
            "which is {chi} for generic exponents, is lower at two random rational points of it "
            "than at a random point of the family. That is evidence that the Euler characteristic "
            "drops there, not a proof: the points are random, and the counts are taken modulo two "
            "primes."
        )
        generic = limits.surfaces[0].generic_count
        pairs = ", ".join(" and ".join(str(c) for c in record.counts) for record in limits.surfaces)
        counts = f"The counts at the two points of each, in the order listed, are {pairs}, against {generic}."
    if limits.candidates:
        either = " either" if limits.surfaces else ""
        candidates = (
            f"The following factors of those restrictions are not among the factors above{either}, "
            "and each is only a candidate:"
        )
        reasons = (
            "Why each is not confirmed, in the order listed: "
            + "; ".join(record.reason or "" for record in limits.candidates)
            + "."
        )
    if not limits.surfaces and not limits.candidates:
        closing = (
            "Every factor of those restrictions is among the factors above, or the restriction "
            "vanishes identically."
        )
    if limits.parent_skipped:
        n = limits.parent_skipped
        skipped = (
            f"The analysis of the generic family skipped {count_noun(n, 'face')}, too large to "
            "eliminate or past the time limit, so the limit surfaces may be incomplete."
        )
        closing = skipped if closing is None else f"{closing} {skipped}"
    return LimitSentences(intro, surfaces, counts, candidates, reasons, closing)


def skipped_faces(landau: Landau) -> str | None:
    """The sentence naming the faces the Landau analysis skipped, or None if none was."""
    timed_out = landau.timed_out or (False,) * len(landau.skipped)
    skipped = sorted(zip(landau.skipped, timed_out, strict=True))
    if not skipped:
        return None
    names = [
        (
            f"the whole polytope, {points} points"
            if whole
            else f"a face of dimension {dimension} with {points} points"
        )
        + (" (past the time limit)" if late else "")
        for (dimension, points, whole), late in skipped
    ]
    n = len(skipped)
    why = (
        ", too large to eliminate or past the time limit,"
        if any(timed_out)
        else " as too large to eliminate,"
    )
    return (
        f"{count_noun(n, 'face')} {'was' if n == 1 else 'were'} skipped{why} and "
        f"{'its discriminant is' if n == 1 else 'their discriminants are'} missing from the "
        f"list: {join_words(names)}."
    )


def _text_factor_list(kind: str, factors: Sequence[sp.Expr]) -> tuple[str, str | None]:
    """'the <kind> factors are' and their display, or a note that there are none."""
    if not factors:
        return f"there are no {kind} factors", None
    return f"the {kind} factors are", _expressions(factors)


def text(report: AnalysisReport, landau: Landau, doc: TextDocument) -> str:
    scale = report.conventions.energy_scale
    intro = (
        "The reduced principal A-determinant of G, each irreducible kinematic factor taken "
        "once, is the product of the discriminants of G restricted to the faces of P"
        f"{doc.cite('gkz1994', 'dhpt2023', 'fmt2023')}."
    )
    factors = landau_factors(landau, scale)
    blocks = []
    if factors.by_dimension:
        blocks += [
            _paragraph(
                intro + " Each discriminant that is not identically one factorises as follows, "
                "by face dimension, with one factor per line; numerical factors and powers of "
                "the energy scale mu, which only normalise the coefficients z_j, are left out:"
            ),
            "\n".join(
                f"- Dimension {dimension}:\n{_expressions(listed)}"
                for dimension, listed in factors.by_dimension
            ),
        ]
    else:
        blocks.append(_paragraph(intro + " No face discriminant has a kinematic factor."))
    if factors.closed_form:
        first, first_display = _text_factor_list("first-type (Cayley)", factors.first_type)
        second, second_display = _text_factor_list("second-type (Gram)", factors.second_type)
        sentence = (
            "By comparison with the closed form from the modified Cayley matrix"
            f"{doc.cite('dhpt2023')}, {first}"
        )
        if first_display is not None:
            blocks += [_paragraph(sentence), first_display]
            sentence = f"and {second}"
        else:
            sentence += f" and {second}"
        blocks.append(_paragraph(sentence if second_display is not None else sentence + "."))
        if second_display is not None:
            blocks.append(second_display)
        shared = factors.in_both
        if shared:
            blocks.append(
                _paragraph(
                    "A factor can arise from both a Cayley minor and a Gram minor, which is why "
                    f"{count_noun(shared, 'factor')} {'appears' if shared == 1 else 'appear'} "
                    "in both lists."
                )
            )
        if factors.bridge_poles:
            lead = (
                "The bridge pole, which the faces give as well, is"
                if len(factors.bridge_poles) == 1
                else "The bridge poles, which the faces give as well, are"
            )
            blocks += [
                _paragraph(
                    "A propagator on no loop is a bridge. The closed form is that of the graph's "
                    "cycle, with the legs of each tree attached to the cycle moved to the vertex "
                    "where the tree meets it, and each bridge b adds the pole m_b^2 = q_b^2 of "
                    f"its propagator, q_b being the momentum through it. {lead}"
                ),
                _expressions(factors.bridge_poles),
            ]
    blocks.append(
        _paragraph(
            "The factors are candidate codimension-one singular loci on all sheets of the "
            "integral: a point on one of them may or may not be singular on the physical "
            f"sheet, and the list is not guaranteed complete{doc.cite('fmt2023')}. The "
            "coefficients are specialised to physical kinematics before each face "
            "discriminant is computed, so beyond one loop this is the principal Landau "
            "determinant rather than the principal A-determinant of the generic polynomial; "
            "multiplicities are dropped."
        )
    )
    skipped = skipped_faces(landau)
    if skipped is not None:
        blocks.append(_paragraph(skipped))
    blocks += _text_limits(landau, doc)
    return _blocks(*blocks)


def _text_limits(landau: Landau, doc: TextDocument) -> list[str]:
    """The paragraphs on the limit surfaces and candidates, none without a parent family."""
    limits = limit_factors(landau)
    if limits is None:
        return []
    maths = {
        "cite": doc.cite("fmt2024"),
        "function": "sum_e nu_e log u_e - (D/2) log G",
        "complement": "the complement of {G = 0} in the torus",
        "chi": "|chi|",
    }
    sentences = limit_sentences(limits)
    blocks = [_paragraph(sentences.intro.format(**maths))]
    if sentences.surfaces is not None:
        blocks += [
            _paragraph(sentences.surfaces.format(**maths)),
            _expressions([record.surface for record in limits.surfaces]),
            _paragraph(sentences.counts or ""),
        ]
    if sentences.candidates is not None:
        blocks += [
            _paragraph(sentences.candidates),
            _expressions([record.surface for record in limits.candidates]),
            _paragraph(sentences.reasons or ""),
        ]
    if sentences.closing is not None:
        blocks.append(_paragraph(sentences.closing))
    return blocks


# TeX sets an align* whole before breaking it across pages, and the massless hexagon's list of
# over 9,000 lines ran out of pdflatex's memory; 3,967 lines still fitted.
_DISPLAY_LINES = 500


def _factor_lines(factors: Sequence[sp.Expr]) -> str:
    """``align*`` displays with each factor on its own line, long factors broken further.

    A list longer than _DISPLAY_LINES lines is split into several displays,
    between factors where one fits, and a longer factor across displays.
    """
    blocks: list[list[str]] = [[]]
    for factor in factors:
        lines = to_latex_lines(factor, max_length=100)
        rows = ["&" + lines[0]] + ["&\\quad {}" + line for line in lines[1:]]
        if blocks[-1] and len(blocks[-1]) + len(rows) > _DISPLAY_LINES:
            blocks.append([])
        for row in rows:
            if len(blocks[-1]) == _DISPLAY_LINES:
                blocks.append([])
            blocks[-1].append(row)
    return "\n".join(
        "\\begin{align*}\n" + " \\\\\n".join(block) + "\n\\end{align*}" for block in blocks
    )


def _latex_factor_list(kind: str, factors: Sequence[sp.Expr]) -> str:
    """'the <kind> factors are' followed by their display, or a note that there are none."""
    if not factors:
        return f"there are no {kind} factors"
    return f"the {kind} factors are\n" + _factor_lines(factors)


def latex(report: AnalysisReport, landau: Landau, doc: LatexDocument) -> str:
    scale = report.conventions.energy_scale
    intro = (
        "The reduced principal $A$-determinant of $G$, each irreducible kinematic factor taken "
        "once, is the product of the discriminants of $G$ restricted to the faces of $P$"
        f"{doc.cite('gkz1994', 'dhpt2023', 'fmt2023')}."
    )
    factors = landau_factors(landau, scale)
    parts: list[str] = []
    if factors.by_dimension:
        parts += [
            intro + " Each discriminant that is not identically one factorises as follows, by "
            "face dimension, with one factor per line; numerical factors and powers of the "
            "energy scale $\\mu$, which only normalise the coefficients $z_j$, are left out:",
            "\\begin{itemize}",
        ]
        for dimension, listed in factors.by_dimension:
            parts += [f"\\item Dimension {dimension}:", _factor_lines(listed)]
        parts.append("\\end{itemize}")
    else:
        parts.append(intro + " No face discriminant has a kinematic factor.")
    if factors.closed_form:
        text = (
            "By comparison with the closed form from the modified Cayley matrix"
            f"{doc.cite('dhpt2023')}, {_latex_factor_list('first-type (Cayley)', factors.first_type)}"
            f"\nand {_latex_factor_list('second-type (Gram)', factors.second_type)}"
        )
        if not factors.second_type:
            text += "."
        shared = factors.in_both
        if shared:
            text += (
                "\nA factor can arise from both a Cayley minor and a Gram minor, which is why "
                f"{count_noun(shared, 'factor')} {'appears' if shared == 1 else 'appear'} in both "
                "lists."
            )
        if factors.bridge_poles:
            lead = (
                "The bridge pole, which the faces give as well, is"
                if len(factors.bridge_poles) == 1
                else "The bridge poles, which the faces give as well, are"
            )
            text += (
                "\nA propagator on no loop is a bridge. The closed form is that of the graph's "
                "cycle, with the legs of each tree attached to the cycle moved to the vertex "
                "where the tree meets it, and each bridge $b$ adds the pole $m_b^2 = q_b^2$ of "
                f"its propagator, $q_b$ being the momentum through it. {lead}\n"
                + _factor_lines(factors.bridge_poles)
            )
        parts.append(text)
    parts.append(
        "The factors are candidate codimension-one singular loci on all sheets of the "
        "integral: a point on one of them may or may not be singular on the physical sheet, "
        f"and the list is not guaranteed complete{doc.cite('fmt2023')}. The coefficients are "
        "specialised to physical kinematics before each face discriminant is computed, so "
        "beyond one loop this is the principal Landau determinant rather than the principal "
        "$A$-determinant of the generic polynomial; multiplicities are dropped."
    )
    skipped = skipped_faces(landau)
    if skipped is not None:
        parts.append(skipped)
    parts += _latex_limits(landau, doc)
    return "\n".join(parts)


def _latex_limits(landau: Landau, doc: LatexDocument) -> list[str]:
    """The paragraphs on the limit surfaces and candidates, none without a parent family."""
    limits = limit_factors(landau)
    if limits is None:
        return []
    maths = {
        "cite": doc.cite("fmt2024"),
        "function": "$\\sum_e \\nu_e \\log u_e - (D/2) \\log G$",
        "complement": "the complement of $\\{G = 0\\}$ in the torus",
        "chi": "$|\\chi|$",
    }
    sentences = limit_sentences(limits)
    parts = ["", sentences.intro.format(**maths)]
    if sentences.surfaces is not None:
        parts += [
            sentences.surfaces.format(**maths),
            _factor_lines([record.surface for record in limits.surfaces]),
            sentences.counts or "",
        ]
    if sentences.candidates is not None:
        parts += [
            sentences.candidates,
            _factor_lines([record.surface for record in limits.candidates]),
            _escape(sentences.reasons or ""),
        ]
    if sentences.closing is not None:
        parts.append(sentences.closing)
    return parts


def summary(landau: Landau) -> list[tuple[str, str]]:
    analysis = landau.analysis
    rows = [("Landau surfaces", str(len(analysis.landau_surfaces)))]
    if analysis.parent is not None:
        rows.append(("Limit surfaces", str(len(analysis.limit_surfaces))))
        rows.append(("Limit candidates", str(len(analysis.limit_candidates))))
        rows.append(("Parent skipped faces", str(len(analysis.parent.skipped_faces))))
    return rows


SECTION = Section(
    name="landau",
    heading="Landau surfaces",
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(100, summary),),
)
