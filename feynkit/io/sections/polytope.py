"""
The Newton polytope of G, its lattice invariants and its figure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...lattice_invariants import LatticeInvariants
from ...polytope import PolytopeData
from .._latex_kit import LatexDocument
from .._report_shared import TORUS_HEADING, count_noun, join_words
from .._text_kit import TextDocument, _blocks, _paragraph
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


# The budget and Normaliz timeout of the lattice invariants in the report and in fk analyse:
# about five seconds of pure-Python work, beyond which Normaliz takes over or the fields that
# need more are not computed. The massless three-loop box would need about 10^10 steps.
LATTICE_BUDGET = 2 * 10**7


NORMALIZ_TIMEOUT = 60.0


@dataclass(frozen=True)
class Polytope:
    """The Newton polytope of G, its figure and its lattice invariants.

    Attributes
    ----------
    data
        Vertices, faces, facet inequalities and normalised volume.
    figure
        TikZ source, or None when the polytope has too many vertices to
        draw legibly.
    invariants
        The lattice invariants of FeynmanIntegral.lattice_invariants, in the
        lattice the points generate, within LATTICE_BUDGET and
        NORMALIZ_TIMEOUT; None only for a Polytope built by hand.

    Point indices in ``data`` (``vertex_indices``, ``faces``, each facet's
    ``point_indices``) follow the order of ``newton_polytope.points``, which
    is not the z-index order of ``gkz.support`` that :class:`ZEntry` uses: the
    two enumerate the same monomials independently and need not agree.
    """

    data: PolytopeData
    figure: str | None
    invariants: LatticeInvariants | None = None


def build(ctx: BuildContext) -> Polytope:
    fi = ctx.integral
    data = ctx.polytope_data()
    figure_max_vertices = ctx.figure_max_vertices
    draw = len(data.vertex_indices) <= figure_max_vertices
    return Polytope(
        data=data,
        figure=fi.visualise_polytope() if draw else None,
        invariants=fi.lattice_invariants(budget=LATTICE_BUDGET, timeout=NORMALIZ_TIMEOUT),
    )


# What the Newton polytope section says feynkit leaves uncomputed when P is full-dimensional.
NOT_COMPUTED = (
    "feynkit computes neither the holonomic rank at the physical point nor the Euler "
    "characteristic, and produces no series solutions, Pfaffian system or restriction to "
    "physical kinematics."
)


def not_computed(report: AnalysisReport, pointer: str) -> str:
    """What the Newton polytope section says feynkit leaves uncomputed.

    Without the point-count section this is NOT_COMPUTED. With it, ``pointer``
    names that section as the renderer refers to it, and the sentence says that
    the Euler characteristic there is only a candidate, or that the counts give
    no candidate. It does not say why: the section gives the reason, and the
    volume bound and the guard on characters of higher order refuse counts
    that may well be polynomial.
    """
    if report.torus is None:
        return NOT_COMPUTED
    rest = "It produces no series solutions, Pfaffian system or restriction to physical kinematics."
    if report.torus.candidate_master_count is not None:
        return (
            "feynkit does not compute the holonomic rank at the physical point, and gives the "
            f"Euler characteristic only as a candidate from the point counts of {pointer}. {rest}"
        )
    return (
        "feynkit computes neither the holonomic rank at the physical point nor the Euler "
        f"characteristic, and the point counts of {pointer} give no candidate. {rest}"
    )


def lattice_normality(polytope: Polytope, cite: Callable[..., str], *, latex: bool) -> str:
    """The sentence on the normality of NA: the certificate, or why there is none.

    NA is normal exactly when the monomials of G are all the lattice points of
    P and P has IDP. Then C[NA] is Cohen-Macaulay, and when P is
    full-dimensional there are no rank jumps; a monoid that is not normal can
    still be Cohen-Macaulay.
    """
    found = polytope.invariants
    assert found is not None
    if latex:
        na, ring = "$\\mathbb{N}A$", "$\\mathbb{C}[\\mathbb{N}A]$"
        p, g, beta = "$P$", "$G$", "$\\beta$"
    else:
        na, ring, p, g, beta = "NA", "C[NA]", "P", "G", "beta"
    if found.normal is None:
        call = "\\texttt{lattice\\_invariants}" if latex else "lattice_invariants"
        return (
            f"Whether {p} has the integer decomposition property{cite('bgt1997')}, and so "
            f"whether {na} is normal, was not computed within the report's budget: install "
            f"PyNormaliz, or call {call} directly."
        )
    if found.normal:
        head = (
            f"{p} has the integer decomposition property{cite('bgt1997')} and the monomials of "
            f"{g} are all its lattice points, so {na} is normal"
        )
        if not polytope.data.is_full_dimensional:
            return f"{head} and {ring} is Cohen-Macaulay{cite('hochster1972')}."
        return (
            f"{head}, {ring} is Cohen-Macaulay{cite('hochster1972')} and there are no rank "
            f"jumps: the holonomic rank is the normalised volume for every {beta}"
            f"{cite('mmw2005')}."
        )
    reasons = []
    if found.idp is False:
        reasons.append(f"{p} lacks the integer decomposition property{cite('bgt1997')}")
    if not found.support_is_saturated:
        missing = found.lattice_points - len(set(polytope.data.points))
        reasons.append(f"the monomials of {g} miss {missing} of its lattice points")
    return (
        f"{na} is not normal, since {' and '.join(reasons)}; this does not decide whether "
        f"{ring} is Cohen-Macaulay."
    )


def face_names(dimension: int) -> list[tuple[str, str]]:
    """Singular and plural names of the faces of each dimension below the top."""
    names = []
    for k in range(dimension):
        if k == 0:
            names.append(("vertex", "vertices"))
        elif k == 1:
            names.append(("edge", "edges"))
        elif k == dimension - 1:
            names.append(("facet", "facets"))
        else:
            names.append((f"{k}-face", f"{k}-faces"))
    return names


# How the Newton polytope section refers to the point-count section, which has no number.
_TEXT_TORUS_REFERENCE = f'the section "{TORUS_HEADING}"'


def text(report: AnalysisReport, polytope: Polytope, doc: TextDocument) -> str:
    data = polytope.data
    counts = [
        count_noun(n, singular, plural)
        for n, (singular, plural) in zip(
            data.f_vector[:-1], face_names(data.dimension), strict=True
        )
    ]
    shape = f", with {join_words(counts)}" if counts else ""
    vertices = join_words(
        [
            f"v_{k} = ({', '.join(str(x) for x in vertex)})"
            for k, vertex in enumerate(data.vertices, start=1)
        ]
    )
    volume = data.normalized_volume
    description = _paragraph(
        "The Newton polytope P of G is the convex hull of the exponent vectors of its "
        f"monomials. It has dimension {data.dimension} in R^{data.ambient_dimension} and "
        f"normalised volume {volume}{shape}. Its vertices are {vertices}."
    )

    def lattice() -> list[str]:
        # Called where the paragraph goes, so that its works are cited in reading order.
        return [] if polytope.invariants is None else [_paragraph(_text_lattice(polytope, doc))]

    if not data.is_full_dimensional:
        return _blocks(
            description,
            _paragraph(
                "Since P is not full-dimensional, the rows of A are linearly dependent. For "
                "generic beta the Euler equations are then inconsistent, and the GKZ system has "
                "no non-zero solutions: its holonomic rank is 0, not the normalised volume."
            ),
            *lattice(),
            _paragraph(_text_scaleless(report, doc)),
        )
    return _blocks(
        description,
        _paragraph(
            "For generic coefficients and non-resonant beta the holonomic rank of the GKZ "
            "system equals the normalised volume, here "
            f"{volume}{doc.cite('adolphson1994', 'chestnov2022')}. The rank equals the volume "
            "for every beta exactly when the toric ring C[NA] is "
            f"Cohen-Macaulay{doc.cite('mmw2005')}. This holds when the graph is one-particle "
            "irreducible and one-vertex irreducible, its external momenta are generic enough "
            "that no monomial of G cancels, and its propagators are all massive, all massless, "
            "or such that every vertex reaches an external leg along massive propagators "
            "alone; the configuration is then normal and so Cohen-Macaulay"
            f"{doc.cite('klausen2023', 'tellander2023', 'walther2022')}. The number of master "
            "integrals is, up to sign, the Euler characteristic of the complement of "
            "{G = 0} in the torus, at most N! times the Euclidean volume of P, with equality "
            "for generic coefficients; graph-polynomial coefficients are rarely generic"
            f"{doc.cite('bbkp2017')}. The bound equals the normalised volume when the exponent "
            f"differences span Z^N. {not_computed(report, _TEXT_TORUS_REFERENCE)}"
        ),
        *lattice(),
    )


def _text_lattice(polytope: Polytope, doc: TextDocument) -> str:
    """The lattice invariants of P and what they certify about C[NA]."""
    found = polytope.invariants
    assert found is not None
    counts = f"{found.lattice_points} lattice points ({found.interior_points} interior)"
    if found.h_star is None:
        shape = f"{counts} and"
    else:
        shape = f"{counts}, h* vector ({', '.join(str(h) for h in found.h_star)}) and"
    parts = [
        f"In the lattice generated by its points, P has {shape} lattice width "
        f"{found.lattice_width}{doc.cite('beckrobins2015')}."
    ]
    index = found.gorenstein_index
    if index == 1:
        parts.append(
            "It is reflexive: its one interior lattice point lies at lattice distance 1 from "
            f"every facet{doc.cite('batyrev1994')}."
        )
    elif index is not None:
        parts.append(
            f"It is Gorenstein of index {index}: {index}P has a lattice point at lattice "
            "distance 1 from every facet."
        )
    else:
        parts.append(
            "It is not Gorenstein: no dilate of P has a lattice point at lattice distance 1 "
            "from every facet."
        )
    parts.append(lattice_normality(polytope, doc.cite, latex=False))
    return " ".join(parts)


def _text_scaleless(report: AnalysisReport, doc: TextDocument) -> str:
    """Whether the integral is scaleless by Lee's criterion, for a P that is not full-dimensional."""
    cite = doc.cite("lee2013")
    if report.polynomials.scaleless:
        return (
            f"The origin does not lie in the affine hull of P, so the integral is scaleless{cite}: "
            "for an equation h_0 + h . x = 0 of the affine hull with h_0 != 0, substituting "
            "lambda^(h_e) u_e for u_e multiplies the Lee-Pomeransky integral by "
            "lambda^(h_0 D/2 + sum_e h_e nu_e), and dimensional regularisation sets it to zero."
        )
    return (
        "The origin lies in the affine hull of P, so the integral is not scaleless by Lee's "
        f"criterion{cite}: for every equation h . x = 0 of the affine hull, substituting "
        "lambda^(h_e) u_e for u_e multiplies the Lee-Pomeransky integral by "
        "lambda^(sum_e h_e nu_e), which does not involve D, and dimensional regularisation "
        "does not regulate it."
    )


# How the Newton polytope section refers to the point-count section.
_LATEX_TORUS_REFERENCE = "Section~\\ref{sec:torus-counts}"


def latex(report: AnalysisReport, polytope: Polytope, doc: LatexDocument) -> str:
    data = polytope.data
    counts = [
        count_noun(n, singular, plural)
        for n, (singular, plural) in zip(
            data.f_vector[:-1], face_names(data.dimension), strict=True
        )
    ]
    shape = f", with {join_words(counts)}" if counts else ""
    vertices = join_words(
        [
            f"$v_{{{k}}} = ({', '.join(str(x) for x in vertex)})$"
            for k, vertex in enumerate(data.vertices, start=1)
        ]
    )
    volume = data.normalized_volume
    parts = [
        "\\label{sec:newton-polytope}",
        "The Newton polytope $P$ of $G$ is the convex hull of the exponent vectors of its "
        f"monomials. It has dimension {data.dimension} in "
        f"$\\mathbb{{R}}^{{{data.ambient_dimension}}}$ and normalised volume {volume}{shape}. "
        f"Its vertices are {vertices}.",
    ]
    if polytope.figure is not None:
        parts += [
            "Figure~\\ref{fig:newton-polytope} draws it.",
            "\\begin{figure}[htbp]",
            "\\centering",
            polytope.figure,
            "\\caption{The Newton polytope $P$ of $G$.}",
            "\\label{fig:newton-polytope}",
            "\\end{figure}",
        ]

    def lattice() -> list[str]:
        # Called where the paragraph goes, so that its works are cited in reading order.
        return [] if polytope.invariants is None else ["", _latex_lattice(polytope, doc)]

    if not data.is_full_dimensional:
        parts += [
            "",
            "Since $P$ is not full-dimensional, the rows of $A$ are linearly dependent. For "
            "generic $\\beta$ the Euler equations are then inconsistent, and the GKZ system has "
            "no non-zero solutions: its holonomic rank is 0, not the normalised volume.",
            *lattice(),
            "",
            _latex_scaleless(report, doc),
        ]
        return "\n".join(parts)
    parts += [
        "",
        "For generic coefficients and non-resonant $\\beta$ the holonomic rank of the GKZ "
        "system equals the normalised volume, here "
        f"{volume}{doc.cite('adolphson1994', 'chestnov2022')}. The rank equals the volume for "
        "every $\\beta$ exactly when the toric ring $\\mathbb{C}[\\mathbb{N}A]$ is "
        f"Cohen-Macaulay{doc.cite('mmw2005')}. This holds when the graph is one-particle "
        "irreducible and one-vertex irreducible, its external momenta are generic enough that "
        "no monomial of $G$ cancels, and its propagators are all massive, all massless, or "
        "such that every vertex reaches an external leg along massive propagators alone; the "
        "configuration is then normal and so Cohen-Macaulay"
        f"{doc.cite('klausen2023', 'tellander2023', 'walther2022')}. The number of master "
        "integrals is, up to sign, the Euler characteristic of the complement of "
        "$\\{G = 0\\}$ in the torus, at most $N!$ times the Euclidean volume of $P$, with "
        "equality for generic coefficients; graph-polynomial coefficients are rarely generic"
        f"{doc.cite('bbkp2017')}. The bound equals the normalised volume when the exponent "
        f"differences span $\\mathbb{{Z}}^N$. {not_computed(report, _LATEX_TORUS_REFERENCE)}",
        *lattice(),
    ]
    return "\n".join(parts)


def _latex_lattice(polytope: Polytope, doc: LatexDocument) -> str:
    """The lattice invariants of $P$ and what they certify about $\\mathbb{C}[\\mathbb{N}A]$."""
    found = polytope.invariants
    assert found is not None
    counts = f"{found.lattice_points} lattice points ({found.interior_points} interior)"
    if found.h_star is None:
        shape = f"{counts} and"
    else:
        # A long h* vector may break after any comma.
        h_star = ",\\allowbreak ".join(str(h) for h in found.h_star)
        shape = f"{counts}, $h^*$ vector $({h_star})$ and"
    parts = [
        f"In the lattice generated by its points, $P$ has {shape} lattice width "
        f"{found.lattice_width}{doc.cite('beckrobins2015')}."
    ]
    index = found.gorenstein_index
    if index == 1:
        parts.append(
            "It is reflexive: its one interior lattice point lies at lattice distance 1 from "
            f"every facet{doc.cite('batyrev1994')}."
        )
    elif index is not None:
        parts.append(
            f"It is Gorenstein of index {index}: ${index}P$ has a lattice point at lattice "
            "distance 1 from every facet."
        )
    else:
        parts.append(
            "It is not Gorenstein: no dilate of $P$ has a lattice point at lattice distance 1 "
            "from every facet."
        )
    parts.append(lattice_normality(polytope, doc.cite, latex=True))
    return " ".join(parts)


def _latex_scaleless(report: AnalysisReport, doc: LatexDocument) -> str:
    """Whether the integral is scaleless by Lee's criterion, for a $P$ that is not full-dimensional."""
    cite = doc.cite("lee2013")
    if report.polynomials.scaleless:
        return (
            "The origin does not lie in the affine hull of $P$, so the integral is scaleless"
            f"{cite}: for an equation $h_0 + h \\cdot x = 0$ of the affine hull with "
            "$h_0 \\neq 0$, substituting $\\lambda^{h_e} u_e$ for $u_e$ multiplies the "
            "Lee-Pomeransky integral by $\\lambda^{h_0 D/2 + \\sum_e h_e \\nu_e}$, and "
            "dimensional regularisation sets it to zero."
        )
    return (
        "The origin lies in the affine hull of $P$, so the integral is not scaleless by "
        f"Lee's criterion{cite}: for every equation $h \\cdot x = 0$ of the affine hull, "
        "substituting $\\lambda^{h_e} u_e$ for $u_e$ multiplies the Lee-Pomeransky integral by "
        "$\\lambda^{\\sum_e h_e \\nu_e}$, which does not involve $D$, and dimensional "
        "regularisation does not regulate it."
    )


def summary(polytope: Polytope) -> list[tuple[str, str]]:
    return [
        ("Polytope vertices", str(len(polytope.data.vertex_indices))),
        ("Normalised volume", str(polytope.data.normalized_volume)),
    ]


def summary_invariants(polytope: Polytope) -> list[tuple[str, str]]:
    """The lattice rows, which follow the point-count row in the summary."""
    found = polytope.invariants
    if found is None:
        return []
    index = found.gorenstein_index
    normal = "not computed" if found.normal is None else "yes" if found.normal else "no"
    return [
        ("Lattice points", str(found.lattice_points)),
        ("Interior lattice points", str(found.interior_points)),
        ("Gorenstein index", "none" if index is None else str(index)),
        ("Normal configuration", normal),
    ]


SECTION = Section(
    name="polytope",
    heading="Newton polytope",
    build=build,
    text=text,
    latex=latex,
    summary=(
        SummaryPart(40, summary),
        SummaryPart(60, summary_invariants),
    ),
)
