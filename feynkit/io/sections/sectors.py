"""
The sector hierarchy: which sectors vanish, and how many master integrals each one has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.exceptions import ValidationError
from ...sectors import Sector, SectorHierarchy, sector_hierarchy
from .._latex_kit import LatexDocument, _longtable
from .._report_shared import count_noun, join_words
from .._text_kit import TextDocument, _blocks, _paragraph, _table
from ._base import Section, SummaryPart
from .resonance import integer_powers

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext

# The most propagators for which the hierarchy is built: it has 2^N sectors, each with its
# face, its volume and its place in the symmetry search.
MAX_PROPAGATORS = 10

# The limit of Singular for each count at a point, in seconds.
_TIMEOUT = 120

_RESONANCE = {
    "all": "every epsilon",
    "progression": "a progression",
    "point": "one value",
    "never": "never",
}


@dataclass(frozen=True)
class Sectors:
    """The sectors of the integral, and where the face of each is resonant.

    Attributes
    ----------
    hierarchy
        The hierarchy of :func:`feynkit.sectors.sector_hierarchy`, with generic counts and
        symmetries, and the counts at a point when they were asked for; None when the graph has
        more than ``limit`` propagators.
    propagators
        The number of internal edges.
    limit
        The most propagators for which the section is built.
    resonance
        For each sector with a face F_S, the kind of the set of epsilon at which that face is
        resonant ("all", "progression", "point" or "never") in the face lattice of the face-lattice
        section; None when the hierarchy was left out or the lattice could not be built.
    """

    hierarchy: SectorHierarchy | None
    propagators: int
    limit: int
    resonance: dict[int, str] | None


def build(ctx: BuildContext) -> Sectors:
    fi = ctx.integral
    n = len(fi.graph.get_internal_edges())
    if n > MAX_PROPAGATORS:
        return Sectors(None, n, MAX_PROPAGATORS, None)
    hierarchy = sector_hierarchy(
        fi, counts=ctx.sectors_counts, seed=ctx.torus_seed, timeout=_TIMEOUT
    )
    powers, _ = integer_powers(fi)
    edges = [e.idx for e in fi.graph.get_internal_edges()]
    resonance: dict[int, str] | None
    try:
        lattice = fi.face_lattice(ctx.d0, nu=dict(zip(edges, powers, strict=True)))
        resonance = {i: face.resonant.kind for i, face in hierarchy.faces(lattice).items()}
    except ValidationError:
        resonance = None
    return Sectors(hierarchy, n, MAX_PROPAGATORS, resonance)


def _ordered(h: SectorHierarchy) -> list[Sector]:
    """The non-zero sectors, the largest first, and by identity within a size."""
    return sorted(h.non_zero(), key=lambda s: (-len(s.propagators), s.id))


def _number(value: object) -> str:
    return "-" if value is None else str(value)


def _forms(sector: Sector) -> str:
    """The count with symmetries, in the absolute and the signed form."""
    if sector.symmetric_count is None:
        return "-"
    if sector.orbit != sector.id:
        return "-"
    if sector.signs_consistent is None:
        return str(sector.symmetric_count)
    absolute, signed = sector.symmetric_count, sector.signed_symmetric_count
    return str(absolute) if sector.signs_consistent else f"{absolute} / {signed}"


def table_data(section: Sectors, *, latex: bool) -> tuple[list[str], list[list[str]]]:
    """The header and the rows of the table of non-zero sectors."""
    assert section.hierarchy is not None
    h = section.hierarchy
    counted = h.counts_source in ("torus", "critical")
    header = [
        "$N_{\\mathrm{id}}$" if latex else "N_id",
        "Propagators",
        "$\\dim F_S$" if latex else "dim F_S",
        "Resonance",
        "$t_{\\mathrm{gen}}$" if latex else "t_gen",
    ]
    if counted:
        header += ["$t(T)$" if latex else "t(T)", "$m(T)$" if latex else "m(T)"]
    if h.symmetric:
        header += ["Orbit", "$|G|$" if latex else "|G|"]
        if counted:
            header += ["$N_T$" if latex else "N_T"]
    rows = []
    for s in _ordered(h):
        kind = None if section.resonance is None else section.resonance.get(s.id)
        row = [
            str(s.id),
            " ".join(str(e) for e in s.propagators),
            str(s.dimension),
            "-" if kind is None else _RESONANCE[kind],
            _number(s.generic_count),
        ]
        if counted:
            row += [_number(s.count), _number(s.sector_count)]
        if h.symmetric:
            row += [_number(s.orbit), _number(s.stabiliser_order)]
            if counted:
                row += [_forms(s)]
        rows.append(row)
    return header, rows


def opening(section: Sectors) -> str:
    h = section.hierarchy
    assert h is not None
    totals = h.totals()
    return (
        f"{count_noun(totals.non_zero, 'sector')} of {2 ** section.propagators} "
        f"{'are' if totals.non_zero != 1 else 'is'} non-zero, {totals.scaleless} "
        f"{'are' if totals.scaleless != 1 else 'is'} scaleless and {totals.cycle} "
        f"{'are' if totals.cycle != 1 else 'is'} cut by a cycle."
    )


def notes(section: Sectors, *, latex: bool) -> list[str]:
    h = section.hierarchy
    assert h is not None
    totals = h.totals()
    notes = []
    if totals.unique is not None:
        notes.append(f"The non-zero sectors fall into {count_noun(totals.unique, 'orbit')}.")
    if h.counts_source in ("torus", "critical"):
        point = join_words([f"{key} = {value}" for key, value in h.point or ()])
        notes.append(
            f"The counts are taken at the kinematic point {point or 'of a graph without symbols'}."
            if not latex
            else "The counts are taken at one rational kinematic point."
        )
        if totals.negative:
            names = join_words(["{" + ",".join(map(str, t)) + "}" for t in totals.negative])
            notes.append(
                f"The count m(T) of the sectors {names} is negative; it is not a number of "
                "master integrals."
                if not latex
                else f"The count $m(T)$ of the sectors {names} is negative; it is not a number "
                "of master integrals."
            )
        if totals.signs_consistent is False:
            notes.append(
                "For some sectors the absolute and the signed form of N_T differ, and the table "
                "gives both."
                if not latex
                else "For some sectors the absolute and the signed form of $N_T$ differ, and "
                "the table gives both."
            )
    return notes


def text(_report: AnalysisReport, section: Sectors, doc: TextDocument) -> str:
    h = section.hierarchy
    if h is None:
        return _paragraph(
            f"The graph has {section.propagators} propagators, more than the {section.limit} "
            "for which the hierarchy is built, so the sectors are left out."
        )
    counted = h.counts_source in ("torus", "critical")
    intro = (
        "A sector is the set T of propagators with a positive index; the others, S, are "
        "contracted. N_id is the sum of 2^(j-1) over the propagators j of T, the first "
        f"propagator being the least significant bit{doc.cite('weinzierl2022')}. A sector is "
        "cut by a cycle when S contains one, and scaleless when S is a forest but the origin "
        f"lies outside the affine hull of the support of G_T{doc.cite('lee2013')}. Both are "
        "zero. G_T is G with the terms in the contracted parameters removed, and F_S is the "
        "face of the Newton polytope that carries them; the table gives its dimension and the "
        "epsilon at which it is resonant, each beside its sector. The count t_gen is the "
        "count for generic coefficients on the support of G_T."
    )
    blocks = [_paragraph(intro), _paragraph(" ".join([opening(section)]))]
    if counted:
        blocks.append(
            _paragraph(
                "The count t(T) is the number of master integrals of T with its subsectors, and "
                "m(T) that of T alone, by inclusion and exclusion over its subsectors"
                f"{doc.cite('bbkp2017')}."
            )
        )
    if h.symmetric:
        blocks.append(
            _paragraph(
                "Sectors are in one orbit when a bijection of their propagators maps the terms "
                "of one G_T onto those of the other with equal coefficients; the orbit is "
                "named by its least N_id and |G| is the order of the group of a sector"
                + (
                    f", with N_T the count with symmetries from the Euler characteristics of "
                    f"the fixed sets{doc.cite('duhr2026')}."
                    if counted
                    else f"{doc.cite('duhr2026')}."
                )
            )
        )
    header, rows = table_data(section, latex=False)
    blocks.append(_table(header, rows))
    remarks = notes(section, latex=False)
    if remarks:
        blocks.append(_paragraph(" ".join(remarks)))
    return _blocks(*blocks)


def latex(_report: AnalysisReport, section: Sectors, doc: LatexDocument) -> str:
    h = section.hierarchy
    if h is None:
        return (
            f"The graph has {section.propagators} propagators, more than the {section.limit} "
            "for which the hierarchy is built, so the sectors are left out."
        )
    counted = h.counts_source in ("torus", "critical")
    intro = (
        "A sector is the set $T$ of propagators with a positive index; the others, $S$, are "
        "contracted. $N_{\\mathrm{id}}$ is the sum of $2^{j-1}$ over the propagators $j$ of $T$, "
        f"the first propagator being the least significant bit{doc.cite('weinzierl2022')}. A "
        "sector is cut by a cycle when $S$ contains one, and scaleless when $S$ is a forest "
        "but the origin lies outside the affine hull of the support of $G_T$"
        f"{doc.cite('lee2013')}. Both are zero. $G_T$ is $G$ with the terms in the contracted "
        "parameters removed, and $F_S$ is the face of the Newton polytope that carries them; the "
        "table gives its dimension and the $\\varepsilon$ at which it is resonant, each beside "
        "its sector. The count $t_{\\mathrm{gen}}$ is the count for generic coefficients on "
        "the support of $G_T$."
    )
    blocks = [intro, opening(section)]
    if counted:
        blocks.append(
            "The count $t(T)$ is the number of master integrals of $T$ with its subsectors, and "
            "$m(T)$ that of $T$ alone, by inclusion and exclusion over its subsectors"
            f"{doc.cite('bbkp2017')}."
        )
    if h.symmetric:
        tail = (
            ", with $N_T$ the count with symmetries from the Euler characteristics of the "
            f"fixed sets{doc.cite('duhr2026')}."
            if counted
            else f"{doc.cite('duhr2026')}."
        )
        blocks.append(
            "Sectors are in one orbit when a bijection of their propagators maps the terms of "
            "one $G_T$ onto those of the other with equal coefficients; the orbit is named by "
            "its least $N_{\\mathrm{id}}$ and $|G|$ is the order of the group of a sector" + tail
        )
    header, rows = table_data(section, latex=True)
    blocks.append(_longtable("rlrl" + "r" * (len(header) - 4), header, rows))
    remarks = notes(section, latex=True)
    if remarks:
        blocks.append(" ".join(remarks))
    return "\n\n".join(blocks)


def summary(section: Sectors) -> list[tuple[str, str]]:
    h = section.hierarchy
    if h is None:
        return [("Sectors", f"left out: {section.propagators} propagators")]
    totals = h.totals()
    rows = [
        ("Nonzero sectors", str(totals.non_zero)),
        ("Scaleless sectors", str(totals.scaleless)),
        ("Sectors cut by a cycle", str(totals.cycle)),
    ]
    if totals.unique is not None:
        rows.append(("Unique sectors", str(totals.unique)))
    if totals.generic is not None:
        rows.append(("Generic masters of the top sector", str(totals.generic)))
    if totals.count is not None:
        rows.append(("Masters of the top sector", str(totals.count)))
    if totals.symmetric is not None:
        rows.append(("Masters with symmetries", str(totals.symmetric)))
    return rows


SECTION = Section(
    name="sectors",
    heading="Sector hierarchy",
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(77, summary),),
)
