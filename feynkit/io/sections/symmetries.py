"""
The automorphisms of the Newton polytope and the symmetries of the integral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ...a_configuration import SymmetryPair
from ...normal_forms.polytope_automorphisms import coefficient_preserving_indices
from .._latex_kit import _LATEX_EULER_MELLIN, LatexDocument
from .._report_shared import count_noun, join_words
from .._text_kit import (
    _TEXT_EULER_MELLIN,
    _WIDTH,
    TextDocument,
    _blocks,
    _matrix,
    _paragraph,
    _text_vector,
)
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Symmetries:
    """Symmetries of the polytope, of the graph and of the A-configuration.

    Attributes
    ----------
    automorphism_order, vertex_orbits
        The order of Aut(P) and its orbits on the vertices.
    graph_automorphisms
        Vertex permutations of the graph preserving topology and masses.
    coefficient_preserving
        How many polytope automorphisms also preserve the coefficients of
        G, the subgroup relevant for functional equations.
    symmetry_pairs
        Every integer affine self-map of the A-configuration.
    full_dimensional
        Whether P is full-dimensional. Below full dimension Aut(P) is the
        group of P in its affine hull, and the identities the symmetry pairs
        give hold only trivially.

    ``vertex_orbits`` indexes ``PolytopeData.vertices``, the list the Newton
    polytope section numbers v_1, v_2, ..., not the z-index order of
    ``gkz.support``.
    """

    automorphism_order: int
    vertex_orbits: tuple[tuple[int, ...], ...]
    graph_automorphisms: tuple[tuple[int, ...], ...]
    coefficient_preserving: int
    symmetry_pairs: tuple[SymmetryPair, ...]
    full_dimensional: bool


def build(ctx: BuildContext) -> Symmetries:
    fi = ctx.integral
    data = ctx.polytope_data()
    automorphisms = fi.polytope_automorphisms
    return Symmetries(
        automorphism_order=automorphisms.order,
        vertex_orbits=tuple(tuple(orbit) for orbit in automorphisms.vertex_orbits),
        graph_automorphisms=tuple(tuple(p) for p in fi.graph_automorphisms),
        coefficient_preserving=len(coefficient_preserving_indices(fi, automorphisms)),
        symmetry_pairs=tuple(fi.symmetry_pairs),
        full_dimensional=data.is_full_dimensional,
    )


# The symmetry section writes out at most this many symmetry pairs.
MAX_PAIRS_SHOWN = 10


def text(report: AnalysisReport, symmetries: Symmetries, doc: TextDocument) -> str:
    orbits = symmetries.vertex_orbits
    if report.polytope is not None:
        listed = join_words(
            ["{" + ", ".join(f"v_{i + 1}" for i in orbit) + "}" for orbit in orbits]
        )
        action = (
            "acts on the vertices listed in the Newton polytope section with "
            f"{'the orbit' if len(orbits) == 1 else 'orbits'} {listed}"
        )
    else:
        sizes = join_words([str(len(orbit)) for orbit in orbits])
        action = (
            f"acts on the vertices of P with {count_noun(len(orbits), 'orbit')} of "
            f"{'size' if len(orbits) == 1 else 'sizes'} {sizes}"
        )
    if report.gkz is not None:
        definition = ""
        integral = " for the Euler-Mellin integral I_A of the GKZ system section"
    else:
        definition = (
            f"Write G = sum_j z_j u^alpha_j and let {_TEXT_EULER_MELLIN} be the Lee-Pomeransky "
            "integral without its prefactor, where beta = (-D/2, -nu_1, ..., -nu_N). "
        )
        integral = ""
    preserving = symmetries.coefficient_preserving
    pairs = symmetries.symmetry_pairs
    group = (
        "unimodular affine maps taking P to itself"
        if symmetries.full_dimensional
        else "affine maps of the affine hull of P that preserve its integer points and take P "
        "to itself"
    )
    blocks = [
        _paragraph(
            f"The group Aut(P) of {group} has order "
            f"{symmetries.automorphism_order} and {action}. The graph has "
            f"{count_noun(len(symmetries.graph_automorphisms), 'automorphism')}. Of the "
            f"polytope automorphisms, {preserving} "
            f"{'preserves' if preserving == 1 else 'preserve'} the coefficients of G."
        )
    ]
    if not pairs:
        blocks.append("The configuration has no symmetry pairs.")
        return _blocks(*blocks)
    if not symmetries.full_dimensional:
        blocks.append(
            _paragraph(
                f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
                "(T, sigma): integer matrices T and permutations sigma of the columns of A such "
                "that T takes column j of A to column sigma(j). Since P is not full-dimensional, "
                f"the identities I_A(beta, z_sigma) = I_A(T beta, z){integral} that they give "
                "hold only trivially: the integral converges absolutely for no D and nu_e, "
                "T beta depends on how T is extended off the affine hull of P, and for generic D "
                "and nu_e the GKZ system has no non-zero solutions."
            )
        )
        return _blocks(*blocks)
    shown = pairs[:MAX_PAIRS_SHOWN]
    n_columns = report.polynomials.monomials_g
    blocks.append(
        _paragraph(
            f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
            "(T, sigma): integer matrices T and permutations sigma of the columns of A such "
            "that T takes column j of A to column sigma(j). Each gives the identity "
            f"I_A(beta, z_sigma) = I_A(T beta, z){integral}, with "
            f"z_sigma = (z_{{sigma(1)}}, ..., z_{{sigma({n_columns})}}). Forsgaard, Matusevich "
            f"and Sobieska{doc.cite('fms2019')} and de la Cruz{doc.cite('delacruz2024')} print "
            "the permutation on the other side, but the substitution in the proof of "
            f"Corollary 4.1 of{doc.cite('fms2019')} gives the form stated here. For I itself "
            "the prefactors at beta and T beta enter as well. "
            + ("They are" if len(pairs) == len(shown) else f"The first {len(shown)} are")
            + ", with sigma listed as (sigma(1), sigma(2), ...):"
        )
    )
    for k, pair in enumerate(shown, start=1):
        sigma = ", ".join(str(j + 1) for j in pair.column_permutation)
        display = _matrix(f"T_{k}", pair.homogenized_map, f", sigma_{k} = ({sigma})")
        if max(len(line) for line in display.splitlines()) > _WIDTH:
            # sigma_k moves to a line of its own, broken after commas to fit.
            images = [sp.Integer(j + 1) for j in pair.column_permutation]
            display = "\n".join(
                [
                    _matrix(f"T_{k}", pair.homogenized_map, ","),
                    _text_vector(f"sigma_{k}", images, ""),
                ]
            )
        blocks.append(display)
    if len(pairs) > len(shown):
        rest = len(pairs) - len(shown)
        blocks.append(f"and {rest} further {'pair' if rest == 1 else 'pairs'}.")
    return _blocks(*blocks)


def latex(report: AnalysisReport, symmetries: Symmetries, doc: LatexDocument) -> str:
    orbits = symmetries.vertex_orbits
    if report.polytope is not None:
        # A long orbit may break after any of its commas.
        listed = join_words(
            [
                "$\\{" + ",\\allowbreak ".join(f"v_{{{i + 1}}}" for i in orbit) + "\\}$"
                for orbit in orbits
            ]
        )
        action = (
            f"acts on the vertices listed in Section~\\ref{{sec:newton-polytope}} with "
            f"{'the orbit' if len(orbits) == 1 else 'orbits'} {listed}"
        )
    else:
        sizes = join_words([str(len(orbit)) for orbit in orbits])
        action = (
            f"acts on the vertices of $P$ with {count_noun(len(orbits), 'orbit')} of "
            f"{'size' if len(orbits) == 1 else 'sizes'} {sizes}"
        )
    if report.gkz is not None:
        definition = ""
        integral = " for the Euler-Mellin integral $I_A$ of Section~\\ref{sec:gkz-system}"
    else:
        definition = (
            f"Write $G = \\sum_j z_j u^{{\\alpha_j}}$ and let ${_LATEX_EULER_MELLIN}$ be the "
            "Lee-Pomeransky integral without its prefactor, where "
            "$\\beta = (-D/2, -\\nu_1, \\ldots, -\\nu_N)$. "
        )
        integral = ""
    preserving = symmetries.coefficient_preserving
    pairs = symmetries.symmetry_pairs
    group = (
        "unimodular affine maps taking $P$ to itself"
        if symmetries.full_dimensional
        else "affine maps of the affine hull of $P$ that preserve its integer points and take "
        "$P$ to itself"
    )
    parts = [
        f"The group $\\mathrm{{Aut}}(P)$ of {group} has "
        f"order {symmetries.automorphism_order} and {action}. The graph has "
        f"{count_noun(len(symmetries.graph_automorphisms), 'automorphism')}. Of the polytope "
        f"automorphisms, {preserving} {'preserves' if preserving == 1 else 'preserve'} the "
        "coefficients of $G$.",
    ]
    if not pairs:
        parts.append("The configuration has no symmetry pairs.")
        return "\n".join(parts)
    if not symmetries.full_dimensional:
        parts.append(
            f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
            "$(T, \\sigma)$: integer matrices $T$ and permutations $\\sigma$ of the columns of "
            "$A$ such that $T$ takes column $j$ of $A$ to column $\\sigma(j)$. Since $P$ is not "
            "full-dimensional, the identities $I_A(\\beta, z_\\sigma) = I_A(T\\beta, z)$"
            f"{integral} that they give hold only trivially: the integral converges absolutely "
            "for no $D$ and $\\nu_e$, $T\\beta$ depends on how $T$ is extended off the affine "
            "hull of $P$, and for generic $D$ and $\\nu_e$ the GKZ system has no non-zero "
            "solutions."
        )
        return "\n".join(parts)
    shown = pairs[:MAX_PAIRS_SHOWN]
    n_columns = report.polynomials.monomials_g
    parts.append(
        f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
        "$(T, \\sigma)$: "
        "integer matrices $T$ and permutations $\\sigma$ of the columns of $A$ such that $T$ "
        "takes column $j$ of $A$ to column $\\sigma(j)$. Each gives the identity "
        f"$I_A(\\beta, z_\\sigma) = I_A(T\\beta, z)${integral}, with "
        f"$z_\\sigma = (z_{{\\sigma(1)}}, \\ldots, z_{{\\sigma({n_columns})}})$. "
        f"Forsg\\aa rd, Matusevich and Sobieska{doc.cite('fms2019')} and de la Cruz"
        f"{doc.cite('delacruz2024')} print the permutation on the other side, but the "
        f"substitution in the proof of Corollary~4.1 of{doc.cite('fms2019')} gives the form "
        "stated here. For $I$ itself the prefactors at $\\beta$ and $T\\beta$ enter as well. "
        + ("They are" if len(pairs) == len(shown) else f"The first {len(shown)} are")
        + ", with $\\sigma$ listed as $(\\sigma(1), \\sigma(2), \\ldots)$:"
    )
    for k, pair in enumerate(shown, start=1):
        sigma = ", ".join(str(j + 1) for j in pair.column_permutation)
        parts += [
            "\\begin{equation*}",
            f"T_{{{k}}} = {doc.matrix(pair.homogenized_map)}, \\qquad \\sigma_{{{k}}} = ({sigma})",
            "\\end{equation*}",
        ]
    if len(pairs) > len(shown):
        rest = len(pairs) - len(shown)
        parts.append(f"and {rest} further {'pair' if rest == 1 else 'pairs'}.")
    return "\n".join(parts)


def summary(symmetries: Symmetries) -> list[tuple[str, str]]:
    return [("Polytope automorphisms", str(symmetries.automorphism_order))]


SECTION = Section(
    name="symmetries",
    heading="Symmetries",
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(80, summary),),
)
