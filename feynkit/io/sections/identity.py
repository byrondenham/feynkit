"""
The graph: its Nickel indices, edges, masses and exponents, and the figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from .._latex_kit import LatexDocument, _math
from .._report_shared import count_noun
from .._text_kit import TextDocument, _blocks, _paragraph, _str, _table
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Identity:
    """What the graph is.

    Attributes
    ----------
    cnickel, nickel
        Coloured and bare Nickel indices of the topology.
    loop_count, propagators, external_legs
        L, the number of internal edges and the number of external legs.
    edge_indices, edge_endpoints
        The index e of each internal edge, which names its parameters a_e and
        u_e and its exponent nu_e, and the two vertices it joins, in
        internal-edge order. The indices need not run from 1 to N.
    edge_masses, edge_exponents
        The mass m_e and the exponent nu_e of each internal edge, in
        internal-edge order; a massless edge has mass 0.
    graph_tikz
        TikZ source for the graph figure.
    """

    cnickel: str
    nickel: str
    loop_count: int
    propagators: int
    external_legs: int
    edge_indices: tuple[int, ...]
    edge_endpoints: tuple[tuple[int, int], ...]
    edge_masses: tuple[sp.Expr, ...]
    edge_exponents: tuple[sp.Expr, ...]
    graph_tikz: str


def build(ctx: BuildContext) -> Identity:
    fi = ctx.integral
    edges = fi.graph.get_internal_edges()
    exponents = fi.propagator_exponents
    return Identity(
        cnickel=fi.cnickel,
        nickel=fi.nickel_index,
        loop_count=fi.loop_count,
        propagators=len(edges),
        external_legs=fi.graph.external_legs,
        edge_indices=tuple(e.idx for e in edges),
        edge_endpoints=tuple((e.v1, e.v2) for e in edges),
        edge_masses=tuple(sp.sympify(e.get_mass()) for e in edges),
        edge_exponents=tuple(exponents[e.idx] for e in edges),
        graph_tikz=fi.tikz(),
    )


def text(_report: AnalysisReport, identity: Identity, _doc: TextDocument) -> str:
    rows = [
        (str(e), f"{v1}-{v2}", _str(nu), _str(mass))
        for e, (v1, v2), nu, mass in zip(
            identity.edge_indices,
            identity.edge_endpoints,
            identity.edge_exponents,
            identity.edge_masses,
            strict=True,
        )
    ]
    return _blocks(
        _paragraph(
            f"It has L = {identity.loop_count} "
            f"{'loop' if identity.loop_count == 1 else 'loops'}, "
            f"N = {identity.propagators} "
            f"{'propagator' if identity.propagators == 1 else 'propagators'} and "
            f"{count_noun(identity.external_legs, 'external leg')}."
        ),
        _table(("e", "Vertices", "nu_e", "m_e"), rows),
        _paragraph(
            "Propagator e joins the vertices listed and carries the exponent nu_e and the "
            "mass m_e shown in the table; massless propagators have m_e = 0. Its index e also "
            "names its parameters a_e and u_e."
        ),
    )


def latex(_report: AnalysisReport, identity: Identity, _doc: LatexDocument) -> str:
    label = re.search(r"\\label\{([^}]*)\}", identity.graph_tikz)
    shown = f"Figure~\\ref{{{label.group(1)}}} shows the graph. " if label else ""
    rows = [
        f"{e} & {v1}--{v2} & {_math(nu)} & {_math(mass)} \\\\"
        for e, (v1, v2), nu, mass in zip(
            identity.edge_indices,
            identity.edge_endpoints,
            identity.edge_exponents,
            identity.edge_masses,
            strict=True,
        )
    ]
    return "\n".join(
        [
            identity.graph_tikz,
            "",
            f"{shown}It has $L = {identity.loop_count}$ "
            f"{'loop' if identity.loop_count == 1 else 'loops'}, "
            f"$N = {identity.propagators}$ "
            f"{'propagator' if identity.propagators == 1 else 'propagators'} and "
            f"{count_noun(identity.external_legs, 'external leg')}.",
            "\\begin{center}",
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "$e$ & Vertices & $\\nu_e$ & $m_e$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
            "Propagator $e$ joins the vertices listed and carries the exponent $\\nu_e$ and the "
            "mass $m_e$ shown in the table; massless propagators have $m_e = 0$. The figure "
            "labels each propagator with its index $e$, which also names its parameters $a_e$ "
            "and $u_e$.",
        ]
    )


def summary(identity: Identity) -> list[tuple[str, str]]:
    return [
        ("Loops", str(identity.loop_count)),
        ("Propagators", str(identity.propagators)),
        ("External legs", str(identity.external_legs)),
    ]


SECTION = Section(
    name="identity",
    heading="Graph",
    always=True,
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(10, summary),),
)
