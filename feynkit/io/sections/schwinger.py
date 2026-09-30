"""
The Schwinger-representation GKZ system and its restriction to the F block.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ...resonance import admissible
from ...systems.cayley import CayleyGKZSystem, lp_to_cayley
from ...systems.complete import GKZSystem
from .._latex_kit import LatexDocument, _latex_vector
from .._text_kit import TextDocument, _blocks, _matrix, _paragraph, _text_vector
from ._base import Section

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Schwinger:
    """The GKZ system of the Schwinger representation.

    Attributes
    ----------
    system
        The two-block Cayley system of U~ and F~.
    f_block
        Its face subsystem on the F~ block.
    lp_to_cayley
        The unimodular T taking the Lee-Pomeransky A-matrix to the Cayley
        one up to column order, and beta_LP to beta_Cayley.
    columns_match
        Whether the columns of ``T A_LP`` are a permutation of those of the
        Cayley matrix.
    f_block_admissible
        Whether beta_Cayley lies in the span of the columns of the F~ block,
        for all values of any symbols in it, as
        :func:`feynkit.resonance.admissible` decides: the condition under
        which the restriction is a true subsystem.
    """

    system: CayleyGKZSystem
    f_block: GKZSystem
    lp_to_cayley: sp.Matrix
    columns_match: bool
    f_block_admissible: bool


def build(ctx: BuildContext) -> Schwinger:
    fi = ctx.integral
    system = fi.schwinger_gkz
    t = lp_to_cayley(len(fi.symanzik.lp_parameters), fi.loop_count)
    block = range(len(system.u_support), system.a_matrix.cols)
    # The section says whether beta_Cayley lies in the span of the block, which is what the
    # warning of restrict_to_f_block would say.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        f_block = system.restrict_to_f_block()
    return Schwinger(
        system=system,
        f_block=f_block,
        lp_to_cayley=t,
        columns_match=_columns(t * fi.gkz.a_matrix) == _columns(system.a_matrix),
        f_block_admissible=admissible(system.a_matrix, block, system.beta_parameters),
    )


def _columns(matrix: sp.Matrix) -> list[tuple[int, ...]]:
    return sorted(tuple(int(x) for x in matrix.col(j)) for j in range(matrix.cols))


def f_block_sentence(schwinger: Schwinger, cite: Callable[..., str], *, latex: bool) -> str:
    """When the F~-block restriction is a true subsystem, and whether it is one here.

    The condition is that beta_Cayley lies in the span of the block's columns;
    it reads nu = (L+1)D/2 only when those columns span the hyperplane y_0 = 0.
    """

    def m(text: str, tex: str) -> str:
        return f"${tex}$" if latex else text

    beta = m("beta_Cayley", r"\beta_{\text{Cayley}}")
    hyperplane = m("y_0 = 0", "y_0 = 0")
    condition = m("nu = (L+1)D/2", r"\nu = (L+1)D/2")
    head = (
        f"its solutions solve the full system when {beta} lies in the span of the face's "
        f"columns{cite('britto2026')}; if those columns span the hyperplane {hyperplane}, as "
        f"they do when the block is a facet, the condition reads {condition}."
    )
    if schwinger.f_block_admissible:
        return f"{head} Here {beta} lies in that span, so the restriction is a true subsystem."
    symbolic = any(sp.sympify(b).free_symbols for b in schwinger.system.beta_parameters)
    generic = " for generic values of its symbols" if symbolic else ""
    return (
        f"{head} Here {beta} does not lie in that span{generic}, and the relation between the "
        "two systems is not established."
    )


def text(_report: AnalysisReport, schwinger: Schwinger, doc: TextDocument) -> str:
    system = schwinger.system
    f_block = schwinger.f_block
    check = "verified" if schwinger.columns_match else "not verified"
    return _blocks(
        _paragraph(
            "The Schwinger representation gives a second GKZ system on the Cayley "
            "configuration of (U~, F~), the Symanzik polynomials with a_N = 1 and a_e = u_e "
            f"otherwise{doc.cite('jimenez2026', 'klausen2023')}:"
        ),
        _matrix("A_Cayley", system.a_matrix),
        _text_vector("beta_Cayley", system.beta_parameters, "."),
        _paragraph(
            "It is unimodularly equivalent to the Lee-Pomeransky configuration, the matrix "
            "A_LP and parameter vector beta_LP = (-D/2, -nu_1, ..., -nu_N) of the GKZ system "
            "of G, through"
        ),
        _matrix("T", schwinger.lp_to_cayley),
        _paragraph(
            "which maps the parameter vectors as beta_Cayley = T beta_LP and the columns of "
            "A_LP to those of A_Cayley up to order; the column correspondence is "
            f"{check} for this integral. Restricting to the F~ block gives the system"
        ),
        _matrix("A_F", f_block.a_matrix),
        _text_vector("beta_F", f_block.beta_parameters, ";"),
        _paragraph(f_block_sentence(schwinger, doc.cite, latex=False)),
    )


def latex(_report: AnalysisReport, schwinger: Schwinger, doc: LatexDocument) -> str:
    system = schwinger.system
    f_block = schwinger.f_block
    check = "verified" if schwinger.columns_match else "not verified"
    return "\n".join(
        [
            "The Schwinger representation gives a second GKZ system on the Cayley "
            "configuration of $(\\tilde U, \\tilde F)$, the Symanzik polynomials with "
            "$a_N = 1$ and $a_e = u_e$ otherwise"
            f"{doc.cite('jimenez2026', 'klausen2023')}:",
            "\\begin{gather*}",
            "A_{\\text{Cayley}} = " + doc.matrix(system.a_matrix) + ", \\\\",
            "\\beta_{\\text{Cayley}} = " + _latex_vector(system.beta_parameters) + ".",
            "\\end{gather*}",
            "It is unimodularly equivalent to the Lee-Pomeransky configuration, the matrix "
            "$A_{\\text{LP}}$ and parameter vector "
            "$\\beta_{\\text{LP}} = (-D/2, -\\nu_1, \\ldots, -\\nu_N)$ of the GKZ system of $G$, "
            "through",
            "\\begin{equation*}",
            "T = " + doc.matrix(schwinger.lp_to_cayley) + ",",
            "\\end{equation*}",
            "which maps the parameter vectors as $\\beta_{\\text{Cayley}} = T "
            "\\beta_{\\text{LP}}$ and the columns of $A_{\\text{LP}}$ to those of "
            "$A_{\\text{Cayley}}$ up to order; the column correspondence is "
            f"{check} for this integral. Restricting to the $\\tilde F$ block gives the system",
            "\\begin{gather*}",
            "A_F = " + doc.matrix(f_block.a_matrix) + ", \\\\",
            "\\beta_F = " + _latex_vector(f_block.beta_parameters) + ";",
            "\\end{gather*}",
            f_block_sentence(schwinger, doc.cite, latex=True),
        ]
    )


SECTION = Section(
    name="schwinger",
    heading="Schwinger-representation system",
    build=build,
    text=text,
    latex=latex,
)
