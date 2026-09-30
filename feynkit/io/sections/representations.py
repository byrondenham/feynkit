"""
The Schwinger, Feynman and Lee-Pomeransky representations and where the last converges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ...core.exceptions import ValidationError
from ...parametrisations.base import ParametrisationResult
from .._latex_kit import LatexDocument
from .._report_shared import integrand_templates
from .._text_kit import _INDENT, _WIDTH, TextDocument, _blocks, _heading, _lines, _paragraph, _str
from ..latex import to_latex
from ._base import Section

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Representations:
    """The parametric representations and where the last one converges.

    Attributes
    ----------
    schwinger, feynman, lee_pomeransky
        The three parametric representations.
    convergence
        Expressions that must all be positive for the Lee-Pomeransky
        integral to converge, one per facet of the Newton polytope; None
        when the polytope is not full-dimensional, in which case the
        integral converges nowhere.
    """

    schwinger: ParametrisationResult
    feynman: ParametrisationResult
    lee_pomeransky: ParametrisationResult
    convergence: tuple[sp.Expr, ...] | None


def build(ctx: BuildContext) -> Representations:
    fi = ctx.integral
    data = ctx.polytope_data()
    convergence: tuple[sp.Expr, ...] | None = None
    if data.is_full_dimensional:
        edges = fi.graph.get_internal_edges()
        exponents = [fi.propagator_exponents[e.idx] for e in edges]
        half_dimension = fi.dimension / 2
        inequalities = []
        for facet in data.facets:
            if len(facet.normal) != len(exponents):
                raise ValidationError(
                    f"Facet normal of length {len(facet.normal)} does not match "
                    f"{len(exponents)} propagator exponents"
                )
            inequalities.append(
                facet.offset * half_dimension
                - sum(m * nu for m, nu in zip(facet.normal, exponents, strict=True))
            )
        convergence = tuple(inequalities)
    return Representations(
        schwinger=fi.schwinger,
        feynman=fi.feynman,
        lee_pomeransky=fi.lee_pomeransky,
        convergence=convergence,
    )


def _quotient_lines(expr: sp.Expr, width: int) -> list[str]:
    """``expr`` over lines, a quotient too long for one line split at its bar."""
    text = _str(expr)
    if len(text) <= width:
        return [text]
    numerator, denominator = sp.fraction(expr)
    if denominator == 1:
        return _lines(expr, width)
    below = _lines(denominator, width - 4)
    below = [f"/ ({below[0]}", *(f"   {line}" for line in below[1:])]
    below[-1] += ")"
    return [*_lines(numerator, width), *below]


def _text_representation(
    prefactor: sp.Expr,
    parameters: Sequence[sp.Symbol],
    measure: sp.Expr,
    constraint: str,
    integrand: sp.Expr,
) -> str:
    """Display ``I`` as its prefactor times the integral, one factor per line."""
    width = _WIDTH - 2 * len(_INDENT)
    differentials = " ".join(f"d{_str(p)}" for p in parameters)
    head = _quotient_lines(prefactor, width)
    lines = [
        f"I = {head[0]}",
        *(f"    {line}" for line in head[1:]),
        f"    * int_{{R_+^{len(parameters)}}} {differentials}",
    ]
    if constraint:
        lines.append(f"    * {constraint}")
    for factor in ([] if measure == 1 else [measure]) + [integrand]:
        first, *rest = _lines(factor, width - 2)
        lines.append(f"    * {first}")
        lines.extend(f"      {line}" for line in rest)
    return "\n".join(_INDENT + line for line in lines)


def text(report: AnalysisReport, representations: Representations, doc: TextDocument) -> str:
    schwinger = representations.schwinger
    feynman = representations.feynman
    lee_pomeransky = representations.lee_pomeransky
    templates = integrand_templates(report.identity.loop_count, report.conventions.dimension)
    # The Schwinger result names its parameters alpha_e; the document uses a_e
    # throughout, the names of the Feynman result and of U and F.
    rename = dict(zip(schwinger.parameters, feynman.parameters, strict=True))

    blocks = [
        _heading("Schwinger representation", "~"),
        _paragraph(
            "With the Schwinger parameters a_e integrated over [0, infinity)"
            f"{doc.cite('weinzierl2022')},"
        ),
        _text_representation(
            schwinger.prefactor,
            feynman.parameters,
            schwinger.measure.subs(rename),
            "",
            templates.schwinger,
        ),
        _heading("Feynman representation", "~"),
        _paragraph(
            "With the same parameters restricted to the simplex sum_e a_e = 1"
            f"{doc.cite('weinzierl2022')},"
        ),
        _text_representation(
            feynman.prefactor,
            feynman.parameters,
            feynman.measure,
            "delta(1 - sum_e a_e)",
            templates.feynman,
        ),
        _heading("Lee-Pomeransky representation", "~"),
        _paragraph(
            "With the Lee-Pomeransky parameters u_e integrated over [0, infinity)"
            f"{doc.cite('leepomeransky2013')},"
        ),
        _text_representation(
            lee_pomeransky.prefactor,
            lee_pomeransky.parameters,
            lee_pomeransky.measure,
            "",
            templates.lee_pomeransky,
        ),
    ]
    euclidean = "For Euclidean kinematics, where every coefficient of G has positive real part"
    if representations.convergence is None:
        blocks.append(
            _paragraph(
                f"Convergence. The Newton polytope P of G is not full-dimensional. {euclidean}, "
                "the Lee-Pomeransky integral therefore converges absolutely for no D and nu_e"
                f"{doc.cite('klausen2023')}."
            )
        )
        return _blocks(*blocks)
    blocks += [
        _paragraph(
            f"Convergence. {euclidean}, and for Re D > 0, the Lee-Pomeransky integral "
            "converges absolutely when the real parts of (nu_1, ..., nu_N), divided by "
            "Re(D/2), lie in the interior of the Newton polytope P of G, that is when for "
            "every facet m_j . x <= b_j of P the real part of b_j D/2 - sum_e m_{je} nu_e is "
            f"positive{doc.cite('klausen2023')}:"
        ),
        "\n".join(f"{_INDENT}Re({_str(c)}) > 0" for c in representations.convergence),
        _paragraph(
            "For such kinematics it converges absolutely for no D and nu_e when P is not "
            "full-dimensional."
        ),
    ]
    return _blocks(*blocks)


def _latex_representation(
    prefactor: sp.Expr,
    parameters: Sequence[sp.Symbol],
    measure: sp.Expr,
    constraint: str,
    integrand: sp.Expr,
) -> str:
    differentials = "\\, ".join(f"\\mathrm{{d}}{to_latex(p)}" for p in parameters)
    factors = [f for f in (constraint, "" if measure == 1 else to_latex(measure)) if f]
    factors.append(to_latex(integrand, fold_short_frac=True))
    return "\n".join(
        [
            "\\begin{equation*}",
            "\\begin{split}",
            f"I &= {to_latex(prefactor)} \\\\",
            f"&\\quad \\times \\int_{{\\mathbb{{R}}_{{+}}^{{{len(parameters)}}}}} "
            f"{differentials}\\; " + "\\, ".join(factors),
            "\\end{split}",
            "\\end{equation*}",
        ]
    )


def latex(report: AnalysisReport, representations: Representations, doc: LatexDocument) -> str:
    schwinger = representations.schwinger
    feynman = representations.feynman
    lee_pomeransky = representations.lee_pomeransky
    templates = integrand_templates(report.identity.loop_count, report.conventions.dimension)
    # The Schwinger result names its parameters alpha_e; the document uses a_e
    # throughout, the names of the Feynman result and of U and F.
    rename = dict(zip(schwinger.parameters, feynman.parameters, strict=True))

    parts = [
        "\\subsection{Schwinger representation}",
        "With the Schwinger parameters $a_e$ integrated over $[0, \\infty)$"
        f"{doc.cite('weinzierl2022')},",
        _latex_representation(
            schwinger.prefactor,
            feynman.parameters,
            schwinger.measure.subs(rename),
            "",
            templates.schwinger,
        ),
        "\\subsection{Feynman representation}",
        "With the same parameters restricted to the simplex $\\sum_e a_e = 1$"
        f"{doc.cite('weinzierl2022')},",
        _latex_representation(
            feynman.prefactor,
            feynman.parameters,
            feynman.measure,
            "\\delta\\Bigl(1 - \\sum_{e} a_{e}\\Bigr)",
            templates.feynman,
        ),
        "\\subsection{Lee-Pomeransky representation}",
        "With the Lee-Pomeransky parameters $u_e$ integrated over $[0, \\infty)$"
        f"{doc.cite('leepomeransky2013')},",
        _latex_representation(
            lee_pomeransky.prefactor,
            lee_pomeransky.parameters,
            lee_pomeransky.measure,
            "",
            templates.lee_pomeransky,
        ),
        "\\paragraph{Convergence.}",
    ]
    euclidean = "For Euclidean kinematics, where every coefficient of $G$ has positive real part"
    if representations.convergence is None:
        parts.append(
            f"The Newton polytope $P$ of $G$ is not full-dimensional. {euclidean}, the "
            "Lee-Pomeransky integral therefore converges absolutely for no $D$ and $\\nu_e$"
            f"{doc.cite('klausen2023')}."
        )
        return "\n".join(parts)
    lines = " \\\\\n".join(
        f"\\mathrm{{Re}}\\left({to_latex(c)}\\right) &> 0" for c in representations.convergence
    )
    parts += [
        f"{euclidean}, and for $\\mathrm{{Re}}\\,D > 0$, the Lee-Pomeransky integral converges "
        "absolutely when the real parts of $(\\nu_1, \\ldots, \\nu_N)$, divided by "
        "$\\mathrm{Re}(D/2)$, lie in the interior of the Newton polytope $P$ of $G$, that is "
        "when for every facet $m_j \\cdot x \\le b_j$ of $P$ the real part of "
        f"$b_j D/2 - \\sum_e m_{{je}} \\nu_e$ is positive{doc.cite('klausen2023')}:",
        "\\begin{align*}",
        lines,
        "\\end{align*}",
        "For such kinematics it converges absolutely for no $D$ and $\\nu_e$ when $P$ is not "
        "full-dimensional.",
    ]
    return "\n".join(parts)


SECTION = Section(
    name="representations",
    heading="Parametric representations",
    build=build,
    text=text,
    latex=latex,
)
