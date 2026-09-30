"""
The candidate Euler characteristic from finite-field point counts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

import sympy as sp

from ...point_count import TorusCount
from .._latex_kit import LatexDocument, _escape, _latex_equation
from .._report_shared import TORUS_HEADING, count_noun, join_words
from .._text_kit import (
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


def build(ctx: BuildContext) -> TorusCount:
    return ctx.integral.torus_count(
        seed=ctx.torus_seed, max_evaluations=ctx.torus_budget, landau=ctx.analysis()
    )


def count_polynomial(coefficients: Sequence[int]) -> sp.Expr:
    """P(q) from its coefficients, constant term first."""
    q = sp.Symbol("q")
    return sp.Add(*(c * q**i for i, c in enumerate(coefficients)))


def primes_left_out(torus: TorusCount) -> str:
    """The sentence listing the primes the counts leave out, up to the largest they could use."""
    primes = torus.excluded_primes
    if len(primes) == 1:
        return f"Up to {torus.max_prime} only {primes[0]} is left out."
    listed = join_words([str(p) for p in primes])
    return f"Up to {torus.max_prime} the primes left out are {listed}."


def in_squared_masses(torus: TorusCount) -> str:
    """The clause ", written in the squared masses," if the point has a squared mass, else "".

    The counts write the face discriminants in the squares of the symbols that
    occur in G only to even powers, the masses of a Feynman graph, which the
    point lists as m_e**2. A factor odd in a mass is covered through its norm
    f(m) f(-m), so the check's sentence says so, but only where a mass occurs.
    """
    squared = any(isinstance(key, sp.Pow) for key, _ in torus.point)
    return ", written in the squared masses," if squared else ""


def torus_skipped_faces(torus: TorusCount) -> str | None:
    """The sentence naming the faces the Landau analysis skipped and the steps of the count
    that miss their discriminants, or None if none was skipped.

    Only the steps that ran are named: the draw for a drawn point, the test for a
    vanishing discriminant for a given one, and the check at further primes when
    the fit reached it.
    """
    if not torus.skipped_faces:
        return None
    drawn = torus.seed is not None
    steps = [
        *(["the draw" if drawn else "the test of the given point"] if torus.point else []),
        "the choice of excluded primes",
        *(["the check"] if torus.verification_primes else []),
    ]
    whose = "whose discriminant is" if torus.skipped_faces == 1 else "whose discriminants are"
    return (
        f"The Landau analysis skipped {count_noun(torus.skipped_faces, 'face')}, {whose} "
        f"left out of {join_words(steps)}."
    )


def _text_rational(value: Fraction) -> str:
    return str(_str(sp.Rational(value.numerator, value.denominator)))


def text(_report: AnalysisReport, torus: TorusCount, doc: TextDocument) -> str:
    intro = _paragraph(
        "Let X be the complement of V = {G = 0} in the torus (C^*)^N, with mu = 1. The number "
        "of master integrals, with subsectors included, symmetries unused and D symbolic, is "
        f"C = (-1)^N chi(X) by Corollary 37 of{doc.cite('bbkp2017')}. By Theorem 44 of the same "
        "paper, C is at most N! Vol(Newt G), where Newt G is the Newton polytope of G and Vol "
        "its Euclidean volume. If the number of points "
        "of V in (F_q^*)^N is a polynomial P(q) for every finite field F_q whose characteristic "
        "avoids a finite set, then chi(V) = P(1) by Theorem 6.1.2(3) of Katz's appendix to the "
        f"paper of Hausel and Rodriguez-Villegas{doc.cite('katz2008')}, and chi(X) = -P(1), "
        "since the Euler characteristic is additive and vanishes on the torus."
    )
    origin = []
    if torus.on_shell:
        settings = join_words([f"{_str(key)} = {_str(v)}" for key, v in torus.on_shell])
        origin.append(f"The counts set {settings} in the momentum products first.")
    if not torus.point:
        then = " then" if torus.on_shell else ""
        origin.append(f"G{then} has no kinematic symbols, so the counts need no drawn point.")
    else:
        where = join_words([f"{_str(key)} = {_text_rational(v)}" for key, v in torus.point])
        if torus.seed is None:
            origin.append(
                f"The counts are taken at the kinematic point {where}, given by the caller."
            )
        else:
            origin.append(
                f"The counts are taken at the kinematic point {where}, drawn with seed "
                f"{torus.seed} so that every coefficient of G and every face discriminant is "
                "non-zero. The value of |chi(X)| is the same on an open dense set of kinematics"
                f"{doc.cite('fmt2024')}, and the draw avoids the Landau surfaces found, on which "
                "it can be smaller."
            )
    if torus.on_landau_surface:
        origin.append(
            "A coefficient of G or a face discriminant vanishes at the point, where C can be "
            "smaller than for generic kinematics."
        )
    skipped = torus_skipped_faces(torus)
    if skipped is not None:
        origin.append(skipped)
    at = " at the point" if torus.point else ""
    n = len(torus.variables)
    if torus.verification_primes:
        # A fit that fails no test before the check; the check stops at a disagreement.
        stops = "" if torus.candidate_polynomial is not None else ", unless a count disagrees first"
        fit = (
            "The check covers the quadratic characters (d/p) for d in the group generated by -1 "
            f"and the non-zero values{at} of the vertex coefficients of G, the factors of the "
            f"face discriminants{in_squared_masses(torus)} and the discriminants of G on its "
            "edges. Characters outside that group are not covered; they can come from the "
            "constant factors of the discriminants of faces of dimension 2 or more, or from the "
            f"discriminants of skipped faces. The polynomial of degree at most {n} through the "
            f"counts at the first {n + 1} primes not left out is checked at further primes, at "
            "least until every non-trivial one of these characters has taken both signs"
            f"{stops}:"
        )
    else:
        fit = (
            f"The counts at the first {n + 1} primes not left out determine a polynomial of "
            f"degree at most {n}:"
        )
    fit_primes = set(torus.fit_primes)
    rows = [(str(p), str(c), "fit" if p in fit_primes else "check") for p, c in torus.counts]
    blocks = [
        intro,
        _paragraph(" ".join(origin)),
        _paragraph(
            f"G is solved for {_str(torus.eliminated)}, in which it has degree at most 2, so each "
            "count is a sum of numbers of roots in F_p^* of quadratics. A prime is left out when "
            "it is 2 or divides the numerator or the denominator of a non-zero value"
            f"{at} of a coefficient of G, a face discriminant, a factor of one or the "
            f"discriminant of G on an edge. {primes_left_out(torus)}"
        ),
        _paragraph(fit),
        _table(("p", "#V(F_p)", "Use"), rows),
    ]
    if torus.candidate_polynomial is None:
        refusal = f"The counts give no candidate: {torus.reason}."
        if not torus.verification_primes:
            refusal += " The fit is not checked at further primes."
        blocks.append(_paragraph(refusal))
    else:
        blocks += [
            "The counts fit the candidate for P",
            _text_equation(
                "P(q)", _lines(count_polynomial(torus.candidate_polynomial), _room("P(q)"))
            ),
            _paragraph(
                "at every prime counted, which gives the candidates "
                f"chi(X) = -P(1) = {torus.candidate_euler_characteristic} and "
                f"C = (-1)^N chi(X) = {torus.candidate_master_count}."
            ),
        ]
    blocks.append(
        _paragraph(
            "A fit on finitely many primes is evidence, not a proof: Katz's theorem needs the "
            "count to be polynomial for every finite field of all but finitely many "
            f"characteristics{doc.cite('katz2008')}. The rule that excludes primes is heuristic, "
            "and a bad prime it misses would, provided some check prime is good, make the fit or "
            "its check fail rather than give a wrong candidate."
        )
    )
    return _blocks(*blocks)


# N! Vol(Newt G), the bound on the master count, as the point-count section sets it.
_VOLUME_BOUND = "N!\\,\\mathrm{Vol}(\\mathrm{Newt}\\,G)"


# The maths in point_count's reasons for no candidate, which it writes as plain text: a
# power of q, a prime p = 41, the bound [0, N! Vol(Newt G)] = [0, 6], any other interval
# and a negative number.
_REASON_MATHS = re.compile(
    r"q\^\d+|p = \d+|\[0, N! Vol\(Newt G\)\] = \[-?\d+, -?\d+\]|\[-?\d+, -?\d+\]|(?<![\w-])-\d+"
)


def _reason(reason: str) -> str:
    """A reason for no candidate, escaped, with its maths set as maths."""
    pieces = []
    start = 0
    for match in _REASON_MATHS.finditer(reason):
        maths = re.sub(r"\^(\d+)", r"^{\1}", match.group())
        maths = maths.replace("N! Vol(Newt G)", _VOLUME_BOUND)
        pieces += [_escape(reason[start : match.start()]), f"${maths}$"]
        start = match.end()
    pieces.append(_escape(reason[start:]))
    return "".join(pieces)


def _latex_rational(value: Fraction) -> str:
    return to_latex(sp.Rational(value.numerator, value.denominator))


def latex(_report: AnalysisReport, torus: TorusCount, doc: LatexDocument) -> str:
    intro = (
        "Let $X$ be the complement of $V = \\{G = 0\\}$ in the torus $(\\mathbb{C}^*)^N$, with "
        "$\\mu = 1$. The number of master integrals, with subsectors included, symmetries unused "
        f"and $D$ symbolic, is $C = (-1)^N \\chi(X)$ by Corollary~37 of{doc.cite('bbkp2017')}. "
        f"By Theorem~44 of the same paper, $C$ is at most ${_VOLUME_BOUND}$, where "
        "$\\mathrm{Newt}\\,G$ is the Newton polytope of $G$ and $\\mathrm{Vol}$ its Euclidean "
        "volume. "
        "If the number of points of $V$ in $(\\mathbb{F}_q^*)^N$ is a polynomial $P(q)$ for "
        "every finite field $\\mathbb{F}_q$ whose characteristic avoids a finite set, then "
        "$\\chi(V) = P(1)$ by Theorem~6.1.2(3) of Katz's appendix to the paper of Hausel and "
        f"Rodriguez-Villegas{doc.cite('katz2008')}, and $\\chi(X) = -P(1)$, since the Euler "
        "characteristic is additive and vanishes on the torus."
    )
    origin = []
    if torus.on_shell:
        settings = join_words([f"${to_latex(key)} = {to_latex(v)}$" for key, v in torus.on_shell])
        origin.append(f"The counts set {settings} in the momentum products first.")
    if not torus.point:
        then = " then" if torus.on_shell else ""
        origin.append(f"$G${then} has no kinematic symbols, so the counts need no drawn point.")
    else:
        where = join_words([f"${to_latex(key)} = {_latex_rational(v)}$" for key, v in torus.point])
        if torus.seed is None:
            origin.append(
                f"The counts are taken at the kinematic point {where}, given by the caller."
            )
        else:
            origin.append(
                f"The counts are taken at the kinematic point {where}, drawn with seed "
                f"{torus.seed} so that every coefficient of $G$ and every face discriminant is "
                "non-zero. The value of $|\\chi(X)|$ is the same on an open dense set of "
                f"kinematics{doc.cite('fmt2024')}, and the draw avoids the Landau surfaces found, "
                "on which it can be smaller."
            )
    if torus.on_landau_surface:
        origin.append(
            "A coefficient of $G$ or a face discriminant vanishes at the point, where $C$ can be "
            "smaller than for generic kinematics."
        )
    skipped = torus_skipped_faces(torus)
    if skipped is not None:
        origin.append(skipped)
    at = " at the point" if torus.point else ""
    n = len(torus.variables)
    if torus.verification_primes:
        # A fit that fails no test before the check; the check stops at a disagreement.
        stops = "" if torus.candidate_polynomial is not None else ", unless a count disagrees first"
        fit = (
            "The check covers the quadratic characters $(d/p)$ for $d$ in the group generated by "
            f"$-1$ and the non-zero values{at} of the vertex coefficients of $G$, the factors of "
            f"the face discriminants{in_squared_masses(torus)} and the discriminants of $G$ "
            "on its edges. Characters outside that group are not covered; they can come from the "
            "constant factors of the discriminants of faces of dimension 2 or more, or from the "
            f"discriminants of skipped faces. The polynomial of degree at most {n} through the "
            f"counts at the first {n + 1} primes not left out is checked at further primes, at "
            "least until every non-trivial one of these characters has taken both signs"
            f"{stops}:"
        )
    else:
        fit = (
            f"The counts at the first {n + 1} primes not left out determine a polynomial of "
            f"degree at most {n}:"
        )
    fit_primes = set(torus.fit_primes)
    rows = [f"{p} & {c} & {'fit' if p in fit_primes else 'check'} \\\\" for p, c in torus.counts]
    parts = [
        "\\label{sec:torus-counts}",
        intro,
        "",
        " ".join(origin),
        "",
        f"$G$ is solved for ${to_latex(torus.eliminated)}$, in which it has degree at most 2, so "
        "each count is a sum of numbers of roots in $\\mathbb{F}_p^*$ of quadratics. A prime is "
        "left out when it is 2 or divides the numerator or the denominator of a non-zero value"
        f"{at} of a coefficient of $G$, a face discriminant, a factor of one or the discriminant "
        f"of $G$ on an edge. {primes_left_out(torus)}",
        "",
        fit,
        "\\begin{center}",
        "\\begin{tabular}{rrl}",
        "\\toprule",
        "$p$ & $\\#V(\\mathbb{F}_p)$ & Use \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{center}",
    ]
    if torus.candidate_polynomial is None:
        refusal = f"The counts give no candidate: {_reason(torus.reason or '')}."
        if not torus.verification_primes:
            refusal += " The fit is not checked at further primes."
        parts.append(refusal)
    else:
        parts += [
            "The counts fit the candidate for $P$",
            _latex_equation("P(q)", to_latex_lines(count_polynomial(torus.candidate_polynomial))),
            "at every prime counted, which gives the candidates "
            f"$\\chi(X) = -P(1) = {torus.candidate_euler_characteristic}$ and "
            f"$C = (-1)^N \\chi(X) = {torus.candidate_master_count}$.",
        ]
    parts += [
        "",
        "A fit on finitely many primes is evidence, not a proof: Katz's theorem needs the count "
        "to be polynomial for every finite field of all but finitely many characteristics"
        f"{doc.cite('katz2008')}. The rule that excludes primes is heuristic, and a bad prime it "
        "misses would, provided some check prime is good, make the fit or its check fail rather "
        "than give a wrong candidate.",
    ]
    return "\n".join(parts)


def summary(torus: TorusCount) -> list[tuple[str, str]]:
    master = torus.candidate_master_count
    return [("Candidate master count", "none" if master is None else str(master))]


SECTION = Section(
    name="torus",
    heading=TORUS_HEADING,
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(50, summary),),
)
