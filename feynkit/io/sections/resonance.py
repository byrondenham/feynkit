"""
The resonant values of the dimensional-regularisation parameter of each facet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import sympy as sp

from ... import _exact
from ...core.exceptions import ValidationError
from ...resonance import D0Source, EpsilonSet, FacetResonance, classify_facets, span_epsilons
from .._latex_kit import LatexDocument, _longtable, _math
from .._text_kit import TextDocument, _blocks, _paragraph, _str, _table
from ..latex import to_latex
from ._base import Section

if TYPE_CHECKING:
    from ...integral import FeynmanIntegral
    from ..report import AnalysisReport, BuildContext


# The interval of eps in which the resonance section lists the resonant values of each facet.
RESONANCE_WINDOW = (Fraction(-1), Fraction(1))


@dataclass(frozen=True)
class Resonance:
    """Which facets of P are resonant or admissible as D = D_0 - 2 eps varies.

    Attributes
    ----------
    d0
        D_0.
    d0_source
        Where D_0 comes from: "given", "dimension", read from a dimension of
        the integral of the form D_0 - 2 eps, or "default", 4; see
        :func:`feynkit.resonance.choose_d0`.
    powers
        The integer power of each edge, in internal-edge order.
    unit_powers
        Whether ``powers`` are all 1 because the exponents of the integral are
        not all integers.
    span
        The eps at which beta lies in the span of A: "all" when P is
        full-dimensional.
    facets
        One record per facet of P, relative to its affine hull when P is not
        full-dimensional, in the order of ``PolytopeData.relative_facets``;
        see :func:`feynkit.resonance.classify_facets`.
    full_dimensional
        Whether P is full-dimensional.
    """

    d0: Fraction
    d0_source: D0Source
    powers: tuple[int, ...]
    unit_powers: bool
    span: EpsilonSet
    facets: tuple[FacetResonance, ...]
    full_dimensional: bool


def integer_powers(fi: FeynmanIntegral) -> tuple[tuple[int, ...], bool]:
    """The powers of the internal edges, in edge order, and whether they are all 1 because the
    exponents of the integral are not all integers."""
    exponents = [fi.propagator_exponents[e.idx] for e in fi.graph.get_internal_edges()]
    try:
        return tuple(_exact._as_int(x, "the exponents") for x in exponents), False
    except ValidationError:
        return (1,) * len(exponents), True


def build(ctx: BuildContext) -> Resonance:
    fi = ctx.integral
    data = ctx.polytope_data()
    d0 = ctx.d0
    source = ctx.d0_source
    powers, unit = integer_powers(fi)
    return Resonance(
        d0=d0,
        d0_source=source,
        powers=powers,
        unit_powers=unit,
        span=span_epsilons(data, powers, d0),
        facets=classify_facets(data, powers, d0),
        full_dimensional=data.is_full_dimensional,
    )


def _number(value: Fraction, latex: bool) -> str:
    """A rational as the text prints it, -1/2, or in LaTeX, -\\tfrac{1}{2}."""
    if value.denominator == 1 or not latex:
        return str(value)
    sign = "-" if value < 0 else ""
    return f"{sign}\\tfrac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def epsilon_set(found: EpsilonSet, *, latex: bool) -> str:
    """An EpsilonSet as a table cell: every eps, a progression, one value, or none.

    A progression is written offset + period Z with the offset eps_F as stored,
    Z/q for the period 1/q, and without the offset when it is 0.
    """
    eps = "$\\varepsilon$" if latex else "epsilon"
    if found.kind == "all":
        return f"every {eps}"
    if found.kind == "never":
        return f"no {eps}"
    assert found.offset is not None
    value = _number(found.offset, latex)
    if found.kind == "point":
        return f"$\\varepsilon = {value}$" if latex else f"epsilon = {value}"
    period = found.period
    assert period is not None
    if latex:
        lattice = ("" if period == 1 else _number(period, True)) + "\\mathbb{Z}"
    elif period == 1:
        lattice = "Z"
    elif period.numerator == 1:
        lattice = f"Z/{period.denominator}"
    else:
        lattice = f"({period})Z"
    text = lattice if found.offset == 0 else f"{value} + {lattice}"
    return f"${text}$" if latex else text


def resonance_forms(
    report: AnalysisReport, section: Resonance
) -> list[tuple[int, sp.Expr, int, sp.Expr]]:
    """Per facet: its number k in F_k, the left side m . x and right side b of its
    inequality, and l_F(beta) in D and the nu_e.

    The coordinates x_e and the powers nu_e are named by the edge indices.
    """
    edges = report.identity.edge_indices
    x = [sp.Symbol(f"x_{e}") for e in edges]
    nu = [sp.Symbol(f"nu_{e}") for e in edges]
    dimension = sp.Symbol("D")
    return [
        (
            k,
            sp.Add(*(m * v for m, v in zip(record.facet.normal, x, strict=True))),
            record.facet.offset,
            record.form.expression(dimension, nu),
        )
        for k, record in enumerate(section.facets, start=1)
    ]


def resonance_windows(section: Resonance, *, latex: bool) -> list[tuple[int, str]]:
    """The facets resonant on a progression or at one value, and their resonant values in
    RESONANCE_WINDOW, with k as in F_k; "none" when no value falls in it."""
    rows = []
    for k, record in enumerate(section.facets, start=1):
        if record.resonant.kind in ("progression", "point"):
            values = record.resonant.window(*RESONANCE_WINDOW)
            text = ", ".join(_number(v, latex) for v in values) or "none"
            rows.append((k, f"${text}$" if latex and values else text))
    return rows


def resonance_rows(section: Resonance, *, latex: bool) -> list[tuple[str, str, str, str]]:
    """Per facet: where it is resonant, whether at eps = 0, where admissible, reducible or not.

    A dash marks a facet that does not decide reducibility.
    """

    def row(record: FacetResonance) -> tuple[str, str, str, str]:
        return (
            epsilon_set(record.resonant, latex=latex),
            "yes" if record.resonant_at_zero else "no",
            epsilon_set(record.admissible, latex=latex),
            "yes" if record.reducible else "-",
        )

    return [row(record) for record in section.facets]


# The maths of the resonance section, as the text prints it and in LaTeX.
_RESONANCE_MATHS = {
    "eps": ("epsilon", r"\varepsilon"),
    "D": ("D = D_0 - 2 epsilon", r"D = D_0 - 2\varepsilon"),
    "nu_e": ("nu_e = 1", r"\nu_e = 1"),
    "F": ("F", "F"),
    "P": ("P", "P"),
    "A": ("A", "A"),
    "G": ("G", "G"),
    "A_F": ("A_F", "A_F"),
    "beta": ("beta", r"\beta"),
    "resonant": ("span(A_F) + ZA", r"\operatorname{span}_{\mathbb{C}} A_F + \mathbb{Z}A"),
    "span": ("span(A_F)", r"\operatorname{span}_{\mathbb{C}} A_F"),
    "system": ("(A_F, beta)", r"(A_F, \beta)"),
    "facet": ("m . x <= b", r"m \cdot x \le b"),
    "g_F": ("g_F", "g_F"),
    "l_F": ("l_F", "l_F"),
    "functional": ("l_F = (b, -m)/g_F", "l_F = (b, -m)/g_F"),
    "ZA": ("ZA", r"\mathbb{Z}A"),
    "Z": ("Z", r"\mathbb{Z}"),
    "value": ("l_F(beta) = (m . nu - b D/2)/g_F", r"l_F(\beta) = (m \cdot \nu - bD/2)/g_F"),
    "full": ("ZA = Z^(N+1)", r"\mathbb{Z}A = \mathbb{Z}^{N+1}"),
    "b0": ("b = 0", "b = 0"),
    "b": ("b != 0", r"b \ne 0"),
    "progression": ("epsilon_F + (g_F/|b|) Z", r"\varepsilon_F + (g_F/|b|)\mathbb{Z}"),
    "eps_F": ("epsilon_F = D_0/2 - m . nu/b", r"\varepsilon_F = D_0/2 - m \cdot \nu/b"),
    "centre": ("ZA + span(A_G)", r"\mathbb{Z}A + \operatorname{span}_{\mathbb{C}} A_G"),
    "hull": ("h_0 + h . x = 0", r"h_0 + h \cdot x = 0"),
    "asks": ("h_0 D/2 + h . nu = 0", r"h_0 D/2 + h \cdot \nu = 0"),
    "zero": ("epsilon = 0", r"\varepsilon = 0"),
    "D_0": ("D = D_0", "D = D_0"),
}


def resonance_paragraphs(
    section: Resonance, cite: Callable[..., str], *, latex: bool
) -> tuple[list[str], str, str]:
    """The paragraphs before the tables of the resonance section, and the two captions.

    Returns the opening paragraphs, the sentence before the table of where each
    facet is resonant, and the sentence before the list of resonant values in
    the window; the renderer prints the facets and their forms after the
    opening paragraphs.
    """

    def m(text: str, tex: str) -> str:
        return f"${tex}$" if latex else text

    x = {key: m(text, tex) for key, (text, tex) in _RESONANCE_MATHS.items()}
    d0 = _number(section.d0, latex)
    d0_is = m(f"D_0 = {d0}", f"D_0 = {d0}")
    if section.d0_source == "given":
        d0_clause = f"{d0_is} as given"
    elif section.d0_source == "dimension":
        d0_clause = f"{d0_is}, read from the dimension of the integral"
    else:
        two_eps = m("2 epsilon", "2\\varepsilon")
        d0_clause = (
            f"{d0_is} by default, since the dimension of the integral is not a number minus "
            f"{two_eps}"
        )
    if section.unit_powers:
        powers = (
            f"{x['nu_e']} on every edge, since the exponents of the integral are not all "
            "integers"
        )
    else:
        vector = "(" + ", ".join(str(p) for p in section.powers) + ")"
        nu = m(f"nu = {vector}", r"\nu = " + vector)
        powers = f"the powers {nu} of the integral"
    opening = (
        f"Let {x['D']} with {d0_clause}, and {powers}. A face {x['F']} of {x['P']}, with columns "
        f"{x['A_F']}, is resonant when {x['beta']} lies in {x['resonant']}, and admissible when "
        f"it lies in {x['span']}; the face system {x['system']} is then a true subsystem, its "
        f"solutions solving the full system{cite('britto2026')}. For a facet {x['facet']} with "
        f"lattice index {x['g_F']}, the functional {x['functional']} is zero on the facet, "
        f"positive on the other columns of {x['A']} and maps {x['ZA']} onto {x['Z']}. The facet "
        f"is resonant exactly when {x['value']} is an integer and admissible exactly when it is "
        f"zero; Britto, Grimm and Hoefnagels prove this for {x['full']}, and the proof needs "
        f"only that {x['l_F']} maps {x['ZA']} onto {x['Z']}{cite('britto2026')}. With the "
        f"powers fixed, a facet with {x['b0']} is resonant for every {x['eps']} or for none, "
        f"and one with {x['b']} exactly on the progression {x['progression']}, where "
        f"{x['eps_F']} is the one value of {x['eps']} at which it is admissible."
    )
    paragraphs = [opening]
    if section.full_dimensional:
        paragraphs.append(
            f"A resonant facet with at least two columns of {x['A']} off it makes the GKZ "
            f"system reducible{cite('britto2026')}: it contains a resonance centre, a minimal "
            f"face {x['G']} with {x['beta']} in {x['centre']}, over which {x['A']} is not a "
            f"pyramid either, and Theorem 4.1 of{cite('schulze2012')} applies. When one column "
            f"lies off the facet, {x['A']} is a pyramid over it and the facet does not decide "
            "reducibility; a dash in the table marks such a facet, and one that is never "
            "resonant."
        )
    else:
        if section.span.kind == "all":
            where = f"for every {x['eps']}"
        elif section.span.kind == "never":
            where = f"for no {x['eps']}"
        else:
            assert section.span.offset is not None
            value = _number(section.span.offset, latex)
            where = "only at " + m(f"epsilon = {value}", r"\varepsilon = " + value)
        paragraphs.append(
            f"{x['P']} is not full-dimensional, so {x['beta']} must first lie in the span of "
            f"{x['A']}: every equation {x['hull']} of the affine hull of {x['P']} asks "
            f"{x['asks']}, and elsewhere the Euler equations are inconsistent and the GKZ "
            f"system has no non-zero solutions. Here {x['beta']} lies in the span of {x['A']} "
            f"{where}. The facets below are relative to the affine hull, their functionals "
            f"fixed modulo its equations, and the sets hold only where {x['beta']} lies in the "
            "span. Below full dimension the report does not decide reducibility, and the table "
            "shows a dash."
        )
    classes = (
        f"The table gives, for each facet, the values of {x['eps']} at which it is resonant, "
        f"whether {x['zero']}, that is {x['D_0']}, is one of them, where it is admissible and "
        "whether it makes the system reducible:"
    )
    low, high = (_number(v, latex) for v in RESONANCE_WINDOW)
    interval = m(f"{low} <= epsilon <= {high}", f"{low} \\le \\varepsilon \\le {high}")
    window = (
        f"In the window {interval}, the facets resonant on a progression or at one value are "
        "resonant at:"
    )
    return paragraphs, classes, window


def text(report: AnalysisReport, section: Resonance, doc: TextDocument) -> str:
    paragraphs, classes, window = resonance_paragraphs(section, doc.cite, latex=False)
    blocks = [_paragraph(p) for p in paragraphs]
    if not section.facets:
        return _blocks(*blocks, _paragraph("P is a point and has no facets."))
    forms = [
        (f"F_{k}", f"{_str(lhs)} <= {b}", _str(form))
        for k, lhs, b, form in resonance_forms(report, section)
    ]
    rows = [(f"F_{k}", *row) for k, row in enumerate(resonance_rows(section, latex=False), start=1)]
    blocks += [
        _table(("Facet", "Inequality", "l_F(beta)"), forms),
        _paragraph(classes),
        _table(("Facet", "Resonant for", "At 0", "Admissible at", "Reducible"), rows),
    ]
    windows = resonance_windows(section, latex=False)
    if windows:
        blocks += [_paragraph(window), _table(None, [(f"F_{k}", v) for k, v in windows])]
    return _blocks(*blocks)


def latex(report: AnalysisReport, section: Resonance, doc: LatexDocument) -> str:
    paragraphs, classes, window = resonance_paragraphs(section, doc.cite, latex=True)
    parts = list(paragraphs)
    if not section.facets:
        return "\n\n".join([*parts, "$P$ is a point and has no facets."])
    forms = [
        (f"$F_{{{k}}}$", f"${to_latex(lhs)} \\le {b}$", _math(form))
        for k, lhs, b, form in resonance_forms(report, section)
    ]
    rows = [
        (f"$F_{{{k}}}$", *row) for k, row in enumerate(resonance_rows(section, latex=True), start=1)
    ]
    parts += [
        _longtable("lll", ("Facet", "Inequality", "$l_F(\\beta)$"), forms),
        classes,
        _longtable(
            "lllll",
            ("Facet", "Resonant for", "At $0$", "Admissible at", "Reducible"),
            rows,
        ),
    ]
    windows = resonance_windows(section, latex=True)
    if windows:
        parts += [window, _longtable("ll", (), [(f"$F_{{{k}}}$", v) for k, v in windows])]
    return "\n\n".join(parts)


SECTION = Section(
    name="resonance",
    heading="Resonance",
    build=build,
    text=text,
    latex=latex,
)
