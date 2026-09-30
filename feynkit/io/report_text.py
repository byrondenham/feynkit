"""
Plain-text rendering of an analysis report.

:func:`render_text` states the facts of :func:`~feynkit.io.report_latex.render_latex`
in the same order, with the same sentences and citations, as plain ASCII:
maths is written out in plain names (``beta = (-D/2, -nu_1, ..., -nu_N)``),
computed expressions are printed like ``sympy.sstr`` but with ``^`` for
powers, as the prose writes them, matrices by ``sympy.pretty`` in ASCII, and
citations as ``[key]``, listed at the end with their entries stripped of
LaTeX. The text has no figures, so the sentences
that point at a figure are left out and a cross-reference names the section
it refers to. Prose is wrapped at 79 columns without breaking inline maths;
displays are indented and broken to fit the same width wherever a sum, a
product, a quotient or a vector allows it.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

import sympy as sp

from ..point_count import TorusCount
from ._report_shared import (
    MAX_PAIRS_SHOWN,
    MAX_UNIDENTIFIED_SHOWN,
    TORUS_HEADING,
    append_signed,
    count_noun,
    count_polynomial,
    f_block_sentence,
    face_counts,
    face_names,
    face_paragraphs,
    facet_graph,
    facet_graphs,
    in_squared_masses,
    integrand_templates,
    join_words,
    kinematic_class_sentence,
    landau_factors,
    lattice_normality,
    limit_factors,
    limit_sentences,
    more_faces,
    not_computed,
    primes_left_out,
    render_sections,
    resonance_forms,
    resonance_paragraphs,
    resonance_rows,
    resonance_windows,
    signed_terms,
    skipped_faces,
    split_g,
    torus_skipped_faces,
    unverified_label,
)
from ._text_kit import (
    _INDENT,
    _TEXT_EULER_MELLIN,
    _WIDTH,
    TextDocument,
    _blocks,
    _break,
    _expressions,
    _heading,
    _lines,
    _matrix,
    _paragraph,
    _room,
    _str,
    _table,
    _text_equation,
    _text_vector,
)
from .report import (
    GKZ,
    Conventions,
    Faces,
    Identity,
    Landau,
    Polytope,
    Representations,
    Resonance,
    Schwinger,
    Symmetries,
)

if TYPE_CHECKING:
    from .report import AnalysisReport

__all__ = ["render_text"]


# How the Newton polytope section refers to the point-count section, which has no number.
_TORUS_REFERENCE = f'the section "{TORUS_HEADING}"'


# --- small helpers -----------------------------------------------------------


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


def _scaled_lines(expr: sp.Expr, scale: sp.Expr, width: int) -> list[str]:
    """The lines of ``expr`` written as ``(1/scale)*(...)``."""
    prefix = f"(1/{_str(scale)})*("
    lines = _lines(expr, width - len(prefix) - 1)
    lines[0] = prefix + lines[0]
    lines[-1] += ")"
    return lines


def _factor_list(kind: str, factors: Sequence[sp.Expr]) -> tuple[str, str | None]:
    """'the <kind> factors are' and their display, or a note that there are none."""
    if not factors:
        return f"there are no {kind} factors", None
    return f"the {kind} factors are", _expressions(factors)


# --- sections ----------------------------------------------------------------


def _summary(report: AnalysisReport) -> str:
    return _table(None, report.summary())


def _graph(identity: Identity) -> str:
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


def _kinematics(conventions: Conventions) -> str:
    invariants = conventions.invariants
    if invariants is None:
        return "the dot products p_i . p_j"
    symbols = list(dict.fromkeys(invariants.symbols))
    listed = join_words([_str(s) for s in symbols])
    noun = "invariant" if len(symbols) == 1 else "invariants"
    if invariants.n_external == 2:
        return f"the {noun} {listed}: s = p^2 for p = p_1 = -p_2, so that p_1 . p_2 = -s"
    clause = "the external masses p_i^2"
    if invariants.mandelstam:
        clause += " and the planar invariants s_{i...j-1} = (p_i + ... + p_{j-1})^2"
    return f"the {noun} {listed}: {clause}"


def _conventions(report: AnalysisReport, doc: TextDocument) -> str:
    conventions = report.conventions
    dimension = conventions.dimension
    is_d = isinstance(dimension, sp.Symbol) and dimension.name == "D"
    in_d = "D" if is_d else f"D = {_str(dimension)}"
    nu_total = sp.Add(*report.identity.edge_exponents)
    if report.identity.external_legs:
        momenta = (
            "External momenta are incoming, and the kinematics is written in "
            f"{_kinematics(conventions)}."
        )
    else:
        momenta = "There are no external momenta."
    kinematic_class = kinematic_class_sentence(
        conventions,
        report.identity.external_legs,
        report.identity.edge_masses,
        name=str,
        math=lambda text: text.replace(" \\cdot ", " . "),
        printer=_str,
    )
    return _blocks(
        "The integral is",
        f"{_INDENT}I = exp(L epsilon gamma_E) (mu^2)^(nu - L D/2)\n"
        f"{_INDENT}    * int prod_{{r=1}}^{{L}} d^D k_r/(i pi^(D/2))\n"
        f"{_INDENT}    * prod_e 1/(-q_e^2 + m_e^2)^nu_e",
        _paragraph(
            f"in {in_d} dimensions{doc.cite('weinzierl2022')}, with loop momenta k_r, the "
            "momentum q_e of propagator e fixed by momentum conservation at the vertices, the "
            "metric signature (+,-,...,-) and nu = sum_e nu_e, here "
            f"nu = {_str(nu_total)}. Dimensional regularisation sets D = D_0 - 2 epsilon "
            "for an even integer D_0 chosen when the result is expanded. The energy scale mu "
            f"makes I and the Symanzik polynomials dimensionless. {momenta} {kinematic_class} "
            "The second Symanzik polynomial is "
            "F = -sum_{T_2} s_{T_2} prod_{e not in T_2} a_e + U sum_e m_e^2 a_e, divided by "
            "mu^2, the sum running over spanning two-forests T_2 and s_{T_2} being the square "
            "of the momentum flowing from one tree of T_2 to the "
            f"other{doc.cite('weinzierl2022')}."
        ),
    )


def _polynomials(report: AnalysisReport, doc: TextDocument) -> str:
    polynomials = report.polynomials
    power = polynomials.f_scale_power
    scale = report.conventions.energy_scale
    g_room = _room("G = U + F")
    if power:
        f_lines = _scaled_lines(polynomials.f_numerator, scale**power, _room("F"))
        u_part, f_part, g_power = split_g(polynomials.g, scale)
        g_f_lines = _scaled_lines(f_part, scale**g_power, g_room - 2)
        g_lines = [*_lines(u_part, g_room), "+ " + g_f_lines[0], *g_f_lines[1:]]
    else:
        f_lines = _lines(polynomials.f_numerator, _room("F"))
        g_lines = _lines(polynomials.g, g_room)
    rows = [
        (str(entry.index), _str(entry.monomial), _str(entry.coefficient))
        for entry in polynomials.z_table
    ]
    rank = polynomials.monomials_g - polynomials.codimension
    return _blocks(
        _paragraph(
            "In the Schwinger parameters a_e the first Symanzik polynomial, the sum over "
            "spanning trees T of prod_{e not in T} a_e, is"
        ),
        _text_equation("U", _lines(polynomials.u, _room("U"))),
        "and the second is",
        _text_equation("F", f_lines),
        "In the Lee-Pomeransky parameters u_e their sum is",
        _text_equation("G = U + F", g_lines),
        _paragraph(
            f"U has degree L = {polynomials.degree_u} and F degree "
            f"L+1 = {polynomials.degree_f}; F has "
            f"{count_noun(polynomials.monomials_f, 'monomial')} and G has "
            f"{polynomials.monomials_g}. The coefficients of G are the GKZ variables z_j, here "
            "at their physical values:"
        ),
        _table(("j", "Monomial", "z_j"), rows),
        _paragraph(
            f"The configuration has codimension {polynomials.monomials_g} - rank A = "
            f"{polynomials.monomials_g} - {rank} = {polynomials.codimension}, with A the "
            "matrix whose columns are the exponent vectors of the monomials of G below a row "
            "of ones, against "
            f"{count_noun(polynomials.independent_invariants, 'independent kinematic invariant')}"
            f" in the coefficients{doc.cite('klausen2023')}."
        ),
    )


def _representation(
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


def _representations(
    report: AnalysisReport, representations: Representations, doc: TextDocument
) -> str:
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
        _representation(
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
        _representation(
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
        _representation(
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


def _polytope(report: AnalysisReport, polytope: Polytope, doc: TextDocument) -> str:
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
        return [] if polytope.invariants is None else [_paragraph(_lattice(polytope, doc))]

    if not data.is_full_dimensional:
        return _blocks(
            description,
            _paragraph(
                "Since P is not full-dimensional, the rows of A are linearly dependent. For "
                "generic beta the Euler equations are then inconsistent, and the GKZ system has "
                "no non-zero solutions: its holonomic rank is 0, not the normalised volume."
            ),
            *lattice(),
            _paragraph(_scaleless(report, doc)),
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
            f"differences span Z^N. {not_computed(report, _TORUS_REFERENCE)}"
        ),
        *lattice(),
    )


def _lattice(polytope: Polytope, doc: TextDocument) -> str:
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


def _scaleless(report: AnalysisReport, doc: TextDocument) -> str:
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


def _rational(value: Fraction) -> str:
    return str(_str(sp.Rational(value.numerator, value.denominator)))


def _torus(torus: TorusCount, doc: TextDocument) -> str:
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
        where = join_words([f"{_str(key)} = {_rational(v)}" for key, v in torus.point])
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


def _gkz(gkz: GKZ, doc: TextDocument) -> str:
    operators = []
    for r, (row, beta_r) in enumerate(gkz.euler_rows):
        terms = signed_terms([(a, f"theta_{j}") for j, a in enumerate(row, start=1)])
        operator = append_signed(terms, -beta_r, _str)
        operators.append(_text_equation(f"E_{r}", _break(operator, _room(f"E_{r}"))))
    blocks = [
        _paragraph(
            "Writing G = sum_j z_j u^alpha_j and treating the coefficients z_j as independent "
            "variables gives the Euler-Mellin integral"
        ),
        f"{_INDENT}{_TEXT_EULER_MELLIN},",
        _paragraph(
            "the Lee-Pomeransky integral without its prefactor, which the GKZ system "
            "annihilates. The system has the matrix"
        ),
        _matrix("A", gkz.a_matrix),
        _paragraph(
            "whose column j is (1, alpha_j). The parameter vector is "
            "beta = (-D/2, -nu_1, ..., -nu_N), the value the Euler equations require for the "
            f"Lee-Pomeransky integral{doc.cite('delacruz2019', 'klausen2020')}. Here"
        ),
        _text_vector("beta", gkz.beta, "."),
        _paragraph(
            "The Euler operators are built from theta_j = z_j d/dz_j, which measures the "
            "degree in z_j. The operators E_r = sum_j A_{rj} theta_j - beta_r annihilate I_A: "
            "row 0 states that I_A is homogeneous of degree -D/2 in the z_j, and row i that "
            "it has degree -nu_i under z_j -> lambda^(A_{ij}) z_j, which a rescaling of u_i "
            "absorbs:"
        ),
        "\n".join(operators),
    ]
    generators = gkz.toric_generators
    if not generators:
        blocks.append("The toric ideal is trivial.")
        return _blocks(*blocks)
    blocks += [
        _paragraph(
            f"The toric ideal of A is generated by {count_noun(len(generators), 'binomial')}. "
            "Each binomial z^k - z^l, with A k = A l, gives the operator "
            "(d/dz)^k - (d/dz)^l; these annihilators of I_A form an analogue of "
            f"integration-by-parts relations for it{doc.cite('chestnov2022')}, and the "
            "specialisation to physical coefficients is a separate step. The generators are"
        ),
        _expressions(generators),
    ]
    return _blocks(*blocks)


def _resonance(report: AnalysisReport, section: Resonance, doc: TextDocument) -> str:
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


def _facet_table(rows: Sequence[tuple[str, str, str]]) -> str:
    """The facets with their inequalities and graphs.

    A graph too long for the rest of its line goes on the lines below, under
    the inequality, broken between its factors.
    """
    header = ("Facet", "Inequality", "Graph")
    first = max(len(row[0]) for row in [header, *rows])
    second = max(len(row[1]) for row in [header, *rows])
    lead = len(_INDENT) + first + 2
    lines = []
    for k, (facet, inequality, graph) in enumerate([header, *rows]):
        start = f"{_INDENT}{facet.ljust(first)}  {inequality.ljust(second)}  "
        if len(start) + len(graph) <= _WIDTH:
            lines.append((start + graph).rstrip())
        else:
            lines.append(start.rstrip())
            below = [""]
            for factor in graph.split(" "):
                if below[-1] and lead + len(below[-1]) + 1 + len(factor) > _WIDTH:
                    below.append("")
                below[-1] = f"{below[-1]} {factor}".strip()
            lines.extend(" " * lead + part for part in below)
        if k == 0:
            rules = ("-" * first, "-" * second, "-" * max(len(row[2]) for row in [header, *rows]))
            lines.append(f"{_INDENT}{rules[0]}  {rules[1]}  {rules[2]}"[:_WIDTH])
    return "\n".join(lines)


def _faces(report: AnalysisReport, section: Faces, doc: TextDocument) -> str:
    paragraphs, classes, intro = face_paragraphs(section, doc.cite, latex=False)
    blocks = [_paragraph(p) for p in paragraphs]
    if not section.full_dimensional:
        return _blocks(*blocks)
    graphs = facet_graphs(report, section)
    rows = [
        (f"F_{k}", f"{_str(lhs)} <= {b}", facet_graph(face, latex=False))
        for k, lhs, b, face in graphs
    ]
    counts = face_counts(section)
    header = ("Class", *(f"Codim {c}" for c in range(section.max_codimension + 1)))
    blocks += [
        _paragraph(
            "The graph of each facet, a support product marked (support) and a facet that is "
            "not identified by a dash:"
        ),
        _facet_table(rows),
        _paragraph(f"{classes}:"),
        _table(header, [(name, *map(str, row)) for name, row in counts]),
    ]
    unidentified = [face for face in section.faces if not face.verified]
    if unidentified:
        number = {id(face): k for k, _, _, face in graphs}
        blocks.append(_paragraph(intro))
        for face in unidentified[:MAX_UNIDENTIFIED_SHOWN]:
            k = number.get(id(face))
            label = f"F_{k}" if k is not None else f"A face of dimension {face.dimension}"
            assert face.prediction is not None
            blocks += [
                _paragraph(unverified_label(face, label, latex=False)),
                _text_equation("G|_F", _lines(face.polynomial, _room("G|_F"))),
                _text_equation("prediction", _lines(face.prediction, _room("prediction"))),
            ]
        left = len(unidentified) - MAX_UNIDENTIFIED_SHOWN
        if left > 0:
            blocks.append(_paragraph(more_faces(left)))
    return _blocks(*blocks)


def _symmetries(report: AnalysisReport, symmetries: Symmetries, doc: TextDocument) -> str:
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


def _landau(landau: Landau, scale: sp.Symbol, doc: TextDocument) -> str:
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
        first, first_display = _factor_list("first-type (Cayley)", factors.first_type)
        second, second_display = _factor_list("second-type (Gram)", factors.second_type)
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
    blocks += _limits(landau, doc)
    return _blocks(*blocks)


def _limits(landau: Landau, doc: TextDocument) -> list[str]:
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


def _schwinger(schwinger: Schwinger, doc: TextDocument) -> str:
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


# --- the document ------------------------------------------------------------


def render_text(report: AnalysisReport, *, title: str | None = None) -> str:
    """Render an analysis report as plain text.

    Parameters
    ----------
    report
        The report to render; sections it does not carry are omitted.
    title
        Title line, used as given; by default "Feynman integral" followed by
        the CNickel string.

    Returns
    -------
    str
        The report, ASCII only: the title underlined with ``=``, each section
        heading with ``-``, and the references last.
    """
    doc = TextDocument()
    heading = f"Feynman integral {report.identity.cnickel}" if title is None else title
    sections = render_sections(
        report,
        summary=lambda: _summary(report),
        graph=lambda: _graph(report.identity),
        conventions=lambda: _conventions(report, doc),
        polynomials=lambda: _polynomials(report, doc),
        representations=lambda section: _representations(report, section, doc),
        polytope=lambda section: _polytope(report, section, doc),
        torus=lambda section: _torus(section, doc),
        gkz=lambda section: _gkz(section, doc),
        resonance=lambda section: _resonance(report, section, doc),
        faces=lambda section: _faces(report, section, doc),
        symmetries=lambda section: _symmetries(report, section, doc),
        landau=lambda section: _landau(section, report.conventions.energy_scale, doc),
        schwinger=lambda section: _schwinger(section, doc),
    )
    blocks = [
        _heading(heading, "="),
        *(f"{_heading(name, '-')}\n\n{body}" for name, body in sections),
        f"{_heading('References', '-')}\n\n{doc.references()}",
    ]
    return "\n\n".join(blocks) + "\n"
