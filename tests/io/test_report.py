"""Tests for the analysis report data model.

Oracles: the objects the report gathers. Every count and expression the
report stores must equal the one the corresponding feynkit object reports,
so the assertions compare against :class:`FeynmanIntegral` attributes and
:func:`landau_analysis` rather than against literals, except where a literal
pins a known value: the massless bubble's convergence region
(Klausen 2023, Thm FIconvergence) and the three invariants of the massless
triangle.
"""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit import landau as landau_module
from feynkit import point_count as point_count_module
from feynkit.core.exceptions import ValidationError
from feynkit.face_identification import identify_faces
from feynkit.io import report as report_module
from feynkit.io.report import DEFAULT_SECTIONS, FACE_CODIMENSION, SECTION_NAMES, AnalysisReport
from feynkit.landau import (
    landau_analysis,
    landau_analysis_from_polynomial,
    one_loop_bridge_poles,
    one_loop_landau_surfaces_by_type,
)
from feynkit.polytope import polytope_data
from feynkit.resonance import EpsilonSet, classify_facets
from feynkit.systems.monomial import extract_monomial_support

SUMMARY_LABELS = (
    "Loops",
    "Propagators",
    "External legs",
    "Kinematic class",
    "Monomials of F",
    "Monomials of G",
    "Independent invariants",
    "Codimension",
    "Scaleless",
    "Polytope vertices",
    "Normalised volume",
    "Lattice points",
    "Interior lattice points",
    "Gorenstein index",
    "Normal configuration",
    "Unidentified faces",
    "Polytope automorphisms",
    "Toric generators",
    "Landau surfaces",
)


def _monic(expr: sp.Expr) -> sp.Expr:
    """Normalise a polynomial factor up to a constant, for set comparison."""
    expr = sp.expand(expr)
    return sp.Poly(expr, *sorted(expr.free_symbols, key=str)).monic().as_expr()


@pytest.fixture(scope="module")  # type: ignore[misc]
def bubble() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("11e|e|:nn")


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


@pytest.fixture(scope="module")  # type: ignore[misc]
def sunrise() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("111e|e|:nnn")


@pytest.fixture(scope="module")  # type: ignore[misc]
def box() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")


@pytest.fixture(scope="module")  # type: ignore[misc]
def bubble_report(bubble: FeynmanIntegral) -> AnalysisReport:
    return AnalysisReport.from_integral(bubble)


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle_report(triangle: FeynmanIntegral) -> AnalysisReport:
    return AnalysisReport.from_integral(triangle)


@pytest.fixture(scope="module")  # type: ignore[misc]
def sunrise_report(sunrise: FeynmanIntegral) -> AnalysisReport:
    return AnalysisReport.from_integral(sunrise)


@pytest.fixture(scope="module")  # type: ignore[misc]
def box_report(box: FeynmanIntegral) -> AnalysisReport:
    # Only the always-built sections (identity, conventions, polynomials) are
    # needed here, so the automorphism and Landau computations, the slow part
    # of a full box report, are skipped.
    return AnalysisReport.from_integral(box, [])


class TestIdentity:
    def test_topology_and_edges(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        identity = bubble_report.identity
        assert identity.cnickel == "11e|e|:nn"
        assert identity.nickel == bubble.nickel_index
        assert (identity.loop_count, identity.propagators, identity.external_legs) == (1, 2, 2)
        edges = bubble.graph.get_internal_edges()
        assert identity.edge_masses == tuple(e.get_mass() for e in edges)
        assert identity.edge_exponents == tuple(bubble.propagator_exponents[e.idx] for e in edges)
        assert identity.edge_indices == tuple(e.idx for e in edges)
        assert identity.edge_endpoints == tuple((e.v1, e.v2) for e in edges)

    def test_edge_indices_need_not_start_at_one(self) -> None:
        graph = Graph(
            internal_vertices=2,
            external_legs=2,
            edges=[
                Edge(idx=2, v1=1, v2=2, is_internal=True),
                Edge(idx=3, v1=2, v2=1, is_internal=True),
                Edge(idx=4, v1=1, v2=3, is_internal=False),
                Edge(idx=5, v1=2, v2=4, is_internal=False),
            ],
        )
        identity = AnalysisReport.from_integral(FeynmanIntegral(graph), []).identity
        assert identity.edge_indices == (2, 3)
        assert identity.edge_endpoints == ((1, 2), (2, 1))

    def test_massless_edges_have_zero_mass(self, triangle_report: AnalysisReport) -> None:
        assert triangle_report.identity.edge_masses == (0, 0, 0)

    def test_graph_figure_is_tikz(self, triangle_report: AnalysisReport) -> None:
        assert "\\begin{tikzpicture}" in triangle_report.identity.graph_tikz


class TestConventions:
    def test_invariants_and_products(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        conventions = triangle_report.conventions
        assert conventions.dimension == triangle.dimension
        assert conventions.energy_scale == triangle.graph.energy_scale
        assert conventions.invariants is not None
        assert conventions.invariants.n_external == 3
        assert conventions.momentum_products == tuple(sorted(triangle.momentum_products.items()))

    def test_no_invariants_without_mandelstam(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", use_mandelstam=False)
        report = AnalysisReport.from_integral(fi, ["gkz"])
        assert report.conventions.invariants is None


class TestPolynomials:
    def test_z_table_matches_gkz_support(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        table = triangle_report.polynomials.z_table
        assert [e.exponent for e in table] == [alpha for alpha, _ in triangle.gkz.support]
        assert [e.index for e in table] == list(range(1, len(table) + 1))
        physical = dict(triangle.newton_polytope.support)
        assert [e.coefficient for e in table] == [physical[e.exponent] for e in table]
        u = triangle.symanzik.lp_parameters
        for e in table:
            assert e.monomial == sp.Mul(*[v**k for v, k in zip(u, e.exponent, strict=True)])

    def test_z_table_indices_match_the_a_matrix_columns(
        self, triangle_report: AnalysisReport
    ) -> None:
        """Catches a z-index / A-matrix-column misalignment.

        The exponent test above compares against ``gkz.support`` directly,
        the very list the z-table is built from, so it cannot see a index
        shift against the matrix the GKZ section renders. Column j - 1 of
        the report's own ``gkz.a_matrix`` (built independently, from
        ``fi.gkz.a_matrix``) must carry the same exponent as ``z_table[j-1]``.
        """
        table = triangle_report.polynomials.z_table
        assert triangle_report.gkz is not None
        a_matrix = triangle_report.gkz.a_matrix
        for j in range(1, len(table) + 1):
            assert table[j - 1].exponent == tuple(int(x) for x in a_matrix.col(j - 1))[1:]

    def test_z_table_coefficients_match_g_independently(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        """Coefficients checked against a fresh `sp.Poly` of G, not against
        `newton_polytope.support`, the dict the implementation itself reads.
        """
        lp_parameters = triangle.symanzik.lp_parameters
        g_poly = sp.Poly(sp.expand(triangle.symanzik.g), *lp_parameters)
        for entry in triangle_report.polynomials.z_table:
            assert entry.coefficient == g_poly.coeff_monomial(entry.monomial)

    def test_degrees_counts_codimension(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        p = triangle_report.polynomials
        assert (p.degree_u, p.degree_f) == (1, 2)
        assert p.monomials_g == len(triangle.gkz.support)
        # The polytope is full-dimensional, so rank A = N + 1 = 4.
        assert p.codimension == p.monomials_g - triangle.gkz.a_matrix.rank() == p.monomials_g - 4
        assert p.f_scale_power == 2
        assert sp.expand(p.f_numerator / triangle.graph.energy_scale**2 - triangle.symanzik.f) == 0

    def test_codimension_uses_the_rank_of_a(self) -> None:
        # A massless vacuum bubble has G = u_1 + u_2, a segment in the plane:
        # A has rank 2, not N + 1 = 3, and the codimension is 2 - 2 = 0.
        graph = Graph(
            internal_vertices=2,
            external_legs=0,
            edges=[
                Edge(idx=1, v1=1, v2=2, is_internal=True, mass=0),
                Edge(idx=2, v1=1, v2=2, is_internal=True, mass=0),
            ],
        )
        fi = FeynmanIntegral(graph, use_mandelstam=False)
        p = AnalysisReport.from_integral(fi, []).polynomials
        assert fi.gkz.a_matrix.rank() == 2
        assert (p.monomials_g, p.codimension) == (2, 0)

    def test_monomials_f_counts_the_support_not_the_terms(
        self,
        bubble_report: AnalysisReport,
        triangle_report: AnalysisReport,
        box_report: AnalysisReport,
    ) -> None:
        """A monomial support entry can be several `Add` terms.

        The box's F has monomials with composite coefficients (sums of
        several kinematic products), so `sp.expand(F)` splits some of them
        into more than one term; counting terms would overcount. Expected
        numbers verified independently with `extract_monomial_support`
        before pinning: bubble 3 of 5, triangle 3 of 6, box 6 of 10.
        """
        assert (bubble_report.polynomials.monomials_f, bubble_report.polynomials.monomials_g) == (
            3,
            5,
        )
        assert (
            triangle_report.polynomials.monomials_f,
            triangle_report.polynomials.monomials_g,
        ) == (3, 6)
        assert (box_report.polynomials.monomials_f, box_report.polynomials.monomials_g) == (
            6,
            10,
        )

    def test_monomials_f_matches_extract_monomial_support(
        self, box: FeynmanIntegral, box_report: AnalysisReport
    ) -> None:
        f_expanded = sp.expand(box.symanzik.f)
        support = extract_monomial_support(f_expanded, list(box.symanzik.schwinger_parameters))
        assert box_report.polynomials.monomials_f == len(support)

    def test_monomials_f_never_exceeds_monomials_g(
        self,
        bubble_report: AnalysisReport,
        triangle_report: AnalysisReport,
        sunrise_report: AnalysisReport,
        box_report: AnalysisReport,
    ) -> None:
        """G = U + F, so G has at least as many monomials as F."""
        for report in (bubble_report, triangle_report, sunrise_report, box_report):
            assert report.polynomials.monomials_f <= report.polynomials.monomials_g

    def test_independent_invariants_massless_triangle(
        self, triangle_report: AnalysisReport
    ) -> None:
        assert triangle_report.polynomials.independent_invariants == 3  # p1^2, p2^2, p3^2

    def test_independent_invariants_massive_bubble(self, bubble_report: AnalysisReport) -> None:
        assert bubble_report.polynomials.independent_invariants == 3  # m_1, m_2, s

    def test_polynomials_are_the_symanzik_ones(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        p = bubble_report.polynomials
        assert p.u == bubble.symanzik.u
        assert p.g == bubble.symanzik.g


class TestRepresentations:
    def test_bubble_convergence_region(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        rep = AnalysisReport.from_integral(fi, ["representations"]).representations
        assert rep is not None
        D = fi.dimension
        n1, n2 = (fi.propagator_exponents[i] for i in (1, 2))
        assert rep.convergence is not None
        assert {sp.expand(c) for c in rep.convergence} == {
            sp.expand(D / 2 - n1),
            sp.expand(D / 2 - n2),
            sp.expand(-D / 2 + n1 + n2),
        }

    def test_one_inequality_per_facet(self, triangle_report: AnalysisReport) -> None:
        assert triangle_report.representations is not None
        assert triangle_report.polytope is not None
        assert len(triangle_report.representations.convergence) == len(
            triangle_report.polytope.data.facets
        )

    def test_parametrisations_are_the_integral_s(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        rep = triangle_report.representations
        assert rep is not None
        assert rep.schwinger == triangle.schwinger
        assert rep.feynman == triangle.feynman
        assert rep.lee_pomeransky == triangle.lee_pomeransky

    def test_no_convergence_region_when_not_full_dimensional(self) -> None:
        """A massless tadpole has G = u_1, a point in the line: it converges nowhere."""
        graph = Graph(
            internal_vertices=1,
            external_legs=0,
            edges=[Edge(idx=1, v1=1, v2=1, is_internal=True, mass=0)],
        )
        fi = FeynmanIntegral(graph, momentum_products={})
        rep = AnalysisReport.from_integral(fi, ["representations"]).representations
        assert rep is not None
        assert rep.convergence is None


class TestPolytope:
    def test_data_matches_the_newton_polytope(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        assert triangle_report.polytope is not None
        data = triangle_report.polytope.data
        assert list(data.points) == [tuple(p) for p in triangle.newton_polytope.points]
        assert data.dimension == 3
        assert data.normalized_volume == 4

    def test_figure_is_drawn_for_small_polytopes(self, triangle_report: AnalysisReport) -> None:
        assert triangle_report.polytope is not None
        figure = triangle_report.polytope.figure
        assert figure is not None and "\\begin{tikzpicture}" in figure

    def test_figure_omitted_when_too_many_vertices(self, triangle: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(triangle, ["polytope"], figure_max_vertices=2)
        assert report.polytope is not None
        assert report.polytope.figure is None


class TestGKZ:
    def test_matrix_beta_and_generators(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        gkz = triangle_report.gkz
        assert gkz is not None
        assert gkz.a_matrix == triangle.gkz.a_matrix
        assert gkz.beta == tuple(triangle.gkz.beta_parameters)
        assert gkz.z_variables == tuple(triangle.gkz.z_variables)
        assert gkz.toric_generators == tuple(triangle.toric_ideal.generators)

    def test_euler_rows_are_the_rows_of_a(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        gkz = triangle_report.gkz
        assert gkz is not None
        a = triangle.gkz.a_matrix
        assert len(gkz.euler_rows) == a.rows
        for r, (row, beta_r) in enumerate(gkz.euler_rows):
            assert row == tuple(int(x) for x in a.row(r))
            assert beta_r == triangle.gkz.beta_parameters[r]


class TestSymmetries:
    def test_group_data_matches_the_integral(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        symmetries = triangle_report.symmetries
        assert symmetries is not None
        auts = triangle.polytope_automorphisms
        assert symmetries.automorphism_order == auts.order
        assert symmetries.vertex_orbits == tuple(tuple(o) for o in auts.vertex_orbits)
        assert symmetries.graph_automorphisms == tuple(
            tuple(p) for p in triangle.graph_automorphisms
        )
        assert symmetries.symmetry_pairs == tuple(triangle.symmetry_pairs)

    def test_coefficient_preserving_is_a_subgroup_size(
        self, triangle_report: AnalysisReport
    ) -> None:
        symmetries = triangle_report.symmetries
        assert symmetries is not None
        assert 1 <= symmetries.coefficient_preserving <= symmetries.automorphism_order
        # Aut(P) permutes the three quadratic monomials of G, whose coefficients
        # are the distinct invariants p_1^2, p_2^2, p_3^2, so only the identity
        # preserves them.
        assert symmetries.coefficient_preserving == 1
        assert symmetries.automorphism_order > 1

    def test_segment(self) -> None:
        # The massive tadpole's Newton polytope is a segment: the identity and its reflection.
        tadpole = FeynmanIntegral.from_cnickel("0|:n", use_mandelstam=False)
        report = AnalysisReport.from_integral(tadpole)
        assert report.polytope is not None
        assert report.polytope.data.dimension == 1
        symmetries = report.symmetries
        assert symmetries is not None
        assert symmetries.full_dimensional
        assert (symmetries.automorphism_order, symmetries.vertex_orbits) == (2, ((0, 1),))
        assert len(symmetries.symmetry_pairs) == 2
        assert dict(report.summary())["Polytope automorphisms"] == "2"

    def test_below_full_dimension(self) -> None:
        # G is u_1 times the G of the massive triangle, so P has dimension 3 in R^4, and the
        # permutations of u_2, u_3 and u_4 are its automorphisms.
        fi = FeynmanIntegral.from_cnickel("012e|2e|e|:znnn")
        report = AnalysisReport.from_integral(fi, ["polytope", "symmetries"])
        assert report.polytope is not None
        assert (report.polytope.data.dimension, report.polytope.data.ambient_dimension) == (3, 4)
        symmetries = report.symmetries
        assert symmetries is not None
        assert not symmetries.full_dimensional
        assert symmetries.automorphism_order == 6
        assert symmetries.vertex_orbits == ((0, 1, 2), (3, 4, 5))
        assert len(symmetries.graph_automorphisms) == 2
        assert len(symmetries.symmetry_pairs) == 6
        assert dict(report.summary())["Polytope automorphisms"] == "6"

    def test_not_built_when_not_asked_for(self, triangle: FeynmanIntegral) -> None:
        assert AnalysisReport.from_integral(triangle, ["gkz"]).symmetries is None


class TestLandau:
    def test_matches_landau_analysis(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        direct = landau_analysis(bubble)
        assert bubble_report.landau is not None
        assert bubble_report.landau.analysis.landau_surfaces == direct.landau_surfaces

    def test_types_cover_the_kinematic_surfaces(self, bubble_report: AnalysisReport) -> None:
        """The closed form and the face computation give the same surfaces.

        Compared as sets of irreducible factors normalised up to a
        constant; both already exclude the energy scale, so no further
        filtering is needed.
        """
        landau = bubble_report.landau
        assert landau is not None
        closed = {_monic(f) for f in landau.first_type + landau.second_type}
        faces = {_monic(f) for f in landau.analysis.landau_surfaces}
        assert closed == faces

    def test_first_and_second_type_come_from_the_closed_form(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        assert bubble_report.landau is not None
        first, second = one_loop_landau_surfaces_by_type(bubble)
        assert bubble_report.landau.first_type == first
        assert bubble_report.landau.second_type == second

    def test_by_dimension_is_ascending_and_non_trivial(self, bubble_report: AnalysisReport) -> None:
        landau = bubble_report.landau
        assert landau is not None
        dimensions = [d for d, _ in landau.by_dimension]
        assert dimensions == sorted(dimensions)
        assert all(discriminants for _, discriminants in landau.by_dimension)
        assert all(d != 1 for _, discriminants in landau.by_dimension for d in discriminants)

    def test_two_loops_have_no_closed_form(self, sunrise_report: AnalysisReport) -> None:
        assert sunrise_report.landau is not None
        assert sunrise_report.landau.first_type == ()
        assert sunrise_report.landau.second_type == ()
        assert sunrise_report.landau.bridge_poles == ()

    def test_bridge_poles(self, bubble_report: AnalysisReport) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|2|e|:nnn")
        landau = AnalysisReport.from_integral(fi, ["landau"]).landau
        assert landau is not None
        assert landau.bridge_poles == one_loop_bridge_poles(fi) != ()
        assert set(landau.bridge_poles) <= set(landau.analysis.landau_surfaces)
        closed = {_monic(f) for f in landau.first_type + landau.second_type + landau.bridge_poles}
        assert closed == {_monic(f) for f in landau.analysis.landau_surfaces}
        assert bubble_report.landau is not None
        assert bubble_report.landau.bridge_poles == ()

    def test_skipped_faces_carry_dimension_and_size(self, bubble: FeynmanIntegral) -> None:
        landau = AnalysisReport.from_integral(bubble, ["landau"], max_face_points=2).landau
        assert landau is not None
        points = [tuple(p) for p in bubble.newton_polytope.points]
        dimension = {frozenset(points[i] for i in idx): d for d, idx in polytope_data(points).faces}
        expected = tuple(
            (dimension[frozenset(face)], len(face), len(face) == len(points))
            for face in landau.analysis.skipped_faces
        )
        assert landau.skipped == expected
        # The edge u_1^2, u_1 u_2, u_2^2 and the whole polytope.
        assert sorted(expected) == [(1, 3, False), (2, 5, True)]

    def test_nothing_skipped_by_default(self, bubble_report: AnalysisReport) -> None:
        assert bubble_report.landau is not None
        assert bubble_report.landau.skipped == ()


class TestSchwinger:
    def test_the_f_block_admissibility(self, bubble: FeynmanIntegral) -> None:
        # The massless bubble at D = 3 and nu = (1, 2): the F~ block is the one column
        # (0, 1, 1) and beta_Cayley = (0, -3/2, -1) is not in its span, although
        # nu = (L+1) D/2.
        massless = FeynmanIntegral.from_cnickel("11e|e|:zz")
        first, second = sorted(massless.propagator_exponents)
        fi = massless.with_(
            dimension=sp.Integer(3),
            propagator_exponents={first: sp.Integer(1), second: sp.Integer(2)},
        )
        section = AnalysisReport.from_integral(fi, ["schwinger"]).schwinger
        assert section is not None and section.f_block_admissible is False
        # The massive bubble at D = 2 with unit powers: the block spans y_0 = 0.
        unit = bubble.with_(
            dimension=sp.Integer(2),
            propagator_exponents=dict.fromkeys(bubble.propagator_exponents, sp.Integer(1)),
        )
        section = AnalysisReport.from_integral(unit, ["schwinger"]).schwinger
        assert section is not None and section.f_block_admissible is True
        section = AnalysisReport.from_integral(bubble, ["schwinger"]).schwinger
        assert section is not None and section.f_block_admissible is False

    def test_sunrise_columns_match(self, sunrise_report: AnalysisReport) -> None:
        assert sunrise_report.schwinger is not None
        assert sunrise_report.schwinger.columns_match

    def test_map_sends_the_lee_pomeransky_beta_to_the_cayley_one(
        self, sunrise_report: AnalysisReport, sunrise: FeynmanIntegral
    ) -> None:
        schwinger = sunrise_report.schwinger
        assert schwinger is not None
        beta = sp.Matrix(sunrise.gkz.beta_parameters)
        assert list(schwinger.lp_to_cayley * beta) == sunrise.schwinger_gkz.beta_parameters
        assert abs(schwinger.lp_to_cayley.det()) == 1

    def test_f_block_is_the_face_subsystem(
        self, sunrise_report: AnalysisReport, sunrise: FeynmanIntegral
    ) -> None:
        schwinger = sunrise_report.schwinger
        assert schwinger is not None
        system = sunrise.schwinger_gkz
        assert schwinger.system.a_matrix == system.a_matrix
        assert schwinger.f_block.beta_parameters == [system.beta_parameters[1]] + list(
            system.beta_parameters[2:]
        )


class TestResonance:
    def test_the_facets_of_the_polytope_section(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        section = bubble_report.resonance
        assert section is not None and bubble_report.polytope is not None
        # The exponents are symbols, so the section takes unit powers and says so.
        assert (section.d0, section.powers, section.unit_powers) == (4, (1, 1), True)
        assert section.span == EpsilonSet("all")
        assert section.facets == classify_facets(bubble_report.polytope.data, [1, 1])

    def test_integer_exponents_and_d0(self, bubble: FeynmanIntegral) -> None:
        fi = bubble.with_(propagator_exponents={e: 2 - e for e in bubble.propagator_exponents})
        section = AnalysisReport.from_integral(fi, ["resonance"], d0=3).resonance
        assert section is not None
        assert (section.d0, section.powers, section.unit_powers) == (3, (1, 0), False)
        data = polytope_data(fi.newton_polytope.points)
        assert section.facets == classify_facets(data, [1, 0], 3)

    def test_d0_must_be_exact(self, bubble: FeynmanIntegral) -> None:
        with pytest.raises(ValidationError):
            AnalysisReport.from_integral(bubble, ["resonance"], d0=4.0)  # type: ignore[arg-type]

    def test_d0_from_the_dimension_of_the_integral(self, bubble: FeynmanIntegral) -> None:
        # D = 6 - 2 eps: the GKZ section has beta_0 = eps - 3, and F_F, l_F(beta) = D/2 - 2,
        # is admissible at eps = 1, not at eps = 0.
        eps = sp.Symbol("epsilon")
        powers = dict.fromkeys(bubble.propagator_exponents, sp.Integer(1))
        fi = bubble.with_(dimension=6 - 2 * eps, propagator_exponents=powers)
        report = AnalysisReport.from_integral(fi, ["gkz", "resonance"])
        section = report.resonance
        assert report.gkz is not None and report.gkz.beta[0] == eps - 3
        assert section is not None and (section.d0, section.d0_source) == (6, "dimension")
        f_f = next(r for r in section.facets if r.facet.offset == -1)
        assert f_f.admissible == EpsilonSet("point", 1)
        given = AnalysisReport.from_integral(fi, ["resonance"], d0=4).resonance
        assert given is not None and (given.d0, given.d0_source) == (4, "given")

    def test_d0_falls_back_to_four(self, bubble: FeynmanIntegral) -> None:
        for fi in (bubble, bubble.with_(dimension=4 - 2 * sp.Symbol("delta"))):
            section = AnalysisReport.from_integral(fi, ["resonance"]).resonance
            assert section is not None and (section.d0, section.d0_source) == (4, "default")

    def test_below_full_dimension(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematics="massless_on_shell")
        section = AnalysisReport.from_integral(fi, ["resonance"]).resonance
        assert section is not None
        # D/2 = nu_1 + nu_2 = 2 at eps = 0.
        assert section.span == EpsilonSet("point", 0)
        assert len(section.facets) == 2


class TestFaces:
    def test_the_faces_up_to_codimension_two(self, triangle: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(triangle, ["faces"])
        section = report.faces
        assert section is not None and section.full_dimensional
        assert section.max_codimension == FACE_CODIMENSION == 2
        assert section.faces == identify_faces(triangle)
        assert report.polytope is None
        assert ("Unidentified faces", "0") in report.summary()

    def test_below_full_dimension(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematics="massless_on_shell")
        section = AnalysisReport.from_integral(fi, ["faces"]).faces
        assert section is not None and not section.full_dimensional
        assert all(face.kind == "unidentified" for face in section.faces)


class TestSections:
    def test_default_builds_every_section_but_the_point_counts(
        self, triangle_report: AnalysisReport
    ) -> None:
        assert tuple(name for name in SECTION_NAMES if name != "torus") == DEFAULT_SECTIONS
        for name in DEFAULT_SECTIONS:
            assert getattr(triangle_report, name) is not None
        assert triangle_report.torus is None

    def test_one_section_leaves_the_others_empty(self, triangle: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(triangle, ["gkz"])
        assert report.gkz is not None
        assert report.polytope is None
        assert report.representations is None
        assert report.symmetries is None
        assert report.landau is None
        assert report.schwinger is None

    def test_the_first_three_sections_are_always_built(self, triangle: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(triangle, [])
        assert report.identity is not None
        assert report.conventions is not None
        assert report.polynomials is not None
        assert report.gkz is None

    def test_unknown_section_is_rejected(self, triangle: FeynmanIntegral) -> None:
        with pytest.raises(ValidationError, match="thermodynamics"):
            AnalysisReport.from_integral(triangle, ["gkz", "thermodynamics"])


def _triangle_with_massless_leg() -> FeynmanIntegral:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    p1 = sp.Symbol("p1^2", real=True)
    products = {k: sp.expand(sp.sympify(v).subs(p1, 0)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


class TestLimitSurfaces:
    def test_the_report_carries_them(self) -> None:
        report = AnalysisReport.from_integral(
            _triangle_with_massless_leg(), ["landau"], limits=True
        )
        assert report.landau is not None
        (limit,) = report.landau.analysis.limit_surfaces
        assert str(limit.surface) == "p2^2 - p3^2"
        rows = report.summary()
        labels = [label for label, _ in rows]
        assert labels[labels.index("Landau surfaces") + 1 :] == [
            "Limit surfaces",
            "Limit candidates",
            "Parent skipped faces",
        ]
        assert dict(rows)["Limit surfaces"] == "1"
        assert dict(rows)["Limit candidates"] == "0"
        assert dict(rows)["Parent skipped faces"] == "0"

    def test_generic_kinematics_have_no_rows(self, triangle_report: AnalysisReport) -> None:
        assert "Limit surfaces" not in dict(triangle_report.summary())

    def test_the_torus_section_alone_leaves_the_parent_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: object) -> None:
            raise AssertionError("the report analysed the parent family")

        monkeypatch.setattr(landau_module, "_generic_parent", refuse)
        report = AnalysisReport.from_integral(_triangle_with_massless_leg(), ["torus"])
        assert report.torus is not None


class TestSummary:
    def test_summary_numbers_match_sections(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        rows = dict(triangle_report.summary())
        assert rows["Loops"] == "1"
        assert rows["Propagators"] == "3"
        assert rows["External legs"] == "3"
        assert rows["Monomials of G"] == str(len(triangle.gkz.support))
        assert triangle_report.polytope is not None
        assert rows["Normalised volume"] == str(triangle_report.polytope.data.normalized_volume)
        assert rows["Polytope automorphisms"] == str(triangle.polytope_automorphisms.order)
        assert rows["Toric generators"] == str(len(triangle.toric_ideal.generators))
        assert triangle_report.landau is not None
        assert rows["Landau surfaces"] == str(len(triangle_report.landau.analysis.landau_surfaces))

    def test_lattice_rows_come_from_the_invariants(
        self, triangle_report: AnalysisReport, triangle: FeynmanIntegral
    ) -> None:
        assert triangle_report.polytope is not None
        invariants = triangle_report.polytope.invariants
        assert invariants == triangle.lattice_invariants()
        rows = dict(triangle_report.summary())
        assert rows["Lattice points"] == str(invariants.lattice_points)
        assert rows["Interior lattice points"] == str(invariants.interior_points)
        index = invariants.gorenstein_index
        assert rows["Gorenstein index"] == ("none" if index is None else str(index))
        assert rows["Normal configuration"] == ("yes" if invariants.normal else "no")
        none = dataclasses.replace(invariants, gorenstein_index=None, normal=False)
        report = dataclasses.replace(
            triangle_report,
            polytope=dataclasses.replace(triangle_report.polytope, invariants=none),
        )
        rows = dict(report.summary())
        assert (rows["Gorenstein index"], rows["Normal configuration"]) == ("none", "no")
        unknown = dataclasses.replace(invariants, idp=None, normal=None)
        report = dataclasses.replace(
            triangle_report,
            polytope=dataclasses.replace(triangle_report.polytope, invariants=unknown),
        )
        assert dict(report.summary())["Normal configuration"] == "not computed"

    def test_labels_are_in_the_documented_order(self, triangle_report: AnalysisReport) -> None:
        assert tuple(label for label, _ in triangle_report.summary()) == SUMMARY_LABELS

    def test_absent_sections_drop_their_rows(self, triangle: FeynmanIntegral) -> None:
        rows = dict(AnalysisReport.from_integral(triangle, ["gkz"]).summary())
        assert "Toric generators" in rows
        assert "Normalised volume" not in rows
        assert "Lattice points" not in rows
        assert "Normal configuration" not in rows
        assert "Polytope vertices" not in rows
        assert "Polytope automorphisms" not in rows
        assert "Landau surfaces" not in rows
        assert rows["Codimension"] == "2"


class TestTorus:
    def test_built_only_when_named(self, bubble: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(bubble, ["torus"])
        assert report.torus == bubble.torus_count()
        assert report.torus is not None and report.torus.candidate_master_count == 3
        assert report.landau is None
        assert report.polytope is None

    def test_seed_and_budget_reach_the_count(self, bubble: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(bubble, ["torus"], torus_seed=1)
        assert report.torus == bubble.torus_count(seed=1)
        with pytest.raises(ValidationError, match="max_evaluations=10"):
            AnalysisReport.from_integral(bubble, ["torus"], torus_budget=10)

    def test_one_landau_analysis_serves_both_sections(
        self, bubble: FeynmanIntegral, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # landau_analysis runs landau_analysis_from_polynomial once; count_torus_points would
        # run it again if it did not use the analysis it is given.
        calls: list[FeynmanIntegral] = []
        eliminations: list[sp.Expr] = []

        def counting(fi: FeynmanIntegral, **kwargs: int) -> object:
            calls.append(fi)
            return landau_analysis(fi, **kwargs)

        def eliminating(g: sp.Expr, *args: object, **kwargs: object) -> object:
            eliminations.append(g)
            return landau_analysis_from_polynomial(g, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(report_module, "landau_analysis", counting)
        monkeypatch.setattr(landau_module, "landau_analysis", counting)
        monkeypatch.setattr(landau_module, "landau_analysis_from_polynomial", eliminating)
        monkeypatch.setattr(point_count_module, "landau_analysis_from_polynomial", eliminating)
        report = AnalysisReport.from_integral(bubble, ["landau", "torus"])
        assert len(calls) == 1
        assert len(eliminations) == 1
        assert report.landau is not None and report.torus is not None

    def test_kinematic_constraints_are_rejected_before_the_landau_analysis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = sp.Symbol("s", real=True)
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])
        calls: list[FeynmanIntegral] = []

        def counting(fi: FeynmanIntegral, **kwargs: int) -> object:
            calls.append(fi)
            return landau_analysis(fi, **kwargs)

        monkeypatch.setattr(report_module, "landau_analysis", counting)
        with pytest.raises(ValidationError, match="kinematic_constraints"):
            AnalysisReport.from_integral(fi, ["landau", "torus"])
        assert calls == []
        assert AnalysisReport.from_integral(fi, ["landau"]).landau is not None
        assert len(calls) == 1

    def test_summary_has_a_candidate_row_only_with_the_section(
        self, bubble: FeynmanIntegral, bubble_report: AnalysisReport
    ) -> None:
        assert "Candidate master count" not in dict(bubble_report.summary())
        report = AnalysisReport.from_integral(bubble, ["polytope", "torus"])
        labels = [label for label, _ in report.summary()]
        assert labels.index("Candidate master count") == labels.index("Normalised volume") + 1
        assert dict(report.summary())["Candidate master count"] == "3"
        assert report.torus is not None
        none = dataclasses.replace(
            report.torus,
            candidate_polynomial=None,
            candidate_euler_characteristic=None,
            candidate_master_count=None,
            reason="the polynomial through the fit counts has non-integer coefficients",
        )
        rows = dict(dataclasses.replace(report, torus=none).summary())
        assert rows["Candidate master count"] == "none"


class TestScaleless:
    @pytest.mark.parametrize(
        ("cnickel", "scaleless"),
        [("12e|2e|e|:nnn", "no"), ("012e|2e|e|:znnn", "yes"), ("1ee|1|:zn", "no")],
    )
    def test_summary_row(self, cnickel: str, scaleless: str) -> None:
        # The row follows the codimension whatever sections are built.
        report = AnalysisReport.from_integral(FeynmanIntegral.from_cnickel(cnickel), ["gkz"])
        labels = [label for label, _ in report.summary()]
        assert labels[labels.index("Codimension") + 1] == "Scaleless"
        assert dict(report.summary())["Scaleless"] == scaleless
        assert report.polynomials.scaleless is (scaleless == "yes")
