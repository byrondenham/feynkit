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

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.core.exceptions import ValidationError
from feynkit.io.report import SECTION_NAMES, AnalysisReport
from feynkit.landau import landau_analysis, one_loop_landau_surfaces_by_type
from feynkit.systems.monomial import extract_monomial_support

SUMMARY_LABELS = (
    "Loops",
    "Propagators",
    "External legs",
    "Monomials of F",
    "Monomials of G",
    "Independent invariants",
    "Codimension",
    "Polytope vertices",
    "Normalised volume",
    "Polytope automorphisms",
    "Toric generators",
    "Landau surfaces",
)


def _monic(expr: sp.Expr) -> sp.Expr:
    """Normalise a polynomial factor up to a constant, for set comparison."""
    expr = sp.expand(expr)
    return sp.Poly(expr, *sorted(expr.free_symbols, key=str)).monic().as_expr()


def _kinematic_symbols(fi: FeynmanIntegral) -> set[sp.Symbol]:
    """Masses and invariants of the integral, without the energy scale."""
    return (
        fi.symanzik.f.free_symbols - set(fi.symanzik.schwinger_parameters) - {fi.graph.energy_scale}
    )


def _factor_set(factors: tuple[sp.Expr, ...], kinematic: set[sp.Symbol]) -> set[sp.Expr]:
    """The given factors that carry kinematics, normalised up to a constant."""
    return {_monic(f) for f in factors if f.free_symbols & kinematic}


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
        assert p.codimension == p.monomials_g - 3 - 1
        assert p.f_scale_power == 2
        assert sp.expand(p.f_numerator / triangle.graph.energy_scale**2 - triangle.symanzik.f) == 0

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


class TestLandau:
    def test_matches_landau_analysis(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        direct = landau_analysis(bubble)
        assert bubble_report.landau is not None
        assert bubble_report.landau.analysis.landau_surfaces == direct.landau_surfaces

    def test_types_cover_the_kinematic_surfaces(
        self, bubble_report: AnalysisReport, bubble: FeynmanIntegral
    ) -> None:
        """The closed form and the face computation give the same surfaces.

        Both are compared as sets of irreducible factors normalised up to a
        constant, and restricted to the factors carrying kinematics: the face
        computation also reports the energy scale, which is not a surface.
        """
        landau = bubble_report.landau
        assert landau is not None
        kinematic = _kinematic_symbols(bubble)
        assert _factor_set(landau.first_type + landau.second_type, kinematic) == _factor_set(
            landau.analysis.landau_surfaces, kinematic
        )

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


class TestSchwinger:
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


class TestSections:
    def test_default_builds_every_section(self, triangle_report: AnalysisReport) -> None:
        for name in SECTION_NAMES:
            assert getattr(triangle_report, name) is not None

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

    def test_labels_are_in_the_documented_order(self, triangle_report: AnalysisReport) -> None:
        assert tuple(label for label, _ in triangle_report.summary()) == SUMMARY_LABELS

    def test_absent_sections_drop_their_rows(self, triangle: FeynmanIntegral) -> None:
        rows = dict(AnalysisReport.from_integral(triangle, ["gkz"]).summary())
        assert "Toric generators" in rows
        assert "Normalised volume" not in rows
        assert "Polytope vertices" not in rows
        assert "Polytope automorphisms" not in rows
        assert "Landau surfaces" not in rows
        assert rows["Codimension"] == "2"
