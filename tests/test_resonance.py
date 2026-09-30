"""Tests for feynkit.resonance: which facets are resonant or admissible, and reducibility.

The oracles are the resonance statements of Britto, Grimm and Hoefnagels,
arXiv:2606.09978 (BGH26 below, page numbers of v1), with D = 4 - 2 eps and
unit powers unless a test says otherwise, and small configurations worked
by hand. BGH26 write the parameter as nu = (D/2, nu_1, ..., nu_N), which is
-beta in feynkit's convention; their functionals L_F are the functionals
here, and L_F(nu) = -l_F(beta).
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.polytope import PolytopeData, polytope_data
from feynkit.resonance import (
    EpsilonSet,
    FacetResonance,
    ParameterForm,
    admissible,
    classify_configuration,
    classify_facets,
    lee_pomeransky_beta,
    span_epsilons,
)

F = Fraction

# The ten graphs on which BGH26's statements are checked.
GRAPHS = {
    "bubble": "11e|e|:nn",
    "bubble-m2-0": "11e|e|:nz",
    "bubble-massless": "11e|e|:zz",
    "triangle": "12e|2e|e|:nnn",
    "triangle-massless": "12e|2e|e|:zzz",
    "box": "12e|3e|3e|e|:nnnn",
    "sunrise": "111e|e|:nnn",
    "sunrise-m3-0": "111e|e|:nnz",
    "sunrise-m2-m3-0": "111e|e|:nzz",
    "banana": "1111e|e|:nnnn",
}


def _data(cnickel: str) -> PolytopeData:
    return polytope_data(FeynmanIntegral.from_cnickel(cnickel).newton_polytope.points)


def _classify(cnickel: str, d0: int | Fraction = 4) -> dict[tuple[Fraction, ...], FacetResonance]:
    """The facets of Newt(G) with unit powers, keyed by their functionals."""
    data = _data(cnickel)
    records = classify_facets(data, [1] * data.ambient_dimension, d0)
    return {r.functional: r for r in records}


def _functional(*entries: int) -> tuple[Fraction, ...]:
    return tuple(F(x) for x in entries)


def _edge(n: int, e: int) -> tuple[Fraction, ...]:
    """L_{F_e} = delta_e, which reads row e of A (BGH26 Eq. 21, p. 11)."""
    return _functional(*(1 if k == e else 0 for k in range(n + 1)))


def _symanzik(n: int, loops: int) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """L_F = (-L, 1, ..., 1) and L_U = (L + 1, -1, ..., -1) (BGH26 Sec. 8.1, p. 46)."""
    return _functional(-loops, *[1] * n), _functional(loops + 1, *[-1] * n)


SAMPLES = [F(0), F(1, 2), F(1, 3), F(1, 4), F(-1), F(3, 2), F(2, 3)]


def _is_integer(x: Fraction) -> bool:
    return x.denominator == 1


# --- BGH26 -------------------------------------------------------------------


class TestBubble:
    def test_four_facets_with_the_functionals_of_eq_68(self) -> None:
        # BGH26 Eqs. 63, 67-68, pp. 21-22.
        found = _classify(GRAPHS["bubble"])
        lf, lu = _symanzik(2, 1)
        assert set(found) == {_edge(2, 1), _edge(2, 2), lf, lu}

    def test_resonance_as_bgh26_state_it(self) -> None:
        # p. 22: F_F is resonant only if D/2 is an integer, F_U only if D is, and the edge
        # facets for any D.
        found = _classify(GRAPHS["bubble"])
        lf, lu = _symanzik(2, 1)
        for eps in SAMPLES:
            d = 4 - 2 * eps
            assert (eps in found[lf].resonant) is _is_integer(d / 2)
            assert (eps in found[lu].resonant) is _is_integer(d)
            assert eps in found[_edge(2, 1)].resonant and eps in found[_edge(2, 2)].resonant
        assert found[_edge(2, 1)].resonant == EpsilonSet("all")
        assert found[lf].resonant == EpsilonSet("progression", F(0), F(1))
        assert found[lu].resonant == EpsilonSet("progression", F(1), F(1, 2))

    def test_forms_in_d_and_nu(self) -> None:
        found = _classify(GRAPHS["bubble"])
        lf, lu = _symanzik(2, 1)
        # l_F(beta) = -L_F(nu) with nu = (D/2, nu_1, nu_2).
        assert found[lf].form == ParameterForm(F(1, 2), (F(-1), F(-1)))
        assert found[lu].form == ParameterForm(F(-1), (F(1), F(1)))
        assert found[_edge(2, 1)].form == ParameterForm(F(0), (F(-1), F(0)))

    def test_one_massless_line(self) -> None:
        # BGH26 Eq. 105, p. 29: F_1 is no longer a facet; the new facet F_{2,(1,2)} is not
        # resonant (for generic D); F_2 is still a resonant facet.
        found = _classify(GRAPHS["bubble-m2-0"])
        assert _edge(2, 1) not in found
        assert found[_edge(2, 2)].kind == "all"
        new = found[_functional(1, 0, -1)]  # x_2 <= 1, form nu_2 - D/2
        assert new.kind == "progression"
        assert new.form == ParameterForm(F(-1, 2), (F(0), F(1)))


class TestOneLoop:
    @pytest.mark.parametrize(("name", "n"), [("triangle", 3), ("box", 4)])
    def test_edge_facets_and_the_symanzik_facets(self, name: str, n: int) -> None:
        # Triangle: BGH26 Eqs. 119-120, pp. 31-32. Box, and any massive n-gon: Sec. 6.2, p. 37.
        found = _classify(GRAPHS[name])
        lf, lu = _symanzik(n, 1)
        assert set(found) == {*(_edge(n, e) for e in range(1, n + 1)), lf, lu}
        for e in range(1, n + 1):
            assert found[_edge(n, e)].kind == "all"
        for eps in SAMPLES:
            d = 4 - 2 * eps
            assert (eps in found[lf].resonant) is _is_integer(d / 2)
            assert (eps in found[lu].resonant) is _is_integer(d)


class TestSunriseAndBanana:
    def test_three_masses(self) -> None:
        # BGH26 Eq. 152, p. 41: assuming integer powers, all edge faces are resonant.
        found = _classify(GRAPHS["sunrise"])
        assert len(found) == 8
        for e in (1, 2, 3):
            assert found[_edge(3, e)].kind == "all"

    def test_one_massless_line(self) -> None:
        # BGH26 p. 43: F_1 and F_2 are no longer facets; F_3 is still a resonant facet.
        found = _classify(GRAPHS["sunrise-m3-0"])
        assert _edge(3, 1) not in found and _edge(3, 2) not in found
        assert found[_edge(3, 3)].kind == "all"

    def test_two_massless_lines(self) -> None:
        # BGH26 p. 44: five facets, none of which are resonant (for generic D), and no edge
        # facet.
        found = _classify(GRAPHS["sunrise-m2-m3-0"])
        assert len(found) == 5
        assert all(r.kind == "progression" for r in found.values())
        assert not any(_edge(3, e) in found for e in (1, 2, 3))

    def test_banana_edges(self) -> None:
        # BGH26 pp. 44-45: one reduction operator per edge; its edge faces are resonant
        # facets.
        found = _classify(GRAPHS["banana"])
        assert len(found) == 16
        for e in (1, 2, 3, 4):
            assert found[_edge(4, e)].kind == "all"

    @pytest.mark.parametrize(
        ("name", "n", "loops"), [("bubble", 2, 1), ("sunrise", 3, 2), ("banana", 4, 3)]
    )
    def test_symanzik_facets_at_any_loop_order(self, name: str, n: int, loops: int) -> None:
        # BGH26 p. 46: F_F is resonant if L D/2 - nu is an integer, F_U if (L + 1) D/2 - nu
        # is, nu the sum of the powers.
        found = _classify(GRAPHS[name])
        lf, lu = _symanzik(n, loops)
        for eps in SAMPLES:
            d = 4 - 2 * eps
            assert (eps in found[lf].resonant) is _is_integer(loops * d / 2 - n)
            assert (eps in found[lu].resonant) is _is_integer((loops + 1) * d / 2 - n)


class TestAdmissibleFacets:
    def test_edge_facets_are_admissible_exactly_at_zero_power(self) -> None:
        # BGH26 Eq. 53, p. 18: the contraction appears when nu_e = 0.
        data = _data(GRAPHS["triangle"])
        for nu, expected in (([1, 1, 1], "never"), ([0, 1, 1], "all")):
            records = classify_facets(data, nu)
            edge = next(r for r in records if r.functional == _edge(3, 1))
            assert edge.admissible.kind == expected

    def test_f_u_is_admissible_where_nu_equals_l_plus_one_d_over_two(self) -> None:
        # BGH26 p. 46: the span of F_U is where (L + 1) D/2 - nu vanishes.
        for name, n, loops in (("bubble", 2, 1), ("sunrise", 3, 2), ("banana", 4, 3)):
            lu = _symanzik(n, loops)[1]
            record = _classify(GRAPHS[name])[lu]
            # (L + 1)(4 - 2 eps)/2 = n.
            point = F(2) - F(n, loops + 1)
            assert record.admissible == EpsilonSet("point", point)
            assert record.resonant.offset == point


class TestReducibility:
    def test_non_pyramids_are_reducible_wherever_resonant(self) -> None:
        # BGH26 p. 12 say the non-pyramid condition holds for all their faces.
        for name, cnickel in GRAPHS.items():
            if name == "bubble-massless":
                continue
            for record in _classify(cnickel).values():
                assert record.columns_off >= 2 and not record.pyramid, name
                assert record.reducible is True, name

    def test_the_massless_bubble_is_a_pyramid_over_every_facet(self) -> None:
        for record in _classify(GRAPHS["bubble-massless"]).values():
            assert record.columns_off == 1 and record.pyramid
            assert record.reducible is None


# --- dimensions and epsilon ----------------------------------------------------


class TestD0:
    def test_every_facet_is_resonant_at_zero_in_four_dimensions(self) -> None:
        for cnickel in GRAPHS.values():
            assert all(r.resonant_at_zero for r in _classify(cnickel).values())

    def test_three_dimensions(self) -> None:
        # With g_F = 1, eps = 0 is resonant exactly when b D_0/2 is an integer: at D_0 = 3,
        # F_F of the bubble (b = -1) is not and F_U (b = 2) is.
        found = _classify(GRAPHS["bubble"], d0=3)
        lf, lu = _symanzik(2, 1)
        assert not found[lf].resonant_at_zero
        assert found[lu].resonant_at_zero
        assert found[lf].resonant == EpsilonSet("progression", F(-1, 2), F(1))
        assert found[_edge(2, 1)].resonant_at_zero

    def test_half_integer_d0(self) -> None:
        found = _classify(GRAPHS["banana"], d0=F(7, 2))
        lu = _symanzik(4, 3)[1]  # b = 4
        assert found[lu].resonant_at_zero
        assert not found[_symanzik(4, 3)[0]].resonant_at_zero  # b = -3

    def test_d0_must_be_exact(self) -> None:
        data = _data(GRAPHS["bubble"])
        with pytest.raises(ValidationError):
            classify_facets(data, [1, 1], 4.0)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            classify_facets(data, [1, F(1, 2)])  # type: ignore[list-item]
        with pytest.raises(ValidationError):
            classify_facets(data, [1, 1, 1])


class TestEpsilonSet:
    def test_window(self) -> None:
        progression = EpsilonSet("progression", F(1), F(1, 2))
        assert progression.window(-1, 1) == (F(-1), F(-1, 2), F(0), F(1, 2), F(1))
        assert progression.window(F(1, 3), F(2, 3)) == (F(1, 2),)
        assert EpsilonSet("point", F(2)).window(-1, 1) == ()
        assert EpsilonSet("point", F(0)).window(-1, 1) == (F(0),)
        assert EpsilonSet("never").window(-1, 1) == ()
        with pytest.raises(ValidationError):
            EpsilonSet("all").window(-1, 1)

    def test_membership(self) -> None:
        assert F(7, 3) in EpsilonSet("progression", F(1, 3), F(1))
        assert F(7, 4) not in EpsilonSet("progression", F(1, 3), F(1))
        assert 5 in EpsilonSet("all") and 5 not in EpsilonSet("never")

    def test_invalid(self) -> None:
        with pytest.raises(ValidationError):
            EpsilonSet("progression", F(0), F(0))
        with pytest.raises(ValidationError):
            EpsilonSet("point")
        with pytest.raises(ValidationError):
            EpsilonSet("some")  # type: ignore[arg-type]


# --- configurations that no Feynman graph gives ------------------------------


class TestLatticeIndex:
    """(0, 0), (2, 0), (0, 1): the facets y >= 0, x >= 0 and x + 2y <= 2 have g_F = 1, 2, 2."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls) -> PolytopeData:
        return polytope_data([(0, 0), (2, 0), (0, 1)])

    def _by_normal(
        self, data: PolytopeData, nu: list[int]
    ) -> dict[tuple[int, ...], FacetResonance]:
        return {r.facet.normal: r for r in classify_facets(data, nu)}

    def test_odd_power_is_never_resonant(self, data: PolytopeData) -> None:
        found = self._by_normal(data, [1, 0])
        assert found[(-1, 0)].lattice_index == 2
        assert found[(-1, 0)].functional == _functional(0, F(1, 2), 0)
        assert found[(-1, 0)].kind == "never"
        assert found[(-1, 0)].admissible.kind == "never"
        assert found[(0, -1)].kind == "all"

    def test_even_power_is_always_resonant(self, data: PolytopeData) -> None:
        found = self._by_normal(data, [2, 0])
        assert found[(-1, 0)].kind == "all"
        assert found[(-1, 0)].admissible.kind == "never"
        assert self._by_normal(data, [0, 0])[(-1, 0)].admissible.kind == "all"

    def test_period_g_over_b(self, data: PolytopeData) -> None:
        # l(beta) = eps + nu_1/2 + nu_2 - 2: period g_F/|b| = 1, offset D_0/2 - m . nu/b.
        record = self._by_normal(data, [1, 0])[(1, 2)]
        assert record.lattice_index == 2
        assert record.resonant == EpsilonSet("progression", F(3, 2), F(1))
        assert record.admissible == EpsilonSet("point", F(3, 2))

    def test_a_simplex_is_a_pyramid_over_each_facet(self, data: PolytopeData) -> None:
        assert all(r.pyramid and r.reducible is None for r in classify_facets(data, [0, 0]))


class TestLowerDimensional:
    def test_scaleless_bubble(self) -> None:
        # The massless bubble on shell: G = u_1 + u_2, a segment on x_1 + x_2 = 1. beta lies
        # in the span of A only where D/2 = nu_1 + nu_2.
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematics="massless_on_shell")
        assert fi.is_scaleless
        data = polytope_data(fi.newton_polytope.points)
        assert span_epsilons(data, [1, 2]) == EpsilonSet("point", F(-1))
        records = classify_facets(data, [1, 2])
        assert len(records) == 2
        for record in records:
            assert record.resonant == EpsilonSet("point", F(-1))
            assert record.admissible.kind == "never"
            assert record.reducible is None
        at_zero = classify_facets(data, [0, 1], d0=2)
        # eps = 0 is where D/2 = 1 = nu_1 + nu_2, and there the facet x_1 >= 0, the edge
        # face of u_1 with nu_1 = 0, is admissible.
        assert {r.admissible.kind for r in at_zero} == {"point", "never"}

    def test_the_span_condition_without_d(self) -> None:
        # 1ee|1|:zn: every equation of the affine hull passes through the origin, so beta lies
        # in the span of A for every eps or for none, as nu_1 = 0 or not.
        data = _data("1ee|1|:zn")
        assert span_epsilons(data, [1, 1]) == EpsilonSet("never")
        assert all(r.kind == "never" for r in classify_facets(data, [1, 1]))
        assert span_epsilons(data, [0, 1]) == EpsilonSet("all")
        assert {r.kind for r in classify_facets(data, [0, 1])} == {"progression"}

    def test_full_dimensional_span(self) -> None:
        assert span_epsilons(_data(GRAPHS["bubble"]), [1, 1]) == EpsilonSet("all")


def test_exported_from_the_package() -> None:
    import feynkit

    assert (feynkit.EpsilonSet, feynkit.FacetResonance) == (EpsilonSet, FacetResonance)
    assert feynkit.classify_facets is classify_facets


# --- the general configuration -------------------------------------------------


class TestConfiguration:
    def test_agrees_with_the_newton_polytope(self) -> None:
        for cnickel in (GRAPHS["bubble-m2-0"], GRAPHS["sunrise"]):
            fi = FeynmanIntegral.from_cnickel(cnickel)
            n = len(fi.symanzik.lp_parameters)
            direct = classify_configuration(fi.gkz.a_matrix, lee_pomeransky_beta(n), [1] * n)
            points = [
                tuple(int(x) for x in fi.gkz.a_matrix[1:, j]) for j in range(fi.gkz.a_matrix.cols)
            ]
            via_data = classify_facets(polytope_data(points), [1] * n)
            key = lambda r: r.facet.point_indices  # noqa: E731
            for a, b in zip(sorted(direct, key=key), sorted(via_data, key=key), strict=True):
                assert a.facet.point_indices == b.facet.point_indices
                assert a.functional == b.functional
                assert (a.form, a.resonant, a.admissible) == (b.form, b.resonant, b.admissible)
                assert (a.columns_off, a.reducible) == (b.columns_off, b.reducible)

    def test_inhomogeneous_matrix(self) -> None:
        with pytest.raises(ValidationError):
            classify_configuration([[1, 0, 1], [0, 1, 1]], lee_pomeransky_beta(1), [1])

    def test_one_form_per_row(self) -> None:
        with pytest.raises(ValidationError):
            classify_configuration([[1, 1], [0, 1]], lee_pomeransky_beta(2), [1, 1])


class TestAdmissible:
    A = [[1, 1, 1, 1, 1], [1, 0, 1, 2, 0], [0, 1, 1, 0, 2]]  # BGH26 Eq. 63

    def test_numbers(self) -> None:
        # F_1 = {a_2, a_{2,2}}, columns 1 and 4, spans y_1 = 0.
        assert admissible(self.A, [1, 4], [-2, 0, -1])
        assert not admissible(self.A, [1, 4], [-2, -1, -1])
        assert admissible(self.A, [1, 4], [F(-3, 2), 0, F(1, 3)])

    def test_symbols_must_vanish_identically(self) -> None:
        d, nu1, nu2 = sp.symbols("D nu_1 nu_2")
        beta = [-d / 2, -nu1, -nu2]
        # F_U = {a_{1,1}, a_{1,2}, a_{2,2}}: the span of F_U is where D = nu_1 + nu_2.
        assert not admissible(self.A, [2, 3, 4], beta)
        assert admissible(self.A, [2, 3, 4], [-(nu1 + nu2) / 2, -nu1, -nu2])
        assert admissible(self.A, list(range(5)), beta)

    def test_empty_face(self) -> None:
        assert admissible(self.A, [], [0, 0, 0])
        assert not admissible(self.A, [], [1, 0, 0])

    def test_bad_input(self) -> None:
        with pytest.raises(ValidationError):
            admissible(self.A, [5], [0, 0, 0])
        with pytest.raises(ValidationError):
            admissible(self.A, [0], [0, 0])
