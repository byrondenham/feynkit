"""Kinematic axes and classes, and imposing a class by substitution."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import pytest
import sympy as sp

from feynkit import FeynmanIntegral, Graph, ValidationError
from feynkit.kinematics import (
    CLASS_OF_AXES,
    IMPOSABLE_CLASSES,
    KINEMATIC_CLASSES,
    impose_kinematics,
    kinematic_axes,
    kinematic_class,
)

BOX = "12e|3e|3e|e|"
MU = sp.Symbol("mu", positive=True, real=True)
M = sp.Symbol("M", real=True)
S = sp.Symbol("s", real=True)
S12 = sp.Symbol("s12", real=True)
S23 = sp.Symbol("s23", real=True)


def p2(i: int) -> sp.Symbol:
    """The invariant p_i^2 of standard_invariants."""
    return sp.Symbol(f"p{i}^2", real=True)


def mass(code: str | int) -> sp.Symbol:
    return sp.Symbol(f"m_{code}", nonnegative=True, real=True)


def substituted(fi: FeynmanIntegral, values: Mapping[sp.Symbol, sp.Expr]) -> FeynmanIntegral:
    """fi with the invariants replaced in every momentum product."""
    products = {k: sp.expand(v.subs(values)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


def with_masses(fi: FeynmanIntegral, masses: list[sp.Expr]) -> FeynmanIntegral:
    """fi with the internal edges given these masses, in internal-edge order."""
    graph = fi.graph
    given = dict(zip([e.idx for e in graph.get_internal_edges()], masses, strict=True))
    edges = [dataclasses.replace(e, mass=given[e.idx]) if e.is_internal else e for e in graph.edges]
    return fi.with_(graph=Graph(graph.internal_vertices, graph.external_legs, edges))


ON_SHELL = {p2(i): 0 for i in range(1, 5)}
EXTERNAL = {
    "off_shell": {},
    "on_shell": ON_SHELL,
    "equal": {p2(i): M for i in range(1, 5)},
    "other": {p2(4): 0},
}
INTERNAL = {"zero": "zzzz", "equal": "aaaa", "generic": "nnnn", "other": "aann"}


class TestAxes:
    @pytest.mark.parametrize("internal", list(INTERNAL))
    @pytest.mark.parametrize("external", list(EXTERNAL))
    def test_every_cell_of_the_table(self, internal: str, external: str) -> None:
        fi = substituted(
            FeynmanIntegral.from_cnickel(BOX + ":" + INTERNAL[internal]), EXTERNAL[external]
        )
        assert kinematic_axes(fi) == (internal, external)
        assert kinematic_class(fi) == CLASS_OF_AXES.get((internal, external), "other")

    def test_the_named_cells(self) -> None:
        assert dict(CLASS_OF_AXES) == {
            ("zero", "off_shell"): "massless_off_shell",
            ("zero", "on_shell"): "massless_on_shell",
            ("equal", "off_shell"): "equal_masses",
            ("generic", "off_shell"): "generic",
        }
        assert KINEMATIC_CLASSES == (
            "generic",
            "massless_off_shell",
            "massless_on_shell",
            "equal_masses",
            "other",
        )
        assert KINEMATIC_CLASSES[:4] == IMPOSABLE_CLASSES

    @pytest.mark.parametrize(
        ("cnickel", "values", "axes"),
        [
            # Equal external masses on the triangle; s_12 = 0 on the box.
            ("12e|2e|e|:zzz", {p2(i): M for i in range(1, 4)}, ("zero", "equal")),
            (BOX + ":zzzz", {S12: 0}, ("zero", "other")),
            # On-shell legs with massive lines: in no class.
            (BOX + ":nnnn", ON_SHELL, ("generic", "on_shell")),
            # p_1^2 = m_1^2 is not linear in the invariants.
            ("12e|2e|e|:nzz", {p2(1): mass(1) ** 2}, ("generic", "other")),
            ("11e|e|:zz", {S: 0}, ("zero", "on_shell")),
            ("11e|e|:nn", {}, ("generic", "off_shell")),
            ("12e|2e|e|:aab", {}, ("other", "off_shell")),
            ("12e|2e|e|:aaz", {}, ("other", "off_shell")),
            ("12e|2e|e|:sss", {}, ("equal", "off_shell")),
            ("12e|2e|e|:nzz", {}, ("generic", "off_shell")),
        ],
    )
    def test_axes(
        self, cnickel: str, values: Mapping[sp.Symbol, sp.Expr], axes: tuple[str, str]
    ) -> None:
        fi = substituted(FeynmanIntegral.from_cnickel(cnickel), values)
        assert kinematic_axes(fi) == axes

    def test_a_numerical_mass_and_a_float_zero(self) -> None:
        triangle = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        for masses in (
            [sp.Integer(3), mass(2), mass(3)],
            [sp.Float(0.0), sp.Integer(0), sp.Integer(0)],
            [mass(1) + mass(2), mass(2), mass(3)],
        ):
            assert kinematic_axes(with_masses(triangle, masses))[0] == "other"
        assert kinematic_axes(with_masses(triangle, [mass(1), mass(1), mass(1)]))[0] == "equal"

    def test_vacuum_graphs_and_a_single_massive_line(self) -> None:
        tadpole = FeynmanIntegral.from_cnickel("0|:n", use_mandelstam=False)
        assert kinematic_axes(tadpole) == ("generic", "off_shell")
        assert kinematic_class(tadpole) == "generic"
        massless = FeynmanIntegral.from_cnickel("0|:z", use_mandelstam=False)
        assert kinematic_class(massless) == "massless_off_shell"

    def test_dot_products(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz", use_mandelstam=False)
        assert kinematic_class(box) == "massless_off_shell"
        triangle = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz", use_mandelstam=False)
        zero = {key: sp.Integer(0) for key in triangle.momentum_products}
        assert kinematic_class(triangle.with_(momentum_products=zero)) == "massless_on_shell"

    def test_mass_symbols_in_a_product(self) -> None:
        # Linear in m_1, but a mass is not an invariant.
        fi = substituted(FeynmanIntegral.from_cnickel("12e|2e|e|:nzz"), {p2(1): mass(1)})
        assert kinematic_axes(fi) == ("generic", "other")


class TestNeverRaises:
    @pytest.mark.parametrize(
        "values",
        [
            {S12: 1 / S23},
            {p2(1): sp.sqrt(S23)},
            {p2(1): MU},
            {p2(1): MU**2},
            {p2(1): sp.Float(0.5) * S23},
            {p2(1): sp.sqrt(2) * S23},
            {p2(1): sp.Float(2.0)},
        ],
        ids=["inverse", "root", "mu", "mu-squared", "float", "irrational", "float-constant"],
    )
    def test_special_products_are_other(self, values: Mapping[sp.Symbol, sp.Expr]) -> None:
        fi = substituted(FeynmanIntegral.from_cnickel(BOX + ":zzzz"), values)
        assert kinematic_axes(fi) == ("zero", "other")
        assert kinematic_class(fi) == "other"

    def test_kinematic_constraints_are_other(self) -> None:
        fi = FeynmanIntegral.from_cnickel(BOX + ":zzzz", kinematic_constraints=[S12 - S23])
        assert kinematic_axes(fi) == ("zero", "other")

    def test_reversed_keys(self) -> None:
        box = substituted(FeynmanIntegral.from_cnickel(BOX + ":zzzz"), ON_SHELL)
        reversed_keys = {(j, i): v for (i, j), v in box.momentum_products.items()}
        assert kinematic_axes(box.with_(momentum_products=reversed_keys)) == ("zero", "on_shell")

    def test_a_missing_product_counts_as_zero(self) -> None:
        bubble = FeynmanIntegral.from_cnickel("11e|e|:zz")
        assert kinematic_axes(bubble.with_(momentum_products={})) == ("zero", "on_shell")


class TestImpose:
    @pytest.mark.parametrize(
        ("cnickel", "classes"),
        [
            (BOX + ":zzzz", ("massless_off_shell", "massless_on_shell")),
            (BOX + ":nnnn", ("generic", "equal_masses")),
            ("12e|2e|e|:zzz", ("massless_off_shell", "massless_on_shell")),
            ("12e|2e|e|:nnn", ("generic", "equal_masses")),
            ("12e|2e|e|:aab", ("equal_masses",)),
            ("11e|e|:zz", ("massless_off_shell", "massless_on_shell")),
            ("12ee|22e|e|:zzzz", ("massless_off_shell", "massless_on_shell")),
            ("12e|23|3|e|:nnnnn", ("generic", "equal_masses")),
        ],
    )
    def test_the_class_imposed_is_the_class_derived(
        self, cnickel: str, classes: tuple[str, ...]
    ) -> None:
        fi = FeynmanIntegral.from_cnickel(cnickel)
        for name in classes:
            assert kinematic_class(impose_kinematics(fi, name)) == name

    def test_the_integral_itself_when_the_class_holds(self) -> None:
        for cnickel in (BOX + ":zzzz", BOX + ":nnnn", "12e|2e|e|:sss"):
            fi = FeynmanIntegral.from_cnickel(cnickel)
            assert impose_kinematics(fi, kinematic_class(fi)) is fi
        sss = impose_kinematics(FeynmanIntegral.from_cnickel("12e|2e|e|:sss"), "equal_masses")
        assert {e.get_mass() for e in sss.graph.get_internal_edges()} == {mass("s")}

    def test_on_shell_box_products(self) -> None:
        box = impose_kinematics(FeynmanIntegral.from_cnickel(BOX + ":zzzz"), "massless_on_shell")
        assert box.momentum_products == {
            (1, 2): S12 / 2,
            (3, 4): S12 / 2,
            (1, 4): S23 / 2,
            (2, 3): S23 / 2,
            (1, 3): -(S12 + S23) / 2,
            (2, 4): -(S12 + S23) / 2,
        }
        assert len(box.newton_polytope.points) == 6

    def test_on_shell_keeps_the_keys(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz")
        reversed_keys = {(j, i): v for (i, j), v in box.momentum_products.items()}
        on_shell = impose_kinematics(
            box.with_(momentum_products=reversed_keys), "massless_on_shell"
        )
        assert set(on_shell.momentum_products) == set(reversed_keys)
        assert kinematic_class(on_shell) == "massless_on_shell"

    def test_equal_masses_from_distinct_and_shared_codes(self) -> None:
        nnn = impose_kinematics(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), "equal_masses")
        aaa = FeynmanIntegral.from_cnickel("12e|2e|e|:aaa")
        assert nnn.cnickel == aaa.cnickel == "12e|2e|e|:aaa"
        assert [e.get_mass() for e in nnn.graph.get_internal_edges()] == [mass("a")] * 3
        assert nnn.symanzik.g == aaa.symanzik.g
        aab = impose_kinematics(FeynmanIntegral.from_cnickel("12e|2e|e|:aab"), "equal_masses")
        assert aab.cnickel == "12e|2e|e|:aaa"
        numerical = with_masses(aaa, [sp.Integer(3), sp.Integer(3), mass("a")])
        assert impose_kinematics(numerical, "equal_masses").cnickel == "12e|2e|e|:aaa"

    def test_equal_masses_keeps_the_energy_scale_and_products(self) -> None:
        scale = sp.Symbol("Q", positive=True)
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        graph = Graph(fi.graph.internal_vertices, fi.graph.external_legs, fi.graph.edges, scale)
        equal = impose_kinematics(fi.with_(graph=graph), "equal_masses")
        assert equal.graph.energy_scale == scale
        assert equal.momentum_products == fi.momentum_products

    def test_derived_again_after_with(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz")
        on_shell = impose_kinematics(box, "massless_on_shell")
        undone = on_shell.with_(momentum_products=box.momentum_products)
        assert kinematic_class(undone) == "massless_off_shell"
        assert kinematic_class(on_shell.with_(dimension=sp.Integer(4))) == "massless_on_shell"

    @pytest.mark.parametrize("name", ["on_shell", "other", "", "Generic"])
    def test_unknown_names_and_other(self, name: str) -> None:
        fi = FeynmanIntegral.from_cnickel(BOX + ":zzzz")
        with pytest.raises(ValidationError) as caught:
            impose_kinematics(fi, name)
        message = str(caught.value)
        assert repr(name) in message
        assert "generic, massless_off_shell, massless_on_shell, equal_masses" in message

    @pytest.mark.parametrize(
        ("cnickel", "name", "fragments"),
        [
            (BOX + ":zzzz", "generic", ("makes no substitution", "massless_off_shell")),
            (BOX + ":nnnn", "massless_off_shell", ("makes no substitution", "generic")),
            (BOX + ":nzzn", "massless_on_shell", ("propagators 1, 4 are massive",)),
            ("12e|2e|e|:nzz", "equal_masses", ("propagators 2, 3 are massless",)),
        ],
    )
    def test_refusals(self, cnickel: str, name: str, fragments: tuple[str, ...]) -> None:
        with pytest.raises(ValidationError) as caught:
            impose_kinematics(FeynmanIntegral.from_cnickel(cnickel), name)
        for fragment in fragments:
            assert fragment in str(caught.value)

    def test_on_shell_needs_two_legs(self) -> None:
        tadpole = FeynmanIntegral.from_cnickel("0|:z", use_mandelstam=False)
        with pytest.raises(ValidationError, match="fewer than two external legs"):
            impose_kinematics(tadpole, "massless_on_shell")

    def test_on_shell_needs_invariants(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz", use_mandelstam=False)
        with pytest.raises(ValidationError, match="use_mandelstam=True"):
            impose_kinematics(box, "massless_on_shell")

    def test_a_result_outside_the_class_is_refused(self) -> None:
        # Equal masses on a box with legs on shell: axes (equal, on_shell), class other.
        box = substituted(FeynmanIntegral.from_cnickel(BOX + ":nnnn"), ON_SHELL)
        with pytest.raises(ValidationError, match="internal axis equal and external axis on_shell"):
            impose_kinematics(box, "equal_masses")
        constrained = FeynmanIntegral.from_cnickel(BOX + ":zzzz", kinematic_constraints=[S12])
        with pytest.raises(ValidationError, match="external axis other"):
            impose_kinematics(constrained, "massless_on_shell")


class TestFacade:
    def test_the_properties_are_the_functions_cached(self) -> None:
        fi = substituted(FeynmanIntegral.from_cnickel(BOX + ":nnnn"), ON_SHELL)
        assert fi.kinematic_axes == kinematic_axes(fi) == ("generic", "on_shell")
        assert fi.kinematic_class == kinematic_class(fi) == "other"
        assert fi.kinematic_axes is fi.kinematic_axes

    def test_with_kinematics(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz")
        assert box.with_kinematics("massless_off_shell") is box
        on_shell = box.with_kinematics("massless_on_shell")
        assert on_shell.kinematic_class == "massless_on_shell"
        assert (
            on_shell.momentum_products
            == impose_kinematics(box, "massless_on_shell").momentum_products
        )
        with pytest.raises(ValidationError, match="makes no substitution"):
            box.with_kinematics("generic")

    @pytest.mark.parametrize(
        ("cnickel", "classes"),
        [
            (BOX + ":zzzz", ("massless_off_shell", "massless_on_shell")),
            (BOX + ":nnnn", ("generic", "equal_masses")),
            ("12e|2e|e|:nzz", ("generic",)),
            ("12e|2e|e|:aaz", ()),
            ("11e|e|:nn", ("generic", "equal_masses")),
            ("111e|e|:zzz", ("massless_off_shell", "massless_on_shell")),
            ("12e|23|3|e|:zzzzz", ("massless_off_shell", "massless_on_shell")),
            ("15e|24|3e|4e|5|e|:nnnnnnn", ("generic", "equal_masses")),
        ],
    )
    def test_from_cnickel_gives_the_class_asked_for(
        self, cnickel: str, classes: tuple[str, ...]
    ) -> None:
        for name in IMPOSABLE_CLASSES:
            if name in classes:
                fi = FeynmanIntegral.from_cnickel(cnickel, kinematics=name)
                assert fi.kinematic_class == name
            else:
                with pytest.raises(ValidationError, match=f"cannot impose {name}"):
                    FeynmanIntegral.from_cnickel(cnickel, kinematics=name)

    def test_from_cnickel_without_kinematics_is_unchanged(self) -> None:
        fi = FeynmanIntegral.from_cnickel(BOX + ":zzzz", kinematics=None)
        assert fi.momentum_products == FeynmanIntegral.from_cnickel(BOX + ":zzzz").momentum_products

    def test_from_cnickel_forwards_the_other_keywords(self) -> None:
        fi = FeynmanIntegral.from_cnickel(
            BOX + ":zzzz", kinematics="massless_on_shell", dimension=sp.Integer(4)
        )
        assert fi.dimension == 4
        assert fi.kinematic_class == "massless_on_shell"
        with pytest.raises(ValidationError, match="use_mandelstam=True"):
            FeynmanIntegral.from_cnickel(
                BOX + ":zzzz", kinematics="massless_on_shell", use_mandelstam=False
            )

    def test_from_nickel_and_other(self) -> None:
        fi = FeynmanIntegral.from_nickel(BOX, kinematics="massless_on_shell")
        assert fi.kinematic_class == "massless_on_shell"
        with pytest.raises(ValidationError, match="'other'"):
            FeynmanIntegral.from_cnickel(BOX + ":zzzz", kinematics="other")

    def test_sss_keeps_its_mass(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:sss", kinematics="equal_masses")
        assert {e.get_mass() for e in fi.graph.get_internal_edges()} == {mass("s")}

    def test_on_shell_point_counts(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz")
        by_class = box.with_kinematics("massless_on_shell").torus_count()
        by_hand = box.torus_count(on_shell=ON_SHELL)
        assert by_class.counts == by_hand.counts
        assert by_class.candidate_master_count == 3


class TestMassesZeroAfterExpansion:
    """A mass that expands to 0 is massless, as G reads it."""

    ZERO = (sp.Symbol("x") + 1) ** 2 - sp.Symbol("x") ** 2 - 2 * sp.Symbol("x") - 1

    def test_the_axis_is_that_of_the_massless_box(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":zzzz")
        fi = with_masses(box, [self.ZERO, 0, 0, 0])
        assert fi.symanzik.g == box.symanzik.g
        assert kinematic_axes(fi) == ("zero", "off_shell")
        assert kinematic_class(fi) == "massless_off_shell"

    def test_it_can_be_put_on_shell(self) -> None:
        fi = with_masses(FeynmanIntegral.from_cnickel(BOX + ":zzzz"), [self.ZERO, 0, 0, 0])
        on_shell = impose_kinematics(fi, "massless_on_shell")
        assert kinematic_class(on_shell) == "massless_on_shell"
        assert len(on_shell.newton_polytope.points) == 6

    def test_equal_masses_leaves_it_massless(self) -> None:
        box = FeynmanIntegral.from_cnickel(BOX + ":nnnn")
        fi = with_masses(box, [self.ZERO, mass(2), mass(3), mass(4)])
        assert len(fi.newton_polytope.points) == 13
        with pytest.raises(ValidationError, match="propagator 1 is massless"):
            impose_kinematics(fi, "equal_masses")


def test_a_single_propagator_cannot_have_equal_masses() -> None:
    tadpole = FeynmanIntegral.from_cnickel("0|:n", use_mandelstam=False)
    with pytest.raises(ValidationError, match="a single propagator cannot have equal masses"):
        impose_kinematics(tadpole, "equal_masses")
