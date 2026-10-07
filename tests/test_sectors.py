"""The sector hierarchy: kinds, the down-set of zero sectors and the counts of each sector.

Lee14 is R. N. Lee, LiteRed 1.4: a powerful tool for the reduction of the multiloop
integrals, arXiv:1310.1145; BBKP19 is Bitoun, Bogner, Klausen and Panzer, Feynman integral
relations from parametric annihilators, arXiv:1712.09215. Counts are those of BBKP19, with
the page of the statement.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary
from feynkit.sectors import (
    FixedPointClass,
    Sector,
    SectorHierarchy,
    fixed_point_euler_characteristic,
    parameter_permutations,
    sector_hierarchy,
    stabiliser,
)

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

BOX = ("13e|2e|3e|e|:zzzz", "massless_on_shell")
BOX_OFF_SHELL = ("13e|2e|3e|e|:zzzz", None)
SUNRISE = ("111e|e|:nnn", None)
KITE = ("12e|23|3e|e|:zzzzz", None)
TWO_POINT_KITE = ("12e|23|3|e|:zzzzz", None)
MASSIVE_KITE = ("12e|23|3e|e|:nnnnn", None)
DOUBLE_BOX = ("15e|24|3e|4e|5|e|:zzzzzzz", "massless_on_shell")
DOUBLE_BOX_OFF_SHELL = ("15e|24|3e|4e|5|e|:zzzzzzz", None)

# The instances whose zero sectors are checked against the scaleless test of every
# contraction: 302 forests in all.
ZERO_INSTANCES = [
    BOX,
    BOX_OFF_SHELL,
    KITE,
    MASSIVE_KITE,
    DOUBLE_BOX,
    DOUBLE_BOX_OFF_SHELL,
]


def _integral(instance: tuple[str, str | None]) -> FeynmanIntegral:
    cnickel, kinematics = instance
    return FeynmanIntegral.from_cnickel(cnickel, kinematics=kinematics)  # type: ignore[arg-type]


def _on_shell(cnickel: str, names: list[str]) -> FeynmanIntegral:
    """The integral with the named invariants set to zero in its momentum products."""
    fi = FeynmanIntegral.from_cnickel(cnickel)
    symbols = {str(x): x for v in fi.momentum_products.values() for x in sp.sympify(v).free_symbols}
    zero = {symbols[name]: 0 for name in names}
    return fi.with_(
        momentum_products={
            k: sp.expand(sp.sympify(v).subs(zero)) for k, v in fi.momentum_products.items()
        }
    )


def _by_edges(h: SectorHierarchy, field: str) -> dict[tuple[int, ...], int]:
    """A count of each sector with a non-zero value, keyed by its propagators."""
    return {
        s.propagators: getattr(s, field) for s in h.sectors if getattr(s, field) not in (None, 0)
    }


class TestKinds:
    @pytest.mark.parametrize("instance", ZERO_INSTANCES)
    def test_zero_sectors_are_those_of_the_scaleless_test(
        self, instance: tuple[str, str | None]
    ) -> None:
        fi = _integral(instance)
        h = sector_hierarchy(fi, counts=None)
        checked = 0
        for s in h.sectors:
            if not s.propagators or s.kind == "cycle":
                continue
            checked += 1
            if s.contracted:
                minor = fi.with_(graph=fi.graph.contract(list(s.contracted)))
                assert (s.kind == "scaleless") == minor.is_scaleless, s
            else:
                assert (s.kind == "scaleless") == fi.is_scaleless
        assert checked > 0

    def test_forest_counts_of_the_six_instances(self) -> None:
        forests = [
            sum(1 for s in sector_hierarchy(_integral(i), counts=None).sectors if s.kind != "cycle")
            for i in (BOX, BOX_OFF_SHELL, KITE, MASSIVE_KITE)
        ]
        # 302 forests with the double boxes' 112 each; the empty sector is a cycle.
        assert forests == [15, 15, 24, 24]

    @pytest.mark.parametrize(
        ("instance", "non_zero", "scaleless", "cycle"),
        [
            (BOX, 7, 8, 1),
            (BOX_OFF_SHELL, 11, 4, 1),
            (KITE, 9, 15, 8),
            (MASSIVE_KITE, 24, 0, 8),
        ],
    )
    def test_kind_totals(
        self, instance: tuple[str, str | None], non_zero: int, scaleless: int, cycle: int
    ) -> None:
        # The empty sector, whose contraction holds the loop, is counted among the cycles.
        totals = sector_hierarchy(_integral(instance), counts=None).totals()
        assert (totals.non_zero, totals.scaleless, totals.cycle) == (non_zero, scaleless, cycle)

    @pytest.mark.parametrize(
        ("instance", "non_zero", "scaleless", "cycle"),
        [(DOUBLE_BOX, 43, 69, 16), (DOUBLE_BOX_OFF_SHELL, 63, 49, 16)],
    )
    def test_double_box_totals(
        self, instance: tuple[str, str | None], non_zero: int, scaleless: int, cycle: int
    ) -> None:
        totals = sector_hierarchy(_integral(instance), counts=None).totals()
        assert (totals.non_zero, totals.scaleless, totals.cycle) == (non_zero, scaleless, cycle)

    @pytest.mark.parametrize("instance", [BOX, KITE, MASSIVE_KITE, SUNRISE])
    def test_zero_sectors_form_a_down_set(self, instance: tuple[str, str | None]) -> None:
        h = sector_hierarchy(_integral(instance), counts=None)
        for s in h.sectors:
            if s.kind == "non_zero":
                continue
            for r in range(len(s.propagators)):
                for sub in itertools.combinations(s.propagators, r):
                    assert h.sector(sub).kind != "non_zero"

    def test_a_set_containing_a_cycle_has_no_face(self) -> None:
        h = sector_hierarchy(_integral(KITE), counts=None)
        cycles = [s for s in h.sectors if s.kind == "cycle"]
        assert cycles
        for s in cycles:
            assert s.point_indices == ()
            assert s.dimension == -1

    def test_non_zero_sectors_form_an_up_set(self) -> None:
        h = sector_hierarchy(_integral(KITE), counts=None)
        for s in h.non_zero():
            for other in h.sectors:
                if set(s.propagators) <= set(other.propagators) and other.kind != "cycle":
                    assert other.kind == "non_zero"

    def test_the_massless_bridge_is_non_zero_with_a_degenerate_face(self) -> None:
        # A massive tadpole joined to its vertex by a massless line that carries no momentum:
        # not scaleless by Lee's criterion, but its polytope is not full-dimensional.
        fi = FeynmanIntegral.from_cnickel("1ee|1|:zn")
        assert not fi.is_scaleless
        top = sector_hierarchy(fi, counts=None).sector(0b11)
        assert top.kind == "non_zero"
        assert top.dimension < 2


class TestIdentity:
    def test_ids_have_the_first_propagator_as_least_significant_bit(self) -> None:
        h = sector_hierarchy(_integral(SUNRISE), counts=None)
        assert [s.id for s in h.sectors] == list(range(8))
        assert h.sector(0b011).propagators == (1, 2)
        assert h.sector(0b101).propagators == (1, 3)
        assert h.sector(0b110).propagators == (2, 3)
        assert h.sector(0b111).contracted == ()
        assert h.sector(0b001).contracted == (2, 3)

    def test_lookup_by_edges_or_id(self) -> None:
        h = sector_hierarchy(_integral(SUNRISE), counts=None)
        assert h.sector((3, 1)) is h.sector(0b101)
        assert h.sector([1, 2, 3]).id == 7

    def test_lookup_rejects_unknown_edges_and_ids(self) -> None:
        h = sector_hierarchy(_integral(SUNRISE), counts=None)
        with pytest.raises(ValidationError):
            h.sector(8)
        with pytest.raises(ValidationError):
            h.sector((1, 4))
        with pytest.raises(ValidationError):
            h.sector((1, 1))

    def test_sector_ids_agree_with_the_family(self) -> None:
        from feynkit.family import IntegralFamily

        fi = _integral(DOUBLE_BOX)
        family = IntegralFamily.from_integral(fi)
        h = sector_hierarchy(fi, counts=None)
        for s in h.sectors:
            n = [1 if e in s.propagators else 0 for e in range(1, 8)] + [0] * (family.size - 7)
            assert family.sector_id(n) == s.id

    def test_subsectors_and_covers(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts=None)
        top = h.sector(0b1111)
        names = {h.sector(i).propagators for i in h.subsectors(top.id)}
        assert names == {s.propagators for s in h.non_zero()} - {top.propagators}
        assert len(names) == 6
        # The covers of the top are the sub-sectors that no other sub-sector contains.
        assert {h.sector(i).propagators for i in h.covers(top.id)} == {
            n for n in names if len(n) == 3
        }
        bubble = h.sector((1, 4))
        assert h.subsectors(bubble.id) == () and h.covers(bubble.id) == ()

    def test_the_face_of_a_sector_is_its_support_with_the_contracted_exponents_zero(self) -> None:
        fi = _integral(BOX)
        h = sector_hierarchy(fi, counts=None)
        points = [tuple(int(x) for x in p) for p in fi.newton_polytope.points]
        for s in h.sectors:
            expected = tuple(
                j for j, p in enumerate(points) if all(p[e - 1] == 0 for e in s.contracted)
            )
            if s.kind != "cycle":
                assert s.point_indices == expected


class TestGenericCounts:
    def test_sunrise_three_masses(self) -> None:
        # The top has normalised volume 10; each two-line sector is a segment of length 1.
        h = sector_hierarchy(_integral(SUNRISE), counts="generic")
        assert h.counts_source == "generic"
        assert h.sector(0b111).generic_count == 10
        assert [h.sector(i).generic_count for i in (0b011, 0b101, 0b110)] == [1, 1, 1]
        assert h.sector(0b001).generic_count == 0  # contains a cycle

    def test_generic_is_the_default_and_needs_no_kinematics(self) -> None:
        h = sector_hierarchy(_integral(BOX))
        assert h.counts_source == "generic"
        assert h.point is None
        assert h.sector(0b1111).count is None

    def test_generic_count_bounds_the_count_of_a_point(self) -> None:
        h = sector_hierarchy(_integral(KITE), counts="generic")
        assert h.sector(0b11111).generic_count == 48
        assert all(s.generic_count == 0 for s in h.sectors if s.kind != "non_zero")

    def test_no_counts(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts=None)
        assert h.counts_source is None
        assert all(s.generic_count is None and s.count is None for s in h.sectors)

    def test_sectors_with_a_degenerate_polytope_get_zero(self) -> None:
        h = sector_hierarchy(FeynmanIntegral.from_cnickel("1ee|1|:zn"), counts="generic")
        assert all(s.generic_count == 0 for s in h.sectors if s.dimension < len(s.propagators))

    def test_unknown_counts_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="counts"):
            sector_hierarchy(_integral(BOX), counts="exact")  # type: ignore[arg-type]

    def test_a_point_needs_a_point_count(self) -> None:
        with pytest.raises(ValidationError, match="point"):
            sector_hierarchy(_integral(BOX), counts="generic", point={})


@requires_singular
class TestPointCounts:
    def test_sunrise_three_masses(self) -> None:
        # BBKP19, Prop. 55, p. 34: 7 = 4 + 1 + 1 + 1.
        h = sector_hierarchy(_integral(SUNRISE), counts="critical", timeout=120)
        assert h.counts_source == "critical"
        assert h.sector(0b111).count == 7
        assert _by_edges(h, "sector_count") == {
            (1, 2, 3): 4,
            (1, 2): 1,
            (1, 3): 1,
            (2, 3): 1,
        }
        assert h.totals().sector_count == 7
        assert h.point is not None

    def test_the_torus_count_agrees(self) -> None:
        a = sector_hierarchy(_integral(BOX), counts="critical", timeout=120)
        b = sector_hierarchy(_integral(BOX), counts="torus")
        assert [s.count for s in a.sectors] == [s.count for s in b.sectors]
        assert b.counts_source == "torus"

    def test_massless_on_shell_box(self) -> None:
        # Three masters: the box and the s- and t-bubbles; the triangles have none.
        h = sector_hierarchy(_integral(BOX), counts="critical", timeout=120)
        assert h.sector(0b1111).count == 3
        assert _by_edges(h, "sector_count") == {(1, 2, 3, 4): 1, (1, 4): 1, (2, 3): 1}
        assert all(s.sector_count == 0 for s in h.sectors if len(s.propagators) == 3)
        assert h.totals().sector_count == 3

    def test_off_shell_box(self) -> None:
        # BBKP19, Table 2, p. 37: 11 massless, 15 massive.
        assert sector_hierarchy(_integral(BOX_OFF_SHELL), counts="critical").sector(15).count == 11
        massive = FeynmanIntegral.from_cnickel("13e|2e|3e|e|:nnnn")
        assert sector_hierarchy(massive, counts="critical").sector(15).count == 15

    def test_zero_sectors_get_zero_without_a_computation(self) -> None:
        h = sector_hierarchy(_integral(KITE), counts="critical", timeout=120)
        assert all(s.count == 0 for s in h.sectors if s.kind != "non_zero")
        assert all(s.sector_count == 0 for s in h.sectors if s.kind != "non_zero")

    def test_a_given_point(self) -> None:
        fi = _integral(SUNRISE)
        drawn = sector_hierarchy(fi, counts="critical", timeout=120)
        assert drawn.point is not None
        again = sector_hierarchy(fi, counts="critical", point=dict(drawn.point), timeout=120)
        assert [s.count for s in again.sectors] == [s.count for s in drawn.sectors]
        assert again.point == drawn.point

    def test_a_point_that_misses_a_symbol_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="point"):
            sector_hierarchy(_integral(SUNRISE), counts="critical", point={}, timeout=120)

    def test_two_point_kites(self) -> None:
        # BBKP19, Ex. 53, p. 33 and Table 2, p. 37: 3 massless, 30 massive.
        assert (
            sector_hierarchy(_integral(TWO_POINT_KITE), counts="critical", timeout=120)
            .totals()
            .count
            == 3
        )
        massive = FeynmanIntegral.from_cnickel("12e|23|3|e|:nnnnn")
        assert sector_hierarchy(massive, counts="critical", timeout=120).sector(31).count == 30

    def test_negative_sector_count_is_flagged_and_never_clipped(self) -> None:
        # BBKP19, Fig. 4.4: C = 15 and the sector alone of Gamma/{1,2} counts -1 (Ex. 59,
        # p. 38; Rem. 60, p. 39). Propagators 3, 4, 5 of feynkit's labels.
        fi = _on_shell("13e|23e|3e|e|:zzaaa", ["p1^2", "p3^2", "p4^2"])
        h = sector_hierarchy(fi, counts="critical", timeout=120)
        assert h.sector(0b11111).count == 15
        negative = h.sector((3, 4, 5))
        assert negative.sector_count == -1
        assert negative.negative_sector_count
        assert [s.propagators for s in h.sectors if s.negative_sector_count] == [(3, 4, 5)]
        assert h.totals().sector_count == 15
        assert h.totals().negative == ((3, 4, 5),)

    def test_equal_mass_box_with_massless_legs(self) -> None:
        # The four bubbles on adjacent lines count -1 and the top counts 7.
        fi = _on_shell("13e|2e|3e|e|:aaaa", ["p1^2", "p2^2", "p3^2", "p4^2"])
        h = sector_hierarchy(fi, counts="critical", timeout=120)
        assert h.sector(15).count == 7
        assert h.totals().non_zero == 15
        assert {s.propagators for s in h.sectors if s.negative_sector_count} == {
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
        }

    def test_massive_kite_total(self) -> None:
        h = sector_hierarchy(_integral(MASSIVE_KITE), counts="critical", timeout=120)
        assert h.sector(31).count == 43 and h.totals().sector_count == 43

    def test_double_box(self) -> None:
        # CMMT23, Sec. 5.5.4, p. 52: twelve masters; the top has two and ten sectors one each.
        h = sector_hierarchy(_integral(DOUBLE_BOX), counts="critical", timeout=120)
        assert h.totals().count == 12
        values = sorted(v for v in _by_edges(h, "sector_count").values())
        assert values == [1] * 10 + [2]
        assert h.sector(127).sector_count == 2
        assert h.totals().sector_count == 12
        assert not h.totals().negative


class TestPointCheck:
    @pytest.fixture
    def install(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[object]]:
        """Patch the sector count to raise by one at every point after the first.

        Returns a function that installs the patch and gives the list of the points the
        count was called at.
        """
        import feynkit.sectors as module

        real = module._sector_count

        def install() -> list[object]:
            seen: list[object] = []

            def wrapper(point: object, *args: object, **kwargs: object) -> int:
                seen.append(point)
                value = real(point, *args, **kwargs)  # type: ignore[arg-type]
                return value + 1 if point is not seen[0] else value

            monkeypatch.setattr(module, "_sector_count", wrapper)
            return seen

        return install

    def test_the_top_count_is_checked_at_a_second_point(
        self, install: Callable[[], list[object]]
    ) -> None:
        seen = install()
        with pytest.raises(ComputationError, match="second point") as excinfo:
            sector_hierarchy(_integral(SUNRISE), counts="critical", symmetries=False, timeout=120)
        assert "7" in str(excinfo.value) and "8" in str(excinfo.value)
        assert len({id(p) for p in seen}) == 2

    def test_a_given_point_is_used_as_given(self, install: Callable[[], list[object]]) -> None:
        fi = _integral(SUNRISE)
        point = dict(sector_hierarchy(fi, counts="critical", symmetries=False).point or ())
        seen = install()
        h = sector_hierarchy(fi, counts="critical", point=point, symmetries=False, timeout=120)
        assert len({id(p) for p in seen}) == 1
        assert h.sector(7).count == 7

    def test_the_drawn_point_is_wide(self) -> None:
        h = sector_hierarchy(_integral(SUNRISE), counts="critical", symmetries=False, timeout=120)
        assert h.point is not None
        assert max(abs(v) for _, v in h.point) > 1000
        assert all(1 <= abs(v) <= 2**20 for _, v in h.point)


class TestPermutations:
    X = sp.symbols("x0:3")
    Y = sp.symbols("y0:3")

    def test_the_symmetric_polynomial_has_the_whole_group(self) -> None:
        x = self.X
        f = x[0] * x[1] + x[1] * x[2] + x[0] * x[2]
        maps = parameter_permutations(f, x, f, x)
        assert len(maps) == 6
        assert sorted(maps) == sorted(itertools.permutations(range(3)))

    def test_a_map_between_two_polynomials(self) -> None:
        x, y = self.X, self.Y
        f = 2 * x[0] * x[1] + x[1] * x[2] + 3 * x[2]
        # x_0, x_1, x_2 go to y_1, y_2, y_0, and sigma[i] is where x_i goes.
        g = 2 * y[1] * y[2] + y[2] * y[0] + 3 * y[0]
        assert parameter_permutations(f, x, g, y) == [(1, 2, 0)]

    def test_coefficients_are_compared_exactly(self) -> None:
        x = self.X
        a = sp.Symbol("a")
        f = x[0] * x[1] + a * x[1] * x[2]
        assert parameter_permutations(f, x, x[0] * x[1] + 2 * x[1] * x[2], x) == []
        assert parameter_permutations(f, x, x[1] * x[2] + a * x[0] * x[1], x) == [(2, 1, 0)]

    def test_different_sizes_have_no_map(self) -> None:
        x = self.X
        assert parameter_permutations(x[0] + x[1], x[:2], x[0] + x[1] + x[2], x) == []

    def test_the_stabiliser_is_the_maps_to_itself(self) -> None:
        x = self.X
        f = x[0] * x[1] + x[2]
        assert stabiliser(f, x) == parameter_permutations(f, x, f, x)
        assert len(stabiliser(f, x)) == 2


class TestOrbits:
    def test_the_massless_box(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts=None)
        top = h.stabiliser(15)
        assert len(top) == 4
        assert h.sector(15).stabiliser_order == 4
        # The bubbles are alone; the triangles fall into two orbits of two.
        assert h.orbits() == {6: (6,), 7: (7, 14), 9: (9,), 11: (11, 13), 15: (15,)}
        assert h.orbit(h.sector((1, 2, 3)).id) == h.orbit(h.sector((2, 3, 4)).id)

    def test_orbits_hold_equal_size_sectors_and_start_at_the_least_id(self) -> None:
        h = sector_hierarchy(_integral(DOUBLE_BOX), counts=None)
        orbits = h.orbits()
        assert sum(len(m) for m in orbits.values()) == 43
        for rep, members in orbits.items():
            assert rep == min(members)
            assert len({len(h.sector(i).propagators) for i in members}) == 1
            assert all(h.sector(i).orbit == rep for i in members)
        assert len(orbits) == 18
        assert all(s.orbit is None for s in h.zero())

    def test_maps_between_sectors_are_bijections_of_their_propagators(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts=None)
        a, b = h.sector((1, 2, 3)), h.sector((2, 3, 4))
        maps = h.maps(a.id, b.id)
        assert maps and all(
            set(m) == set(a.propagators) and set(m.values()) == set(b.propagators) for m in maps
        )
        assert h.maps(h.sector((1, 4)).id, h.sector((2, 3)).id) == []
        assert h.maps(a.id, h.sector(15).id) == []

    def test_a_zero_sector_has_no_maps(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts=None)
        with pytest.raises(ValidationError, match="non-zero"):
            h.maps(0, 15)

    def test_no_symmetries(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts=None, symmetries=False)
        assert all(s.orbit is None and s.stabiliser_order is None for s in h.sectors)
        assert h.totals().unique is None
        with pytest.raises(ValidationError):
            h.orbits()

    def test_equal_mass_box_has_seven_unique_sectors(self) -> None:
        # The top group is Z_2 x Z_2, and the groupoid has 7 orbits where the top group alone
        # has 8 (Sec. 3.6, Eq. 3.60 of DMSS26).
        fi = _on_shell("13e|2e|3e|e|:aaaa", ["p1^2", "p2^2", "p3^2", "p4^2"])
        h = sector_hierarchy(fi, counts=None)
        assert h.totals().non_zero == 15
        assert h.totals().unique == 7
        group = h.stabiliser(15)
        assert len(group) == 4
        assert all(sorted(g.values()) == [1, 2, 3, 4] for g in group)
        # Z_2 x Z_2: every element squares to the identity.
        assert all(g[g[e]] == e for g in group for e in g)
        non_zero = [s.propagators for s in h.non_zero()]
        seen: set[frozenset[int]] = set()
        classes = 0
        for t in non_zero:
            if frozenset(t) in seen:
                continue
            classes += 1
            for g in group:
                seen.add(frozenset(g[e] for e in t))
        assert classes == 8

    def test_double_box_matches_weinzierl_numbering(self) -> None:
        # Wei22, Eq. 6.83: with his labels the masters sit in sectors 28, 73, 54, 57, 79, 93
        # and 127. Each belongs to an orbit, and the top is alone.
        from tests.test_family import _weinzierl_double_box

        h = sector_hierarchy(_weinzierl_double_box(), counts=None)
        for nid in (28, 73, 54, 57, 79, 93, 127):
            assert h.sector(nid).kind == "non_zero"
        assert h.sector(127).orbit == 127
        assert len(h.orbits()[127]) == 1
        assert h.totals().unique == 18


@requires_singular
class TestSymmetricCounts:
    def test_equal_mass_sunrise(self) -> None:
        # DMSS26, Eqs. 2.82-2.84, p. 30: the top has 2 masters, the family 3 (Wei22, Eq. 6.85).
        h = sector_hierarchy(
            FeynmanIntegral.from_cnickel("111e|e|:aaa"), counts="critical", timeout=120
        )
        top = h.sector(7)
        assert top.stabiliser_order == 6
        assert sorted((c.cycle_type, c.size, c.euler_characteristic) for c in top.fixed_points) == [
            ((1, 1, 1), 1, -4),
            ((2, 1), 3, 2),
            ((3,), 2, -1),
        ]
        assert top.symmetric_count == 2 and top.signed_symmetric_count == 2
        assert top.signs_consistent is True
        assert h.totals().symmetric == 3
        assert h.totals().unique == 2

    def test_two_equal_masses(self) -> None:
        h = sector_hierarchy(
            FeynmanIntegral.from_cnickel("111e|e|:aab"), counts="critical", timeout=120
        )
        assert h.sector(7).symmetric_count == 3
        assert h.totals().symmetric == 5

    def test_massless_box(self) -> None:
        h = sector_hierarchy(_integral(BOX), counts="critical", timeout=120)
        assert h.totals().symmetric == 3 and h.totals().signed_symmetric == 3
        assert h.sector(15).stabiliser_order == 4

    def test_equal_mass_box_signs(self) -> None:
        # Eq. 8.18 gives 7 for the family and the signed form 6: the four adjacent bubbles
        # have chi = -1, the wrong sign for c = 2, so the two forms disagree there.
        fi = _on_shell("13e|2e|3e|e|:aaaa", ["p1^2", "p2^2", "p3^2", "p4^2"])
        h = sector_hierarchy(fi, counts="critical", timeout=120)
        totals = h.totals()
        assert (totals.symmetric, totals.signed_symmetric) == (7, 6)
        assert totals.signs_consistent is False
        bad = {s.id for s in h.sectors if s.signs_consistent is False}
        assert bad == {h.sector(t).orbit for t in [(1, 2), (1, 3), (2, 4), (3, 4)]}
        assert all(
            h.sector(i).symmetric_count == 1 and h.sector(i).signed_symmetric_count == 0
            for i in bad
        )
        assert h.sector(15).signs_consistent is True

    def test_double_box(self) -> None:
        # CMMT23, Sec. 5.5.4, p. 52: eight with symmetries. Six unique sectors with one
        # master each, of sizes 3, 3, 4, 4, 5, 5, and the top with two.
        h = sector_hierarchy(_integral(DOUBLE_BOX), counts="critical", timeout=120)
        counted = {s.id: s for s in h.sectors if s.symmetric_count}
        sizes = sorted(len(s.propagators) for s in counted.values() if s.symmetric_count == 1)
        assert sizes == [3, 3, 4, 4, 5, 5]
        assert counted[127].symmetric_count == 2
        assert h.totals().symmetric == 8 and h.totals().signed_symmetric == 8
        assert h.totals().signs_consistent is True
        assert h.totals().sector_count == 12

    def test_the_helper_agrees_with_the_hierarchy(self) -> None:
        fi = FeynmanIntegral.from_cnickel("111e|e|:aaa")
        h = sector_hierarchy(fi, counts="critical", timeout=120)
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        variables = list(fi.symanzik.lp_parameters)
        point = dict(h.point or ())
        assert fixed_point_euler_characteristic(g, variables, (0, 1, 2), point, timeout=120) == -4
        assert fixed_point_euler_characteristic(g, variables, (1, 0, 2), point, timeout=120) == 2
        assert fixed_point_euler_characteristic(g, variables, (1, 2, 0), point, timeout=120) == -1

    def test_a_non_symmetry_is_refused(self) -> None:
        fi = FeynmanIntegral.from_cnickel("111e|e|:aab")
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        variables = list(fi.symanzik.lp_parameters)
        point = dict(
            sector_hierarchy(fi, counts="critical", symmetries=False, timeout=120).point or ()
        )
        with pytest.raises(ValidationError, match="symmetry"):
            fixed_point_euler_characteristic(g, variables, (2, 1, 0), point, timeout=120)


BANANA_3 = [
    ("1111e|e|:aaaa", 3),
    ("1111e|e|:aaab", 5),
    ("1111e|e|:aabb", 6),
    ("1111e|e|:aabc", 8),
    ("1111e|e|:abcd", 11),
]
BANANA_4 = [
    ("11111e|e|:aaaaa", 4),
    ("11111e|e|:aaaab", 7),
    ("11111e|e|:aaabb", 9),
    ("11111e|e|:aaabc", 12),
    ("11111e|e|:aabbc", 14),
    ("11111e|e|:aabcd", 19),
    ("11111e|e|:abcde", 26),
]


@requires_singular
@pytest.mark.slow
class TestBananas:
    # DMSS26, Tables 1 and 2, p. 31: the masters of the top sector with symmetries.
    @pytest.mark.parametrize(("cnickel", "masters"), BANANA_3)
    def test_three_loops(self, cnickel: str, masters: int) -> None:
        h = sector_hierarchy(FeynmanIntegral.from_cnickel(cnickel), counts="critical", timeout=120)
        assert h.sector(15).symmetric_count == masters

    @pytest.mark.parametrize(("cnickel", "masters"), BANANA_4)
    def test_four_loops(self, cnickel: str, masters: int) -> None:
        h = sector_hierarchy(FeynmanIntegral.from_cnickel(cnickel), counts="critical", timeout=120)
        assert h.sector(31).symmetric_count == masters


def test_a_sector_is_a_frozen_value() -> None:
    h = sector_hierarchy(_integral(BOX), counts=None)
    s = h.sector(0b1111)
    assert isinstance(s, Sector)
    with pytest.raises(AttributeError):
        s.kind = "cycle"  # type: ignore[misc]
    assert s.symmetric_count is None
    assert FixedPointClass((1,), 1, 0).size == 1


class TestAccessor:
    def test_the_integral_caches_its_hierarchy_for_each_choice_of_arguments(self) -> None:
        fi = _integral(BOX)
        first = fi.sectors()
        assert first is fi.sectors()
        assert first.totals() == sector_hierarchy(fi).totals()
        assert fi.sectors(symmetries=False) is not first
        assert fi.sectors(counts=None) is not first
        assert fi.sectors(counts=None) is fi.sectors(counts=None)

    @requires_singular
    def test_a_point_is_part_of_the_key(self) -> None:
        fi = _integral(SUNRISE)
        drawn = fi.sectors(counts="critical", symmetries=False, timeout=120)
        point = dict(drawn.point or ())
        given = fi.sectors(counts="critical", point=point, symmetries=False, timeout=120)
        assert given is fi.sectors(
            counts="critical", point=dict(point), symmetries=False, timeout=120
        )
        assert given.sector(7).count == 7
