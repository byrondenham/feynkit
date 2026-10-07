"""The sector hierarchy: kinds, the down-set of zero sectors and the counts of each sector.

Lee14 is R. N. Lee, LiteRed 1.4: a powerful tool for the reduction of the multiloop
integrals, arXiv:1310.1145; BBKP19 is Bitoun, Bogner, Klausen and Panzer, Feynman integral
relations from parametric annihilators, arXiv:1712.09215. Counts are those of BBKP19, with
the page of the statement.
"""

from __future__ import annotations

import itertools

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.landau import _singular_binary
from feynkit.sectors import Sector, SectorHierarchy, sector_hierarchy

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


def test_a_sector_is_a_frozen_value() -> None:
    h = sector_hierarchy(_integral(BOX), counts=None)
    s = h.sector(0b1111)
    assert isinstance(s, Sector)
    with pytest.raises(AttributeError):
        s.kind = "cycle"  # type: ignore[misc]
    assert s.symmetric_count is None
