"""Tests for the face degeneracy of feynkit.degeneracy.

Oracles: faces whose singular points are known by hand (the massive bubble at
lambda = 0, a square on an edge), the Gröbner path itself against the
shortcuts for vertices, simplices and edges, the count of critical points,
which equals the normalised volume when no face is degenerate, the dominant
components of the Landau analysis, and the published examples of Fevola,
Mizera and Telen (arXiv:2311.16219) and Klausen (arXiv:1910.08651).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from fractions import Fraction

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import degeneracy as degeneracy_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.degeneracy import (
    DegeneracyAnalysis,
    FaceDegeneracy,
    face_degeneracy,
    face_degeneracy_from_polynomial,
)
from feynkit.landau import _singular_binary, landau_analysis_from_polynomial
from feynkit.point_count import critical_point_count

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

# SymPy's factorisation, where a test uses it, runs from a fixed state of its generator.
pytestmark = pytest.mark.usefixtures("sympy_seeded")

X, Y, Z = sp.symbols("x y z")


def symbol(fi: FeynmanIntegral, name: str) -> sp.Symbol:
    """The kinematic symbol of G with this name."""
    (found,) = [x for x in fi.symanzik.g.free_symbols if x.name == name]
    return found


def on_shell(fi: FeynmanIntegral, *names: str) -> FeynmanIntegral:
    """The integral with the momentum-product symbols of these names set to 0."""
    zeros = {symbol(fi, name): 0 for name in names}
    return fi.with_(
        momentum_products={
            key: sp.expand(sp.sympify(value).subs(zeros))
            for key, value in fi.momentum_products.items()
        }
    )


def degenerate(analysis: DegeneracyAnalysis) -> list[tuple[int, int, int | None, int | None]]:
    """(dimension, points, singular dimension, Tjurina number) of each degenerate face."""
    return sorted(
        (f.dimension, len(f.point_indices), f.singular_dimension, f.tjurina)
        for f in analysis.degenerate_faces
    )


def exponents(analysis: DegeneracyAnalysis, face: FaceDegeneracy) -> frozenset[tuple[int, ...]]:
    return frozenset(analysis.points[i] for i in face.point_indices)


def assert_modular_agrees(analysis: DegeneracyAnalysis) -> None:
    """The finite-field verdict, where it ran, is the exact one."""
    for face in analysis.faces:
        if face.modular is not None and face.degenerate is not None:
            assert face.modular == face.degenerate, face


def bubble_g(s: int, m1: int, m2: int) -> sp.Expr:
    """G of the massive bubble at mu = 1, with m1 and m2 the squared masses."""
    return X + Y + m1 * X**2 + m2 * Y**2 + (m1 + m2 - s) * X * Y


# --- no computation -------------------------------------------------------------------------


class TestWithoutSingular:
    """Vertices, simplices and edges are decided without Singular."""

    @pytest.fixture(autouse=True)
    def no_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: None)

    def test_a_square_on_an_edge_is_degenerate(self) -> None:
        # (x - 1)^2 (x + 2): one double root, of Tjurina number 1.
        analysis = face_degeneracy_from_polynomial(sp.expand((X - 1) ** 2 * (X + 2)), [X])
        (top,) = [f for f in analysis.faces if f.dimension == 1]
        assert (top.method, top.degenerate, top.singular_dimension, top.tjurina) == (
            "edge",
            True,
            0,
            1,
        )
        assert [f.method for f in analysis.faces if f.dimension == 0] == ["vertex", "vertex"]
        assert analysis.volume == 3
        assert analysis.non_degenerate is False

    def test_a_cube_has_tjurina_number_two(self) -> None:
        analysis = face_degeneracy_from_polynomial(sp.expand((X + 3) ** 3 * (X - 2)), [X])
        assert degenerate(analysis) == [(1, 5, 0, 2)]

    def test_an_edge_without_a_double_root(self) -> None:
        analysis = face_degeneracy_from_polynomial(X**2 + 3 * X + 1, [X])
        (top,) = analysis.faces[-1:]
        assert (top.method, top.degenerate, top.singular_dimension, top.tjurina) == (
            "edge",
            False,
            -1,
            0,
        )
        assert analysis.non_degenerate is True

    def test_a_root_at_zero_is_not_in_the_torus(self) -> None:
        # x^3 + x^2 = x^2 (x + 1): the edge is the segment [2, 3], a simplex.
        analysis = face_degeneracy_from_polynomial(X**3 + X**2, [X])
        assert {f.method for f in analysis.faces} == {"vertex", "simplex"}
        assert analysis.non_degenerate is True

    def test_the_massless_bubble_needs_no_computation(self) -> None:
        analysis = face_degeneracy(FeynmanIntegral.from_cnickel("11e|e|:zz"))
        assert {f.method for f in analysis.faces} == {"vertex", "simplex"}
        assert all(f.degenerate is False and f.modular is None for f in analysis.faces)
        assert (analysis.mode, analysis.volume, analysis.support_loss) == ("generic", 1, 0)

    def test_an_edge_in_generic_mode(self) -> None:
        a, b = sp.symbols("a b")
        # a x^2 + 2 a b x + a b^2 = a (x + b)^2 for every a and b.
        analysis = face_degeneracy_from_polynomial(a * X**2 + 2 * a * b * X + a * b**2, [X])
        assert degenerate(analysis) == [(1, 3, 0, 1)]
        # a x^2 + b x + 1 has a double root only on b^2 = 4a.
        analysis = face_degeneracy_from_polynomial(a * X**2 + b * X + 1, [X])
        assert analysis.non_degenerate is True
        point = face_degeneracy_from_polynomial(a * X**2 + b * X + 1, [X], {a: 1, b: 2})
        assert degenerate(point) == [(1, 3, 0, 1)]

    def test_groebner_faces_need_singular(self) -> None:
        with pytest.raises(RuntimeError, match="Singular"):
            face_degeneracy_from_polynomial(bubble_g(9, 1, 4), [X, Y])


class TestArguments:
    def test_the_zero_polynomial_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="zero"):
            face_degeneracy_from_polynomial(sp.Integer(0), [X])

    def test_variables_must_be_distinct_symbols(self) -> None:
        with pytest.raises(ValidationError, match="distinct symbols"):
            face_degeneracy_from_polynomial(X + 1, [X, X])
        with pytest.raises(ValidationError, match="distinct symbols"):
            face_degeneracy_from_polynomial(X + 1, [])

    @pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), "5", 3e6, None])
    def test_timeout_must_be_a_number_of_seconds(self, timeout: object) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            face_degeneracy_from_polynomial(X + 1, [X], timeout=timeout)  # type: ignore[arg-type]

    def test_a_point_needs_every_symbol(self) -> None:
        a, b = sp.symbols("a b")
        with pytest.raises(ValidationError, match="no value for b"):
            face_degeneracy_from_polynomial(a * X**2 + b * X + 1, [X], {a: 1})
        with pytest.raises(ValidationError, match="not rational"):
            face_degeneracy_from_polynomial(a * X**2 + b * X + 1, [X], {a: 1, b: sp.sqrt(2)})

    def test_a_point_where_the_polynomial_vanishes_is_refused(self) -> None:
        a = sp.Symbol("a")
        with pytest.raises(ValidationError, match="vanishes"):
            face_degeneracy_from_polynomial(a * X**2 + a * X, [X], {a: 0})

    def test_coefficients_must_be_rational_functions(self) -> None:
        with pytest.raises(ValidationError, match="rational"):
            face_degeneracy_from_polynomial(sp.sqrt(2) * X**2 + X + 1, [X])

    def test_kinematic_constraints_are_refused(self) -> None:
        s = sp.Symbol("s", real=True)
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])
        with pytest.raises(ValidationError, match="kinematic_constraints"):
            face_degeneracy(fi)


class TestPointMode:
    def test_support_loss(self) -> None:
        a = sp.Symbol("a")
        # At a = 0 the vertex x^2 goes, and the edge [0, 2] shrinks to [0, 1].
        analysis = face_degeneracy_from_polynomial(a * X**2 + X + 1, [X], {a: 0})
        assert analysis.mode == "point"
        assert analysis.points == ((0,), (1,))
        assert (analysis.volume, analysis.support_loss) == (1, 1)
        assert analysis.point == ((a, Fraction(0)),)

    def test_point_indices_follow_the_newton_polytope(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        analysis = face_degeneracy(fi)
        assert analysis.points == tuple(tuple(int(x) for x in p) for p in fi.newton_polytope.points)

    def test_masses_are_given_by_their_squares(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        m1, m2, s = symbol(fi, "m_1"), symbol(fi, "m_2"), symbol(fi, "s")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(degeneracy_module, "_singular_binary", lambda: None)
            with pytest.raises(RuntimeError):
                face_degeneracy(fi, {m1**2: 1, m2: 2, s: 9})
        with pytest.raises(ValidationError, match="the keys are"):
            face_degeneracy(fi, {m1**2: 1, m2**2: 4, s: 9, sp.Symbol("mu"): 1})


# --- Singular ------------------------------------------------------------------------------


@requires_singular
class TestUnit:
    def test_massive_bubble_at_lambda_zero(self) -> None:
        # m_1^2 = 1, m_2^2 = 4, s = 9: lambda = 0, and G has the edge (u_1 - 2 u_2)^2.
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        m1, m2, s = symbol(fi, "m_1"), symbol(fi, "m_2"), symbol(fi, "s")
        point = {m1**2: 1, m2**2: 4, s: 9}
        analysis = face_degeneracy(fi, point)
        assert degenerate(analysis) == [(1, 3, 0, 1)]
        assert (analysis.volume, analysis.support_loss) == (3, 0)
        assert_modular_agrees(analysis)
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        assert critical_point_count(g, fi.symanzik.lp_parameters, point) == 2

    def test_massive_bubble_is_generically_non_degenerate(self) -> None:
        analysis = face_degeneracy(FeynmanIntegral.from_cnickel("11e|e|:nn"))
        assert analysis.mode == "generic" and analysis.point is None
        assert analysis.non_degenerate is True
        assert analysis.volume == 3

    def test_massive_triangle_with_a_massless_leg(self) -> None:
        fi = on_shell(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), "p1^2")
        analysis = face_degeneracy(fi)
        # One facet, the bubble face with five points, with one node.
        assert degenerate(analysis) == [(2, 5, 0, 1)]
        assert analysis.volume == 7
        assert_modular_agrees(analysis)

    @pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "111e|e|:zzz"])
    def test_massless_triangle_and_sunrise_have_none(self, cnickel: str) -> None:
        analysis = face_degeneracy(FeynmanIntegral.from_cnickel(cnickel))
        assert analysis.non_degenerate is True
        assert_modular_agrees(analysis)

    def test_massless_sunrise_volume_is_its_master_count(self) -> None:
        fi = FeynmanIntegral.from_cnickel("111e|e|:zzz")
        assert face_degeneracy(fi).volume == 1
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        assert critical_point_count(g, fi.symanzik.lp_parameters, {symbol(fi, "s"): 7}) == 1

    def test_double_bubble_has_singular_curves(self) -> None:
        analysis = face_degeneracy(FeynmanIntegral.from_cnickel("1122|e|e|:zzzz"))
        found = degenerate(analysis)
        assert found
        assert any(dim is not None and dim >= 1 for _, _, dim, _ in found)
        # The Tjurina number is given only for a finite singular locus.
        assert all((tau is None) == (dim != 0) for _, _, dim, tau in found)
        assert_modular_agrees(analysis)

    def test_the_check_can_be_left_out(self) -> None:
        fi = on_shell(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), "p1^2")
        analysis = face_degeneracy(fi, check=False)
        assert analysis.prime is None
        assert all(f.modular is None for f in analysis.faces)
        assert degenerate(analysis) == [(2, 5, 0, 1)]


def random_polynomial(rng: random.Random, n: int) -> sp.Expr:
    """A polynomial in n of x, y, z with exponents in {0, 1, 2}, with a constant term, and with
    a square on a random edge of the box now and then."""
    variables = [X, Y, Z][:n]
    terms = {(0,) * n: rng.choice([1, -1, 2])}
    for _ in range(rng.randint(2, 6)):
        terms[tuple(rng.randint(0, 2) for _ in range(n))] = rng.choice([-3, -2, -1, 1, 2, 3])
    g = sum(c * sp.prod(v**e for v, e in zip(variables, m, strict=True)) for m, c in terms.items())
    if rng.random() < 0.3:
        k = rng.randrange(n)
        g += sp.expand((variables[k] - rng.choice([1, 2, -1])) ** 2) - 1
    return sp.expand(g)


@requires_singular
class TestShortcuts:
    def test_shortcuts_agree_with_the_groebner_path(self) -> None:
        rng = random.Random(3)
        compared = 0
        degenerate_edges = 0
        for _ in range(25):
            n = rng.randint(1, 3)
            g = random_polynomial(rng, n)
            variables = [X, Y, Z][:n]
            fast = face_degeneracy_from_polynomial(g, variables, check=False)
            slow = degeneracy_module._analysis(
                g, variables, None, scale=None, check=False, timeout=120, shortcuts=False
            )
            assert fast.points == slow.points
            for one, other in zip(fast.faces, slow.faces, strict=True):
                assert one.point_indices == other.point_indices
                if one.dimension == 0:
                    continue
                assert other.method == "groebner"
                assert (one.degenerate, one.singular_dimension, one.tjurina) == (
                    other.degenerate,
                    other.singular_dimension,
                    other.tjurina,
                ), (g, one, other)
                compared += one.method != "groebner"
                degenerate_edges += one.method == "edge" and bool(one.degenerate)
        assert compared > 50
        assert degenerate_edges > 0


# --- properties ---------------------------------------------------------------------------


@requires_singular
class TestProperties:
    def test_no_degenerate_face_means_the_count_is_the_volume(self) -> None:
        rng = random.Random(5)
        seen = {True: 0, False: 0}
        for _ in range(20):
            n = rng.randint(1, 3)
            g = random_polynomial(rng, n)
            variables = [X, Y, Z][:n]
            if n >= 2 and rng.random() < 0.4:
                # Singular along {1 + x = 0 = g}, which meets the torus for most g.
                g = sp.expand((1 + rng.choice(variables)) * g)
            analysis = face_degeneracy_from_polynomial(g, variables)
            assert analysis.non_degenerate is not None
            assert_modular_agrees(analysis)
            seen[analysis.non_degenerate] += 1
            count = critical_point_count(g, variables, {})
            if analysis.non_degenerate:
                assert count == analysis.volume, g
            else:
                assert count <= analysis.volume, g
        assert seen[True] >= 5 and seen[False] >= 3

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(
                lambda: on_shell(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), "p1^2"),
                id="triangle",
            ),
            pytest.param(lambda: FeynmanIntegral.from_cnickel("111e|e|:nnn"), id="sunrise"),
            pytest.param(
                lambda: FeynmanIntegral.from_cnickel("1122|e|e|:zzzz"), id="double bubble"
            ),
            pytest.param(
                lambda: on_shell(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"), "p1^2", "p2^2"),
                id="box",
                marks=pytest.mark.slow,
            ),
            pytest.param(
                lambda: FeynmanIntegral.from_cnickel("12e|23|3|e|:nnzzz"),
                id="kite",
                marks=pytest.mark.slow,
            ),
        ],
    )
    def test_generic_mode_agrees_with_dominant_components(
        self, build: Callable[[], FeynmanIntegral]
    ) -> None:
        fi = build()
        analysis = face_degeneracy(fi)
        assert_modular_agrees(analysis)
        sym = fi.symanzik
        landau = landau_analysis_from_polynomial(
            sym.g, list(sym.lp_parameters), scale=fi.graph.energy_scale
        )
        kinematic = sym.g.free_symbols - set(sym.lp_parameters) - {fi.graph.energy_scale}
        by_exponents = {exponents(analysis, f): f for f in analysis.faces}
        compared = 0
        for face in landau.face_discriminants:
            if face.dimension < 2 or face.is_simplex:
                continue
            if not set().union(*(sp.sympify(c).free_symbols for c in face.coefficients)) & (
                kinematic
            ):
                continue
            ours = by_exponents[frozenset(face.exponents)]
            assert ours.degenerate == face.dominant, face.exponents
            compared += 1
        assert compared > 0


# --- published examples ----------------------------------------------------------------------


@requires_singular
class TestPublished:
    def test_banana_with_three_edges(self) -> None:
        """Fevola, Mizera and Telen, Ex. 2.5 and Rem. 2.6, pp. 11-13: 7 < 10 on K; Klausen,
        Sec. 4, p. 25: the fully massive sunset has 10 basis solutions, of which 7 remain."""
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        analysis = face_degeneracy(fi)
        assert analysis.volume == 10
        assert analysis.non_degenerate is False
        # The quadrilateral whose discriminant vanishes on K is among them.
        assert any(
            f.dimension == 2 and len(f.point_indices) == 4 for f in analysis.degenerate_faces
        )
        assert_modular_agrees(analysis)
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        point = {symbol(fi, "s"): 13, **{symbol(fi, f"m_{e}") ** 2: e + 1 for e in (1, 2, 3)}}
        assert critical_point_count(g, fi.symanzik.lp_parameters, point) == 7

    @pytest.mark.parametrize(
        "edges",
        [2, 3, pytest.param(4, marks=pytest.mark.slow)],
        ids=["E=2", "E=3", "E=4"],
    )
    def test_bananas(self, edges: int) -> None:
        """Fevola, Mizera and Telen, Sec. 4.2, p. 45: |chi| = 2^E - 1 against
        vol = binom(2E - 1, E). The top face of B_4 takes Singular minutes, and is left
        undecided at the shorter limit."""
        fi = FeynmanIntegral.from_cnickel("1" * edges + "e|e|:" + "n" * edges)
        analysis = face_degeneracy(fi, check=False, timeout=10)
        assert analysis.volume == math.comb(2 * edges - 1, edges)
        g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
        squares = {symbol(fi, f"m_{e}") ** 2: e + 1 for e in range(1, edges + 1)}
        point = {symbol(fi, "s"): 13, **squares}
        count = critical_point_count(g, fi.symanzik.lp_parameters, point)
        assert count == 2**edges - 1
        assert analysis.non_degenerate is (count == analysis.volume)

    def test_massless_banana_has_none(self) -> None:
        """Fevola, Mizera and Telen, Tab. 1, p. 14: B_4 with massless lines, (1, 1)."""
        analysis = face_degeneracy(FeynmanIntegral.from_cnickel("1111e|e|:zzzz"))
        assert analysis.volume == 1
        assert analysis.non_degenerate is True

    @pytest.mark.slow
    def test_parachute(self) -> None:
        """Fevola, Mizera and Telen, Tab. 1, p. 14: the parachute on K, (19, 35)."""
        analysis = face_degeneracy(
            FeynmanIntegral.from_cnickel("112e|2e|e|:nnnn"), check=False, timeout=10
        )
        assert analysis.volume == 35
        assert analysis.non_degenerate is False

    def test_product_with_a_dense_face(self) -> None:
        """Fevola, Mizera and Telen, Ex. 3.9, pp. 22-23: the dense face of
        (1 + x)(a + b x + c y + d x y) is degenerate for all (a, b, c, d), and bc = ad adds a
        singular point."""
        a, b, c, d = sp.symbols("a b c d")
        g = sp.expand((1 + X) * (a + b * X + c * Y + d * X * Y))
        generic = face_degeneracy_from_polynomial(g, [X, Y])
        (top,) = [f for f in generic.faces if f.dimension == 2]
        assert (top.degenerate, top.singular_dimension) == (True, 0)
        assert_modular_agrees(generic)
        off = face_degeneracy_from_polynomial(g, [X, Y], {a: 1, b: 2, c: 3, d: 5})
        on = face_degeneracy_from_polynomial(g, [X, Y], {a: 1, b: 2, c: 3, d: 6})
        (top_off,) = [f for f in off.faces if f.dimension == 2]
        (top_on,) = [f for f in on.faces if f.dimension == 2]
        assert top_off.tjurina == top.tjurina
        assert top_on.tjurina is not None and top.tjurina is not None
        assert top_on.tjurina > top.tjurina


# --- timeouts -----------------------------------------------------------------------------


# The eight points of the unit cube: six square facets and the cube go to Singular.
CUBE = sp.expand((1 + X) * (1 + Y) * (1 + Z) + X * Y * Z)


class TestTimeouts:
    def test_a_face_past_the_limit_is_undecided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run that times out keeps what it printed; the face it was on runs alone, and the
        rest run again together."""
        monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: "Singular")
        calls: list[list[int]] = []
        hang: list[int] = []

        def fake(script: str, binary: str, timeout: float) -> tuple[str, bool]:
            faces = [int(line.split()[1]) for line in script.splitlines() if line.startswith("//F")]
            calls.append(faces)
            out = ""
            for k in faces:
                if k in hang:
                    return out, True
                out += f"F {k} -1 0\n"
            return out, False

        monkeypatch.setattr(degeneracy_module, "_run", fake)
        g = CUBE
        first = face_degeneracy_from_polynomial(g, [X, Y, Z], check=False)
        (batch,) = calls
        assert len(batch) >= 2
        hang.append(batch[0])
        calls.clear()
        analysis = face_degeneracy_from_polynomial(g, [X, Y, Z], check=False)
        assert first.non_degenerate is True
        undecided = [i for i, f in enumerate(analysis.faces) if f.degenerate is None]
        assert undecided == [batch[0]]
        assert "timeout=120" in (analysis.faces[batch[0]].reason or "")
        assert analysis.non_degenerate is None
        # The face that hung had the whole limit in the first run, so it is not run again, and
        # the rest run together.
        assert calls == [batch, batch[1:]]

    def test_a_face_that_hangs_after_others_runs_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: "Singular")
        calls: list[list[int]] = []
        hang: list[int] = []

        def fake(script: str, binary: str, timeout: float) -> tuple[str, bool]:
            faces = [int(line.split()[1]) for line in script.splitlines() if line.startswith("//F")]
            calls.append(faces)
            out = ""
            for k in faces:
                if k in hang and len(faces) > 1:
                    return out, True
                out += f"F {k} 0 1\n" if k in hang else f"F {k} -1 0\n"
            return out, False

        monkeypatch.setattr(degeneracy_module, "_run", fake)
        face_degeneracy_from_polynomial(CUBE, [X, Y, Z], check=False)
        (batch,) = calls
        assert len(batch) >= 3
        hang.append(batch[1])
        calls.clear()
        analysis = face_degeneracy_from_polynomial(CUBE, [X, Y, Z], check=False)
        assert calls == [batch, [batch[1]], batch[2:]]
        assert analysis.faces[batch[1]].degenerate is True
        assert all(f.degenerate is not None for f in analysis.faces)

    def test_a_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: "Singular")
        monkeypatch.setattr(degeneracy_module, "_run", lambda *args: ("   ? error\n", False))
        with pytest.raises(ComputationError, match="Singular"):
            face_degeneracy_from_polynomial(CUBE, [X, Y, Z], check=False)


@requires_singular
class TestIntegral:
    def test_the_method_is_cached_for_each_choice_of_the_arguments(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        m1, m2, s = symbol(fi, "m_1"), symbol(fi, "m_2"), symbol(fi, "s")
        point = {m1**2: 1, m2**2: 4, s: 9}
        at_point = fi.face_degeneracy(point)
        assert at_point == face_degeneracy(fi, point)
        assert fi.face_degeneracy({s: 9, m2**2: 4, m1**2: Fraction(1)}) is at_point
        assert fi.face_degeneracy(point, check=False) is not at_point
        generic = fi.face_degeneracy()
        assert generic.mode == "generic"
        assert fi.face_degeneracy() is generic
        assert fi.with_().face_degeneracy() is not generic

    def test_kinematic_constraints_are_refused(self) -> None:
        s = sp.Symbol("s", real=True)
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])
        with pytest.raises(ValidationError, match="kinematic_constraints"):
            fi.face_degeneracy()


def test_exports() -> None:
    import feynkit

    assert feynkit.face_degeneracy is face_degeneracy
    assert feynkit.face_degeneracy_from_polynomial is face_degeneracy_from_polynomial
    assert feynkit.DegeneracyAnalysis is DegeneracyAnalysis
    assert feynkit.FaceDegeneracy is FaceDegeneracy
