"""Tests for the principal A-determinant and Landau surfaces.

Oracles: the one-loop closed form of Dlapa, Helmer, Papathanasiou and
Tellander (arXiv:2304.02629, eq. 1LoopEA): the reduced principal
A-determinant of G = U + F is the product of the principal minors of the
modified Cayley matrix. Face-by-face computation must reproduce it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import random
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
import sympy.core.random as sympy_random
from sympy.polys.rings import PolyRing

from feynkit import Edge, FeynmanIntegral, Graph, landau_analysis, landau_analysis_from_polynomial
from feynkit import landau as landau_module
from feynkit import point_count as point_count_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.generate import generate_graphs
from feynkit.io.report import AnalysisReport
from feynkit.io.report_text import render_text
from feynkit.kinematics.mandelstam import standard_invariants
from feynkit.landau import (
    LandauAnalysis,
    _eliminate_singular,
    _elimination_discriminant,
    _factor_list,
    _factor_lists,
    _factorize_flint,
    _factorize_singular,
    _images,
    _modified_cayley_matrix,
    _normalised,
    _normalised_poly,
    _one_loop_cycle,
    _read_singular_polynomial,
    _renaming,
    _singular_binary,
    _univariate_discriminant,
    one_loop_bridge_poles,
    one_loop_landau_surfaces,
    one_loop_landau_surfaces_by_type,
    one_loop_principal_a_determinant,
)
from feynkit.point_count import critical_point_count
from feynkit.polytope import faces as polytope_faces
from feynkit.polytope import lattice_coordinates
from feynkit.systems.monomial import extract_monomial_support
from tests.test_pld import _NAME, FIXTURES, _python

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

# SymPy's factorisation, where a test uses it, runs from a fixed state of its generator.
pytestmark = pytest.mark.usefixtures("sympy_seeded")


def _monic(expr: sp.Expr) -> sp.Expr:
    """Normalise a polynomial factor up to a constant, for set comparison."""
    expr = sp.expand(expr)
    return sp.Poly(expr, *sorted(expr.free_symbols, key=str)).monic().as_expr()


def _factor_set(
    factors: tuple[sp.Expr, ...] | list[sp.Expr], symbols: set[sp.Symbol]
) -> set[sp.Expr]:
    """Normalised set of the given irreducible factors that involve the symbols."""
    return {_monic(f) for f in factors if f.free_symbols & symbols}


def _same_surfaces(fi: FeynmanIntegral) -> bool:
    """The faces give the factors of the one-loop closed form and its bridge poles, as they
    do for generic kinematics; special kinematics can keep a factor out of the faces (see
    test_a_component_only_in_the_limit)."""
    kin = _kin(fi)
    faces = _factor_set(landau_analysis(fi).landau_surfaces, kin)
    closed = _factor_set(one_loop_landau_surfaces(fi), kin)
    assert faces == closed, faces ^ closed
    return True


def _kin(fi: FeynmanIntegral) -> set[sp.Symbol]:
    return (
        fi.symanzik.f.free_symbols - set(fi.symanzik.schwinger_parameters) - {fi.graph.energy_scale}
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def massive_bubble() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("11e|e|:nn")


@pytest.fixture(scope="module")  # type: ignore[misc]
def massless_triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


class TestMassiveBubble:
    def test_face_result_matches_one_loop_closed_form(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        assert _same_surfaces(massive_bubble)

    def test_threshold_and_pseudothreshold_are_surfaces(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        """The Kallen factor splits over the masses into s = (m1 + m2)^2 and s = (m1 - m2)^2."""
        m1, m2 = (sp.Symbol(f"m_{i}", nonnegative=True, real=True) for i in (1, 2))
        s = sp.Symbol("s", real=True)
        surfaces = _factor_set(landau_analysis(massive_bubble).landau_surfaces, {s, m1, m2})
        normal = _monic((m1 + m2) ** 2 - s)
        pseudo = _monic((m1 - m2) ** 2 - s)
        assert normal in surfaces and pseudo in surfaces
        assert s in surfaces  # second-type (Gram) singularity p^2 = 0
        assert m1 in surfaces and m2 in surfaces  # vertex factors: mass singularities

    def test_kallen_factor_appears_once(self, massive_bubble: FeynmanIntegral) -> None:
        """Regression: the edge exponent bug squared the Kallen factor."""
        s = sp.Symbol("s", real=True)
        la = landau_analysis(massive_bubble)
        edge = [
            f
            for f in la.face_discriminants
            if f.dimension == 1 and s in f.discriminant.free_symbols
        ]
        assert len(edge) == 1
        assert sp.degree(sp.expand(edge[0].discriminant), s) == 2


class TestUnivariateDiscriminant:
    """The edge discriminant is the standard polynomial discriminant."""

    @pytest.mark.parametrize(
        "exponents",
        [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 3], [0, 2, 3], [0, 1, 2, 4]],
    )
    def test_equals_the_sympy_discriminant(self, exponents: list[int]) -> None:
        z = sp.symbols(f"z0:{len(exponents)}")
        t = sp.Symbol("t")
        expected = sp.discriminant(sum(c * t**e for c, e in zip(z, exponents, strict=True)), t)
        got = _univariate_discriminant(list(z), exponents)
        assert sp.expand(got - expected) == 0
        assert got.is_polynomial(*z)

    def test_the_cubic_is_the_classical_discriminant(self) -> None:
        z0, z1, z2, z3 = sp.symbols("z0:4")
        classical = (
            z1**2 * z2**2
            - 4 * z0 * z2**3
            - 4 * z1**3 * z3
            + 18 * z0 * z1 * z2 * z3
            - 27 * z0**2 * z3**2
        )
        got = _univariate_discriminant([z0, z1, z2, z3], [0, 1, 2, 3])
        assert sp.expand(got - classical) == 0

    def test_an_edge_with_four_lattice_points_has_a_polynomial_discriminant(self) -> None:
        a, b, c, d, e = sp.symbols("a b c d e")
        x, y = sp.symbols("x y")
        g = a + b * x + c * x**2 + d * x**3 + e * y
        analysis = landau_analysis_from_polynomial(g, [x, y])
        (edge,) = [
            f for f in analysis.face_discriminants if f.dimension == 1 and len(f.exponents) == 4
        ]
        expected = sp.discriminant(a + b * x + c * x**2 + d * x**3, x)
        assert sp.expand(edge.discriminant - expected) == 0
        assert edge.discriminant.is_polynomial(a, b, c, d)


class TestMasslessTriangle:
    def test_edges_are_trivial(self, massless_triangle: FeynmanIntegral) -> None:
        """Every edge of the massless triangle's polytope is a two-point simplex."""
        la = landau_analysis(massless_triangle)
        assert all(f.discriminant == 1 for f in la.face_discriminants if f.dimension == 1)

    def test_surfaces_are_external_masses_and_gram_determinant(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        p1, p2, p3 = (sp.Symbol(f"p{i}^2", real=True) for i in (1, 2, 3))
        surfaces = _factor_set(landau_analysis(massless_triangle).landau_surfaces, {p1, p2, p3})
        gram = sp.expand(p1**2 + p2**2 + p3**2 - 2 * p1 * p2 - 2 * p1 * p3 - 2 * p2 * p3)
        assert surfaces == {p1, p2, p3, _monic(gram)}

    def test_face_result_matches_one_loop_closed_form(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        assert _same_surfaces(massless_triangle)


class TestOneMassTriangle:
    @requires_singular
    def test_face_result_matches_one_loop_closed_form(self) -> None:
        assert _same_surfaces(FeynmanIntegral.from_cnickel("12e|2e|e|:nzz"))


class TestMasslessBox:
    @requires_singular
    def test_face_result_matches_one_loop_closed_form(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        assert not landau_analysis(fi).skipped_faces
        assert _same_surfaces(fi)


class TestStructure:
    def test_massless_bubble_is_singular_only_at_p_squared_zero(self) -> None:
        la = landau_analysis(FeynmanIntegral.from_cnickel("11e|e|:zz"))
        assert la.landau_surfaces == (sp.Symbol("s", real=True),)

    def test_simplex_faces_have_unit_discriminant(self, massive_bubble: FeynmanIntegral) -> None:
        la = landau_analysis(massive_bubble)
        assert all(
            f.discriminant == 1 for f in la.face_discriminants if f.is_simplex and f.dimension > 0
        )

    def test_faces_include_every_dimension(self, massive_bubble: FeynmanIntegral) -> None:
        dims = {f.dimension for f in landau_analysis(massive_bubble).face_discriminants}
        assert dims == {0, 1, 2}

    def test_polynomial_interface_matches_integral_interface(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        sym = massive_bubble.symanzik
        la_poly = landau_analysis_from_polynomial(
            sym.g, list(sym.lp_parameters), scale=massive_bubble.graph.energy_scale
        )
        la_int = landau_analysis(massive_bubble)
        assert la_poly.landau_surfaces == la_int.landau_surfaces

    def test_closed_form_rejects_multi_loop(self) -> None:
        with pytest.raises(ValueError):
            one_loop_principal_a_determinant(FeynmanIntegral.from_cnickel("111e|e|:nnn"))

    def test_empty_polynomial_trivial(self) -> None:
        u = sp.symbols("u1:3")
        la = landau_analysis_from_polynomial(sp.Integer(0), list(u))
        assert la.landau_surfaces == ()


class TestBanana:
    @requires_singular
    def test_two_loop_banana_thresholds(self) -> None:
        """PLD example 3.4: the massive banana B_3 has factors m_e^2, s and
        s - (m_1 +/- m_2 +/- m_3)^2."""
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        la = landau_analysis(fi)
        s = sp.Symbol("s", real=True)
        m = [sp.Symbol(f"m_{i}", nonnegative=True, real=True) for i in (1, 2, 3)]
        surfaces = _factor_set(la.landau_surfaces, {s, *m})
        for e2 in (1, -1):
            for e3 in (1, -1):
                assert _monic((m[0] + e2 * m[1] + e3 * m[2]) ** 2 - s) in surfaces
        assert s in surfaces


class TestEnergyScaleExcluded:
    """Task 5b: mu scales every coefficient of F but is not itself a singular locus."""

    @pytest.mark.parametrize(
        "cnickel",
        [
            "11e|e|:nn",
            pytest.param("12e|2e|e|:nzz", marks=requires_singular),
            pytest.param("111e|e|:nnn", marks=requires_singular),
        ],
    )
    def test_no_surface_is_pure_energy_scale(self, cnickel: str) -> None:
        fi = FeynmanIntegral.from_cnickel(cnickel)
        mu = fi.graph.energy_scale
        surfaces = landau_analysis(fi).landau_surfaces
        assert all(not (f.free_symbols <= {mu}) for f in surfaces)

    def test_massive_bubble_surfaces_match_closed_form_exactly(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        """With mu dropped, the full surface sets agree, up to sign, with no filtering needed."""
        faces = {_monic(f) for f in landau_analysis(massive_bubble).landau_surfaces}
        closed = {_monic(f) for f in one_loop_landau_surfaces(massive_bubble)}
        assert faces == closed


class TestOneLoopSurfaceTypes:
    def test_massive_bubble_types(self, massive_bubble: FeynmanIntegral) -> None:
        """Cayley minors (no index 0) against Gram minors (index 0 included).

        subset (1,) gives m_1, (2,) gives m_2 and (1, 2) gives the Kallen
        factor lambda(s, m_1^2, m_2^2), split into its two linear factors in
        s: all three are first type. subset (0, 1, 2) gives s, the only
        second-type factor; (0, 1) and (0, 2) are constants and contribute
        nothing.
        """
        first, second = one_loop_landau_surfaces_by_type(massive_bubble)
        m1, m2 = (e.get_mass() for e in massive_bubble.graph.get_internal_edges())
        s = sp.Symbol("s", real=True)

        assert m1 in first
        assert m2 in first
        assert s in second
        assert all(s != f for f in first)

        kallen = s**2 + m1**4 + m2**4 - 2 * s * m1**2 - 2 * s * m2**2 - 2 * m1**2 * m2**2
        threshold_factors = [f for f in first if s in f.free_symbols]
        assert len(threshold_factors) == 2
        assert sp.expand(sp.prod(threshold_factors) - kallen) == 0

        assert tuple(one_loop_landau_surfaces(massive_bubble)) == first + tuple(
            x for x in second if x not in first
        )

    def test_massless_triangle_first_type_precedence(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        """The three external masses arise from both Cayley and Gram minors.

        Each size-2 Cayley minor (subsets {1, 2}, {1, 3}, {2, 3}, none
        containing index 0) gives one of p1^2, p2^2, p3^2, and each size-3
        Gram minor (subsets {0, 1, 2}, {0, 1, 3}, {0, 2, 3}) gives the same
        factor again, so all three are first type only, per the "both
        types" rule. The size-4 Gram minor {0, 1, 2, 3} gives a fourth,
        genuinely second-type factor: the Kallen function of p1^2, p2^2 and
        p3^2.
        """
        first, second = one_loop_landau_surfaces_by_type(massless_triangle)
        p1sq, p2sq, p3sq = standard_invariants(3).external_masses

        assert set(first) == {p1sq, p2sq, p3sq}

        kallen = p1sq**2 + p2sq**2 + p3sq**2 - 2 * p1sq * p2sq - 2 * p1sq * p3sq - 2 * p2sq * p3sq
        extra = [f for f in second if f not in first]
        assert len(extra) == 1
        gram = extra[0]
        assert sp.expand(gram - kallen) == 0
        assert set(second) == {p1sq, p2sq, p3sq, gram}

        merged = one_loop_landau_surfaces(massless_triangle)
        for p_sq in (p1sq, p2sq, p3sq):
            assert merged.count(p_sq) == 1
        assert tuple(merged) == first + (gram,)


def fake_singular(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
    error: Exception | None = None,
) -> dict[str, object]:
    """Replace Singular by a stub that prints stdout or raises error; returns what it was given."""
    seen: dict[str, object] = {}

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs, script=Path(args[-1]).read_text())
        if error is not None:
            raise error
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(landau_module, "_singular_binary", lambda: "Singular")
    monkeypatch.setattr(landau_module.subprocess, "run", run)
    return seen


W, T, X, Y, Z = sp.symbols("w t x y z")


def eliminate(timeout: float | None = None, *, decompose: bool = False) -> list[sp.Poly]:
    """A system in w and t over x, y and z, which Singular names v0 to v4."""
    return _eliminate_singular(
        [W * T - 1], [W, T], [X, Y, Z], "Singular", points=10, timeout=timeout, decompose=decompose
    ).generators


def singular_term(k: int, exponents: tuple[int, ...]) -> str:
    """The k-th term, (-1)^k (k + 1) x^a y^b z^c, as Singular prints it."""
    factors = [str(k + 1)] + [
        f"v{2 + i}" + (f"^{e}" if e > 1 else "") for i, e in enumerate(exponents) if e
    ]
    return ("-" if k % 2 else "+") + "*".join(factors)


class TestSingularOutput:
    """Singular prints each generator of the elimination ideal expanded on one line."""

    EXPONENTS = list(itertools.product(range(28), repeat=3))[:20000]

    def test_twenty_thousand_terms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # SymPy's parser recurses once per term and fails on such a line.
        line = "".join(singular_term(k, e) for k, e in enumerate(self.EXPONENTS)).lstrip("+")
        expected = {e: (-1) ** k * (k + 1) for k, e in enumerate(self.EXPONENTS)}
        assert _read_singular_polynomial(line, 2, 3) == expected
        fake_singular(monkeypatch, line + "\n")
        (generator,) = eliminate()
        assert generator.gens == (X, Y, Z)
        assert generator.as_dict() == expected

    def test_rational_coefficients_and_several_generators(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_singular(monkeypatch, "1/2*v2^2-3*v3*v4+7\n0\nv4\n")
        assert [g.as_expr() for g in eliminate()] == [X**2 / 2 - 3 * Y * Z + 7, Z]

    def test_an_error_printed_on_stdout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Singular reports an error in the script on stdout and still exits with 0.
        fake_singular(monkeypatch, "   ? error occurred in or before elim.sing line 2\n")
        with pytest.raises(
            ComputationError, match="face with 10 points from 50 characters"
        ) as info:
            eliminate()
        assert info.value.__cause__ is None and info.value.__suppress_context__

    def test_a_variable_that_was_eliminated_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, "v0*v2-1\n")
        with pytest.raises(ComputationError, match="cannot read"):
            eliminate()

    def test_the_message_stays_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, "v2*" * 100000 + "?\n")
        with pytest.raises(ComputationError, match="from 300002 characters of output") as info:
            eliminate()
        assert len(str(info.value)) < 200
        # repr writes a control character as four, so the line is cut after repr.
        fake_singular(monkeypatch, "\x01" * 1000 + "\n")
        with pytest.raises(ComputationError, match=r"at '\\x01\\x01") as info:
            eliminate()
        assert len(str(info.value)) < 200

    def test_a_failure_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, returncode=1, stderr="halt: out of memory\n")
        message = "Singular failed on a face with 10 points, 0 characters of output: halt: out of"
        with pytest.raises(ComputationError, match=message):
            eliminate()
        fake_singular(monkeypatch, returncode=1, stderr="halt:\n  out of memory\t" * 1000)
        with pytest.raises(ComputationError, match="output: halt: out of memory halt: out") as info:
            eliminate()
        assert len(str(info.value)) < 200

    def test_only_ascii_digits_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arabic-Indic three and two, which int() reads as 3 and 2.
        assert _read_singular_polynomial("v\u0663", 2, 3) is None
        assert _read_singular_polynomial("\u0662*v2", 2, 3) is None
        fake_singular(monkeypatch, "v\u0663\n")
        with pytest.raises(ComputationError, match="cannot read"):
            eliminate()

    def test_a_zero_denominator_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _read_singular_polynomial("1/0*v2", 2, 3) is None
        fake_singular(monkeypatch, "1/0*v2\n")
        with pytest.raises(ComputationError, match="cannot read"):
            eliminate()

    def test_output_that_is_not_text_raises(self, tmp_path: Path) -> None:
        # A stand-in for Singular that prints the byte 0xff, which is not UTF-8.
        binary = tmp_path / "Singular"
        binary.write_text("#!/bin/sh\nprintf '\\377\\n'\n")
        binary.chmod(0o755)
        with pytest.raises(ComputationError, match="cannot read"):
            _eliminate_singular([W * T - 1], [W, T], [X, Y, Z], str(binary), points=10)

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = fake_singular(monkeypatch, error=subprocess.TimeoutExpired(["Singular"], 7))
        with pytest.raises(
            ComputationError, match="face with 10 points within timeout=7 s"
        ) as info:
            eliminate(timeout=7)
        assert seen["timeout"] == 7
        assert info.value.__suppress_context__

    def test_a_face_past_the_time_limit_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, massive_bubble: FeynmanIntegral
    ) -> None:
        # The polygon is the bubble's only face that needs an elimination.
        fake_singular(monkeypatch, error=subprocess.TimeoutExpired(["Singular"], 7))
        analysis = landau_analysis(massive_bubble, timeout=7)
        assert [len(face) for face in analysis.skipped_faces] == [5]
        assert all(face.dimension < 2 for face in analysis.face_discriminants)
        assert set(analysis.landau_surfaces) == set(one_loop_landau_surfaces(massive_bubble)) - {S}

    def test_timeout_reaches_singular(
        self, monkeypatch: pytest.MonkeyPatch, massive_bubble: FeynmanIntegral
    ) -> None:
        # A generator in the first kinematic symbol, so that the ideal is not zero.
        seen = fake_singular(monkeypatch, "v3\n")
        landau_analysis(massive_bubble)
        assert seen["timeout"] == landau_module.DEFAULT_FACE_TIMEOUT
        landau_analysis(massive_bubble, timeout=None)
        assert seen["timeout"] is None
        landau_analysis(massive_bubble, timeout=7)
        assert seen["timeout"] == 7
        landau_analysis(massive_bubble, timeout=0.5)
        assert seen["timeout"] == 0.5
        landau_analysis(massive_bubble, timeout=2_000_000)
        assert seen["timeout"] == 2_000_000

    @pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan"), "5", 3e6])
    def test_timeout_must_be_a_number_of_seconds(
        self, timeout: object, massive_bubble: FeynmanIntegral
    ) -> None:
        # subprocess.run cannot wait past about 2.1e6 s.
        with pytest.raises(ValidationError, match="timeout must be None or a number of seconds"):
            landau_analysis(massive_bubble, timeout=timeout)  # type: ignore[arg-type]


S, T_ = (sp.Symbol(name, real=True) for name in ("s", "t"))
M1, M2 = (sp.Symbol(f"m_{i}", positive=True) for i in (1, 2))


class TestRenaming:
    """The distinct coefficients that are not constant get fresh symbols only when they are
    linearly independent linear forms in symbols, or in squares of symbols."""

    def test_independent_forms(self) -> None:
        renaming, squared = _renaming([sp.Integer(1), -S, M1**2 + M2**2 - S, -S, M1**2])
        assert list(renaming) == [-S, M1**2 + M2**2 - S, M1**2]
        assert len(set(renaming.values())) == 3
        assert squared

    def test_forms_in_symbols_alone(self) -> None:
        renaming, squared = _renaming([-S, S - T_])
        assert len(renaming) == 2
        assert not squared

    @pytest.mark.parametrize(
        "coefficients",
        [
            [S, T_, S + T_],  # dependent
            [-S, S],  # dependent
            [M1, M1**2],  # a symbol both alone and squared
            [S * T_],  # not linear
            [S + 1],  # a constant term
            [S / T_],  # not a polynomial
            [S**3],  # neither a symbol nor its square
            [sp.Float(0.5) * S],  # not rational
        ],
    )
    def test_anything_else_is_not_renamed(self, coefficients: list[sp.Expr]) -> None:
        assert _renaming(coefficients) == ({}, False)


# A face of the polytope of the massless hexagon 12e|3e|4e|5e|5e|e|: u_1, u_2, u_3 and u_6 and
# their products in pairs, the massless box that u_4 = u_5 = 0 leaves. Its coefficients are
# 1 on u_i and Y_ij / mu^2 on u_i u_j, where Y_ij is minus the square of the momentum between
# propagators i and j, dense in the hexagon's invariants.
HEXAGON_FACE = (
    (1, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 1),
    (0, 0, 1, 0, 0, 1),
    (0, 1, 0, 0, 0, 1),
    (1, 1, 0, 0, 0, 0),
    (1, 0, 1, 0, 0, 0),
    (0, 1, 1, 0, 0, 0),
)
MU = sp.Symbol("mu", positive=True)
P1, P2, P3, P4, P5, P6 = (sp.Symbol(f"p{i}^2", real=True) for i in range(1, 7))
S12, S23, S34, S45, S123, S234, S345, S1234, S2345 = (
    sp.Symbol(f"s{i}", real=True)
    for i in ("12", "23", "34", "45", "123", "234", "345", "1234", "2345")
)
HEXAGON_Y = {
    (1, 6): (
        -P1 - P2 - P3 - P4 - P5 - P6 + S12 - S123 + S1234 + S23 - S234 + S2345 + S34 - S345 + S45
    ),
    (3, 6): -P4 - P5 - P6 - S123 + S1234 + S45,
    (2, 6): -P3 - P4 - P5 + S34 - S345 + S45,
    (1, 2): -P1,
    (1, 3): -P2,
    (2, 3): -S12,
}


def backends() -> list[object]:
    """Singular when it is installed, and the SymPy fallback."""
    return [pytest.param("singular", marks=requires_singular), "sympy"]


class TestElimination:
    @requires_singular
    def test_a_hexagon_face_is_the_gram_determinant_of_its_box(self) -> None:
        # Eliminated with mu and the fifteen invariants as variables, this face took Singular
        # over four minutes and gave a generator of 13730 terms. Renamed, it takes a fraction
        # of a second.
        coefficients = [sp.Integer(1)] * 4 + [y / MU**2 for y in HEXAGON_Y.values()]
        kinematic = set().union(*(y.free_symbols for y in HEXAGON_Y.values())) | {MU}
        renaming, squared = _renaming(list(HEXAGON_Y.values()))
        assert len(renaming) == 6 and not squared
        factors, principal, dominant = _elimination_discriminant(
            coefficients, lattice_coordinates(np.array(HEXAGON_FACE)), kinematic, scale=MU
        )
        # The modified Cayley matrix of the box: Y bordered by a row and a column of ones.
        edges = (1, 2, 3, 6)
        cayley = sp.zeros(5, 5)
        for a, i in enumerate(edges, start=1):
            cayley[0, a] = cayley[a, 0] = 1
            for b, j in enumerate(edges, start=1):
                if i != j:
                    cayley[a, b] = HEXAGON_Y[min(i, j), max(i, j)]
        assert principal and not dominant
        assert len(factors) == 1
        assert len(sp.Add.make_args(factors[0])) == 145
        assert sp.expand(cayley.det() - 2 * factors[0]) == 0

    @pytest.mark.parametrize("backend", backends())
    def test_faces_are_eliminated_at_mu_one(
        self, backend: str, massive_bubble: FeynmanIntegral, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With mu as a variable the polygon's elimination ideal also had a component at
        # mu = 0, so it had two generators, whose factors added mu and the two thresholds.
        if backend == "sympy":
            monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        mu = massive_bubble.graph.energy_scale
        faces = landau_analysis(massive_bubble).face_discriminants
        (polygon,) = [face for face in faces if face.dimension == 2]
        assert polygon.principal
        assert polygon.discriminant == S
        assert any(mu in face.discriminant.free_symbols for face in faces if face.dimension == 0)

    def test_without_the_scale_mu_is_a_variable(self, massive_bubble: FeynmanIntegral) -> None:
        symanzik = massive_bubble.symanzik
        analysis = landau_analysis_from_polynomial(symanzik.g, list(symanzik.lp_parameters))
        (polygon,) = [face for face in analysis.face_discriminants if face.dimension == 2]
        assert not polygon.principal

    @requires_singular
    @pytest.mark.parametrize(
        "cnickel", ["11e|e|:nn", "12e|2e|e|:nnn", "12e|2e|e|:nzz", "12e|3e|3e|e|:zzzz"]
    )
    def test_the_renaming_leaves_the_analysis_unchanged(
        self, cnickel: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fi = FeynmanIntegral.from_cnickel(cnickel)
        renamed = landau_analysis(fi)
        monkeypatch.setattr(landau_module, "_renaming", lambda coefficients: ({}, False))
        plain = landau_analysis(fi)
        assert renamed.face_discriminants == plain.face_discriminants
        assert set(renamed.landau_surfaces) == set(plain.landau_surfaces)

    @pytest.mark.parametrize("backend", backends())
    def test_a_factor_in_squared_masses_is_factored_again(self, backend: str) -> None:
        # In the fresh symbols the edge's discriminant c_2^2 - 4 c_1 c_3 is irreducible;
        # substituted back it is the Källén function, which splits in the masses.
        coefficients = [M1**2, M1**2 + M2**2 - S, M2**2]
        exponents = [(0,), (1,), (2,)]
        renaming, squared = _renaming(coefficients)
        assert len(renaming) == 3 and squared
        factors, principal, dominant = _elimination_discriminant(
            coefficients, exponents, {S, M1, M2}, backend
        )
        assert principal
        assert {_monic(f) for f in factors} == {
            _monic(S - (M1 + M2) ** 2),
            _monic(S - (M1 - M2) ** 2),
        }
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(landau_module, "_renaming", lambda coefficients: ({}, False))
            plain = _elimination_discriminant(coefficients, exponents, {S, M1, M2}, backend)
        assert plain == (factors, principal, dominant)

    def test_mu_is_set_to_one_only_where_that_is_exact(self) -> None:
        exact = landau_module._unit_scale_is_exact
        u = [((1, 0), sp.Integer(1)), ((0, 1), sp.Integer(1))]
        f = [((2, 0), M1**2 / MU**2), ((1, 1), (M1**2 + M2**2 - S) / MU**2)]
        assert exact(u + f, MU)
        # mu in a numerator, and a denominator that is not a power of mu.
        assert not exact(u + [((1, 1), (MU**2 - S) / MU**2)], MU)
        assert not exact(u + [((1, 1), S / (MU**2 + 1))], MU)
        # Powers of mu that are not an affine function of the exponents: 0 at (2, 0), -2 at
        # (1, 1), with 0 at (1, 0) and (0, 1).
        assert not exact(u + [((2, 0), M1**2), ((1, 1), S / MU**2)], MU)

    @pytest.mark.parametrize("power", [2, 4])
    def test_kinematics_scaled_by_a_power_of_mu_are_eliminated_at_mu_one(
        self, power: int, massive_bubble: FeynmanIntegral, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # With every squared mass and invariant times mu^power the coefficients of F are
        # mu^(power - 2) times kinematics free of mu: free of mu for power 2, and still a power
        # of mu, affine in the exponent, for power 4.
        mu = massive_bubble.graph.energy_scale
        symanzik = massive_bubble.symanzik
        variables = list(symanzik.lp_parameters)
        masses = {edge.get_mass() for edge in massive_bubble.graph.get_internal_edges()}
        scaling = {m: mu ** (power // 2) * m for m in masses}
        scaling.update({x: mu**power * x for x in _kin(massive_bubble) - masses})
        g = sp.expand(symanzik.g.subs(scaling, simultaneous=True))
        support = landau_module.extract_monomial_support(g, variables)
        assert landau_module._unit_scale_is_exact(support, mu)
        scales: list[sp.Symbol | None] = []
        eliminate = landau_module._elimination_discriminant

        def spy(*args: object, **kwargs: object) -> tuple[list[sp.Expr], bool, bool]:
            scales.append(kwargs["scale"])  # type: ignore[arg-type]
            return eliminate(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(landau_module, "_elimination_discriminant", spy)
        analysis = landau_analysis_from_polynomial(g, variables, scale=mu)
        assert scales == [mu]
        assert analysis.landau_surfaces == landau_analysis(massive_bubble).landau_surfaces

    @requires_singular
    @pytest.mark.parametrize(("cnickel", "count"), [("12e|2e|e|:zzz", 3), ("12e|2e|e|:nnn", 15)])
    def test_kinematics_with_the_scale_keep_it_a_variable(self, cnickel: str, count: int) -> None:
        # At p_1^2 = mu^2 a coefficient of F is no longer a power of mu times kinematics free
        # of it, and eliminating at mu = 1 gave lambda(1, p_2^2, p_3^2).
        fi = FeynmanIntegral.from_cnickel(cnickel)
        mu = fi.graph.energy_scale
        p1, p2, p3 = standard_invariants(3).external_masses
        products = {
            pair: sp.expand(sp.sympify(value).subs(p1, mu**2))
            for pair, value in fi.momentum_products.items()
        }
        at_mu = fi.with_(momentum_products=products)
        surfaces = set(landau_analysis(at_mu).landau_surfaces)
        kallen = mu**4 + p2**2 + p3**2 - 2 * mu**2 * p2 - 2 * mu**2 * p3 - 2 * p2 * p3
        assert kallen in surfaces
        assert len(surfaces) == count
        # With mu as a plain kinematic symbol the analysis gives the same factors, and mu.
        symanzik = at_mu.symanzik
        plain = landau_analysis_from_polynomial(symanzik.g, list(symanzik.lp_parameters))
        assert surfaces == set(plain.landau_surfaces) - {mu}


@pytest.mark.slow
class TestMasslessPolygons:
    """The pentagon did not finish in forty minutes, and the hexagon raised RecursionError."""

    @requires_singular
    def test_pentagon(self) -> None:
        analysis = landau_analysis(FeynmanIntegral.from_cnickel("12e|3e|4e|4e|e|:zzzzz"))
        assert len(analysis.landau_surfaces) == 31
        assert [len(face) for face in analysis.skipped_faces] == [15]

    @requires_singular
    def test_hexagon(self) -> None:
        analysis = landau_analysis(FeynmanIntegral.from_cnickel("12e|3e|4e|5e|5e|e|:zzzzzz"))
        assert len(analysis.landau_surfaces) == 71
        assert len(analysis.skipped_faces) == 8


def pld_entry(name: str) -> tuple[sp.Expr, list[sp.Symbol], set[sp.Expr]]:
    """U + F of a committed database entry, its variables, and the irreducible factors of the
    components the database computed from faces, with PLD_sym or PLD_num."""
    text = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
    fields = dict(re.findall(r"^(\w+) = (.+?)\s*$", text.split("# Component 1")[0], re.M))
    names = [_python(n) for n in _NAME.findall(fields["variables"] + fields["parameters"])]
    symbols = {n: sp.Symbol(n) for n in names}
    variables = [symbols[_python(n)] for n in _NAME.findall(fields["variables"])]
    g = sp.sympify(_python(fields["U"]), locals=symbols) + sp.sympify(
        _python(fields["F"]), locals=symbols
    )
    methods = dict(re.findall(r"^computed_with\[(\d+)\] = (.+?)\s*$", text, re.M))
    components: set[sp.Expr] = set()
    for k, component in re.findall(r"^D\[(\d+)\] = (.+?)\s*$", text, re.M):
        if "PLD_sym" in methods[k] or "PLD_num" in methods[k]:
            polynomial = sp.sympify(_python(component), locals=symbols)
            components.update(base for base, _ in sp.factor_list(polynomial)[1])
    return g, variables, components


# The face of the database's massive kite on which x_3 = 0: fourteen monomials of F.
KITE_FACE = (
    (2, 1, 0, 0, 0),
    (2, 0, 0, 0, 1),
    (1, 2, 0, 0, 0),
    (0, 2, 0, 1, 0),
    (0, 2, 0, 0, 1),
    (0, 1, 0, 2, 0),
    (0, 0, 0, 2, 1),
    (1, 0, 0, 0, 2),
    (0, 1, 0, 0, 2),
    (0, 0, 0, 1, 2),
    (1, 1, 0, 1, 0),
    (1, 0, 0, 1, 1),
    (0, 1, 0, 1, 1),
    (1, 1, 0, 0, 1),
)


class TestSeveralGenerators:
    """With several generators, a face contributes the factors of their greatest common
    divisor, the codimension-one part of their common zeros."""

    @requires_singular
    def test_a_kite_face(self) -> None:
        # Every factor of every generator gave lambda(m_1, m_2, m_5), the threshold at zero
        # momentum of the sunrise formed by the three propagators at internal vertex 3, and two
        # factors that are not components; only s divides them all.
        g, variables, _ = pld_entry("kite_generic_generic")
        support = dict(extract_monomial_support(sp.expand(g), variables))
        factors, principal, dominant = _elimination_discriminant(
            [support[point] for point in KITE_FACE],
            lattice_coordinates(np.array(KITE_FACE)),
            g.free_symbols - set(variables),
        )
        assert (factors, principal, dominant) == ([sp.Symbol("s")], False, False)

    @pytest.mark.parametrize("backend", backends())
    def test_generators_without_a_common_factor(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A massive bubble with a massless bridge: on this face the three generators are s^2,
        # s (m_1^2 - m_2^2) and a quartic, which share no factor.
        if backend == "sympy":
            monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        fi = FeynmanIntegral.from_cnickel("11e|2|e|:nnz")
        points = {(2, 0, 0), (0, 2, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)}
        analysis = landau_analysis(fi)
        (face,) = [f for f in analysis.face_discriminants if set(f.exponents) == points]
        assert face.discriminant == 1
        assert not face.principal
        m1, m2, _ = (edge.get_mass() for edge in fi.graph.get_internal_edges())
        # Every factor of every generator added m_1 - m_2, m_1 + m_2 and the quartic
        # 4 m_2^2 s - (m_1^2 - m_2^2)^2.
        assert set(analysis.landau_surfaces) == {
            m1,
            m2,
            S,
            sp.expand(S - (m1 + m2) ** 2),
            sp.expand(S - (m1 - m2) ** 2),
        }

    @requires_singular
    def test_a_component_only_in_the_limit(self) -> None:
        # With p_1^2 = 0 the massive triangle's Gram determinant is proportional to
        # (p_2^2 - p_3^2)^2, but the top face's generators are p_3^2 and p_2^2 - p_3^2, whose
        # locus has codimension two, and no face has p_2^2 = p_3^2 as a component. The closed
        # form keeps the factor; the faces, taking the gcd, leave it out.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        p1, p2, p3 = standard_invariants(3).external_masses
        products = {
            pair: sp.expand(sp.sympify(value).subs(p1, 0))
            for pair, value in fi.momentum_products.items()
        }
        on_shell = fi.with_(momentum_products=products)
        faces = set(landau_analysis(on_shell).landau_surfaces)
        closed = set(one_loop_landau_surfaces(on_shell))
        assert faces < closed
        assert closed - faces == {p2 - p3}

    @staticmethod
    def _triangle_with_massless_leg() -> tuple[FeynmanIntegral, sp.Expr]:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        p1, p2, p3 = standard_invariants(3).external_masses
        products = {
            pair: sp.expand(sp.sympify(value).subs(p1, 0))
            for pair, value in fi.momentum_products.items()
        }
        return fi.with_(momentum_products=products), p2 - p3

    def test_the_closed_form_keeps_the_component_lost_in_the_limit(self) -> None:
        on_shell, component = self._triangle_with_massless_leg()
        assert component in one_loop_landau_surfaces(on_shell)

    @requires_singular
    def test_the_component_lost_in_the_limit_is_a_limit_surface(self) -> None:
        # The Euler characteristic drops from 6 to 5 on p_2^2 = p_3^2.
        on_shell, component = self._triangle_with_massless_leg()
        analysis = landau_analysis(on_shell, limits=True)
        assert component not in analysis.landau_surfaces
        (limit,) = analysis.limit_surfaces
        assert limit.surface == component
        assert limit.confirmed
        assert (limit.generic_count, limit.counts) == (6, (5, 5))
        assert not analysis.limit_candidates
        closed = set(one_loop_landau_surfaces(on_shell))
        assert set(analysis.landau_surfaces) | {component} == closed

    @requires_singular
    @pytest.mark.parametrize(
        "name",
        [
            "par_zero_generic",
            "A4_zero_generic",
            pytest.param("kite_generic_generic", marks=pytest.mark.slow),
        ],
    )
    def test_the_database_components_from_faces(self, name: str) -> None:
        # The kite's polytope itself, 34 points, is skipped here as in the database, whose
        # three other components come from HyperInt alone. Its face of 26 points takes over
        # two minutes to decompose, past the default time limit.
        g, variables, components = pld_entry(name)
        analysis = landau_analysis_from_polynomial(
            g, variables, max_face_points=30, timeout=None, large_face_timeout=None
        )
        assert set(analysis.landau_surfaces) == components


def _massless_legs(cnickel: str, legs: tuple[int, ...]) -> FeynmanIntegral:
    """The integral with p_i^2 = 0 for the given legs."""
    fi = FeynmanIntegral.from_cnickel(cnickel)
    external = standard_invariants(fi.graph.external_legs).external_masses
    zero = {external[i - 1]: 0 for i in legs}
    products = {
        pair: sp.expand(sp.sympify(v).subs(zero)) for pair, v in fi.momentum_products.items()
    }
    return fi.with_(momentum_products=products)


class TestLimitSurfaces:
    """At special kinematics, the factors of the surfaces of the same graph with generic legs,
    restricted, that are not in the principal Landau determinant, each tested for a drop of the
    number of critical points at random points of it."""

    def test_generic_kinematics_have_no_parent(self, massive_bubble: FeynmanIntegral) -> None:
        analysis = landau_analysis(massive_bubble)
        assert analysis.parent is None
        assert analysis.limit_surfaces == analysis.limit_candidates == ()

    @requires_singular
    def test_the_parent_has_the_same_graph_and_generic_legs(self) -> None:
        on_shell = _massless_legs("12e|2e|e|:nnn", (1,))
        analysis = landau_analysis(on_shell, confirm=False, limits=True)
        generic = landau_analysis(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"))
        assert analysis.parent is not None
        assert analysis.parent.landau_surfaces == generic.landau_surfaces
        (candidate,) = analysis.limit_candidates
        p1, p2, p3 = standard_invariants(3).external_masses
        assert candidate.surface == p2 - p3
        # The Gram determinant, lambda(p_1^2, p_2^2, p_3^2), restricts to (p_2^2 - p_3^2)^2.
        assert sp.expand(sp.Mul(*candidate.parent_surfaces).subs(p1, 0)) == sp.expand(
            (p2 - p3) ** 2
        )
        assert candidate.reason == "not tested, since confirm is False"
        assert not candidate.confirmed
        assert analysis.limit_surfaces == ()

    @requires_singular
    def test_the_parent_is_analysed_once_for_its_restrictions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two sets of massless legs of the same triangle share the parent family.
        landau_module._parent_analysis.cache_clear()
        analysed: list[sp.Expr] = []
        analysis = landau_module._analysis

        def spy(g: sp.Expr, *args: object) -> LandauAnalysis:
            analysed.append(g)
            return analysis(g, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(landau_module, "_analysis", spy)
        generic = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn").symanzik.g
        for legs in [(1,), (2,), (1, 2)]:
            landau_analysis(_massless_legs("12e|2e|e|:nnn", legs), confirm=False, limits=True)
        assert analysed.count(generic) == 1
        assert len(analysed) == 4

    @requires_singular
    def test_the_parent_has_the_same_time_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        landau_module._parent_analysis.cache_clear()
        timeouts: list[object] = []
        analysis = landau_module._analysis

        def spy(g: sp.Expr, *args: object) -> LandauAnalysis:
            timeouts.append(args[-3:-1])
            return analysis(g, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(landau_module, "_analysis", spy)
        on_shell = _massless_legs("12e|2e|e|:nnn", (1,))
        landau_analysis(on_shell, confirm=False, limits=True)
        default = (landau_module.DEFAULT_FACE_TIMEOUT, landau_module.DEFAULT_LARGE_FACE_TIMEOUT)
        assert timeouts == [default] * 2
        landau_analysis(on_shell, confirm=False, timeout=30, large_face_timeout=3, limits=True)
        assert timeouts[2:] == [(30, 3), (30, 3)]

    def test_renamed_generic_kinematics_have_no_parent(self) -> None:
        # Invariants renamed by an invertible linear map are the generic family itself.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        p1, p2, p3 = standard_invariants(3).external_masses
        a, b = sp.symbols("a b", real=True)
        renamed = {p1: a + b, p2: a - b}
        products = {
            k: sp.expand(sp.sympify(v).subs(renamed)) for k, v in fi.momentum_products.items()
        }
        assert landau_analysis(fi.with_(momentum_products=products), limits=True).parent is None
        # A mass among the values is a specialisation.
        m1 = fi.graph.get_internal_edges()[0].get_mass()
        products = {
            k: sp.expand(sp.sympify(v).subs(p1, m1**2)) for k, v in fi.momentum_products.items()
        }
        assert landau_analysis(
            fi.with_(momentum_products=products), confirm=False, limits=True
        ).parent

    def test_the_default_of_limits_is_one_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The banana at s = m_1^2 has a parent, the triangle at p_1^2 = 0 too.
        banana = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        m1 = banana.graph.get_internal_edges()[0].get_mass()
        s = standard_invariants(2).external_masses[0]
        at_threshold = banana.with_(
            momentum_products={
                k: sp.sympify(v).subs(s, m1**2) for k, v in banana.momentum_products.items()
            }
        )
        triangle = _massless_legs("12e|2e|e|:nnn", (1,))
        monkeypatch.setattr(landau_module, "DEFAULT_LIMITS", "one-loop")
        assert landau_analysis(at_threshold, confirm=False).parent is None
        assert landau_analysis(triangle, confirm=False).parent is not None
        monkeypatch.setattr(landau_module, "DEFAULT_LIMITS", False)
        assert landau_analysis(triangle, confirm=False).parent is None
        assert landau_analysis(triangle, confirm=False, limits=True).parent is not None
        monkeypatch.setattr(landau_module, "DEFAULT_LIMITS", True)
        assert landau_analysis(at_threshold, confirm=False).parent is not None
        with pytest.raises(ValidationError, match="limits must be True, False or 'one-loop'"):
            landau_analysis(triangle, limits="two-loop")  # type: ignore[arg-type]

    def test_the_default_gives_no_limit_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert landau_module.DEFAULT_LIMITS is False

        def refuse(*args: object) -> None:
            raise AssertionError("analysed the parent family")

        monkeypatch.setattr(landau_module, "_parent_analysis", refuse)
        analysis = landau_analysis(_massless_legs("12e|2e|e|:nnn", (1,)), confirm=False)
        assert analysis.parent is None
        assert analysis.limit_surfaces == analysis.limit_candidates == ()

    def test_limits_can_be_left_out(self) -> None:
        analysis = landau_analysis(_massless_legs("12e|2e|e|:nnn", (1,)), limits=False)
        assert analysis.parent is None
        assert analysis.limit_candidates == ()

    @requires_singular
    def test_a_count_that_does_not_drop_leaves_a_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(point_count_module, "critical_point_count", lambda *a, **k: 6)
        analysis = landau_analysis(_massless_legs("12e|2e|e|:nnn", (1,)), limits=True)
        (candidate,) = analysis.limit_candidates
        assert candidate.counts == (6,)
        assert candidate.reason == "the count, 6, did not drop below 6"

    @requires_singular
    def test_a_count_that_fails_leaves_a_candidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        counts = iter([6])

        def count(*args: object, **kwargs: object) -> int:
            for value in counts:
                return value
            raise ComputationError("Singular did not finish within timeout=60 s")

        monkeypatch.setattr(point_count_module, "critical_point_count", count)
        analysis = landau_analysis(_massless_legs("12e|2e|e|:nnn", (1,)), limits=True)
        (candidate,) = analysis.limit_candidates
        assert candidate.reason == ("the count failed: Singular did not finish within timeout=60 s")

    @requires_singular
    def test_a_surface_without_a_rational_point_is_a_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(landau_module, "_rational_root", lambda poly: None)
        analysis = landau_analysis(_massless_legs("12e|2e|e|:nnn", (1,)), limits=True)
        (candidate,) = analysis.limit_candidates
        assert candidate.reason == "no rational point found on it"
        assert candidate.generic_count == 6

    @requires_singular
    def test_the_time_limit_applies_to_each_candidate(self) -> None:
        with pytest.raises(ValidationError, match="confirm_timeout must be a number"):
            landau_analysis(
                _massless_legs("12e|2e|e|:nnn", (1,)), confirm_timeout=None, limits=True
            )
        analysis = landau_analysis(
            _massless_legs("12e|2e|e|:nnn", (1,)), confirm_timeout=1e-9, limits=True
        )
        (candidate,) = analysis.limit_candidates
        assert candidate.reason is not None

    @requires_singular
    def test_a_generic_count_that_fails_leaves_every_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def count(*args: object, **kwargs: object) -> int:
            raise RuntimeError("critical_point_count needs Singular, which was not found")

        monkeypatch.setattr(point_count_module, "critical_point_count", count)
        analysis = landau_analysis(_massless_legs("12e|2e|e|:nnn", (1,)), limits=True)
        (candidate,) = analysis.limit_candidates
        assert candidate.reason == (
            "the count at a random point of the family failed: critical_point_count needs "
            "Singular, which was not found"
        )
        assert candidate.generic_count is None

    @requires_singular
    def test_the_points_are_on_the_surface(self) -> None:
        analysis = landau_analysis(_massless_legs("12e|3e|3e|e|:nnzz", (1,)), limits=True)
        assert analysis.limit_surfaces
        for limit in analysis.limit_surfaces:
            assert len(limit.points) == 2
            for point in limit.points:
                assert limit.surface.xreplace(dict(point)) == 0

    @requires_singular
    def test_a_polynomial_with_a_parent(self) -> None:
        # FMT24a example 3.9 as the restriction of the generic polynomial on the same support:
        # every factor of the restricted principal A-determinant that does not vanish is in the
        # principal Landau determinant, which also has bc - ad.
        z = sp.symbols("z1:7")
        monomials = [1, A1, A2, A1**2, A1 * A2, A1**2 * A2]
        parent = sum(c * m for c, m in zip(z, monomials, strict=True))
        restriction = dict(zip(z, [A_, A_ + B_, C_, B_, C_ + D_, D_], strict=True))
        f = sp.expand((1 + A1) * (A_ + B_ * A1 + C_ * A2 + D_ * A1 * A2))
        analysis = landau_analysis_from_polynomial(
            f, [A1, A2], parent=parent, restriction=restriction
        )
        assert analysis.parent is not None
        assert len(analysis.parent.landau_surfaces) == 7
        assert analysis.limit_surfaces == analysis.limit_candidates == ()

    def test_the_parent_must_restrict_to_the_polynomial(self) -> None:
        f = sp.expand((1 + A1) * (A_ + B_ * A1))
        with pytest.raises(ValidationError, match="does not restrict to"):
            landau_analysis_from_polynomial(f, [A1], parent=f + A1, restriction={})
        with pytest.raises(ValidationError, match="restriction needs a parent"):
            landau_analysis_from_polynomial(f, [A1], restriction={A_: B_})


def _specialised_one_loop_cases() -> list[tuple[str, tuple[int, ...]]]:
    """Every bubble, triangle and box with each mass massless or its own, and each non-empty
    set of massless legs; a bubble's two legs are massless together, at s = 0."""
    cases = []
    for cnickel in generate_graphs(1, range(2, 5)):
        n = len(cnickel.split(":")[1])
        subsets = (
            [(1, 2)]
            if n == 2
            else [c for r in range(1, n + 1) for c in itertools.combinations(range(1, n + 1), r)]
        )
        cases += [(cnickel, legs) for legs in subsets]
    return cases


@requires_singular
@pytest.mark.slow
@pytest.mark.parametrize(("cnickel", "legs"), _specialised_one_loop_cases())
def test_the_limit_surfaces_complete_the_closed_form(cnickel: str, legs: tuple[int, ...]) -> None:
    # The principal Landau determinant lies within the closed form, and with the limit
    # surfaces, each confirmed, it is the closed form.
    fi = _massless_legs(cnickel, legs)
    analysis = landau_analysis(fi, limits=True)
    closed = set(one_loop_landau_surfaces(fi))
    faces = set(analysis.landau_surfaces)
    assert not analysis.skipped_faces
    assert faces <= closed
    assert not analysis.limit_candidates
    assert faces | {limit.surface for limit in analysis.limit_surfaces} == closed


def test_there_are_121_specialised_one_loop_cases() -> None:
    assert len(_specialised_one_loop_cases()) == 3 + 4 * 7 + 6 * 15


def _symbols(fi: FeynmanIntegral) -> dict[str, sp.Symbol]:
    return {x.name: x for x in fi.symanzik.g.free_symbols}


def _generic_count(fi: FeynmanIntegral) -> int:
    """The number of critical points at a rational point off every Landau surface found."""
    g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
    variables = list(fi.symanzik.lp_parameters)
    kinematic = sorted(g.free_symbols - set(variables), key=str)
    values = dict(zip(kinematic, [sp.Rational(p, 7) for p in (13, 29, 47, 61, 83)], strict=False))
    return critical_point_count(sp.expand(g.subs(values)), variables, {})


class TestTwoLoopExamples:
    """The two-loop examples of Fevola, Mizera and Telen (2024), with the Euler
    characteristics they give."""

    @requires_singular
    def test_the_banana(self) -> None:
        # Example 3.7: m_1 m_2 m_3 s and the four thresholds; |chi| = 7 (remark 2.6).
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        x = _symbols(fi)
        m1, m2, m3, s = x["m_1"], x["m_2"], x["m_3"], x["s"]
        thresholds = [s - (m1 + a * m2 + b * m3) ** 2 for a in (1, -1) for b in (1, -1)]
        analysis = landau_analysis(fi)
        assert _factor_set(analysis.landau_surfaces, set(x.values())) == _factor_set(
            [m1, m2, m3, s, *thresholds], set(x.values())
        )
        assert analysis.parent is None
        assert _generic_count(fi) == 7

    @requires_singular
    def test_the_banana_with_a_massless_line(self) -> None:
        # Example 3.8: m_2 m_3 s lambda(s, m_2^2, m_3^2), and |chi| = 4.
        fi = FeynmanIntegral.from_cnickel("111e|e|:znn")
        x = _symbols(fi)
        m2, m3, s = x["m_2"], x["m_3"], x["s"]
        analysis = landau_analysis(fi)
        expected = [m2, m3, s, s - (m2 + m3) ** 2, s - (m2 - m3) ** 2]
        assert _factor_set(analysis.landau_surfaces, set(x.values())) == _factor_set(
            expected, set(x.values())
        )
        assert _generic_count(fi) == 4

    @requires_singular
    @pytest.mark.slow
    def test_the_parachute_at_s_zero_with_two_massless_lines(self) -> None:
        # Section 3.5: on s = m_1 = m_2 = 0 the principal Landau determinant is
        # m_3 m_4 M_3 M_4 (M_3 - M_4) lambda(M_3, m_3, m_4) lambda(M_4, m_3, m_4), which is the
        # Euler discriminant; here M_3 and M_4 are p_4^2 and p_3^2. The parent family, with s
        # generic, adds nothing; its polytope has a face of 15 points.
        fi = FeynmanIntegral.from_cnickel("12ee|22e|e|:zznn")
        x = _symbols(fi)
        s12 = x["s12"]
        products = {
            k: sp.expand(sp.sympify(v).subs(s12, 0)) for k, v in fi.momentum_products.items()
        }
        on_s = fi.with_(momentum_products=products)
        m3, m4, p3, p4 = x["m_3"], x["m_4"], x["p3^2"], x["p4^2"]
        expected = [m3, m4, p3, p4, p3 - p4] + [
            q - (m3 + a * m4) ** 2 for q in (p3, p4) for a in (1, -1)
        ]
        analysis = landau_analysis(on_s, max_face_points=15, limits=True)
        assert _factor_set(analysis.landau_surfaces, set(x.values())) == _factor_set(
            expected, set(x.values())
        )
        assert analysis.parent is not None
        assert not analysis.skipped_faces and not analysis.parent.skipped_faces
        assert analysis.limit_surfaces == analysis.limit_candidates == ()
        assert _generic_count(on_s) == 7

    @requires_singular
    def test_the_massless_parachute(self) -> None:
        # The database's massless parachute: M_3, M_4, s and lambda(s, M_3, M_4); |chi| = 4.
        fi = FeynmanIntegral.from_cnickel("12ee|22e|e|:zzzz")
        x = _symbols(fi)
        s12, p3, p4 = x["s12"], x["p3^2"], x["p4^2"]
        kallen = s12**2 + p3**2 + p4**2 - 2 * s12 * p3 - 2 * s12 * p4 - 2 * p3 * p4
        analysis = landau_analysis(fi)
        assert _factor_set(analysis.landau_surfaces, set(x.values())) == _factor_set(
            [s12, p3, p4, kallen], set(x.values())
        )
        assert _generic_count(fi) == 4


A1, A2 = sp.symbols("a1 a2")
A_, B_, C_, D_, Z_ = sp.symbols("a b c d z")


class TestDominantFaces:
    """A face whose elimination ideal is zero has a component of its incidence variety that
    projects onto the whole kinematic space. Its other components are found through the
    minimal primes (Fevola, Mizera and Telen 2024, definition 3.5)."""

    @requires_singular
    def test_example_3_9_keeps_the_component_beside_the_dominant_one(self) -> None:
        # FMT24a example 3.9: the dense face has a dominant component and one projecting to
        # bc = ad, which restricting the generic principal A-determinant loses.
        f = sp.expand((1 + A1) * (A_ + B_ * A1 + C_ * A2 + D_ * A1 * A2))
        analysis = landau_analysis_from_polynomial(f, [A1, A2])
        expected = [A_, B_, C_, D_, A_ - B_, C_ - D_, B_ * C_ - A_ * D_]
        assert _factor_set(analysis.landau_surfaces, {A_, B_, C_, D_}) == _factor_set(
            expected, {A_, B_, C_, D_}
        )
        (top,) = [face for face in analysis.face_discriminants if face.dimension == 2]
        assert top.dominant
        assert _monic(top.discriminant) == _monic(B_ * C_ - A_ * D_)

    @requires_singular
    def test_example_3_10_gives_only_the_edge(self) -> None:
        # FMT24a example 3.10: the Euler characteristic drops at z = 0, which no face sees.
        f = sp.expand((A2 - 1) ** 2 - (A1 - Z_) * A1**2)
        analysis = landau_analysis_from_polynomial(f, [A1, A2])
        assert set(analysis.landau_surfaces) == {4 * Z_**3 + 27}

    def test_a_face_with_a_nonzero_ideal_is_not_dominant(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        assert not any(face.dominant for face in landau_analysis(massive_bubble).face_discriminants)

    def test_without_singular_the_other_components_are_not_sought(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        f = sp.expand((1 + A1) * (A_ + B_ * A1 + C_ * A2 + D_ * A1 * A2))
        analysis = landau_analysis_from_polynomial(f, [A1, A2])
        (top,) = [face for face in analysis.face_discriminants if face.dimension == 2]
        assert top.dominant
        assert top.discriminant == 1

    def test_the_decomposition_is_read_component_by_component(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two minimal primes: the first projects onto everything, the second onto x y = z.
        seen = fake_singular(monkeypatch, "@\n@\nv2*v3-v4\n")
        result = _eliminate_singular(
            [W * T - 1], [W, T], [X, Y, Z], "Singular", points=10, decompose=True
        )
        assert result.generators == []
        assert result.components is not None
        assert [[g.as_expr() for g in c] for c in result.components] == [[], [X * Y - Z]]
        # One run: the primes are sought only when the ideal is zero.
        assert "if (size(E) == 0)" in str(seen["script"])
        assert "minAssChar" in str(seen["script"])

    def test_a_nonzero_ideal_has_no_components(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_singular(monkeypatch, "v2\n")
        result = _eliminate_singular(
            [W * T - 1], [W, T], [X, Y, Z], "Singular", points=10, decompose=True
        )
        assert result == ([sp.Poly(X, X, Y, Z)], None)
        fake_singular(monkeypatch, "")
        assert eliminate(decompose=True) == []

    @pytest.mark.parametrize("stdout", ["v2\n@\n", "@\nv0*v2\n", "@\n?\n"])
    def test_output_that_is_not_a_decomposition_raises(
        self, stdout: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_singular(monkeypatch, stdout)
        with pytest.raises(ComputationError, match="cannot read"):
            eliminate(decompose=True)

    @requires_singular
    def test_a_database_entry_is_unchanged(self) -> None:
        # The massless parachute has dominant faces; they add no component.
        g, variables, components = pld_entry("par_zero_generic")
        analysis = landau_analysis_from_polynomial(g, variables)
        assert any(face.dominant for face in analysis.face_discriminants)
        assert set(analysis.landau_surfaces) == components


class TestBridges:
    """A bridge is an internal edge on no cycle. The closed form covers the cycle, with the legs
    of each tree moved to the vertex where it meets the cycle, and each bridge adds its pole."""

    def test_a_self_loop_with_a_bridge(self) -> None:
        # A massless self-loop at vertex 1 and a massive bridge to vertex 2: the old walk
        # stopped after the self-loop, and the leg at vertex 2 raised KeyError.
        fi = FeynmanIntegral.from_cnickel("01e|e|:zn")
        edges, legs, bridges = _one_loop_cycle(fi)
        assert [e.idx for e in edges] == [1]
        assert legs == [[1, 2]]
        assert [(e.idx, far) for e, far in bridges] == [(2, [2])]
        m2 = fi.graph.get_internal_edges()[1].get_mass()
        assert one_loop_bridge_poles(fi) == (S - m2**2,)
        assert one_loop_landau_surfaces_by_type(fi) == ((), ())
        assert one_loop_landau_surfaces(fi) == (S - m2**2,)

    def test_a_bridge_as_the_first_edge(self) -> None:
        # A bridge from vertex 1 to a massive bubble on vertices 2 and 3. The old walk started
        # along the bridge and returned it twice among four edges.
        fi = FeynmanIntegral.from_cnickel("1e|22|e|:nnn")
        edges, legs, bridges = _one_loop_cycle(fi)
        assert [e.idx for e in edges] == [2, 3]
        assert legs == [[1], [2]]
        assert [(e.idx, far) for e, far in bridges] == [(1, [1])]

    def test_a_path_of_bridges(self) -> None:
        # A triangle with two bridges in a row, 3-4 and 4-5, leg 3 at the far end.
        fi = FeynmanIntegral.from_cnickel("12e|2e|3|4|e|:nnnnn")
        edges, legs, bridges = _one_loop_cycle(fi)
        assert sorted(e.idx for e in edges) == [1, 2, 3]
        assert sorted(legs) == [[1], [2], [3]]
        assert [(e.idx, far) for e, far in bridges] == [(4, [3]), (5, [3])]
        assert len(one_loop_bridge_poles(fi)) == 2

    @requires_singular
    def test_a_bridge_to_a_vertex_without_legs(self) -> None:
        # CNickel cannot write it: a massive bridge from vertex 3 of a triangle to a bare
        # vertex 4 carries no momentum, so its pole is m_4^2.
        m4 = sp.Symbol("m_4", positive=True)
        edges = [
            Edge(idx=1, v1=1, v2=2, is_internal=True),
            Edge(idx=2, v1=2, v2=3, is_internal=True),
            Edge(idx=3, v1=1, v2=3, is_internal=True),
            Edge(idx=4, v1=3, v2=4, is_internal=True, mass=m4),
            *(Edge(idx=4 + v, v1=v, v2=4 + v, is_internal=False) for v in (1, 2, 3)),
        ]
        fi = FeynmanIntegral(Graph(internal_vertices=4, external_legs=3, edges=edges))
        assert _one_loop_cycle(fi)[2] == [(edges[3], [])]
        assert one_loop_bridge_poles(fi) == (m4,)
        assert _same_surfaces(fi)

    def test_no_bridges(self, massive_bubble: FeynmanIntegral) -> None:
        assert _one_loop_cycle(massive_bubble)[2] == []
        assert one_loop_bridge_poles(massive_bubble) == ()

    def test_bridge_poles_reject_two_loops(self) -> None:
        with pytest.raises(ValueError, match="one-loop"):
            one_loop_bridge_poles(FeynmanIntegral.from_cnickel("11e|2|33|e|:nnnnn"))

    @pytest.mark.parametrize("cnickel", ["11e|e|33e|e|", "122e|3e|e|3e|e|"])
    def test_a_disconnected_graph_is_rejected(self, cnickel: str) -> None:
        # Two bubbles with no edge between them, and a two-loop part with a vertex apart, have
        # two loops each, and are refused for being disconnected rather than for the loop count.
        # The second gave six factors and no error.
        fi = FeynmanIntegral.from_cnickel(cnickel)
        assert fi.loop_count == 2
        with pytest.raises(ValueError, match="connected"):
            one_loop_landau_surfaces(fi)

    @pytest.mark.parametrize(
        ("internal", "vertices", "legs"),
        [
            # A sunrise on vertices 1 and 2 and a separate edge from 3 to 4: removing leaves
            # emptied vertex 3 and raised an unrelated ValueError.
            ([(1, 2), (1, 2), (1, 2), (3, 4)], 4, [1, 2, 3, 4]),
            # A triangle with a bubble hung from vertex 3, and vertex 6 apart: the walk from
            # vertex 5 circled the triangle without end, its list of edges growing.
            ([(5, 4), (1, 2), (2, 3), (1, 3), (4, 3), (5, 4)], 6, [1, 2, 5, 6]),
        ],
        ids=["tree", "vertex-apart"],
    )
    def test_a_hand_built_disconnected_graph_is_rejected(
        self, internal: list[tuple[int, int]], vertices: int, legs: list[int]
    ) -> None:
        # Each has two loops and is refused for being disconnected. A walk that does not stop
        # would fill the memory, so an alarm stops it where the platform has one.
        edges = [
            Edge(idx=k, v1=a, v2=b, is_internal=True) for k, (a, b) in enumerate(internal, start=1)
        ]
        edges += [
            Edge(idx=len(internal) + k, v1=v, v2=vertices + k, is_internal=False)
            for k, v in enumerate(legs, start=1)
        ]
        graph = Graph(internal_vertices=vertices, external_legs=len(legs), edges=edges)
        fi = FeynmanIntegral(graph)
        assert fi.loop_count == 2

        def expire(signum: int, frame: object) -> None:
            raise TimeoutError("the closed form did not stop")

        alarm = hasattr(signal, "SIGALRM")
        previous = signal.signal(signal.SIGALRM, expire) if alarm else None
        if alarm:
            signal.alarm(10)
        try:
            for closed_form in (one_loop_landau_surfaces, one_loop_bridge_poles):
                with pytest.raises(ValueError, match="connected"):
                    closed_form(fi)
        finally:
            if alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)

    def test_products_are_read_in_either_order(self) -> None:
        # F reads p_a . p_c under (a, c) or (c, a); the closed form read (c, a), c > a, as 0 and
        # lost every kinematic factor.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        swapped = fi.with_(
            momentum_products={(j, i): value for (i, j), value in fi.momentum_products.items()}
        )
        assert swapped.symanzik.f == fi.symanzik.f
        assert one_loop_landau_surfaces(swapped) == one_loop_landau_surfaces(fi)

    def test_the_momenta_come_from_the_products(self) -> None:
        # F sees p_3^2 = 0 set in the momentum products. The closed form took p_3^2 from the
        # invariants, so the bridge carrying p_3 gave the pole m_4^2 - p_3^2, not m_4^2.
        fi = FeynmanIntegral.from_cnickel("12e|2e|3|e|:nnnn")
        p3 = standard_invariants(3).external_masses[2]
        products = {
            pair: sp.expand(sp.sympify(value).subs(p3, 0))
            for pair, value in fi.momentum_products.items()
        }
        on_shell = fi.with_(momentum_products=products)
        m4 = on_shell.graph.get_internal_edges()[3].get_mass()
        assert one_loop_bridge_poles(on_shell) == (m4,)
        kin = _kin(on_shell)
        assert p3 not in kin
        assert set().union(*(f.free_symbols for f in one_loop_landau_surfaces(on_shell))) <= kin

    @requires_singular
    @pytest.mark.parametrize(
        "cnickel",
        [
            "01e|e|:zn",
            "11e|2|e|:nnn",
            "11e|2|e|:nnz",
            "1e|22|e|:nnn",
            "12e|2e|3|e|:nnnz",
            "12e|2e|3|4|e|:zzznn",
        ],
    )
    def test_faces_give_the_closed_form_and_the_bridge_poles(self, cnickel: str) -> None:
        # With a massless bridge, 11e|2|e|:nnz and 12e|2e|3|e|:nnnz need the rule for several
        # generators: every factor of every generator added m_1 - m_2, m_1 + m_2 and a quartic.
        assert _same_surfaces(FeynmanIntegral.from_cnickel(cnickel))


class TestClosedForm:
    """The minors of the modified Cayley matrix are taken with a symbol per distinct entry and
    factored before the entries are substituted back."""

    @pytest.mark.parametrize("backend", backends())
    @pytest.mark.parametrize(
        "cnickel", ["11e|e|:nn", "12e|2e|e|:aaa", "12e|3e|3e|e|:nzzz", "12e|3e|3e|e|:zzzz"]
    )
    def test_the_factors_of_the_expanded_minors(
        self, cnickel: str, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The oracle is the old computation: every minor expanded in the invariants and
        # factored, by Singular when it is installed and the backend is Singular's, else by
        # SymPy. SymPy's factorisation of the Cayley minor on 1, 2, 3, 4 of the one-mass box
        # ran for minutes from some states of its generator, which the module fixes.
        if backend == "sympy":
            monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        fi = FeynmanIntegral.from_cnickel(cnickel)
        y = _modified_cayley_matrix(fi)
        subsets = [
            subset
            for size in range(1, y.rows + 1)
            for subset in itertools.combinations(range(y.rows), size)
        ]
        minors = [sp.expand(y.extract(list(s), list(s)).det()) for s in subsets]
        first: dict[sp.Expr, None] = {}
        second: dict[sp.Expr, None] = {}
        for subset, factors in zip(subsets, _factor_lists(minors, y.free_symbols), strict=True):
            target = second if 0 in subset else first
            for factor in factors:
                target.setdefault(factor, None)
        assert one_loop_landau_surfaces_by_type(fi) == (tuple(first), tuple(second))

    def test_a_landau_variable(self) -> None:
        # The equal-mass bubble in the Landau variable, s = -m^2 (1 - x)^2 / x: the entries are
        # not polynomials, and the images are expanded expressions, as the ring cannot hold them.
        x = sp.Symbol("x", positive=True)
        fi = FeynmanIntegral.from_cnickel("11e|e|:aa")
        m = fi.graph.get_internal_edges()[0].get_mass()
        bubble = fi.with_(
            momentum_products={
                pair: sp.sympify(value).subs(S, -(m**2) * (1 - x) ** 2 / x)
                for pair, value in fi.momentum_products.items()
            }
        )
        assert one_loop_landau_surfaces(bubble) == (m, x - 1, x + 1)
        landau = AnalysisReport.from_integral(bubble, ["landau"]).landau
        assert landau is not None
        assert landau.first_type == (m, x - 1, x + 1)

    @staticmethod
    def bubble_with_mass(mass: sp.Expr) -> FeynmanIntegral:
        """The two-mass bubble with its first mass replaced."""
        graph = FeynmanIntegral.from_cnickel("11e|e|:nn").graph
        first = graph.get_internal_edges()[0].idx
        edges = [
            dataclasses.replace(edge, mass=mass) if edge.idx == first else edge
            for edge in graph.edges
        ]
        return FeynmanIntegral(
            Graph(graph.internal_vertices, graph.external_legs, edges, graph.energy_scale)
        )

    def test_a_mass_over_mu(self) -> None:
        graph = FeynmanIntegral.from_cnickel("11e|e|:nn").graph
        mu = graph.energy_scale
        m1, m2 = (edge.get_mass() for edge in graph.get_internal_edges())
        first, second = one_loop_landau_surfaces_by_type(self.bubble_with_mass(m1 / mu))
        assert first == (
            m1,
            m2,
            sp.expand(mu**2 * S - (m1 + m2 * mu) ** 2),
            sp.expand(mu**2 * S - (m1 - m2 * mu) ** 2),
        )
        assert second == (S,)

    def test_a_float_mass_stays_a_float(self) -> None:
        # The ring over the rationals would turn 0.1 into 1/10; the images keep the Float, and
        # the factors are those the expanded expressions always gave.
        surfaces = one_loop_landau_surfaces(self.bubble_with_mass(sp.Float(0.1)))
        assert [str(f) for f in surfaces] == [
            "m_2",
            "-1.0*m_2**2 - 0.2*m_2 + 1.0*s - 0.01",
            "-1.0*m_2**2 + 0.2*m_2 + 1.0*s - 0.01",
            "1.0*s - 1.73472347597681e-18",
        ]

    @requires_singular
    def test_sympys_random_state_is_left_alone(self) -> None:
        # Singular factors the minors and, with equal masses, the substituted factors, so
        # nothing is drawn from SymPy's generator.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:aaa")
        sympy_random.seed(7)
        state = sympy_random.rng.getstate()
        one_loop_landau_surfaces_by_type(fi)
        assert sympy_random.rng.getstate() == state


def random_expressions(seed: int) -> list[sp.Expr]:
    """Products of powers of random polynomials, some irreducible, with rational
    coefficients and leading terms of either sign, some expanded and some not."""
    rng = random.Random(seed)
    symbols = [*sp.symbols("a b c", real=True), sp.Symbol("m_1", positive=True), sp.Dummy("d")]

    def polynomial() -> sp.Expr:
        chosen = rng.sample(symbols, rng.randint(1, 3))
        return sp.Add(
            *(
                sp.Rational(rng.choice([-1, 1]) * rng.randint(1, 9), rng.choice([1, 1, 2, 3]))
                * sp.prod(s ** rng.randint(0, 2) for s in chosen)
                for _ in range(rng.randint(1, 4))
            )
        )

    expressions = []
    for _ in range(40):
        product = sp.Rational(rng.choice([-3, -1, 1, 2]), rng.choice([1, 5])) * sp.prod(
            polynomial() ** rng.randint(1, 3) for _ in range(rng.randint(1, 3))
        )
        expressions.append(sp.expand(product) if rng.random() < 0.5 else product)
    return expressions


class TestFactorLists:
    """Singular factors a batch of polynomials in one run, and each factor is written and
    ordered as sp.factor_list writes and orders it."""

    @requires_singular
    @pytest.mark.parametrize("seed", range(4))
    def test_the_factors_are_sympys(self, seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
        expressions = random_expressions(seed)
        symbols = set().union(*(e.free_symbols for e in expressions))
        expected = [_factor_list(e, symbols) for e in expressions]
        polys = [sp.Poly(e) for e in expressions if e.free_symbols]
        assert _factorize_singular(polys, "Singular") == {p: p.factor_list()[1] for p in polys}

        def fail(*args: object) -> list[sp.Expr]:
            raise AssertionError("factored by SymPy")

        monkeypatch.setattr(landau_module, "_factor_list", fail)
        assert _factor_lists(expressions, symbols) == expected

    @requires_singular
    @pytest.mark.parametrize("seed", range(2))
    def test_ring_elements_factor_as_their_expressions(
        self, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The closed form's images are ring elements, whose bases are read off the polynomial
        # rather than from together; the factors, their forms and their order are the same.
        a, b = sp.symbols("a b", real=True)
        monomials = [1, a, a**2 * b, a * b, 3 * b**2]
        products = [
            sp.expand(e * monomials[k % len(monomials)])
            for k, e in enumerate(random_expressions(seed))
        ]
        expressions = [e for e in products if e.free_symbols]
        symbols = set().union(*(e.free_symbols for e in expressions))
        ring = PolyRing(sp.Poly(sp.Add(*symbols)).gens, sp.QQ)
        polys = [ring.from_expr(e) for e in expressions]
        assert [poly.as_expr() for poly in polys] == expressions
        expected = _factor_lists(expressions, symbols)
        assert [_normalised_poly(poly) for poly in polys] == [_normalised(e) for e in expressions]

        def fail(*args: object) -> list[sp.Expr]:
            raise AssertionError("factored by SymPy")

        monkeypatch.setattr(landau_module, "_factor_list", fail)
        assert _factor_lists(polys, symbols) == expected

    def test_images_are_the_substituted_expansions(self) -> None:
        m1, m2, s, t = sp.symbols("m_1 m_2 s t", positive=True)
        y1, y2, y3 = sp.Dummy("y1"), sp.Dummy("y2"), sp.Dummy("y3")
        back = {y1: m1**2 + m2**2 - s, y2: 2 * m1**2, y3: s / 2 - t}
        factors = dict.fromkeys([y1 * y2 - y3**2, y1**3 + 2 * y2 * y3 - 5, 7 * y3])
        images = _images(factors, back)
        assert [sp.srepr(images[f].as_expr()) for f in factors] == [
            sp.srepr(sp.expand(f.xreplace(back))) for f in factors
        ]

    @requires_singular
    def test_a_minor_that_stalled_sympy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A Cayley minor of the massive box without Mandelstam variables, 57 terms in eight
        # symbols. From a generator seeded with 0, SymPy's factorisation ran for over ten
        # minutes.
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn", use_mandelstam=False)
        y = _modified_cayley_matrix(fi)
        minor = sp.expand(y.extract([2, 3, 4], [2, 3, 4]).det())
        assert len(minor.args) == 57

        def fail(*args: object) -> list[sp.Expr]:
            raise AssertionError("factored by SymPy")

        monkeypatch.setattr(landau_module, "_factor_list", fail)
        sympy_random.seed(0)
        start = time.perf_counter()
        assert _factor_lists([minor], y.free_symbols) == [[_normalised(minor)]]
        assert time.perf_counter() - start < 5

    @requires_singular
    def test_the_landau_analysis_factors_with_singular(self) -> None:
        # The massive box with p_1^2 = p_2^2 = 0. From a generator seeded with 0, SymPy's
        # factorisation of a face's generator, 92 terms in eight symbols, ran for over ten
        # minutes; Singular factors it in a fiftieth of a second.
        if not hasattr(signal, "SIGALRM"):
            pytest.skip("stopping a stalled factorisation needs SIGALRM")

        def expire(signum: int, frame: object) -> None:
            raise TimeoutError("the Landau analysis took more than 60 s")

        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
        p1, p2, _, _ = standard_invariants(4).external_masses
        products = {
            pair: sp.expand(sp.sympify(value).subs({p1: 0, p2: 0}))
            for pair, value in fi.momentum_products.items()
        }
        on_shell = fi.with_(momentum_products=products)
        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(60)
        try:
            sympy_random.seed(0)
            analysis = landau_analysis(on_shell)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        # 26 from the faces of up to 12 points and one from the polytope, of 14.
        assert len(analysis.landau_surfaces) == 27

    def test_without_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        expressions = random_expressions(0)
        symbols = set().union(*(e.free_symbols for e in expressions))
        expected = [_factor_list(e, symbols) for e in expressions]
        assert _factor_lists(expressions, symbols) == expected

    @pytest.mark.parametrize(
        ("stdout", "returncode"),
        [
            ("", 1),
            ("   ? error occurred in or before STDIN line 3\n", 0),
            ("@2\n1:1\n", 0),
            ("@2\n1:1\n1:v0^2\n", 0),
            ("@2\n1:v0-3*v1\n1:v0+2*v1\n", 0),
            ("@2\n1:v0-v1\n1:v0+2*v1\n@1\n1:v0\n", 0),
        ],
        ids=["status", "error", "short", "degrees", "coefficients", "extra"],
    )
    def test_a_failure_of_singular_leaves_it_to_sympy(
        self, stdout: str, returncode: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a, b = sp.symbols("a b")
        expression = sp.expand((a - b) * (a + 2 * b))
        expected = _factor_list(expression, {a, b})
        seen = fake_singular(monkeypatch, stdout, returncode=returncode)
        assert _factor_lists([expression], {a, b}) == [expected]
        assert "factorize" in str(seen["script"])


# Polynomials that the Landau analysis of one-loop and two-loop families factored.
_CORPUS = (
    "2*_c1*_c2*_c5",
    "2*_c1*_c4*_c7",
    "2*_c2*_c4*_c9",
    "2*_c10*_c3*_c4",
    "2*_c11*_c6*_c8",
    "2*_c13*_c7*_c8",
    "2*_c15*_c8*_c9",
    "2*_c13*_c14*_c15",
    "_c1*_c10 - _c4**2",
    "-_c1 + 2*_c3 - _c8",
    "-_c10 - _c8 + 2*_c9",
    "_c1**2 - 2*_c1*_c2 - 2*_c1*_c6 + _c2**2 - 2*_c2*_c6 + _c6**2",
    "_c1**2 - 2*_c1*_c4 - 2*_c1*_c8 + _c4**2 - 2*_c4*_c8 + _c8**2",
    "_c5**2 - 2*_c5*_c6 - 2*_c5*_c8 + _c6**2 - 2*_c6*_c8 + _c8**2",
    "_c10**2 - 2*_c10*_c6 - 2*_c10*_c7 + _c6**2 - 2*_c6*_c7 + _c7**2",
    "_c12**2 - 2*_c12*_c2 - 2*_c12*_c5 + _c2**2 - 2*_c2*_c5 + _c5**2",
    "_c14**2 - 2*_c14*_c3 - 2*_c14*_c5 + _c3**2 - 2*_c3*_c5 + _c5**2",
    "_c1*_c5*_c8 - _c1*_c6**2 - _c2**2*_c8 + 2*_c2*_c3*_c6 - _c3**2*_c5",
    "-m5**4 + 2*m5**2*m6**2 + 2*m5**2*s12 - m6**4 + 2*m6**2*s12 - s12**2",
    "-p1**4 + 2*p1**2*p4**2 + 2*p1**2*s23 - p4**4 + 2*p4**2*s23 - s23**2",
    "_c1*_c10*_c8 - _c1*_c9**2 - _c10*_c3**2 + 2*_c3*_c4*_c9 - _c4**2*_c8",
    "_c11**2 - 2*_c11*_c12 - 2*_c11*_c15 + _c12**2 - 2*_c12*_c15 + _c15**2",
    "-m1**4 + 2*m1**2*m3**2 + 2*m1**2*p2**2 - m3**4 + 2*m3**2*p2**2 - p2**4",
    "-m5**4 + 2*m5**2*m7**2 + 2*m5**2*p3**2 - m7**4 + 2*m7**2*p3**2 - p3**4",
    "_c1**2*_c10**2 - 2*_c1*_c10*_c2*_c7 - 2*_c1*_c10*_c3*_c6 + _c2**2*_c7**2 - 2*_c2*_c3*_c6*_c7 + _c3**2*_c6**2",
    "_c1**2*_c13**2 - 2*_c1*_c13*_c3*_c8 - 2*_c1*_c13*_c4*_c7 + _c3**2*_c8**2 - 2*_c3*_c4*_c7*_c8 + _c4**2*_c7**2",
    "_c10**2*_c5**2 - 2*_c10*_c5*_c6*_c9 - 2*_c10*_c5*_c7*_c8 + _c6**2*_c9**2 - 2*_c6*_c7*_c8*_c9 + _c7**2*_c8**2",
    "_c10**2*_c9**2 - 2*_c10*_c12*_c7*_c9 - 2*_c10*_c14*_c6*_c9 + _c12**2*_c7**2 - 2*_c12*_c14*_c6*_c7 + _c14**2*_c6**2",
    "_c13**2*_c9**2 - 2*_c13*_c14*_c8*_c9 - 2*_c13*_c15*_c7*_c9 + _c14**2*_c8**2 - 2*_c14*_c15*_c7*_c8 + _c15**2*_c7**2",
    "-_c1*_c10 - _c1*_c8 + 2*_c1*_c9 + 2*_c10*_c3 - _c10*_c8 + _c3**2 - 2*_c3*_c4 - 2*_c3*_c9 + _c4**2 + 2*_c4*_c8 - 2*_c4*_c9 + _c9**2",
    "-2*_c1**2*_c9 - 2*_c1*_c2*_c5 + 2*_c1*_c2*_c7 + 2*_c1*_c2*_c9 + 2*_c1*_c4*_c5 - 2*_c1*_c4*_c7 + 2*_c1*_c4*_c9 + 2*_c1*_c5*_c9 + 2*_c1*_c7*_c9 - 2*_c1*_c9**2 - 2*_c2**2*_c7 + 2*_c2*_c4*_c5 + 2*_c2*_c4*_c7 - 2*_c2*_c4*_c9 + 2*_c2*_c5*_c7 - 2*_c2*_c7**2 + 2*_c2*_c7*_c9 - 2*_c4**2*_c5 - 2*_c4*_c5**2 + 2*_c4*_c5*_c7 + 2*_c4*_c5*_c9 - 2*_c5*_c7*_c9",
    "-2*_c1**2*_c12 - 2*_c1*_c12**2 + 2*_c1*_c12*_c2 + 2*_c1*_c12*_c5 + 2*_c1*_c12*_c6 + 2*_c1*_c12*_c9 - 2*_c1*_c2*_c6 + 2*_c1*_c2*_c9 + 2*_c1*_c5*_c6 - 2*_c1*_c5*_c9 - 2*_c12*_c2*_c5 + 2*_c12*_c2*_c9 + 2*_c12*_c5*_c6 - 2*_c12*_c6*_c9 - 2*_c2**2*_c9 + 2*_c2*_c5*_c6 + 2*_c2*_c5*_c9 + 2*_c2*_c6*_c9 - 2*_c2*_c9**2 - 2*_c5**2*_c6 - 2*_c5*_c6**2 + 2*_c5*_c6*_c9",
    "-2*_c10**2*_c2 - 2*_c10*_c2**2 + 2*_c10*_c2*_c3 + 2*_c10*_c2*_c4 + 2*_c10*_c2*_c8 + 2*_c10*_c2*_c9 - 2*_c10*_c3*_c4 + 2*_c10*_c3*_c9 + 2*_c10*_c4*_c8 - 2*_c10*_c8*_c9 - 2*_c2*_c3*_c8 + 2*_c2*_c3*_c9 + 2*_c2*_c4*_c8 - 2*_c2*_c4*_c9 - 2*_c3**2*_c9 + 2*_c3*_c4*_c8 + 2*_c3*_c4*_c9 + 2*_c3*_c8*_c9 - 2*_c3*_c9**2 - 2*_c4**2*_c8 - 2*_c4*_c8**2 + 2*_c4*_c8*_c9",
    "-2*_c10**2*_c8 - 2*_c10*_c11*_c13 + 2*_c10*_c11*_c7 + 2*_c10*_c11*_c8 + 2*_c10*_c13*_c6 + 2*_c10*_c13*_c8 - 2*_c10*_c6*_c7 + 2*_c10*_c6*_c8 + 2*_c10*_c7*_c8 - 2*_c10*_c8**2 - 2*_c11**2*_c7 + 2*_c11*_c13*_c6 + 2*_c11*_c13*_c7 + 2*_c11*_c6*_c7 - 2*_c11*_c6*_c8 - 2*_c11*_c7**2 + 2*_c11*_c7*_c8 - 2*_c13**2*_c6 - 2*_c13*_c6**2 + 2*_c13*_c6*_c7 + 2*_c13*_c6*_c8 - 2*_c13*_c7*_c8",
    "-2*_c13**2*_c5 - 2*_c13*_c14*_c15 + 2*_c13*_c14*_c4 + 2*_c13*_c14*_c5 + 2*_c13*_c15*_c3 + 2*_c13*_c15*_c5 - 2*_c13*_c3*_c4 + 2*_c13*_c3*_c5 + 2*_c13*_c4*_c5 - 2*_c13*_c5**2 - 2*_c14**2*_c4 + 2*_c14*_c15*_c3 + 2*_c14*_c15*_c4 + 2*_c14*_c3*_c4 - 2*_c14*_c3*_c5 - 2*_c14*_c4**2 + 2*_c14*_c4*_c5 - 2*_c15**2*_c3 - 2*_c15*_c3**2 + 2*_c15*_c3*_c4 + 2*_c15*_c3*_c5 - 2*_c15*_c4*_c5",
    "-m1**4*s12 + m1**2*m2**2*p1**2 - m1**2*m2**2*p2**2 + m1**2*m2**2*s12 - m1**2*m4**2*p1**2 + m1**2*m4**2*p2**2 + m1**2*m4**2*s12 + m1**2*p1**2*s12 + m1**2*p2**2*s12 - m1**2*s12**2 - m2**4*p1**2 + m2**2*m4**2*p1**2 + m2**2*m4**2*p2**2 - m2**2*m4**2*s12 - m2**2*p1**4 + m2**2*p1**2*p2**2 + m2**2*p1**2*s12 - m4**4*p2**2 + m4**2*p1**2*p2**2 - m4**2*p2**4 + m4**2*p2**2*s12 - p1**2*p2**2*s12",
    "-2*_c10**2*_c15 - 2*_c10*_c11*_c13 + 2*_c10*_c11*_c14 + 2*_c10*_c11*_c15 + 2*_c10*_c12*_c13 - 2*_c10*_c12*_c14 + 2*_c10*_c12*_c15 + 2*_c10*_c13*_c15 + 2*_c10*_c14*_c15 - 2*_c10*_c15**2 - 2*_c11**2*_c14 + 2*_c11*_c12*_c13 + 2*_c11*_c12*_c14 - 2*_c11*_c12*_c15 + 2*_c11*_c13*_c14 - 2*_c11*_c14**2 + 2*_c11*_c14*_c15 - 2*_c12**2*_c13 - 2*_c12*_c13**2 + 2*_c12*_c13*_c14 + 2*_c12*_c13*_c15 - 2*_c13*_c14*_c15",
    "-2*m1**4*s12 + 2*m1**2*m2**2*p1**2 - 2*m1**2*m2**2*p2**2 + 2*m1**2*m2**2*s12 - 2*m1**2*m4**2*p1**2 + 2*m1**2*m4**2*p2**2 + 2*m1**2*m4**2*s12 + 2*m1**2*p1**2*s12 + 2*m1**2*p2**2*s12 - 2*m1**2*s12**2 - 2*m2**4*p1**2 + 2*m2**2*m4**2*p1**2 + 2*m2**2*m4**2*p2**2 - 2*m2**2*m4**2*s12 - 2*m2**2*p1**4 + 2*m2**2*p1**2*p2**2 + 2*m2**2*p1**2*s12 - 2*m4**4*p2**2 + 2*m4**2*p1**2*p2**2 - 2*m4**2*p2**4 + 2*m4**2*p2**2*s12 - 2*p1**2*p2**2*s12",
)


requires_flint = pytest.mark.skipif(
    importlib.util.find_spec("flint") is None, reason="python-flint not installed"
)


@requires_flint
class TestFlintFactorLists:
    """python-flint factors what SymPy would, in the same form and order, when Singular is
    absent."""

    def test_the_factors_are_sympys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        expressions = [sp.sympify(text) for text in _CORPUS]
        symbols = set().union(*(e.free_symbols for e in expressions))
        expected = [_factor_list(e, symbols) for e in expressions]
        assert any(len(factors) > 1 for factors in expected)

        def fail(*args: object) -> list[sp.Expr]:
            raise AssertionError("factored by SymPy")

        with monkeypatch.context() as patch:
            patch.setattr(landau_module, "_factor_list", fail)
            assert _factor_lists(expressions, symbols) == expected
        polys = [sp.Poly(e) for e in expressions]
        assert _factorize_flint(polys) == {p: p.factor_list()[1] for p in polys}

    def test_rational_coefficients_and_repeated_factors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        a, b, c = sp.symbols("a b c")
        expression = sp.expand((a - 2 * b) ** 2 * (a * c + b) * (b**2 - c) / 6)
        assert _factor_lists([expression], {a, b, c}) == [_factor_list(expression, {a, b, c})]

    def test_a_polynomial_that_is_not_integral_goes_to_sympy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        a, b = sp.symbols("a b")
        expression = sp.expand((a - sp.sqrt(2) * b) * (a + sp.sqrt(2) * b))
        assert _factor_lists([expression], {a, b}) == [_factor_list(expression, {a, b})]

    def test_a_failure_leaves_it_to_sympy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        monkeypatch.setattr(landau_module, "_factorize_flint", lambda polys: None)
        a, b = sp.symbols("a b")
        expression = sp.expand((a - b) * (a + 2 * b))
        assert _factor_lists([expression], {a, b}) == [_factor_list(expression, {a, b})]


class _Clock:
    """A stand-in for ``landau.time`` whose clock moves only when told to."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _faces_of(fi: FeynmanIntegral) -> list[tuple[int, tuple[tuple[int, ...], ...]]]:
    """The faces of the Newton polytope of G as (dimension, exponents), in the order of
    :func:`feynkit.polytope.faces`."""
    symanzik = fi.symanzik
    support = extract_monomial_support(sp.expand(symanzik.g), list(symanzik.lp_parameters))
    exps = np.array([e for e, _ in support], dtype=int)
    return [
        (dimension, tuple(tuple(int(x) for x in exps[i]) for i in idx))
        for dimension, idx in polytope_faces(exps)
    ]


class TestFaceOrderAndBudget:
    """Every face is attempted, smallest first, each within ``timeout`` and all within
    ``total_timeout``. The massless kite's polytope has 16 points."""

    KITE = "12e|23|3|e|:zzzzz"

    def _spy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        seconds: float = 0.0,
        clock: _Clock | None = None,
        late: int | None = None,
    ) -> list[tuple[int, int, float | None]]:
        """Replace the elimination of a face by one that gives no factor and takes ``seconds``
        on ``clock``. It runs past its time limit when that is shorter, or when the face has
        ``late`` points. Returns (points, dimension, time limit) for each call."""
        calls: list[tuple[int, int, float | None]] = []

        def spy(
            coeffs: list[sp.Expr],
            exps: list[tuple[int, ...]],
            kinematic_syms: set[sp.Symbol],
            backend: str = "auto",
            *,
            scale: sp.Symbol | None = None,
            timeout: float | None = None,
        ) -> object:
            calls.append((len(coeffs), len(exps[0]), timeout))
            if clock is not None:
                clock.now += seconds
            if len(coeffs) == late or (timeout is not None and timeout < seconds):
                raise landau_module._EliminationTimeout("past the limit")
            return landau_module._Elimination([], True, False)

        monkeypatch.setattr(landau_module, "_singular_binary", lambda: "Singular")
        monkeypatch.setattr(landau_module, "_elimination_discriminant", spy)
        if clock is not None:
            monkeypatch.setattr(landau_module, "time", clock)
        return calls

    def _eliminated(self, fi: FeynmanIntegral) -> list[tuple[tuple[int, ...], ...]]:
        """The faces of dimension two or more that are not simplices, in the polytope's order."""
        return [
            face
            for dimension, face in _faces_of(fi)
            if dimension >= 2 and len(face) > dimension + 1
        ]

    def test_faces_are_attempted_smallest_first_and_listed_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._spy(monkeypatch)
        fi = FeynmanIntegral.from_cnickel(self.KITE)
        analysis = landau_analysis(fi)
        order = [(points, dimension) for points, dimension, _ in calls]
        assert order == sorted(order)
        assert len(order) == len(self._eliminated(fi))
        # No face is too large by default: the polytope itself is attempted, last.
        assert order[-1] == (16, 5)
        assert not analysis.skipped_faces
        assert [f.exponents for f in analysis.face_discriminants] == [
            face for _, face in _faces_of(fi)
        ]

    def test_max_face_points_still_limits_the_faces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch)
        analysis = landau_analysis(FeynmanIntegral.from_cnickel(self.KITE), max_face_points=14)
        assert max(points for points, _, _ in calls) <= 14
        assert analysis.skipped_faces
        assert all(len(face) > 14 for face in analysis.skipped_faces)
        assert analysis.timed_out_faces == analysis.unattempted_faces == ()

    def test_a_face_past_its_time_limit_is_timed_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._spy(monkeypatch, late=16)
        analysis = landau_analysis(FeynmanIntegral.from_cnickel(self.KITE))
        assert [len(face) for face in analysis.skipped_faces] == [16]
        assert analysis.timed_out_faces == analysis.skipped_faces
        assert analysis.unattempted_faces == ()

    def test_the_faces_left_when_the_total_time_runs_out_are_not_attempted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each face takes 10 s: the first two finish, and the third is cut short at 25 s.
        clock = _Clock()
        calls = self._spy(monkeypatch, seconds=10, clock=clock)
        fi = FeynmanIntegral.from_cnickel(self.KITE)
        analysis = landau_analysis(fi, timeout=12, total_timeout=25)
        assert [limit for _, _, limit in calls] == [12, 12, 5]
        dimension = {face: d for d, face in _faces_of(fi)}
        eliminated = self._eliminated(fi)
        first = sorted(eliminated, key=lambda face: (len(face), dimension[face]))[:2]
        left = tuple(face for face in eliminated if face not in first)
        assert analysis.unattempted_faces == analysis.skipped_faces == left
        assert analysis.timed_out_faces == ()
        # The vertices and edges need no elimination, and are all there.
        low = [face for face in analysis.face_discriminants if face.dimension < 2]
        assert len(low) == len([face for d, face in _faces_of(fi) if d < 2])

    def test_the_parent_has_the_same_total_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        landau_module._parent_analysis.cache_clear()
        limits: list[object] = []
        analysis = landau_module._analysis

        def spy(g: sp.Expr, *args: object) -> LandauAnalysis:
            limits.append(args[-1])
            return analysis(g, *args)  # type: ignore[arg-type]

        monkeypatch.setattr(landau_module, "_analysis", spy)
        on_shell = _massless_legs("12e|2e|e|:nnn", (1,))
        landau_analysis(on_shell, confirm=False, limits=True)
        assert limits == [None, None]
        landau_analysis(on_shell, confirm=False, total_timeout=100, limits=True)
        assert limits[2:] == [100, 100]

    @pytest.mark.parametrize("total", [0, -1, True, float("inf"), float("nan"), "5", 3e6])
    def test_total_timeout_must_be_a_number_of_seconds(
        self, total: object, massive_bubble: FeynmanIntegral
    ) -> None:
        with pytest.raises(
            ValidationError, match="total_timeout must be None or a number of seconds"
        ):
            landau_analysis(massive_bubble, total_timeout=total)  # type: ignore[arg-type]

    def test_faces_of_more_than_14_points_get_a_short_time_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._spy(monkeypatch)
        fi = FeynmanIntegral.from_cnickel(self.KITE)
        landau_analysis(fi)
        assert landau_module.LARGE_FACE_POINTS == 14
        assert landau_module.DEFAULT_LARGE_FACE_TIMEOUT == 5
        limits = {(points > 14, limit) for points, _, limit in calls}
        assert limits == {(False, 60), (True, 5)}
        calls.clear()
        landau_analysis(fi, timeout=30, large_face_timeout=None)
        assert {(points > 14, limit) for points, _, limit in calls} == {(False, 30), (True, None)}

    @pytest.mark.parametrize("probe", [0, -1, True, float("inf"), float("nan"), "5", 3e6])
    def test_large_face_timeout_must_be_a_number_of_seconds(
        self, probe: object, massive_bubble: FeynmanIntegral
    ) -> None:
        with pytest.raises(
            ValidationError, match="large_face_timeout must be None or a number of seconds"
        ):
            landau_analysis(massive_bubble, large_face_timeout=probe)  # type: ignore[arg-type]

    def test_without_singular_faces_stop_at_14_points(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # SymPy's elimination cannot be stopped, so it keeps the limit the analysis had.
        calls = self._spy(monkeypatch)
        monkeypatch.setattr(landau_module, "_singular_binary", lambda: None)
        analysis = landau_analysis(FeynmanIntegral.from_cnickel(self.KITE))
        assert max(points for points, _, _ in calls) <= 14
        assert analysis.skipped_faces
        assert all(len(face) > 14 for face in analysis.skipped_faces)


class TestFaceLimit:
    """Every face of a one-loop box gets the full time limit for a face; a larger face gets a
    probe."""

    @requires_singular
    @pytest.mark.slow
    def test_the_massive_parachute_gets_a_kallen_function_from_its_polytope(self) -> None:
        # The polytope, 19 points, gives lambda(p_3^2, p_4^2, s_12) within its probe: the
        # component lambda(M_3, M_4, s) that the database of Fevola, Mizera and Telen computes
        # from faces for par on its full kinematic space. Singular had not eliminated the face
        # of 14 points after 400 s, and it runs past its time limit.
        fi = FeynmanIntegral.from_cnickel("12ee|22e|e|:nnnn")
        analysis = landau_analysis(fi)
        assert [len(face) for face in analysis.skipped_faces] == [14]
        assert analysis.timed_out_faces == analysis.skipped_faces
        p3, p4, s12 = (sp.Symbol(name, real=True) for name in ("p3^2", "p4^2", "s12"))
        kallen = p3**2 + p4**2 + s12**2 - 2 * p3 * p4 - 2 * p3 * s12 - 2 * p4 * s12
        assert _monic(kallen) in {_monic(h) for h in analysis.landau_surfaces}

    @requires_singular
    def test_the_default_limit_covers_the_box(self) -> None:
        # Every one-loop box has at most 14 points: 4 on U and 10 on F.
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnz")
        analysis = landau_analysis(fi)
        assert not analysis.skipped_faces
        assert set(analysis.landau_surfaces) == set(one_loop_landau_surfaces(fi))

    @requires_singular
    @pytest.mark.slow
    def test_every_generic_one_loop_colouring_is_the_closed_form(self) -> None:
        # The bubbles, triangles and boxes, each mass massless or its own: 3, 4 and 6.
        cnickels = list(generate_graphs(1, range(2, 5)))
        assert len(cnickels) == 13
        for cnickel in cnickels:
            fi = FeynmanIntegral.from_cnickel(cnickel)
            analysis = landau_analysis(fi)
            assert not analysis.skipped_faces, cnickel
            assert set(analysis.landau_surfaces) == set(one_loop_landau_surfaces(fi)), cnickel


@pytest.mark.slow
class TestPolygonClosedForms:
    """The massless pentagon's closed form took about 40 s, and the hexagon's did not finish
    in 47 minutes."""

    @requires_singular
    def test_pentagon(self) -> None:
        # The face computation skips the polytope itself, 15 points, and misses its factor.
        fi = FeynmanIntegral.from_cnickel("12e|3e|4e|4e|e|:zzzzz")
        faces = set(landau_analysis(fi).landau_surfaces)
        first, second = one_loop_landau_surfaces_by_type(fi)
        closed = set(first) | set(second)
        assert faces < closed
        assert len(closed - faces) == 1
        assert not (closed - faces) & set(first)

    @requires_singular
    def test_hexagon_whatever_sympys_random_state(self) -> None:
        # SymPy's factorisation draws evaluation points from a generator the process shares:
        # from this state it had not factored a Gram minor of 130 terms, which factors in a
        # fifth of a second, after seven minutes. Singular's does not depend on it.
        if not hasattr(signal, "SIGALRM"):
            pytest.skip("stopping a stalled factorisation needs SIGALRM")

        def expire(signum: int, frame: object) -> None:
            raise TimeoutError("the hexagon's closed form took more than 300 s")

        fi = FeynmanIntegral.from_cnickel("12e|3e|4e|5e|5e|e|:zzzzzz")
        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(300)
        try:
            sympy_random.seed(0)
            first, second = one_loop_landau_surfaces_by_type(fi)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        assert (len(first), len(second)) == (37, 57)

    @requires_singular
    def test_a_box_skips_its_own_face_and_says_so(self) -> None:
        # With three massive lines the box's polytope has 13 points, one more than a limit of
        # 12, so even at generic kinematics its own factor is missing from the faces.
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnz")
        report = AnalysisReport.from_integral(fi, ["landau"], max_face_points=12)
        assert report.landau is not None
        analysis = report.landau.analysis
        assert [len(face) for face in analysis.skipped_faces] == [13]
        assert report.landau.skipped == ((4, 13, True),)
        closed = set(one_loop_landau_surfaces(fi))
        faces = set(analysis.landau_surfaces)
        assert faces < closed
        assert len(closed - faces) == 1
        text = " ".join(render_text(report).split())
        assert (
            "1 face was skipped as too large to eliminate, and its discriminant is missing "
            "from the list: the whole polytope, 13 points."
        ) in text

    @requires_singular
    def test_hexagon_report(self) -> None:
        # The eight faces skipped, seven of 15 points and the polytope itself, miss eight factors.
        fi = FeynmanIntegral.from_cnickel("12e|3e|4e|5e|5e|e|:zzzzzz")
        landau = AnalysisReport.from_integral(fi, ["landau"]).landau
        assert landau is not None
        closed = set(landau.first_type) | set(landau.second_type)
        faces = set(landau.analysis.landau_surfaces)
        assert faces < closed
        assert len(closed - faces) == 8
