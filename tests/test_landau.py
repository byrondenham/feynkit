"""Tests for the principal A-determinant and Landau surfaces.

Oracles: the one-loop closed form of Dlapa, Helmer, Papathanasiou and
Tellander (arXiv:2304.02629, eq. 1LoopEA): the reduced principal
A-determinant of G = U + F is the product of the principal minors of the
modified Cayley matrix. Face-by-face computation must reproduce it.
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, landau_analysis, landau_analysis_from_polynomial
from feynkit import landau as landau_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.kinematics.mandelstam import standard_invariants
from feynkit.landau import (
    _eliminate_singular,
    _elimination_discriminant,
    _read_singular_polynomial,
    _renaming,
    _singular_binary,
    one_loop_landau_surfaces,
    one_loop_landau_surfaces_by_type,
    one_loop_principal_a_determinant,
)
from feynkit.polytope import lattice_coordinates

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")


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


def eliminate(timeout: float | None = None) -> list[sp.Poly]:
    """A system in w and t over x, y and z, which Singular names v0 to v4."""
    return _eliminate_singular(
        [W * T - 1], [W, T], [X, Y, Z], "Singular", points=10, timeout=timeout
    )


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

    def test_timeout_reaches_singular(
        self, monkeypatch: pytest.MonkeyPatch, massive_bubble: FeynmanIntegral
    ) -> None:
        seen = fake_singular(monkeypatch)
        landau_analysis(massive_bubble)
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
        factors, principal = _elimination_discriminant(
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
        assert principal
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
        factors, principal = _elimination_discriminant(
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
        assert plain == (factors, principal)

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

        def spy(*args: object, **kwargs: object) -> tuple[list[sp.Expr], bool]:
            scales.append(kwargs["scale"])  # type: ignore[arg-type]
            return eliminate(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(landau_module, "_elimination_discriminant", spy)
        analysis = landau_analysis_from_polynomial(g, variables, scale=mu)
        assert scales == [mu]
        assert analysis.landau_surfaces == landau_analysis(massive_bubble).landau_surfaces

    @requires_singular
    @pytest.mark.parametrize(("cnickel", "count"), [("12e|2e|e|:zzz", 3), ("12e|2e|e|:nnn", 19)])
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
