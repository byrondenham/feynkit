"""The columns of FeynmanIntegral.gkz are the monomials of G at every kinematic point.

A coefficient of F is a sum, and it can vanish while its terms do not: on shell
a tree carrying one leg j has s = p_j^2 = 0, and p_1^2 = m_1^2 cancels a mass.
The oracle is the support of G, the points of fi.newton_polytope, which comes
from expanding G.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
import sympy as sp

from feynkit import FeynmanIntegral, Graph
from feynkit.cli import _print_newton
from feynkit.io.report import AnalysisReport
from feynkit.systems.gkz import construct_gkz_matrix_from_exponents

P = [sp.Symbol(f"p{i}^2", real=True) for i in range(1, 5)]
S = sp.Symbol("s", real=True)
M = sp.Symbol("M", positive=True)
X = sp.Symbol("x", positive=True)


def substitute(fi: FeynmanIntegral, values: dict[sp.Symbol, sp.Expr]) -> FeynmanIntegral:
    """fi with values substituted for the invariants of its momentum products."""
    products = {k: sp.expand(sp.sympify(v).subs(values)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


def on_shell(cnickel: str) -> FeynmanIntegral:
    """The integral of a four-leg CNickel string with p_i^2 = 0."""
    return substitute(FeynmanIntegral.from_cnickel(cnickel), dict.fromkeys(P, 0))


def with_masses(fi: FeynmanIntegral, masses: list[sp.Expr]) -> FeynmanIntegral:
    """fi with these masses on its internal edges, in edge order."""
    graph = fi.graph
    given = iter(masses)
    edges = [dataclasses.replace(e, mass=next(given)) if e.is_internal else e for e in graph.edges]
    return fi.with_(
        graph=Graph(graph.internal_vertices, graph.external_legs, edges, graph.energy_scale)
    )


def mass(fi: FeynmanIntegral, edge: int) -> sp.Expr:
    """The mass of internal edge `edge`, counting from one."""
    return fi.graph.get_internal_edges()[edge - 1].get_mass()


def newton_matrix(fi: FeynmanIntegral) -> sp.Matrix:
    """The A-matrix of the Newton points, in the column order of fi.gkz."""
    points = sorted(fi.newton_polytope.points, key=lambda v: (-sum(v), tuple(-e for e in v)))
    return construct_gkz_matrix_from_exponents(points, len(fi.symanzik.lp_parameters))


def one_mass_triangle_at_p1_squared_m1_squared() -> FeynmanIntegral:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
    return substitute(fi, {P[0]: mass(fi, 1) ** 2})


def bubble_at_s_m1_squared_plus_m2_squared() -> FeynmanIntegral:
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    return substitute(fi, {S: mass(fi, 1) ** 2 + mass(fi, 2) ** 2})


def box_with_p4_squared_zero() -> FeynmanIntegral:
    return substitute(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz"), {P[3]: 0})


def equal_mass_triangle_at_p1_squared_2_m_squared() -> FeynmanIntegral:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:aaa")
    return substitute(fi, {P[0]: 2 * mass(fi, 1) ** 2})


SPECIAL = {
    "on-shell box": lambda: on_shell("12e|3e|3e|e|:zzzz"),
    "on-shell 12ee|22e|e|": lambda: on_shell("12ee|22e|e|:zzzz"),
    "on-shell planar double box": lambda: on_shell("15e|24|3e|4e|5|e|:zzzzzzz"),
    "on-shell massive box": lambda: on_shell("12e|3e|3e|e|:nnnn"),
    "box with p_4^2 = 0": box_with_p4_squared_zero,
    "one-mass triangle at p_1^2 = m_1^2": one_mass_triangle_at_p1_squared_m1_squared,
    "massive bubble at s = m_1^2 + m_2^2": bubble_at_s_m1_squared_plus_m2_squared,
    "equal-mass triangle at p_1^2 = 2 m_a^2": equal_mass_triangle_at_p1_squared_2_m_squared,
}

GENERIC = (
    "11e|e|:zz",
    "11e|e|:nn",
    "111e|e|:nnn",
    "12e|2e|e|:zzz",
    "12e|2e|e|:nzz",
    "12e|2e|e|:aaa",
    "12e|3e|3e|e|:zzzz",
    "12e|3e|3e|e|:nnnn",
    "12ee|22e|e|:zzzz",
    "15e|24|3e|4e|5|e|:zzzzzzz",
)

RANDOM_GRAPHS = (
    "11e|e|:nn",
    "12e|2e|e|:nnn",
    "12e|2e|e|:nzz",
    "12e|3e|3e|e|:nnzz",
    "12e|3e|3e|e|:zzzz",
)


def random_special_point(rng: random.Random, cnickel: str) -> FeynmanIntegral:
    """cnickel at a random special point: numerical and shared masses, and invariants
    set to zero, to small integers or to squared masses."""
    fi = FeynmanIntegral.from_cnickel(cnickel)
    masses: list[sp.Expr] = []
    for e in fi.graph.get_internal_edges():
        m = e.get_mass()
        choice = rng.randrange(4)
        masses.append(m if m == 0 or choice < 2 else M if choice == 2 else sp.Integer(1))
    fi = with_masses(fi, masses)
    squares = [m**2 for m in masses if m != 0]
    invariants = sorted(
        set().union(*(sp.sympify(v).free_symbols for v in fi.momentum_products.values())),
        key=str,
    )
    values: dict[sp.Symbol, sp.Expr] = {}
    for x in invariants:
        options = [x, x, sp.Integer(0), sp.Integer(rng.randint(-2, 2)), *squares]
        values[x] = rng.choice(options)
    return substitute(fi, values)


class TestColumnsAreTheNewtonPoints:
    @pytest.mark.parametrize("name", sorted(SPECIAL))
    def test_at_special_points(self, name: str) -> None:
        fi = SPECIAL[name]()
        assert fi.gkz.a_matrix == newton_matrix(fi)

    @pytest.mark.parametrize("cnickel", GENERIC)
    def test_at_default_kinematics(self, cnickel: str) -> None:
        fi = FeynmanIntegral.from_cnickel(cnickel)
        assert fi.gkz.a_matrix == newton_matrix(fi)

    def test_for_the_massive_tadpole(self) -> None:
        fi = FeynmanIntegral.from_cnickel("0|:n", use_mandelstam=False)
        assert fi.gkz.a_matrix == newton_matrix(fi) == sp.Matrix([[1, 1], [2, 1]])

    def test_at_random_special_points(self) -> None:
        rng = random.Random(0)
        wrong = []
        for cnickel in rng.choices(RANDOM_GRAPHS, k=40):
            fi = random_special_point(rng, cnickel)
            if fi.gkz.a_matrix != newton_matrix(fi):
                wrong.append((cnickel, fi.gkz.a_matrix.cols, len(fi.newton_polytope.points)))
        assert wrong == []

    def test_where_a_rational_coefficient_is_zero_but_does_not_expand_to_zero(self) -> None:
        # m_1^2 - p_1^2 = s/x - s/(x + 1) - s/(x (x + 1)) is 0, but expanding does not show
        # it; the support of G keeps the monomial, and so must gkz.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        fi = with_masses(fi, [sp.sqrt(S / X - S / (X + 1)), 0, 0])
        fi = substitute(fi, {P[0]: S / (X * (X + 1))})
        assert fi.gkz.a_matrix == newton_matrix(fi)
        assert fi.gkz.a_matrix.cols == 7

    def test_where_a_mass_is_zero_only_after_expanding(self) -> None:
        # m_1^2 = x (x + 1) - x^2 - x expands to 0, so G has no u_1^2 monomial, and nor
        # may gkz.
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        fi = with_masses(fi, [sp.sqrt(X * (X + 1) - X**2 - X), 0, 0])
        assert fi.gkz.a_matrix == newton_matrix(fi)
        assert fi.gkz.a_matrix.cols == 6

    def test_where_the_masses_are_the_python_float_0(self) -> None:
        # 0.0 == 0 in Python, but sp.sympify(0.0) is Float(0.0), which SymPy does not treat
        # as 0. Such masses are massless: no u_e^2 columns, at generic kinematics or on shell.
        generic = with_masses(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz"), [0.0] * 4)
        on_shell_box = with_masses(on_shell("12e|3e|3e|e|:zzzz"), [0.0] * 4)
        for fi, columns in ((generic, 10), (on_shell_box, 6)):
            assert fi.gkz.a_matrix == newton_matrix(fi)
            assert fi.gkz.a_matrix.cols == columns


@pytest.fixture(scope="module")  # type: ignore[misc]
def box() -> FeynmanIntegral:
    return on_shell("12e|3e|3e|e|:zzzz")


class TestOnShellBox:
    def test_gkz_has_six_columns(self, box: FeynmanIntegral) -> None:
        assert box.gkz.a_matrix.shape == (5, 6)
        assert len(box.gkz.z_variables) == 6
        assert len(box.gkz.euler_equations) == 5

    def test_toric_ideal_has_one_generator(self, box: FeynmanIntegral) -> None:
        assert len(box.toric_ideal.generators) == 1

    def test_symmetry_pairs_are_the_polytope_automorphisms(self, box: FeynmanIntegral) -> None:
        assert len(box.symmetry_pairs) == box.polytope_automorphisms.order == 72

    def test_report_numbers(self, box: FeynmanIntegral) -> None:
        report = AnalysisReport.from_integral(box, ["polynomials", "gkz", "schwinger"])
        rows = dict(report.summary())
        assert (rows["Monomials of G"], rows["Codimension"], rows["Toric generators"]) == (
            "6",
            "1",
            "1",
        )
        assert all(entry.coefficient != 0 for entry in report.polynomials.z_table)
        assert report.schwinger is not None
        assert report.schwinger.columns_match

    def test_fk_newton_section(
        self, box: FeynmanIntegral, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _print_newton(box)
        lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
        assert "Monomials (A-columns) 6" in lines
        assert "Hull vertices 6" in lines
        assert "Normalised volume 3 (the holonomic rank for generic beta)" in lines
