"""U and F of Feynman-graph minors, in the parameters and kinematics of the graph."""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.polynomials import minor_polynomials

GRAPHS = [
    "11e|e|:nn",
    "12e|2e|e|:nzz",
    "12e|3e|3e|e|:zzzz",
    "111e|e|:nnz",
    "12e|22e|e|:nnnn",
    "12e|23|3|e|:nnnnn",
    "12ee|22e|e|:zzzz",
]


def _minor(fi: FeynmanIntegral, **sets: list[int]) -> tuple[sp.Expr, sp.Expr]:
    return minor_polynomials(fi.graph, fi.momentum_products, **sets)


@pytest.mark.parametrize("cnickel", GRAPHS)
def test_the_whole_graph_gives_feynkits_u_and_f(cnickel: str) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    u, f = _minor(fi)
    assert sp.expand(u - fi.symanzik.u) == 0
    assert sp.expand(f - fi.symanzik.f) == 0


def test_on_shell_kinematics_are_those_of_the_graph() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz", kinematics="massless_on_shell")
    u, f = _minor(fi)
    assert sp.expand(u - fi.symanzik.u) == 0
    assert sp.expand(f - fi.symanzik.f) == 0


@pytest.mark.parametrize("cnickel", GRAPHS)
def test_an_edge_face_is_the_contraction(cnickel: str) -> None:
    """BGH26 (Eq. 21, p. 11): the face F_e of the terms free of x_e is Gamma/e."""
    fi = FeynmanIntegral.from_cnickel(cnickel)
    g = sp.expand(fi.symanzik.u + fi.symanzik.f)
    for edge in fi.graph.get_internal_edges():
        a = fi.graph.schwinger_parameters[edge.idx]
        u, f = _minor(fi, contract=[edge.idx])
        assert sp.expand(g.subs(a, 0) - u - f) == 0, (cnickel, edge.idx)


@pytest.mark.parametrize("gamma", [[3, 4], [1, 2, 3], [1, 2, 4], [1], [1, 2]])
def test_fmt24a_factorisation_on_the_massive_parachute(gamma: list[int]) -> None:
    """FMT24a (Eq. 3.13, p. 27): G(a_gamma -> eps a_gamma) = eps^L_gamma U_gamma G_{Gamma/gamma}
    + O(eps^(L_gamma + 1)) for connected gamma and non-zero masses."""
    fi = FeynmanIntegral.from_cnickel("12e|22e|e|:nnnn")
    eps = sp.Symbol("eps")
    a = fi.graph.schwinger_parameters
    g = sp.expand((fi.symanzik.u + fi.symanzik.f).subs({a[e]: eps * a[e] for e in gamma}))
    rest = [e.idx for e in fi.graph.get_internal_edges() if e.idx not in gamma]
    u_gamma, _ = _minor(fi, delete=rest)
    u_quotient, f_quotient = _minor(fi, contract=gamma)
    loops = fi.graph.delete(rest).get_loop_count()
    leading = sp.Poly(g, eps).coeff_monomial(eps**loops)
    assert all(sp.Poly(g, eps).coeff_monomial(eps**k) == 0 for k in range(loops))
    assert sp.expand(leading - u_gamma * (u_quotient + f_quotient)) == 0


def test_legs_on_a_removed_vertex_are_dropped() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    a3 = fi.graph.schwinger_parameters[3]
    mu = fi.graph.energy_scale
    m3 = fi.graph.get_internal_edges()[2].get_mass()
    u, f = _minor(fi, delete=[1, 2])
    p23 = fi.momentum_products[(2, 3)]
    assert u == 1
    assert sp.expand(f - (p23 + m3**2) * a3 / mu**2) == 0


def test_a_disconnected_minor_follows_the_forest_convention() -> None:
    fi = FeynmanIntegral.from_cnickel("11e|2|33|e|:nnnnn")
    u, f = _minor(fi, delete=[3])
    u_left, f_left = _minor(fi, delete=[3, 4, 5])
    u_right, f_right = _minor(fi, delete=[1, 2, 3])
    assert sp.expand(u - u_left * u_right) == 0
    assert sp.expand(f - f_left * u_right - u_left * f_right) == 0


def test_contraction_moves_the_legs() -> None:
    """The box with two adjacent edges contracted is a bubble with legs 1, 2, 3 on one side."""
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    a = fi.graph.schwinger_parameters
    mu = fi.graph.energy_scale
    u, f = _minor(fi, contract=[1, 2])
    p4 = sp.Symbol("p4^2", real=True)
    assert sp.expand(u - a[3] - a[4]) == 0
    assert sp.expand(f + p4 * a[3] * a[4] / mu**2) == 0


def test_the_parameters_can_be_chosen() -> None:
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    edges = fi.graph.get_internal_edges()
    u_lp = {e.idx: x for e, x in zip(edges, fi.symanzik.lp_parameters, strict=True)}
    u, f = minor_polynomials(fi.graph, fi.momentum_products, parameters=u_lp)
    assert sp.expand(u + f - fi.symanzik.g) == 0


def test_an_edge_contracted_and_deleted_is_rejected() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    with pytest.raises(ValidationError, match="both"):
        _minor(fi, contract=[1], delete=[1, 2])


def test_an_unknown_edge_is_rejected() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    with pytest.raises(ValidationError, match="9"):
        _minor(fi, contract=[9])
