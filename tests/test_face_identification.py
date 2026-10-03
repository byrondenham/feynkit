"""Graphs of the faces of the Newton polytope of G = U + F, checked against the literature.

FMT24a is Fevola, Mizera and Telen, Principal Landau determinants, arXiv:2311.16219;
AHM22 is Arkani-Hamed, Hillman and Mizera, Feynman polytopes and the tropical geometry of
UV and IR divergences, arXiv:2202.12296; BGH26 is Britto, Grimm and Hoefnagels, Resonance
and differential reduction of Feynman integrals, arXiv:2606.09978. Pages are those printed.
"""

from __future__ import annotations

from collections import Counter

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from feynkit import FeynmanIntegral
from feynkit.core import Edge, Graph
from feynkit.core.exceptions import ValidationError
from feynkit.face_identification import FaceIdentification, identify_faces
from feynkit.polynomials import minor_polynomials
from feynkit.polytope import polytope_data


def _lp(fi: FeynmanIntegral) -> dict[int, sp.Symbol]:
    edges = fi.graph.get_internal_edges()
    return {e.idx: u for e, u in zip(edges, fi.symanzik.lp_parameters, strict=True)}


def _minor(fi: FeynmanIntegral, contract=(), delete=()) -> tuple[sp.Expr, sp.Expr]:
    return minor_polynomials(
        fi.graph, fi.momentum_products, contract=contract, delete=delete, parameters=_lp(fi)
    )


def _facets(found: tuple[FaceIdentification, ...]) -> list[FaceIdentification]:
    return [face for face in found if face.facet is not None]


def _inward(face: FaceIdentification) -> tuple[int, ...]:
    assert face.facet is not None
    return tuple(-m for m in face.facet.normal)


class TestParachute:
    """FMT24a, section 3.5: the parachute with distinct non-zero masses."""

    fi = FeynmanIntegral.from_cnickel("12e|22e|e|:nnnn")

    def test_the_nine_rays(self) -> None:
        """The rays of the normal fan (FMT24a, p. 29)."""
        rays = {
            (-1, -1, -1, -1),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 1),
            (1, 1, 1, 0),
            (1, 1, 0, 1),
            (1, 1, 1, 1),
        }
        assert {_inward(face) for face in _facets(identify_faces(self.fi))} == rays

    def test_each_subgraph_facet_is_u_gamma_times_g_of_the_quotient(self) -> None:
        """FMT24a, Eq. 3.15, p. 28: in_{w_gamma}(G) = U_gamma G_{Gamma/gamma}."""
        edges = [1, 2, 3, 4]
        for face in _facets(identify_faces(self.fi)):
            ray = _inward(face)
            if min(ray) < 0:
                continue
            gamma = [e for e, w in zip(edges, ray, strict=True) if w == 1]
            rest = [e for e in edges if e not in gamma]
            if rest:
                expected = _minor(self.fi, delete=rest)[0] * sum(_minor(self.fi, contract=gamma))
            else:
                expected = _minor(self.fi)[0]  # G of Gamma/Gamma is 1
            assert face.verified
            assert sp.expand(face.polynomial - expected) == 0, gamma

    def test_names(self) -> None:
        names = {_inward(face): face.name() for face in _facets(identify_faces(self.fi))}
        assert names[(0, 0, 1, 1)] == "U({3,4}) G(Gamma/{3,4})"
        assert names[(1, 1, 1, 0)] == "U({1,2,3}) G(Gamma/{1,2,3})"
        assert names[(1, 0, 0, 0)] == "G(Gamma/{1})"
        assert names[(1, 1, 1, 1)] == "U(Gamma)"
        assert names[(-1, -1, -1, -1)] == "F(Gamma)"

    def test_kinds(self) -> None:
        kinds = Counter(face.kind for face in _facets(identify_faces(self.fi)))
        assert kinds == {"contraction": 4, "product_uv": 3, "u_layer": 1, "f_layer": 1}


def test_massive_sunrise_has_three_uv_bubble_facets() -> None:
    """AHM22, App. B, Eq. B14, p. 9: six rays, three of them the UV bubbles gamma_ij."""
    fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
    facets = _facets(identify_faces(fi))
    layers = [face for face in facets if face.kind in ("u_layer", "f_layer")]
    assert len(facets) - len(layers) == 6
    bubbles = [face for face in facets if face.kind == "product_uv"]
    assert sorted(face.levels[0].edges for face in bubbles) == [(1, 2), (1, 3), (2, 3)]
    for face in bubbles:
        gamma = face.levels[0].edges
        (rest,) = {1, 2, 3} - set(gamma)
        u_gamma = _minor(fi, delete=[rest])[0]
        assert sp.expand(face.polynomial - u_gamma * sum(_minor(fi, contract=gamma))) == 0


def test_three_mass_box_ir_facet() -> None:
    """AHM22, App. B, p. 10: the facet of gamma_14, with U and F restricted as in B24-B25.

    feynkit's box has leg j at vertex j and the edges 1: v1-v2, 2: v1-v3, 3: v2-v4,
    4: v3-v4. With p_2^2 = 0 the massless corner is vertex 2, between edges 1 and 3,
    so the paper's gamma_14 is {2, 4}.
    """
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    p2 = sp.Symbol("p2^2", real=True)
    fi = fi.with_(momentum_products={k: v.subs(p2, 0) for k, v in fi.momentum_products.items()})
    (face,) = [f for f in _facets(identify_faces(fi)) if f.facet.normal == (1, 0, 1, 0)]
    assert face.facet.offset == 1
    u = _lp(fi)
    mu = fi.graph.energy_scale
    p1, p4 = sp.Symbol("p1^2", real=True), sp.Symbol("p4^2", real=True)
    s12 = sp.Symbol("s12", real=True)
    mp = fi.momentum_products
    p13 = p1 + sp.Symbol("p3^2", real=True) + 2 * mp[(1, 3)]
    expected = (
        u[1]
        + u[3]
        - (p1 * u[1] * u[2] + s12 * u[2] * u[3] + p13 * u[1] * u[4] + p4 * u[3] * u[4]) / mu**2
    )
    assert sp.expand(face.polynomial - expected) == 0
    assert [level.edges for level in face.levels] == [(2, 4), (1, 3)]
    # The coefficients of u_2 u_1, u_2 u_3, u_4 u_1, u_4 u_3 form a matrix of rank 2, so the
    # F part is no product of a polynomial in u_2, u_4 and one in u_1, u_3, but its exponents
    # are those of the predicted product G({2,4}) U(Gamma/{2,4}).
    assert face.kind == "support_product"
    assert face.support_verified and not face.verified
    assert face.name() == "G({2,4}) U(Gamma/{2,4})"


def _three_mass_box() -> FeynmanIntegral:
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    p2 = sp.Symbol("p2^2", real=True)
    return fi.with_(momentum_products={k: v.subs(p2, 0) for k, v in fi.momentum_products.items()})


def _exponents(expr: sp.Expr, fi: FeynmanIntegral) -> set[tuple[int, ...]]:
    return set(sp.Poly(expr, *fi.symanzik.lp_parameters).monoms())


class TestSupportProducts:
    """Faces whose exponents are the product of the supports of the flag's factors, while
    G|_F is not their product."""

    def test_the_three_mass_box(self) -> None:
        fi = _three_mass_box()
        found = [f for f in identify_faces(fi, max_codimension=None) if f.kind == "support_product"]
        assert len(found) == 4
        assert len([face for face in found if face.facet is not None]) == 1
        for face in found:
            # The exponents of G|_F are the sums of one exponent of each factor, computed afresh.
            sums = {(0,) * 4}
            for level in face.levels:
                u, f = _minor(fi, contract=level.contracted, delete=level.deleted)
                factor = {"U": u, "F": f, "G": u + f}[level.kind]
                sums = {
                    tuple(a + b for a, b in zip(x, y, strict=True))
                    for x in sums
                    for y in _exponents(factor, fi)
                }
            assert _exponents(face.polynomial, fi) == sums
            assert sp.expand(face.polynomial - face.prediction) != 0
            assert face.reason is not None

    def test_the_on_shell_double_box(self) -> None:
        fi = FeynmanIntegral.from_cnickel(
            "15e|24|3e|4e|5|e|:zzzzzzz", kinematics="massless_on_shell"
        )
        counts = Counter(face.kind for face in identify_faces(fi, max_codimension=None))
        assert (counts["support_product"], counts["unidentified"]) == (22, 910)

    def test_none_on_the_on_shell_box(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz", kinematics="massless_on_shell")
        assert all(f.kind != "support_product" for f in identify_faces(fi, max_codimension=None))

    def test_support_verified_follows_the_class(self) -> None:
        for face in identify_faces(_three_mass_box(), max_codimension=None):
            assert face.support_verified == (face.kind != "unidentified")
            assert face.verified == (face.kind not in ("unidentified", "support_product"))


# The classes of every face before support products had a class of their own: a face then
# identified keeps its class, and the support products come out of the unidentified faces.
_BEFORE = {
    ("12e|3e|3e|e|:zzzz", None): {
        "contraction": 10,
        "f_layer": 27,
        "product_ir": 28,
        "u_layer": 15,
        "whole": 1,
    },
    ("12e|3e|3e|e|:zzzz", "massless_on_shell"): {
        "contraction": 6,
        "f_layer": 3,
        "product_ir": 4,
        "u_layer": 15,
        "unidentified": 20,
        "whole": 1,
    },
    ("three-mass box", None): {
        "contraction": 9,
        "f_layer": 18,
        "product_ir": 20,
        "u_layer": 15,
        "unidentified": 4,
        "whole": 1,
    },
    ("12e|23|3|e|:zzzzz", None): {
        "contraction": 13,
        "f_layer": 51,
        "product_ir": 95,
        "product_uv": 8,
        "u_layer": 51,
        "whole": 1,
    },
    ("12e|23|3|e|:nnnnn", None): {
        "contraction": 23,
        "f_layer": 81,
        "product_uv": 57,
        "u_layer": 51,
        "whole": 1,
    },
    ("111e|e|:nnz", None): {
        "contraction": 3,
        "f_layer": 9,
        "product_ir": 1,
        "product_uv": 4,
        "u_layer": 7,
        "whole": 1,
    },
    ("12ee|22e|e|:zzzz", None): {
        "contraction": 4,
        "f_layer": 15,
        "product_ir": 27,
        "product_uv": 1,
        "u_layer": 19,
        "whole": 1,
    },
}


@pytest.mark.parametrize(("cnickel", "kinematics"), list(_BEFORE))
def test_identified_faces_keep_their_class(cnickel: str, kinematics: str | None) -> None:
    if cnickel == "three-mass box":
        fi = _three_mass_box()
    else:
        fi = FeynmanIntegral.from_cnickel(cnickel, kinematics=kinematics)
    counts = Counter(face.kind for face in identify_faces(fi, max_codimension=None))
    before = Counter(_BEFORE[(cnickel, kinematics)])
    split = ("unidentified", "support_product")
    assert {k: n for k, n in counts.items() if k not in split} == {
        k: n for k, n in before.items() if k not in split
    }
    assert counts["unidentified"] + counts["support_product"] == before["unidentified"]


class TestEdgeFaces:
    """BGH26, Eq. 21, p. 11: the face F_e of the terms free of x_e is Gamma/e."""

    @pytest.mark.parametrize(
        "cnickel", ["12e|2e|e|:nnn", "12e|3e|3e|e|:zzzz", "12e|23|3|e|:nnnnn", "111e|e|:nnn"]
    )
    def test_edge_facets_are_contractions(self, cnickel: str) -> None:
        fi = FeynmanIntegral.from_cnickel(cnickel)
        for face in _facets(identify_faces(fi)):
            if face.facet.offset != 0:
                continue
            (e,) = [k + 1 for k, m in enumerate(face.facet.normal) if m]
            assert face.kind == "contraction"
            assert face.contracted == (e,)
            assert sp.expand(face.polynomial - sum(_minor(fi, contract=[e]))) == 0

    def test_bubble_with_a_massless_line(self) -> None:
        """BGH26, Eq. 105, p. 29: with m_2 = 0, F_1 is no facet, F_2 still is, and x_2 <= 1 is
        a new one."""
        fi = FeynmanIntegral.from_cnickel("11e|e|:nz")
        found = identify_faces(fi)
        names = {face.facet.normal: (face.kind, face.name()) for face in _facets(found)}
        assert names == {
            (0, -1): ("contraction", "G(Gamma/{2})"),
            (0, 1): ("product_ir", "G({1}) U(Gamma/{1})"),
            (-1, -1): ("u_layer", "U(Gamma)"),
            (1, 1): ("f_layer", "F(Gamma)"),
        }

    def test_sunrise_with_a_massless_line(self) -> None:
        """BGH26, p. 43: with m_3 = 0, F_1 and F_2 are no facets and F_3 is Gamma/3."""
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnz")
        contractions = [f for f in _facets(identify_faces(fi)) if f.facet.offset == 0]
        assert [(f.facet.normal, f.kind, f.contracted) for f in contractions] == [
            ((0, 0, -1), "contraction", (3,))
        ]


def test_a_forest_of_two_components_contracts() -> None:
    """Edges 1 and 4 of the massive box share no vertex: their face is Gamma/{1,4}."""
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    (face,) = [
        f
        for f in identify_faces(fi)
        if f.codimension == 2 and f.weight is not None and f.levels[0].edges == (1, 4)
    ]
    assert face.levels[0].loops == 0
    assert (face.kind, face.contracted, face.name()) == ("contraction", (1, 4), "G(Gamma/{1,4})")


def test_the_polytope_itself_is_the_whole_graph() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    (whole,) = [face for face in identify_faces(fi) if face.codimension == 0]
    assert whole.kind == "whole"
    assert whole.name() == "G(Gamma)"
    assert sp.expand(whole.polynomial - fi.symanzik.g) == 0


def test_on_shell_massless_box() -> None:
    """A regression: the faces of dimension at least 1 that the flag leaves unidentified."""
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz", kinematics="massless_on_shell")
    found = identify_faces(fi, max_codimension=None)
    unidentified = [face for face in found if face.kind == "unidentified"]
    assert len([face for face in unidentified if face.dimension >= 1]) == 20
    assert len([face for face in unidentified if face.facet is not None]) == 4
    for face in unidentified:
        assert face.reason is not None
        assert sp.expand(face.polynomial - face.prediction) != 0


def test_a_polytope_below_full_dimension_is_not_identified() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz", kinematics="massless_on_shell")
    found = identify_faces(fi)
    assert found
    assert all(face.kind == "unidentified" and not face.levels for face in found)
    assert {face.reason for face in found} == {"the Newton polytope is not full-dimensional"}


class TestCodimension:
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")

    def test_the_default_stops_at_codimension_two(self) -> None:
        found = identify_faces(self.fi)
        assert {face.codimension for face in found} == {0, 1, 2}
        data = polytope_data(self.fi.newton_polytope.points)
        assert [face.facet for face in found if face.codimension == 1] == list(data.facets)

    def test_none_gives_every_face(self) -> None:
        found = identify_faces(self.fi, max_codimension=None)
        assert {face.codimension for face in found} == {0, 1, 2, 3, 4}

    @pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
    def test_a_bad_codimension_is_rejected(self, bad: object) -> None:
        with pytest.raises(ValidationError, match="max_codimension"):
            identify_faces(self.fi, max_codimension=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("cnickel", "kinematics"),
    [
        ("12e|23|3|e|:nnnnn", None),
        ("12e|23|3|e|:zzzzz", None),
        ("12ee|22e|e|:zzzz", None),
        ("12e|3e|3e|e|:zzzz", "massless_on_shell"),
    ],
)
def test_verified_identifications_multiply_back(cnickel: str, kinematics: str | None) -> None:
    """Every verified face is the product of its level polynomials, computed afresh."""
    fi = FeynmanIntegral.from_cnickel(cnickel, kinematics=kinematics)
    for face in identify_faces(fi, max_codimension=None):
        if not face.verified:
            continue
        product = sp.Integer(1)
        for level in face.levels:
            u, f = _minor(fi, contract=level.contracted, delete=level.deleted)
            product *= {"U": u, "F": f, "G": u + f}[level.kind]
        assert sp.expand(product - face.polynomial) == 0, face.name()


RELABELLED = ["11e|e|:nz", "12e|2e|e|:nzz", "111e|e|:nnz", "12e|3e|3e|e|:nzzz", "12ee|22e|e|:zzzz"]


@given(st.sampled_from(RELABELLED), st.randoms(use_true_random=False))
def test_class_counts_do_not_depend_on_edge_labels(cnickel: str, rng) -> None:  # type: ignore[no-untyped-def]
    graph = Graph.from_cnickel(cnickel)
    internal = graph.get_internal_edges()
    labels = [e.idx for e in internal]
    shuffled = list(labels)
    rng.shuffle(shuffled)
    relabel = dict(zip(labels, shuffled, strict=True))
    edges = [
        Edge(idx=relabel[e.idx], v1=e.v1, v2=e.v2, is_internal=True, mass=e.get_mass())
        for e in internal
    ] + graph.get_external_edges()
    other = Graph(graph.internal_vertices, graph.external_legs, edges)

    def counts(g: Graph) -> Counter[str]:
        return Counter(face.kind for face in identify_faces(FeynmanIntegral(g)))

    assert counts(other) == counts(graph)


def test_levels_say_which_minors_are_scaleless() -> None:
    """F vanishes on the quotient of the three-mass box's IR facet, a bubble whose external
    momentum is the massless one, and not on the hard factor G({2,4})."""
    fi = _three_mass_box()
    (face,) = [f for f in _facets(identify_faces(fi)) if f.facet.normal == (1, 0, 1, 0)]
    assert [level.scaleless for level in face.levels] == [False, True]
    whole = identify_faces(fi, max_codimension=0)[0]
    assert [level.scaleless for level in whole.levels] == [False]
