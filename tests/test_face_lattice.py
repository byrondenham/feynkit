"""Tests for feynkit.face_lattice: resonance, admissibility and reducibility on every face.

The oracles are :func:`feynkit.resonance.classify_facets` on every facet, the statements of
Schulze and Walther, arXiv:1009.3569 (SW12 below), of Britto, Grimm and Hoefnagels,
arXiv:2606.09978 (BGH26), and of Grimm and Hoefnagels, arXiv:2409.13815 (GH25), with the page
numbers of their arXiv versions, and small configurations worked by hand.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.face_lattice import (
    DecoratedConfiguration,
    DecoratedFace,
    DecoratedFaceLattice,
    decorate_configuration,
    decorate_faces,
)
from feynkit.polytope import polytope_data
from feynkit.resonance import EpsilonSet, classify_facets
from feynkit.systems.cayley import lp_to_cayley
from feynkit.systems.monomial import extract_monomial_support

F = Fraction


def _matrix(columns: list[tuple[int, ...]]) -> list[list[int]]:
    return [[c[i] for c in columns] for i in range(len(columns[0]))]


def _faces(lattice: DecoratedConfiguration) -> dict[tuple[int, ...], DecoratedFace]:
    return {face.point_indices: face for face in lattice.faces}


# --- configurations worked by hand ---------------------------------------------------


QUADRIC_CONE = [(1, 0), (1, 1), (1, 2)]


class TestQuadricCone:
    """SW12's Ex. 3.3 (p. 6): beta = (1/2, 1) on the quadric cone has both extremal rays as
    resonance centres."""

    @pytest.fixture(scope="class")
    @classmethod
    def lattice(cls) -> DecoratedConfiguration:
        return decorate_configuration(_matrix(QUADRIC_CONE), [F(1, 2), F(1)])

    def test_the_faces_with_the_empty_face_first(self, lattice: DecoratedConfiguration) -> None:
        assert [face.point_indices for face in lattice.faces] == [(), (0,), (2,), (0, 1, 2)]
        assert [face.dimension for face in lattice.faces] == [-1, 0, 0, 1]
        assert [face.codimension for face in lattice.faces] == [2, 1, 1, 0]
        assert [lattice.faces[k].point_indices for k in lattice.facets] == [(0,), (2,)]
        assert lattice.rank == 2 and lattice.full_rank

    def test_both_rays_are_resonant_and_the_empty_face_is_not(
        self, lattice: DecoratedConfiguration
    ) -> None:
        faces = _faces(lattice)
        assert faces[(0,)].resonant.kind == faces[(2,)].resonant.kind == "all"
        assert faces[()].resonant.kind == "never"
        assert faces[(0, 1, 2)].resonant.kind == "all"

    def test_the_empty_face_has_defect_two(self, lattice: DecoratedConfiguration) -> None:
        """The facet forms through the empty face, (0, 1) and (2, -1) on ZA = Z^2, span a
        sublattice of index 2, so resonance of the empty face is not that of its facets."""
        faces = _faces(lattice)
        assert faces[()].lattice_defect == 2
        assert all(face.lattice_defect == 1 for face in lattice.faces if face.point_indices)
        assert all(
            lattice.faces[lattice.facets[k]].resonant.kind == "all" for k in faces[()].facets
        )

    def test_two_centres_and_reducible(self, lattice: DecoratedConfiguration) -> None:
        for eps in ("generic", 0, F(1, 3)):
            centres = lattice.centres(eps)
            assert [face.point_indices for face in centres] == [(0,), (2,)]
            assert not any(face.pyramid for face in centres)
            assert lattice.reducible(eps) is True

    def test_a_is_a_pyramid_over_itself_only(self, lattice: DecoratedConfiguration) -> None:
        assert [face.point_indices for face in lattice.faces if face.pyramid] == [(0, 1, 2)]
        assert [face.columns_off for face in lattice.faces] == [3, 2, 2, 0]


def test_a_triangle_with_defects_three_and_nine() -> None:
    """The triangle (0,0), (2,1), (1,2) with its centroid (1,1), for which ZA = Z^3. The edge
    forms (0, -1, 2), (0, 2, -1) and (3, -1, -1) span a sublattice of index 3 in N_G at each
    vertex and of index 9 in Z^3 at the empty face."""
    columns = [(1, 0, 0), (1, 2, 1), (1, 1, 2), (1, 1, 1)]
    lattice = decorate_configuration(_matrix(columns), [F(0)] * 3)
    faces = _faces(lattice)
    assert faces[(0,)].lattice_defect == 3
    assert faces[(1,)].lattice_defect == faces[(2,)].lattice_defect == 3
    assert faces[()].lattice_defect == 9
    assert all(face.lattice_defect == 1 for face in lattice.faces if face.dimension >= 1)


class TestSingleExchange:
    """GH25's single-exchange configuration, Eq. 3.28 (p. 15), at nu = (1, 1, 1, 1 + eps,
    1 + eps) (Eq. 3.25): ZA = Z^5 and F = {1, 2, 3, 4} is a resonant face (Eqs. 4.24-4.27,
    p. 24), over which A is not a pyramid, so the system is reducible (p. 25)."""

    COLUMNS = [
        (1, 0, 0, 0, 0),
        (1, 0, 0, 1, 0),
        (0, 1, 0, 0, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 1, 0, 0),
        (0, 0, 1, 1, 0),
        (0, 0, 1, 0, 1),
    ]

    @pytest.fixture(scope="class")
    @classmethod
    def lattice(cls) -> DecoratedConfiguration:
        return decorate_configuration(
            _matrix(cls.COLUMNS), [F(1), F(1), F(1), F(1), F(1)], [F(0), F(0), F(0), F(1), F(1)]
        )

    def test_eight_facets(self, lattice: DecoratedConfiguration) -> None:
        assert len(lattice.facets) == 8
        assert lattice.full_rank

    def test_the_first_four_columns_are_a_resonant_face(
        self, lattice: DecoratedConfiguration
    ) -> None:
        face = lattice.face((0, 1, 2, 3))
        assert face.resonant.kind == "all"
        assert not face.pyramid

    def test_two_centres_at_generic_eps(self, lattice: DecoratedConfiguration) -> None:
        centres = lattice.centres("generic")
        assert {face.point_indices for face in centres} == {(0, 1, 2, 3), (4, 5, 6)}
        assert not any(face.pyramid for face in centres)
        assert lattice.reducible("generic") is True

    def test_no_defects(self, lattice: DecoratedConfiguration) -> None:
        assert all(face.lattice_defect == 1 for face in lattice.faces)


# --- the guards on reducibility ----------------------------------------------------------

SEGMENT = [(1, 0, 0), (1, 1, 0), (1, 2, 0)]


def test_no_resonant_face_off_the_span_of_a() -> None:
    """beta = (1/2, 1, 1) is not in the span of the segment, so the system has no non-zero
    solutions: no face is resonant and reducibility is not decided."""
    lattice = decorate_configuration(_matrix(SEGMENT), [F(1, 2), F(1), F(1)])
    assert lattice.span.kind == "never"
    assert lattice.resonant_faces("generic") == ()
    assert lattice.centres(0) == ()
    assert lattice.reducible("generic") is None
    assert lattice.reducible(0) is None


def test_reducibility_is_not_decided_below_full_rank() -> None:
    """In the span, the segment has centres, but A has rank 2 with three rows."""
    lattice = decorate_configuration(_matrix(SEGMENT), [F(1, 2), F(1), F(0)])
    assert lattice.span.kind == "all"
    assert lattice.rank == 2 and not lattice.full_rank
    assert [face.point_indices for face in lattice.centres("generic")] == [(0,), (2,)]
    assert lattice.reducible("generic") is None


def test_full_rank_is_the_rank_of_a_not_the_dimension_of_its_columns() -> None:
    """The Cayley matrix of the massive bubble has full rank, while its columns lie on an affine
    hyperplane, so polytope_data never calls them full-dimensional. beta is T beta_LP at
    D = 4 - 2 eps and unit powers."""
    a = FeynmanIntegral.from_cnickel("11e|e|:nn").schwinger_gkz.a_matrix
    columns = [tuple(int(x) for x in a.col(j)) for j in range(a.cols)]
    assert not polytope_data(columns).is_full_dimensional
    t = lp_to_cayley(2, 1)
    beta0 = [sum(int(t[r, k]) * x for k, x in enumerate((F(-2), F(-1), F(-1)))) for r in range(3)]
    beta1 = [int(t[r, 0]) for r in range(3)]
    lattice = decorate_configuration(a.tolist(), beta0, beta1)
    assert lattice.full_rank
    assert lattice.reducible("generic") is True


def test_the_span_can_fix_eps() -> None:
    """With beta(eps) = (1/2, 1, eps), beta lies in the span only at eps = 0."""
    lattice = decorate_configuration(_matrix(SEGMENT), [F(1, 2), F(1), F(0)], [F(0), F(0), F(1)])
    assert lattice.span.kind == "point" and lattice.span.offset == 0
    whole = lattice.face((0, 1, 2))
    assert whole.resonant.kind == "point" and whole.resonant.offset == 0
    assert whole.admissible.kind == "point"


# --- the face test on progressions --------------------------------------------------------

# The unit simplex conv(0, e_1, e_2, e_3), homogenised: ZA = Z^4.
SIMPLEX = [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)]


def test_the_forms_of_a_vertex() -> None:
    """With beta(eps) = (eps, 1/2, 0, 0): at the vertex a_0 the forms y_1, y_2, y_3 take
    1/2, 0, 0, so it is never resonant; the edge {a_0, a_1}, the plane y_2 = y_3 = 0, contains
    beta; at the vertex a_1 the forms y_0 - y_1, y_2, y_3 take eps - 1/2, 0, 0, so it is resonant
    on 1/2 + Z and admissible at 1/2."""
    lattice = decorate_configuration(
        _matrix(SIMPLEX), [F(0), F(1, 2), F(0), F(0)], [F(1), F(0), F(0), F(0)]
    )
    assert lattice.face((0,)).resonant.kind == "never"
    edge = lattice.face((0, 1))
    assert edge.resonant.kind == "all" and edge.admissible.kind == "all"
    vertex = lattice.face((1,))
    assert vertex.resonant == EpsilonSet("progression", F(1, 2), F(1))
    assert vertex.admissible == EpsilonSet("point", F(1, 2))


def test_an_offset_without_an_admissible_point_is_reduced() -> None:
    """With beta(eps) = (2 eps + 5/2, eps, 1, 0), at the vertex a_1 the forms y_0 - y_1, y_2, y_3
    take eps + 5/2, 1, 0: resonant on 1/2 + Z, the offset reduced into [0, 1), and never
    admissible."""
    lattice = decorate_configuration(
        _matrix(SIMPLEX), [F(5, 2), F(0), F(1), F(0)], [F(2), F(1), F(0), F(0)]
    )
    vertex = lattice.face((1,))
    assert vertex.resonant == EpsilonSet("progression", F(1, 2), F(1))
    assert vertex.admissible.kind == "never"


# --- inputs ---------------------------------------------------------------------------


def test_a_must_be_homogeneous() -> None:
    with pytest.raises(ValidationError, match="homogeneous"):
        decorate_configuration(_matrix([(1, 0), (0, 1), (1, 1)]), [F(0), F(0)])


def test_columns_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="repeated"):
        decorate_configuration(_matrix([(1, 0), (1, 0), (1, 1)]), [F(0), F(0)])


def test_beta_needs_one_entry_per_row() -> None:
    with pytest.raises(ValidationError, match="per row"):
        decorate_configuration(_matrix(QUADRIC_CONE), [F(0)])
    with pytest.raises(ValidationError, match="per row"):
        decorate_configuration(_matrix(QUADRIC_CONE), [F(0), F(0)], [F(0)])


def test_beta_and_eps_must_be_exact() -> None:
    with pytest.raises(ValidationError):
        decorate_configuration(_matrix(QUADRIC_CONE), [0.5, F(1)])  # type: ignore[list-item]
    lattice = decorate_configuration(_matrix(QUADRIC_CONE), [F(1, 2), F(1)])
    with pytest.raises(ValidationError):
        lattice.reducible(0.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        lattice.centres("special")  # type: ignore[arg-type]


def test_a_face_is_looked_up_by_its_points() -> None:
    lattice = decorate_configuration(_matrix(QUADRIC_CONE), [F(1, 2), F(1)])
    assert lattice.face([2]).point_indices == (2,)
    assert lattice.face(()).dimension == -1
    assert lattice.resonance((0,)).kind == "all"
    assert lattice.admissible((0,)).kind == "never"
    assert lattice.defect(()) == 2
    with pytest.raises(ValidationError, match="not a face"):
        lattice.face((0, 1))


# --- the face lattice of a Feynman integral ------------------------------------------------


def _box(zero: tuple[int, ...] = (), *, on_shell: bool = False) -> FeynmanIntegral:
    """The box with massless propagators, with p_i^2 = 0 for the legs i in zero."""
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    if on_shell:
        return fi.with_kinematics("massless_on_shell")
    symbols = {s.name: s for v in fi.momentum_products.values() for s in sp.sympify(v).free_symbols}
    rules = {symbols[f"p{i}^2"]: 0 for i in zero}
    products = {k: sp.expand(sp.sympify(v).subs(rules)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


BOXES = {
    "box, p_i^2 != 0": lambda: _box(),
    "box, p_2^2 = 0": lambda: _box((2,)),
    "box, p_1^2 = p_4^2 = 0": lambda: _box((1, 4)),
    "box, p_1^2 = p_2^2 = 0": lambda: _box((1, 2)),
    "box, p_2^2 = p_3^2 = p_4^2 = 0": lambda: _box((2, 3, 4)),
    "box on shell": lambda: _box(on_shell=True),
}

GRAPHS = {
    "bubble": lambda: FeynmanIntegral.from_cnickel("11e|e|:nn"),
    "bubble, m_2 = 0": lambda: FeynmanIntegral.from_cnickel("11e|e|:nz"),
    "bubble, massless": lambda: FeynmanIntegral.from_cnickel("11e|e|:zz"),
    "triangle": lambda: FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"),
    "triangle, massless": lambda: FeynmanIntegral.from_cnickel("12e|2e|e|:zzz"),
    "sunrise": lambda: FeynmanIntegral.from_cnickel("111e|e|:nnn"),
    "sunrise, m_3 = 0": lambda: FeynmanIntegral.from_cnickel("111e|e|:nnz"),
    "sunrise, m_2 = m_3 = 0": lambda: FeynmanIntegral.from_cnickel("111e|e|:nzz"),
    "box, massive": lambda: FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"),
    **BOXES,
    "banana": lambda: FeynmanIntegral.from_cnickel("1111e|e|:nnnn"),
    "kite": lambda: FeynmanIntegral.from_cnickel("12e|23|3|e|:nnnnn"),
    "kite, massless": lambda: FeynmanIntegral.from_cnickel("12e|23|3|e|:zzzzz"),
    "pentagon, massless": lambda: FeynmanIntegral.from_cnickel("12e|3e|4e|4e|e|:zzzzz"),
}

DOUBLE_BOXES = {
    "double box": lambda: FeynmanIntegral.from_cnickel("15e|24|3e|4e|5|e|:zzzzzzz"),
    "double box on shell": lambda: FeynmanIntegral.from_cnickel(
        "15e|24|3e|4e|5|e|:zzzzzzz"
    ).with_kinematics("massless_on_shell"),
}

_BUILT: dict[str, tuple[FeynmanIntegral, DecoratedFaceLattice]] = {}


def _unit(fi: FeynmanIntegral) -> dict[int, int]:
    return {e.idx: 1 for e in fi.graph.get_internal_edges()}


def _lattice(name: str) -> tuple[FeynmanIntegral, DecoratedFaceLattice]:
    """The integral and its face lattice at unit powers and D = 4 - 2 eps, built once."""
    if name not in _BUILT:
        fi = {**GRAPHS, **DOUBLE_BOXES}[name]()
        _BUILT[name] = (fi, decorate_faces(fi, nu=_unit(fi)))
    return _BUILT[name]


def _edge_face(fi: FeynmanIntegral, lattice: DecoratedFaceLattice, e: int) -> DecoratedFace:
    """F_e, the face of the terms of G free of u_e (BGH26 Eq. 21, p. 11)."""
    points = fi.newton_polytope.points
    return lattice.face(j for j, p in enumerate(points) if p[e - 1] == 0)


def _samples(sets: list[EpsilonSet]) -> set[Fraction]:
    """The values of the progressions and points in [-2, 2], and two values in none of them."""
    found = {F(1, 7), F(-3, 11)}
    for s in sets:
        if s.kind != "all":
            found.update(s.window(-2, 2))
    return found


@pytest.mark.parametrize("name", GRAPHS)
def test_every_facet_is_classified_as_by_classify_facets(name: str) -> None:
    fi, lattice = _lattice(name)
    records = classify_facets(lattice.data, [1] * len(lattice.nu), 4)
    assert len(records) == len(lattice.facets)
    for record, position in zip(records, lattice.facets, strict=True):
        face = lattice.faces[position]
        assert face.point_indices == record.facet.point_indices
        assert face.resonant == record.resonant
        assert face.admissible == record.admissible
        assert face.through_origin == (record.facet.offset == 0)
    assert lattice.facet_resonance == records


@pytest.mark.parametrize("name", GRAPHS)
def test_each_face_is_resonant_where_its_facets_are(name: str) -> None:
    """With T_G trivial, G is resonant exactly where every facet containing it is, and it is
    admissible exactly where they all are (Propositions 1 and 2 of the design note). The facets'
    sets come from classify_facets."""
    fi, lattice = _lattice(name)
    records = classify_facets(lattice.data, [1] * len(lattice.nu), 4)
    for face in lattice.faces:
        assert face.lattice_defect == 1
        facets = [records[k] for k in face.facets]
        for eps in _samples([face.resonant, face.admissible, *(r.resonant for r in facets)]):
            assert (eps in face.resonant) == all(eps in r.resonant for r in facets)
            assert (eps in face.admissible) == all(eps in r.admissible for r in facets)


def test_resonance_grows_with_the_face() -> None:
    """G contained in G' makes G' resonant and admissible wherever G is."""
    for name in ("sunrise, m_3 = 0", "box, p_2^2 = 0"):
        fi, lattice = _lattice(name)
        for small in lattice.faces:
            for large in lattice.faces:
                if set(small.point_indices) < set(large.point_indices):
                    for eps in _samples([small.resonant, small.admissible]):
                        assert eps not in small.resonant or eps in large.resonant
                        assert eps not in small.admissible or eps in large.admissible


class TestBritto:
    """BGH26's statements on the faces of the bubble and the sunrise."""

    def test_the_bubble_with_m2_zero(self) -> None:
        """F_1 still exists but is no longer a facet, and is resonant only for D/2 in Z; F_2 is
        still a resonant facet (p. 29)."""
        fi, lattice = _lattice("bubble, m_2 = 0")
        first = _edge_face(fi, lattice, 1)
        assert first.codimension == 2
        assert first.resonant == EpsilonSet("progression", F(0), F(1))
        second = _edge_face(fi, lattice, 2)
        assert second.codimension == 1 and second.resonant.kind == "all"

    def test_the_edge_faces_of_the_massive_sunrise(self) -> None:
        """With integer powers every edge face is resonant (Eq. 152, p. 41)."""
        fi, lattice = _lattice("sunrise")
        for e in (1, 2, 3):
            face = _edge_face(fi, lattice, e)
            assert face.codimension == 1 and face.resonant.kind == "all"

    def test_the_sunrise_with_m3_zero(self) -> None:
        """F_1 and F_2 are no longer facets and are not resonant; F_3 is a resonant facet
        (p. 43)."""
        fi, lattice = _lattice("sunrise, m_3 = 0")
        assert len(lattice.facets) == 6
        for e in (1, 2):
            face = _edge_face(fi, lattice, e)
            assert face.codimension > 1 and face.resonant.kind != "all"
        third = _edge_face(fi, lattice, 3)
        assert third.codimension == 1 and third.resonant.kind == "all"

    def test_the_sunrise_with_m2_and_m3_zero(self) -> None:
        """No edge face is a facet, none of the five facets is resonant, and none of the edge
        faces is (p. 44): only P is resonant, a centre over which A is a pyramid."""
        fi, lattice = _lattice("sunrise, m_2 = m_3 = 0")
        assert len(lattice.facets) == 5
        assert all(lattice.faces[k].resonant.kind != "all" for k in lattice.facets)
        for e in (1, 2, 3):
            face = _edge_face(fi, lattice, e)
            assert face.codimension > 1 and face.resonant.kind != "all"
        (centre,) = lattice.centres("generic")
        assert centre.codimension == 0
        assert lattice.reducible("generic") is False


def test_the_massless_bubble_is_resonant_but_irreducible() -> None:
    """A is a pyramid over every face of the massless bubble, so whichever face is the centre,
    Thm 5.1 of SW12 applies. At eps = 0 every face is resonant and the empty face is the centre.
    The facets alone leave this open."""
    fi, lattice = _lattice("bubble, massless")
    assert all(face.pyramid for face in lattice.faces)
    assert all(record.reducible is None for record in lattice.facet_resonance)
    assert len(lattice.resonant_faces(0)) == len(lattice.faces)
    assert [face.point_indices for face in lattice.centres(0)] == [()]
    for eps in ("generic", 0, F(1, 2), F(1, 3), -1):
        assert lattice.reducible(eps) is False


@pytest.mark.parametrize(
    ("name", "facets", "centres"),
    [
        ("box, p_i^2 != 0", 10, 6),
        ("box, p_2^2 = 0", 9, 5),
        ("box, p_1^2 = p_4^2 = 0", 7, 4),
        ("box, p_1^2 = p_2^2 = 0", 9, 4),
        ("box, p_2^2 = p_3^2 = p_4^2 = 0", 8, 3),
        ("box on shell", 9, 2),
    ],
)
def test_the_boxes(name: str, facets: int, centres: int) -> None:
    """The facets of Newt(G) are one per facet of Newt(U F), the U layer, and the F layer when
    Newt(F) has dimension N - 1 = 3. At generic eps the centres are contractions Gamma/S."""
    fi, lattice = _lattice(name)
    s = fi.symanzik
    product = extract_monomial_support(sp.expand(s.u * s.f), s.schwinger_parameters)
    f_support = extract_monomial_support(sp.expand(s.f), s.schwinger_parameters)
    layers = 1 + (polytope_data([a for a, _ in f_support]).dimension == 3)
    assert len(lattice.facets) == facets
    assert len(polytope_data([a for a, _ in product]).relative_facets) + layers == facets
    found = lattice.centres("generic")
    assert len(found) == centres
    assert all(face.identification is not None for face in found)
    assert all(face.identification.kind == "contraction" for face in found)  # type: ignore[union-attr]
    assert lattice.reducible("generic") is True


@pytest.mark.parametrize("name", ["triangle", "sunrise", "box, p_2^2 = 0", "kite"])
def test_at_d_equal_four_the_empty_face_is_the_centre(name: str) -> None:
    """At eps = 0, beta = (-2, -1, ..., -1) lies in ZA, so every face is resonant."""
    fi, lattice = _lattice(name)
    assert len(lattice.resonant_faces(0)) == len(lattice.faces)
    assert [face.point_indices for face in lattice.centres(0)] == [()]
    assert lattice.reducible(0) is True


@pytest.mark.parametrize("name", GRAPHS)
def test_the_schwinger_side_agrees(name: str) -> None:
    fi, lattice = _lattice(name)
    check = lattice.check_schwinger()
    assert check.agrees
    assert check.bijective and check.same_faces and check.mismatches == ()
    assert check.faces == len(lattice.faces)


@pytest.mark.slow
@pytest.mark.parametrize("name", DOUBLE_BOXES)
def test_the_schwinger_side_agrees_on_the_double_boxes(name: str) -> None:
    fi, lattice = _lattice(name)
    assert lattice.check_schwinger().agrees
    records = classify_facets(lattice.data, [1] * 7, 4)
    assert [lattice.faces[k].resonant for k in lattice.facets] == [r.resonant for r in records]
    assert all(face.lattice_defect == 1 for face in lattice.faces)


def test_a_mismatch_is_reported() -> None:
    """A lattice whose Cayley matrix has been tampered with fails the check."""
    fi, lattice = _lattice("triangle")
    columns = [list(row) for row in lattice.cayley_matrix]
    columns[-1][0] += 1
    broken = dataclasses.replace(lattice, cayley_matrix=tuple(map(tuple, columns)))
    check = broken.check_schwinger()
    assert not check.agrees and not check.bijective


@pytest.mark.parametrize("name", ["triangle", "sunrise, m_3 = 0", "kite"])
def test_the_column_maps(name: str) -> None:
    fi, lattice = _lattice(name)
    points = fi.newton_polytope.points
    a = fi.gkz.a_matrix
    for j, k in enumerate(lattice.gkz_columns):
        assert tuple(a.col(k)) == (1, *points[j])
    assert sorted(lattice.gkz_columns) == list(range(len(points)))
    t = lp_to_cayley(len(lattice.nu), fi.loop_count)
    cayley = fi.schwinger_gkz.a_matrix
    for j, k in enumerate(lattice.cayley_columns):
        assert t * sp.Matrix([1, *points[j]]) == cayley.col(k)
    assert sorted(lattice.cayley_columns) == list(range(len(points)))


def test_the_faces_are_identified_up_to_the_codimension_asked_for() -> None:
    fi, lattice = _lattice("triangle")
    assert lattice.identify_codimension == 2
    known = {face.point_indices: face for face in fi.face_identification(2)}
    for face in lattice.faces:
        if face.point_indices and face.codimension <= 2:
            assert face.identification == known[face.point_indices]
        else:
            assert face.identification is None
    every = decorate_faces(fi, nu=_unit(fi), identify_codimension=None)
    assert all(face.identification is not None for face in every.faces if face.point_indices)
    with pytest.raises(ValidationError):
        decorate_faces(fi, nu=_unit(fi), identify_codimension=-1)


def test_through_the_origin() -> None:
    """The empty face does not contain the origin; P of the bubble does, since u_1 + u_2 has
    degree one."""
    fi, lattice = _lattice("bubble")
    assert lattice.faces[0].through_origin is False
    assert lattice.faces[-1].through_origin is True


class TestBelowFullDimension:
    def test_no_resonance_off_the_span(self) -> None:
        """1ee|1|:zn: every equation of the affine hull of P has h_0 = 0, and beta lies in the
        span of A for no eps at unit powers."""
        fi = FeynmanIntegral.from_cnickel("1ee|1|:zn")
        lattice = decorate_faces(fi, nu=_unit(fi))
        assert lattice.span.kind == "never"
        assert lattice.resonant_faces("generic") == ()
        assert lattice.reducible("generic") is None
        assert lattice.check_schwinger().agrees

    def test_the_massless_bubble_on_shell(self) -> None:
        """The massless on-shell bubble is scaleless: beta lies in the span of A only at
        D = 2(nu_1 + nu_2), eps = 0 here, and A does not have full rank."""
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz").with_kinematics("massless_on_shell")
        lattice = decorate_faces(fi, nu=_unit(fi))
        assert lattice.span == EpsilonSet("point", F(0))
        assert lattice.faces[-1].resonant == EpsilonSet("point", F(0))
        assert not lattice.full_rank
        assert lattice.reducible(0) is None
        assert all(
            face.identification is None or not face.identification.verified
            for face in lattice.faces
        )
        assert lattice.check_schwinger().agrees


class TestInputs:
    def test_d0_and_powers(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        lattice = decorate_faces(fi, 3, nu={1: 2, 2: 1})
        assert lattice.d0 == 3 and lattice.d0_source == "given"
        assert lattice.nu == (2, 1)
        assert lattice.facet_resonance == classify_facets(lattice.data, [2, 1], 3)

    def test_d0_from_the_dimension(self) -> None:
        eps = sp.Symbol("epsilon")
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn").with_(
            dimension=6 - 2 * eps, propagator_exponents={1: 1, 2: 1}
        )
        lattice = decorate_faces(fi)
        assert lattice.d0 == 6 and lattice.d0_source == "dimension"
        assert lattice.nu == (1, 1)

    def test_the_powers_must_be_integers(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        with pytest.raises(ValidationError, match="integer powers"):
            decorate_faces(fi)
        with pytest.raises(ValidationError, match="one power"):
            decorate_faces(fi, nu={1: 1})
        with pytest.raises(ValidationError):
            decorate_faces(fi, 4.0, nu=_unit(fi))  # type: ignore[arg-type]


def test_exported_from_the_package() -> None:
    import feynkit

    assert feynkit.DecoratedFaceLattice is DecoratedFaceLattice
    assert feynkit.DecoratedFace is DecoratedFace
    assert feynkit.decorate_faces is decorate_faces
