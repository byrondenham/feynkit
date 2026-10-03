"""Tests for feynkit.visualisation.hasse, the Hasse diagram of the face lattice in TikZ.

Rows are dimensions, faces in a row follow the order of the lattice, and each node carries a
class for its resonance (``fkall``, ``fkprog``, ``fknever``), ``fkring`` when degenerate,
``fkdash`` when undecided and ``fkhl`` when highlighted.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.face_lattice import DecoratedFaceLattice, decorate_faces
from feynkit.landau import _singular_binary
from feynkit.visualisation.hasse import (
    hasse_document,
    hasse_faces,
    hasse_tikz,
    infrared_facets,
    save_hasse_tikz,
    soft_collinear_cones,
)


def _unit(fi: FeynmanIntegral) -> dict[int, int]:
    return {e.idx: 1 for e in fi.graph.get_internal_edges()}


def _lattice(fi: FeynmanIntegral, **kwargs: object) -> DecoratedFaceLattice:
    return decorate_faces(fi, nu=_unit(fi), **kwargs)  # type: ignore[arg-type]


def _on_shell(fi: FeynmanIntegral, names: tuple[str, ...]) -> FeynmanIntegral:
    symbols = {s.name: s for v in fi.momentum_products.values() for s in sp.sympify(v).free_symbols}
    zeros = {symbols[name]: 0 for name in names}
    products = {k: sp.expand(sp.sympify(v).subs(zeros)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


def _box(legs: tuple[int, ...]) -> FeynmanIntegral:
    """The box with massless propagators and p_j^2 = 0 for each leg j in legs; leg j is at vertex
    j, so that legs 1 and 4, and 2 and 3, are opposite."""
    return _on_shell(
        FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz"), tuple(f"p{j}^2" for j in legs)
    )


def _nodes(code: str) -> dict[int, str]:
    """The options of each node, by the position of its face in the lattice."""
    return {
        int(m.group(2)): m.group(1) for m in re.finditer(r"\\node\[([^\]]*)\] \(f(\d+)\)", code)
    }


def _marks(code: str) -> list[int]:
    """The faces drawn with a ring or a dashed outline."""
    return [k for k, o in _nodes(code).items() if "fkring" in o or "fkdash" in o]


def _edges(code: str) -> set[tuple[int, int]]:
    return {
        (int(a), int(b)) for a, b in re.findall(r"\\draw\[fkedge\] \(f(\d+)\) -- \(f(\d+)\)", code)
    }


# --- the selection of faces ------------------------------------------------------------------


def test_the_default_view_is_codimension_two() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"))
    found = hasse_faces(lattice)
    assert {f.codimension for f in found} == {0, 1, 2}
    assert list(found) == [f for f in lattice.faces if f.codimension <= 2]


def test_up_and_down_sets() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("111e|e|:nnn"))
    face = lattice.faces[lattice.facets[0]]
    down = {f.point_indices for f in hasse_faces(lattice, "downset", face=face)}
    assert down == {
        f.point_indices for f in lattice.faces if set(f.point_indices) <= set(face.point_indices)
    }
    up = {f.point_indices for f in hasse_faces(lattice, "upset", face=face.point_indices)}
    assert up == {
        f.point_indices for f in lattice.faces if set(f.point_indices) >= set(face.point_indices)
    }
    assert face.point_indices in up and face.point_indices in down
    assert hasse_faces(lattice, "upset", face=face)[-1].codimension == 0


def test_filters_select_faces() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"))
    resonant = hasse_faces(lattice, "filter", filter="resonant")
    assert resonant and all(f.resonant.kind == "all" for f in resonant)
    contraction = hasse_faces(lattice, "filter", filter="contraction")
    assert {f.identification.kind for f in contraction} == {"contraction"}
    assert len([f for f in contraction if f.codimension == 1]) == 4


def test_bad_views_are_rejected() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("11e|e|:nn"))
    for kwargs in (
        {"view": "tree"},
        {"view": "upset"},
        {"view": "filter"},
        {"view": "filter", "filter": "red"},
        {"view": "codim", "k": -1},
        {"view": "codim", "k": True},
        {"view": "all", "max_faces": 0},
    ):
        with pytest.raises(ValidationError):
            hasse_faces(lattice, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="not a face"):
        hasse_faces(lattice, "upset", face=(0, 99))


def test_the_size_guard_names_the_count_and_the_views_that_fit() -> None:
    fi = FeynmanIntegral.from_cnickel("15e|24|3e|4e|5|e|:zzzzzzz")
    lattice = _lattice(fi)
    assert len(lattice.faces) > 2000
    with pytest.raises(ValidationError) as caught:
        hasse_tikz(lattice, "all")
    message = str(caught.value)
    assert str(len(lattice.faces)) in message and "max_faces" in message
    assert "codim <= 2" in message
    assert len(hasse_faces(lattice)) <= 200
    assert len(hasse_faces(lattice, "all", max_faces=len(lattice.faces))) == len(lattice.faces)


# --- the picture -----------------------------------------------------------------------------


def test_one_node_per_face_and_one_edge_per_cover() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("111e|e|:nnn"))
    code = hasse_tikz(lattice, "all")
    assert set(_nodes(code)) == set(range(len(lattice.faces)))
    index = {f.point_indices: k for k, f in enumerate(lattice.faces)}
    assert _edges(code) == {
        (index[f.point_indices], index[c.point_indices])
        for f in lattice.faces
        for c in lattice.covers(f)
    }


def test_a_subset_keeps_the_covers_among_its_faces() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"))
    code = hasse_tikz(lattice, "codim", 2)
    chosen = {lattice.faces.index(f) for f in hasse_faces(lattice)}
    assert set(_nodes(code)) == chosen
    assert all(a in chosen and b in chosen for a, b in _edges(code))


def test_a_filtered_view_joins_faces_by_the_induced_order() -> None:
    """The faces dropped from a filtered view leave their covers joined through them."""
    lattice = _lattice(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"))
    code = hasse_tikz(lattice, "filter", filter="resonant")
    chosen = hasse_faces(lattice, "filter", filter="resonant")
    sets = {lattice.faces.index(f): set(f.point_indices) for f in chosen}
    expected = {
        (a, b)
        for a, sa in sets.items()
        for b, sb in sets.items()
        if sa < sb and not any(sa < sc < sb for sc in sets.values())
    }
    assert _edges(code) == expected


def test_rows_are_dimensions_and_the_order_is_the_lattice_order() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("11e|e|:nn"))
    code = hasse_tikz(lattice, "all")
    at = {
        int(m.group(1)): (float(m.group(2)), float(m.group(3)))
        for m in re.finditer(r"\(f(\d+)\) at \(([-\d.]+),([-\d.]+)\)", code)
    }
    for dimension in {f.dimension for f in lattice.faces}:
        row = [k for k, f in enumerate(lattice.faces) if f.dimension == dimension]
        assert len({at[k][1] for k in row}) == 1
        assert [at[k][0] for k in row] == sorted(at[k][0] for k in row)
        assert len({at[k][0] for k in row}) == len(row)
    heights = [at[k][1] for k in range(len(lattice.faces))]
    assert heights == sorted(heights)


def test_labels_are_the_identification_names() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn"))
    code = hasse_tikz(lattice, "codim", 1)
    assert r"$G(\Gamma/\{4\})$" in code
    assert r"$U(\Gamma)$" in code
    assert r"$F(\Gamma)$" in code
    assert "Gamma/{" not in code


def test_unnamed_faces_are_labelled_by_their_points() -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("11e|e|:nn"), identify_codimension=0)
    code = hasse_tikz(lattice, "all")
    assert r"$\emptyset$" in code
    assert r"$p_{0}$" in code


def test_two_runs_are_byte_identical() -> None:
    kwargs = {"highlight": "ir", "eps": 0}
    first = hasse_tikz(_lattice(_box((1, 2, 3))), "codim", 2, **kwargs)  # type: ignore[arg-type]
    second = hasse_tikz(_lattice(_box((1, 2, 3))), "codim", 2, **kwargs)  # type: ignore[arg-type]
    assert first == second
    assert first.endswith("\\end{tikzpicture}")
    assert "\t" not in first and not re.search(r" $", first, re.M)
    assert first.isascii()


# --- colours ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cnickel", ["11e|e|:nn", "12e|2e|e|:nnn", "111e|e|:nnn"])
def test_the_classes_match_facet_resonance(cnickel: str) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    lattice = fi.face_lattice(d0=4, nu=_unit(fi))
    nodes = _nodes(hasse_tikz(lattice, "codim", 1))
    names = {"all": "fkall", "progression": "fkprog", "never": "fknever", "point": "fkprog"}
    for record, position in zip(lattice.facet_resonance, lattice.facets, strict=True):
        assert names[record.kind] in nodes[position].split(", ")
    for position in set(nodes) - set(lattice.facets):
        assert nodes[position].split(", ")[0] in {"fkall", "fkprog", "fknever"}


def test_the_bubble_has_both_kinds_of_facet() -> None:
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    lattice = fi.face_lattice(d0=4, nu={1: 1, 2: 1})
    nodes = _nodes(hasse_tikz(lattice, "codim", 1))
    kinds = {record.kind for record in lattice.facet_resonance}
    assert kinds == {"all", "progression"}
    assert any("fkall" in o for o in nodes.values()) and any("fkprog" in o for o in nodes.values())


def test_eps_makes_the_resonant_faces_bold() -> None:
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    lattice = fi.face_lattice(d0=4, nu={1: 1, 2: 1})
    for eps in (0, 1):
        nodes = _nodes(hasse_tikz(lattice, "all", eps=eps))
        for k, face in enumerate(lattice.faces):
            assert ("fkat" in nodes[k].split(", ")) == (eps in face.resonant)
    assert "fkat" not in "".join(_nodes(hasse_tikz(lattice, "all")).values())
    with pytest.raises(ValidationError):
        hasse_tikz(lattice, eps="x")  # type: ignore[arg-type]


def test_d0_and_nu_decorate_an_integral_and_clash_with_a_lattice() -> None:
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    from_integral = hasse_tikz(fi, "all", d0=4, nu={1: 1, 2: 1})
    assert from_integral == hasse_tikz(fi.face_lattice(d0=4, nu={1: 1, 2: 1}), "all")
    with pytest.raises(ValidationError, match="already decorated"):
        hasse_tikz(fi.face_lattice(nu={1: 1, 2: 1}), "all", d0=4)


# --- degeneracy ------------------------------------------------------------------------------


@pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")
class TestDegeneracy:
    def test_the_degenerate_facet_of_the_triangle_on_a_massless_leg_is_ringed(self) -> None:
        fi = _on_shell(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), ("p1^2",))
        lattice = _lattice(fi, degeneracy=True)
        nodes = _nodes(hasse_tikz(lattice, "all", degeneracy=True))
        ringed = {k for k, o in nodes.items() if "fkring" in o.split(", ")}
        assert ringed == {k for k, f in enumerate(lattice.faces) if f.degenerate}
        assert len(ringed) == 1
        (k,) = ringed
        assert lattice.faces[k].dimension == 2 and len(lattice.faces[k].point_indices) == 5
        assert lattice.faces[k].codimension == 1
        assert not any("fkdash" in o for o in nodes.values())

    def test_no_marks_unless_asked_or_computed(self) -> None:
        fi = _on_shell(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), ("p1^2",))
        plain = _lattice(fi)
        assert not _marks(hasse_tikz(plain, "all"))
        with pytest.raises(ValidationError, match="no degeneracy"):
            hasse_tikz(plain, "all", degeneracy=True)
        computed = _lattice(fi, degeneracy=True)
        assert not _marks(hasse_tikz(computed, "all", degeneracy=False))
        assert _marks(hasse_tikz(computed, "all"))

    def test_the_degenerate_filter(self) -> None:
        fi = _on_shell(FeynmanIntegral.from_cnickel("12e|2e|e|:nnn"), ("p1^2",))
        lattice = _lattice(fi, degeneracy=True)
        found = hasse_faces(lattice, "filter", filter="degenerate")
        assert found and all(f.degenerate for f in found)


def test_undecided_faces_are_dashed() -> None:
    import dataclasses

    lattice = _lattice(FeynmanIntegral.from_cnickel("11e|e|:nn"))
    faces = tuple(
        dataclasses.replace(f, degenerate=None if f.dimension == 1 else bool(f.dimension == 0))
        for f in lattice.faces
    )
    lattice = dataclasses.replace(lattice, faces=faces)
    nodes = _nodes(hasse_tikz(lattice, "all"))
    for k, f in enumerate(lattice.faces):
        options = nodes[k].split(", ")
        assert ("fkdash" in options) == (f.dimension == 1)
        assert ("fkring" in options) == (f.dimension == 0)


# --- the soft-collinear cones of the massless box --------------------------------------------


@pytest.mark.parametrize(
    ("legs", "facets", "cones"),
    [
        ((1,), 1, 0),
        ((1, 2), 2, 1),
        ((1, 4), 2, 0),
        ((2, 3), 2, 0),
        ((1, 2, 3), 3, 2),
        ((1, 2, 3, 4), 4, 4),
    ],
)
def test_soft_collinear_cones_of_the_massless_box(
    legs: tuple[int, ...], facets: int, cones: int
) -> None:
    """Two, three and four massless corners have 1, 2 and 4 intersections of two IR facets;
    opposite corners have none (AHM22, Sec. VIII, p. 6)."""
    lattice = _lattice(_box(legs))
    assert len(infrared_facets(lattice)) == facets
    found = soft_collinear_cones(lattice)
    assert len(found) == cones
    assert all(f.codimension == 2 for f in found)
    code = hasse_tikz(lattice, "codim", 2, highlight="ir")
    nodes = _nodes(code)
    marked = {k for k, o in nodes.items() if "fkhl" in o.split(", ")}
    expected = {lattice.faces.index(f) for f in (*infrared_facets(lattice), *found)}
    assert marked == expected
    assert sum(1 for k in marked if lattice.faces[k].codimension == 2) == cones


def test_the_ir_filter_shows_only_ir_facets_and_cones() -> None:
    lattice = _lattice(_box((1, 2, 3, 4)))
    found = hasse_faces(lattice, "filter", filter="ir")
    assert len(found) == 8
    assert not infrared_facets(_lattice(FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")))


# --- documents -------------------------------------------------------------------------------


def test_the_document_wraps_the_picture(tmp_path: Path) -> None:
    lattice = _lattice(FeynmanIntegral.from_cnickel("11e|e|:nn"))
    document = hasse_document(lattice, "all")
    assert document.startswith("\\documentclass")
    assert hasse_tikz(lattice, "all") in document
    target = save_hasse_tikz(lattice, str(tmp_path / "h.tex"), "all")
    assert Path(target).read_text(encoding="utf-8") == document
    bare = save_hasse_tikz(lattice, str(tmp_path / "b.tex"), "all", standalone=False)
    assert Path(bare).read_text(encoding="utf-8") == hasse_tikz(lattice, "all")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lualatex") is None, reason="lualatex not installed")
@pytest.mark.parametrize(
    "build",
    [
        lambda: _lattice(FeynmanIntegral.from_cnickel("11e|e|:nn")),
        lambda: _lattice(_box((1, 2, 3))),
        lambda: _lattice(FeynmanIntegral.from_cnickel("12e|3e|4e|4e|e|:zzzzz")),
    ],
    ids=["bubble", "box with three massless corners", "pentagon"],
)
def test_every_view_compiles_with_lualatex(build, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    lattice = build()
    views = [
        {"view": "all", "max_faces": 500},
        {"view": "codim", "k": 2, "highlight": "ir", "eps": 0},
    ]
    views.append({"view": "filter", "filter": "contraction"})
    for n, kwargs in enumerate(views):
        path = tmp_path / f"h{n}.tex"
        save_hasse_tikz(lattice, str(path), **kwargs)  # type: ignore[arg-type]
        done = subprocess.run(
            ["lualatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout[-2000:]
        assert (tmp_path / f"h{n}.pdf").exists()
