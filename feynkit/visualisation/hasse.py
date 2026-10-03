"""
Hasse diagrams of the face lattice of a Newton polytope, in TikZ.

The nodes are faces of the polytope of a Feynman integral, drawn one row to a
dimension with the empty face at the bottom and the polytope at the top, and
the edges are the cover relation of
:meth:`~feynkit.face_lattice.DecoratedConfiguration.covers`. A face is
coloured by its resonance at the lattice's D_0 and powers: resonant for every
eps, resonant at a progression of eps, or never. A ring marks a face that is
degenerate, and a dashed outline one that the analysis left undecided. The
label of a face is the name its identification gives, such as G(Gamma/{4}),
and otherwise its points.

Within a row the faces follow the order of the lattice, by point indices, so
that the output is the same on every run and diffs of it are stable.

A full lattice is large, thousands of faces for the double box, so a diagram
shows a view of it: every face, the faces up to a codimension, the faces above
or below one face, or the faces a filter selects. A view of more than
``max_faces`` faces raises instead of drawing.

An infrared facet, in feynkit's term, is a facet whose face is a product of
polynomials of minors with a scaleless factor of at least two edges, such as
U(Gamma/{1,3}) at a massless corner of the box; a purely soft facet, whose
scaleless factor is a single line, is not one. A soft-collinear cone is a face
of codimension 2 on two infrared facets, and on neither the U layer nor the F
layer of the polytope, which two opposite infrared facets of the box share.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

from .._exact import _as_int
from ..core.exceptions import ValidationError
from ..face_lattice import DecoratedFace, DecoratedFaceLattice, Epsilon, _epsilon
from .tikz import TikzDocument

if TYPE_CHECKING:
    from ..integral import FeynmanIntegral

__all__ = [
    "HASSE_FILTERS",
    "HASSE_VIEWS",
    "hasse_document",
    "hasse_faces",
    "hasse_tikz",
    "infrared_facets",
    "save_hasse_tikz",
    "soft_collinear_cones",
]

HasseView = Literal["all", "codim", "upset", "downset", "filter"]
HasseFilter = Literal["resonant", "degenerate", "contraction", "ir"]

HASSE_VIEWS: tuple[HasseView, ...] = ("all", "codim", "upset", "downset", "filter")
HASSE_FILTERS: tuple[HasseFilter, ...] = ("resonant", "degenerate", "contraction", "ir")

# The default limit on the number of faces of a diagram.
MAX_FACES = 200
# Height of a row, in centimetres.
ROW_HEIGHT = 2.0
# Width of a node's column per character of its label, and the least, in tenths of a centimetre.
CHARACTER_WIDTH = 1.3
LEAST_WIDTH = 16

_CLASSES = {
    "all": ("fkall", "resonant for every $\\varepsilon$"),
    "progression": ("fkprog", "resonant at a progression of $\\varepsilon$"),
    "never": ("fknever", "never resonant"),
}

_STYLES = (
    "fk/.style={draw, rounded corners=2pt, inner sep=2pt, font=\\scriptsize, align=center}",
    "fkall/.style={fk, fill=red!35}",
    "fkprog/.style={fk, fill=yellow!45}",
    "fknever/.style={fk, fill=green!25}",
    "fkring/.style={double, double distance=1.2pt}",
    "fkdash/.style={dashed}",
    "fkat/.style={font=\\bfseries\\scriptsize}",
    "fkhl/.style={line width=1.6pt}",
    "fkedge/.style={gray}",
)


# --- infrared facets and cones ---------------------------------------------------------------


def _is_infrared(face: DecoratedFace) -> bool:
    found = face.identification
    if face.codimension != 1 or found is None or not found.levels:
        return False
    if found.kind in ("contraction", "u_layer", "f_layer", "whole"):
        return False
    if not found.levels[0].deleted:
        return False
    return any(level.f_vanishes and len(level.edges) >= 2 for level in found.levels[1:])


def infrared_facets(lattice: DecoratedFaceLattice) -> tuple[DecoratedFace, ...]:
    """The infrared facets of the polytope, in the order of the lattice.

    A facet F is infrared when its flag deletes edges and a later level, the
    quotient of the product G(gamma) U(Gamma/gamma) it is, has F-polynomial
    zero and at least two edges. This is the collinear region at a massless
    corner of a box; a facet whose scaleless level is a single line is a soft
    region and is left out. Facets that are not identified are not infrared.
    """
    return tuple(lattice.faces[k] for k in lattice.facets if _is_infrared(lattice.faces[k]))


def soft_collinear_cones(lattice: DecoratedFaceLattice) -> tuple[DecoratedFace, ...]:
    """The faces of codimension 2 on two infrared facets, in the order of the lattice.

    A face on the U layer or the F layer of the polytope is left out: two
    opposite infrared facets of the massless box meet in the F layer, and
    their intersection is not a cone of the Feynman polytope.
    """
    infrared = {lattice.faces.index(f) for f in infrared_facets(lattice)}
    layered = ("u_layer", "f_layer")
    found = []
    for face in lattice.faces:
        if face.codimension != 2:
            continue
        if face.identification is not None and face.identification.kind in layered:
            continue
        on = [lattice.facets[k] for k in face.facets]
        if any(
            lattice.faces[k].identification is not None
            and lattice.faces[k].identification.kind in layered  # type: ignore[union-attr]
            for k in on
        ):
            continue
        if sum(1 for k in on if k in infrared) >= 2:
            found.append(face)
    return tuple(found)


def _highlighted(lattice: DecoratedFaceLattice, name: str | None) -> set[tuple[int, ...]]:
    if name is None:
        return set()
    return {f.point_indices for f in _filtered(lattice, name)}


def _filtered(lattice: DecoratedFaceLattice, name: str) -> list[DecoratedFace]:
    if name == "resonant":
        found = [f for f in lattice.faces if f.resonant.kind == "all"]
    elif name == "degenerate":
        found = [f for f in lattice.faces if f.degenerate is True]
    elif name == "contraction":
        found = [
            f
            for f in lattice.faces
            if f.identification is not None and f.identification.kind == "contraction"
        ]
    elif name == "ir":
        keep = {
            f.point_indices for f in (*infrared_facets(lattice), *soft_collinear_cones(lattice))
        }
        found = [f for f in lattice.faces if f.point_indices in keep]
    else:
        raise ValidationError(f"filter must be one of {', '.join(HASSE_FILTERS)}, not {name!r}")
    return found


# --- the selection of faces ------------------------------------------------------------------


def _resolve(
    lattice: DecoratedFaceLattice | FeynmanIntegral,
    d0: int | Fraction | None,
    nu: Mapping[int, int] | None,
    degeneracy: bool | None,
) -> DecoratedFaceLattice:
    if isinstance(lattice, DecoratedFaceLattice):
        if d0 is not None or nu is not None:
            raise ValidationError(
                "the lattice is already decorated, so d0 and nu cannot be given; "
                "pass the integral to decorate it at them"
            )
        return lattice
    return lattice.face_lattice(d0, nu=nu, degeneracy=bool(degeneracy))


def _check_limit(max_faces: object) -> int:
    limit = _as_int(max_faces, "max_faces")
    if limit < 1:
        raise ValidationError(f"max_faces must be a positive integer, not {max_faces!r}")
    return limit


def hasse_faces(
    lattice: DecoratedFaceLattice,
    view: HasseView = "codim",
    k: int = 2,
    *,
    face: DecoratedFace | Iterable[int] | None = None,
    filter: HasseFilter | None = None,  # noqa: A002
    max_faces: int = MAX_FACES,
) -> tuple[DecoratedFace, ...]:
    """The faces a view shows, in the order of the lattice.

    Parameters
    ----------
    lattice
        The decorated face lattice.
    view
        "all"; "codim", the faces of codimension at most k, the empty face
        excluded unless k reaches it; "upset" and "downset", the faces
        containing, and contained in, ``face``, itself included; or "filter",
        the faces ``filter`` selects.
    k
        The largest codimension of the "codim" view.
    face
        A face or its point indices, for "upset" and "downset".
    filter
        "resonant" (resonant for every eps), "degenerate", "contraction"
        (the faces identified as contractions) or "ir" (the infrared facets
        and soft-collinear cones), for "filter".
    max_faces
        The most faces a view may have.

    Raises
    ------
    ValidationError
        If an argument is invalid, the point indices are not a face, or the
        view has more than max_faces faces; the message gives its size and the
        codimension views that fit.
    """
    limit = _check_limit(max_faces)
    if view == "all":
        found = list(lattice.faces)
    elif view == "codim":
        if isinstance(k, bool) or _as_int(k, "k") < 0:
            raise ValidationError(f"k must be a non-negative integer, not {k!r}")
        found = [f for f in lattice.faces if f.codimension <= k]
    elif view in ("upset", "downset"):
        if face is None:
            raise ValidationError(f"the {view} view needs a face")
        key = lattice.face(face.point_indices if isinstance(face, DecoratedFace) else face)
        points = set(key.point_indices)
        if view == "upset":
            found = [f for f in lattice.faces if points <= set(f.point_indices)]
        else:
            found = [f for f in lattice.faces if set(f.point_indices) <= points]
    elif view == "filter":
        if filter is None:
            raise ValidationError(f"the filter view needs a filter: {', '.join(HASSE_FILTERS)}")
        found = _filtered(lattice, filter)
    else:
        raise ValidationError(f"view must be one of {', '.join(HASSE_VIEWS)}, not {view!r}")
    if len(found) > limit:
        raise ValidationError(_too_large(lattice, len(found), limit))
    return tuple(found)


def _too_large(lattice: DecoratedFaceLattice, count: int, limit: int) -> str:
    fits = None
    for k in range(0, lattice.faces[0].codimension + 1):
        size = sum(1 for f in lattice.faces if f.codimension <= k)
        if size > limit:
            break
        fits = (k, size)
    hint = f"codim <= {fits[0]} has {fits[1]} faces and fits, or " if fits else ""
    return (
        f"the view has {count} faces, more than max_faces = {limit}; {hint}"
        "pass a smaller view, or a larger max_faces"
    )


# --- labels and layout -----------------------------------------------------------------------


def _tex_name(name: str) -> str:
    """The name of an identification, such as G({3,4}) U(Gamma/{3,4}), as TeX maths."""
    tex = name.replace("Gamma", "\\Gamma").replace("{", "\\{").replace("}", "\\}")
    return "$" + tex.replace(") ", ")\\,") + "$"


def _label(face: DecoratedFace) -> tuple[str, int]:
    """The TeX label of a face and its width in characters."""
    found = face.identification
    if not face.point_indices:
        return "$\\emptyset$", 2
    if found is not None and found.levels:
        name = found.name()
        text = _tex_name(name)
        if not found.verified and found.kind in ("support_product", "unidentified"):
            text = text[:-1] + "^{?}$"
        return text, len(name)
    points = face.point_indices
    if len(points) == 1:
        return f"$p_{{{points[0]}}}$", 3
    shown = ",".join(str(p) for p in points[:3])
    suffix = ",\\ldots" if len(points) > 3 else ""
    return f"$\\{{{shown}{suffix}\\}}$", len(shown) + len(suffix) + 2


def _tenths(value: int) -> str:
    """A number of tenths as a decimal string."""
    sign = "-" if value < 0 else ""
    whole, part = divmod(abs(value), 10)
    return f"{sign}{whole}.{part}"


def _reduction(chosen: list[DecoratedFace]) -> list[tuple[int, int]]:
    """The covers of the order that set containment gives on the chosen faces, by position."""
    masks = [sum(1 << j for j in f.point_indices) for f in chosen]
    pairs = []
    for a, low in enumerate(masks):
        above = [b for b, big in enumerate(masks) if low != big and low & big == low]
        for b in above:
            if not any(
                masks[b] != masks[c] and masks[c] & masks[b] == masks[c] and masks[c] & low == low
                for c in above
                if c != b
            ):
                pairs.append((a, b))
    return pairs


# --- the picture -----------------------------------------------------------------------------


def hasse_tikz(
    lattice: DecoratedFaceLattice | FeynmanIntegral,
    view: HasseView = "codim",
    k: int = 2,
    d0: int | Fraction | None = None,
    nu: Mapping[int, int] | None = None,
    eps: Epsilon | None = None,
    degeneracy: bool | None = None,
    *,
    face: DecoratedFace | Iterable[int] | None = None,
    filter: HasseFilter | None = None,  # noqa: A002
    highlight: HasseFilter | None = None,
    max_faces: int = MAX_FACES,
) -> str:
    """The Hasse diagram of a view of the face lattice, as a TikZ picture.

    Parameters
    ----------
    lattice
        A decorated face lattice, or an integral, which is decorated at d0
        and nu, with degeneracy when ``degeneracy`` is true.
    view, k, face, filter, max_faces
        The view; see :func:`hasse_faces`.
    d0, nu
        D_0 and the powers, to decorate an integral; not for a lattice, which
        has its own.
    eps
        An integer, a Fraction or "generic"; the faces resonant at it are set
        in bold.
    degeneracy
        Whether to mark degenerate and undecided faces. None marks them when
        the lattice carries verdicts; True needs it to; False never marks.
    highlight
        A filter, whose faces are drawn with a heavy outline.

    Returns
    -------
    str
        A ``tikzpicture`` with a legend, which needs the TikZ package and
        compiles in a standalone document.

    Raises
    ------
    ValidationError
        As :func:`hasse_faces`; if d0 and nu are given with a lattice; if
        eps is not a rational number or "generic"; if degeneracy is true and
        the lattice has no verdicts; or if highlight is not a filter.
    """
    resolved = _resolve(lattice, d0, nu, degeneracy)
    chosen = list(hasse_faces(resolved, view, k, face=face, filter=filter, max_faces=max_faces))
    value = None if eps is None else _epsilon(eps)
    verdicts = any(f.degenerate is not None for f in resolved.faces)
    if degeneracy and not verdicts:
        raise ValidationError(
            "the lattice has no degeneracy verdicts; decorate it with degeneracy=True"
        )
    marks = verdicts if degeneracy is None else bool(degeneracy)
    marked = _highlighted(resolved, highlight)

    index = {f.point_indices: n for n, f in enumerate(resolved.faces)}
    position = {f.point_indices: n for n, f in enumerate(chosen)}
    if view == "filter":
        edges = [
            (index[chosen[a].point_indices], index[chosen[b].point_indices])
            for a, b in _reduction(chosen)
        ]
    else:
        edges = [
            (index[f.point_indices], index[c.point_indices])
            for f in chosen
            for c in resolved.covers(f)
            if c.point_indices in position
        ]

    rows: dict[int, list[DecoratedFace]] = {}
    for f in chosen:
        rows.setdefault(f.dimension, []).append(f)
    labels = {f.point_indices: _label(f) for f in chosen}
    width = max([LEAST_WIDTH, *(round(CHARACTER_WIDTH * w) + 6 for _, w in labels.values())])

    doc = TikzDocument()
    doc.begin_picture(options=", ".join(_STYLES))
    doc.add_comment(f"Hasse diagram of the face lattice, {len(chosen)} faces, rows by dimension")
    for dimension in sorted(rows):
        row = rows[dimension]
        y = (dimension + 1) * ROW_HEIGHT
        for n, f in enumerate(row):
            x = (2 * n - (len(row) - 1)) * width // 2
            options = [
                _CLASSES[f.resonant.kind if f.resonant.kind in _CLASSES else "progression"][0]
            ]
            if marks and f.point_indices:
                if f.degenerate is True:
                    options.append("fkring")
                elif f.degenerate is None:
                    options.append("fkdash")
            if value is None and eps is not None:
                if f.resonant.kind == "all":
                    options.append("fkat")
            elif value is not None and value in f.resonant:
                options.append("fkat")
            if f.point_indices in marked:
                options.append("fkhl")
            doc.add_line(
                f"\\node[{', '.join(options)}] (f{index[f.point_indices]}) "
                f"at ({_tenths(x)},{y:.1f}) {{{labels[f.point_indices][0]}}};"
            )
    for a, b in sorted(edges):
        doc.add_line(f"\\draw[fkedge] (f{a}) -- (f{b});")
    _legend(
        doc,
        rows,
        width,
        marks=marks,
        highlighted=bool(marked),
        eps=eps,
        highlight=highlight,
    )
    doc.end_picture()
    return doc.get_code()


def _legend(
    doc: TikzDocument,
    rows: dict[int, list[DecoratedFace]],
    width: int,
    *,
    marks: bool,
    highlighted: bool,
    eps: Epsilon | None,
    highlight: str | None,
) -> None:
    widest = max(len(row) for row in rows.values())
    x = (widest * width) // 2 + 30
    lines = [(f"fk, {name}", text) for name, text in _CLASSES.values()]
    if marks:
        lines.append(("fk, fkall, fkring", "degenerate"))
        lines.append(("fk, fkall, fkdash", "degeneracy undecided"))
    if eps is not None:
        lines.append(("fk, fkat, fill=none", f"resonant at $\\varepsilon = {_eps_tex(eps)}$"))
    if highlighted:
        lines.append(("fk, fkhl, fill=none", f"highlighted: {highlight}"))
    top = (max(rows) + 1) * ROW_HEIGHT
    doc.add_comment("Legend")
    for n, (style, text) in enumerate(lines):
        y = top - n * 0.8
        doc.add_line(
            f"\\node[{style}, minimum width=6mm, minimum height=3mm] (legend{n}) at ({_tenths(x)},{y:.1f}) {{}};"
        )
        doc.add_line(
            f"\\node[anchor=west, font=\\scriptsize] at ([xshift=0.3cm]legend{n}.east) {{{text}}};"
        )


def _eps_tex(eps: Epsilon) -> str:
    if isinstance(eps, str):
        return "\\text{generic}"
    value = _epsilon(eps)
    assert value is not None
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


def hasse_document(*args: object, **kwargs: object) -> str:
    """A standalone LaTeX document holding the diagram of :func:`hasse_tikz`, same arguments."""
    code = hasse_tikz(*args, **kwargs)  # type: ignore[arg-type]
    return "\n".join(
        [
            "\\documentclass[tikz,border=5pt]{standalone}",
            "\\usepackage{amssymb}",
            "",
            "\\begin{document}",
            code,
            "\\end{document}",
            "",
        ]
    )


def save_hasse_tikz(
    lattice: DecoratedFaceLattice | FeynmanIntegral,
    filename: str,
    *args: object,
    standalone: bool = True,
    **kwargs: object,
) -> str:
    """Write the diagram to a file, standalone by default; the other arguments are those of
    :func:`hasse_tikz`. Returns the filename."""
    if standalone:
        content = hasse_document(lattice, *args, **kwargs)
    else:
        content = hasse_tikz(lattice, *args, **kwargs)  # type: ignore[arg-type]
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(content)
    return filename
