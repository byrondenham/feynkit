"""
Newton polytope visualisation for Feynman integrals.

Provides functions to visualise Newton polytopes (convex hulls of monomial support)
using TikZ.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.exceptions import ValidationError
from .geometry import (
    classify_edges_by_visibility,
    compute_convex_hull_3d,
    determine_label_position,
    extract_edges_from_hull,
    project_to_3d_pca,
)
from .tikz import TikzDocument


def visualise_newton_polytope(
    support: Sequence[tuple[tuple[int, ...], Any]],
    title: str = "Newton Polytope",
    scale: float = 1.6,
    show_labels: bool = True,
    show_fill: bool = False,
    camera_direction: np.ndarray | None = None,
) -> str:
    """
    Generate TikZ code for visualising the Newton polytope.

    Creates TikZ picture code for visualising the Newton polytope (convex hull
    of the support points) in 3D. For 2D support, embeds in 3D with z=0.
    For higher dimensions, uses PCA projection to 3D.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], any]]
        Monomial support from extract_monomial_support().
    title : str, default "Newton Polytope"
        Title for the visualisation (as comment).
    scale : float, default 1.6
        Scale factor for the TikZ picture.
    show_labels : bool, default True
        Whether to show coordinate labels at vertices.
    show_fill : bool, default False
        Whether to fill one face with color (set to False to see through).
    camera_direction : Optional[np.ndarray], default None
        Camera viewing direction for determining hidden edges.
        If None, uses [0.5, 0.5, 1] for a nice 3D perspective.

    Returns
    -------
    str
        TikZ code for the Newton polytope.

    Raises
    ------
    ValidationError
        If support is empty or invalid.

    Notes
    -----
    **Newton Polytope:**

    The Newton polytope Newt(G) is the convex hull of the monomial support:
        Newt(G) = conv{alpha : z^alpha appears in G with nonzero coefficient}

    Properties:
    - Vertices correspond to "corner" monomials
    - Dimension equals the rank of the A-matrix
    - Volume relates to the number of solutions (BKK theorem)
    - Facets correspond to principal A-determinants

    **TikZ Output:**

    The function generates 3D TikZ code with:
    - Dashed gray lines for hidden edges (behind faces)
    - Solid black lines for visible edges (in front)
    - Vertex labels with coordinates
    - Optional filled faces with transparency

    **Dimension Handling:**

    - 2D support: Embedded in 3D with z=0
    - 3D support: Used directly
    - Higher dimensions: PCA projection to 3D

    Examples
    --------
    >>> import sympy as sp
    >>> from feynkit.systems import extract_monomial_support
    >>> from feynkit.visualisation import visualise_newton_polytope
    >>>
    >>> u1, u2 = sp.symbols('u1 u2')
    >>> poly = u1**2 + u1*u2 + u2**2
    >>> support = extract_monomial_support(poly, [u1, u2])
    >>> tikz_code = visualise_newton_polytope(support)
    >>> print(tikz_code)

    References
    ----------
    .. [1] Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1994).
            "Discriminants, Resultants and Multidimensional Determinants."
            Birkhäuser.
    """
    if not support:
        raise ValidationError("Support is empty")

    # Set default camera direction for nice 3D perspective
    if camera_direction is None:
        camera_direction = np.array([0.5, 0.5, 1.0])
        camera_direction = camera_direction / np.linalg.norm(camera_direction)

    # Extract exponent vectors as numpy array
    exponent_vectors = np.array([list(alpha) for alpha, _ in support], dtype=float)
    num_dimensions = exponent_vectors.shape[1]

    # === Dimension Handling ===
    if num_dimensions == 1:
        # 1D: Embed in 3D along x-axis
        points_3d = np.column_stack([exponent_vectors, np.zeros((len(exponent_vectors), 2))])
    elif num_dimensions == 2:
        # 2D: Embed in 3D with z=0
        points_3d = np.column_stack([exponent_vectors, np.zeros(len(exponent_vectors))])
    elif num_dimensions == 3:
        # 3D: Use directly
        points_3d = exponent_vectors
    else:
        # Higher dimensions: Project to 3D using PCA
        points_3d = project_to_3d_pca(exponent_vectors)

    # === Compute Convex Hull ===
    try:
        hull = compute_convex_hull_3d(points_3d)
    except Exception:
        # Degenerate case: just show points
        return _generate_degenerate_tikz(points_3d, title, scale, show_labels)

    # === Extract edges (PASS points_3d here!) ===
    edges = extract_edges_from_hull(hull, points_3d)

    # === Classify edges by visibility ===
    visible_edges, hidden_edges = classify_edges_by_visibility(
        points_3d, hull, edges, camera_direction
    )

    # === Generate TikZ code ===
    doc = TikzDocument()
    doc.begin_picture(scale=scale)
    doc.add_comment(title)
    doc.add_line("")

    # Draw filled face first if requested (so edges are on top)
    if show_fill and len(hull.simplices) > 0:
        doc.add_comment("Filled face (for reference)")
        # Find a front-facing facet
        for i, equation in enumerate(hull.equations):
            normal = equation[:3]
            if np.dot(normal, camera_direction) > 0:
                face = hull.simplices[i]
                face_points = [(points_3d[v][0], points_3d[v][1], points_3d[v][2]) for v in face]
                doc.draw_filled_polygon(face_points, fill_color="blue!10", edge_color="none")
                break
        doc.add_line("")

    # Draw HIDDEN edges FIRST (dashed, gray) - these go behind
    if hidden_edges:
        doc.add_comment("Hidden edges (behind the polytope)")
        for v1, v2 in hidden_edges:
            p1, p2 = points_3d[v1], points_3d[v2]
            doc.draw_dashed_line(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], color="gray")
        doc.add_line("")

    # Draw VISIBLE edges LAST (solid, black) - these go in front
    if visible_edges:
        doc.add_comment("Visible edges (in front)")
        for v1, v2 in visible_edges:
            p1, p2 = points_3d[v1], points_3d[v2]
            doc.draw_line(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], style="thick", color="black")
        doc.add_line("")

    # Add vertex labels on top
    if show_labels:
        doc.add_comment("Vertex labels")
        for v_idx in hull.vertices:
            p = points_3d[v_idx]
            # Get original exponent vector for label
            orig_exp = tuple(int(e) for e in exponent_vectors[v_idx])
            label_text = f"$({','.join(str(e) for e in orig_exp)})$"
            label_pos = determine_label_position(p)

            doc.draw_point(p[0], p[1], p[2], label=label_text, label_position=label_pos)

    doc.end_picture()

    return doc.get_code()


def _generate_degenerate_tikz(
    points: np.ndarray,
    title: str,
    scale: float,
    show_labels: bool,
) -> str:
    """
    Generate TikZ code for degenerate cases (points that don't form a polytope).

    Parameters
    ----------
    points : np.ndarray
        3D coordinates of points.
    title : str
        Title for the visualisation.
    scale : float
        Scale factor.
    show_labels : bool
        Whether to show labels.

    Returns
    -------
    str
        TikZ code showing just the points.
    """
    doc = TikzDocument()
    doc.begin_picture(scale=scale)
    doc.add_comment(f"{title} (degenerate - showing points only)")
    doc.add_line("")

    for i, p in enumerate(points):
        label = f"$p_{i}$" if show_labels else None
        doc.draw_point(p[0], p[1], p[2], label=label, label_position="right")

    doc.end_picture()
    return doc.get_code()


def save_polytope_tikz(
    support: Sequence[tuple[tuple[int, ...], Any]],
    filename: str,
    title: str = "Newton Polytope",
    standalone: bool = True,
    show_labels: bool = True,
    show_fill: bool = False,
) -> str:
    """
    Generate and save Newton polytope visualisation to a file.

    Parameters
    ----------
    support : List[Tuple[Tuple[int, ...], any]]
        Monomial support.
    filename : str
        Output filename (e.g., "polytope.tex").
    title : str, default "Newton Polytope"
        Title for the visualisation.
    standalone : bool, default True
        If True, wrap in standalone document for compilation.
    show_labels : bool, default True
        Whether to show vertex labels.
    show_fill : bool, default False
        Whether to fill a face with color.

    Returns
    -------
    str
        The filename where the code was saved.

    Examples
    --------
    >>> support = extract_monomial_support(poly, variables)
    >>> save_polytope_tikz(support, "newton_polytope.tex")
    'newton_polytope.tex'
    """
    tikz_code = visualise_newton_polytope(
        support, title=title, show_labels=show_labels, show_fill=show_fill
    )

    if standalone:
        # Wrap in standalone document
        document = [
            "\\documentclass[tikz,border=5pt]{standalone}",
            "\\usepackage{tikz}",
            "\\usepackage{tikz-3dplot}",
            "",
            "\\begin{document}",
            tikz_code,
            "\\end{document}",
        ]
        content = "\n".join(document)
    else:
        content = tikz_code

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename
