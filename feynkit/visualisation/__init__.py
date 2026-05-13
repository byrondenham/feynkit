"""
Visualisation module for Feynman integrals.

Provides tools for generating TikZ code and visualising geometric structures
such as Newton polytopes.

Classes
-------
TikzDocument
    Builder class for constructing TikZ pictures.

Functions
---------
visualise_newton_polytope
    Generate TikZ code for Newton polytope visualisation.
save_polytope_tikz
    Generate and save polytope visualisation to file.
project_to_3d_pca
    Project high-dimensional points to 3D using PCA.
compute_convex_hull_3d
    Compute convex hull for 3D points.

Examples
--------
>>> from feynkit.visualisation import visualise_newton_polytope, save_polytope_tikz
>>> from feynkit.systems import extract_monomial_support
>>> import sympy as sp
>>>
>>> # Create polynomial and extract support
>>> u1, u2 = sp.symbols('u1 u2')
>>> poly = u1**2 + u1*u2 + u2**2
>>> support = extract_monomial_support(poly, [u1, u2])
>>>
>>> # Generate TikZ code
>>> tikz_code = visualise_newton_polytope(support)
>>> print(tikz_code)
>>>
>>> # Or save directly to file
>>> save_polytope_tikz(support, "polytope.tex", standalone=True)
"""

from .geometry import (
    compute_convex_hull_3d,
    determine_label_position,
    format_coordinate,
    project_to_3d_pca,
)
from .polytope import save_polytope_tikz, visualise_newton_polytope
from .tikz import TikzDocument, graph_to_tikz

__all__ = [
    # TikZ
    "TikzDocument",
    "graph_to_tikz",
    # Polytope visualisation
    "visualise_newton_polytope",
    "save_polytope_tikz",
    # Geometry utilities
    "project_to_3d_pca",
    "compute_convex_hull_3d",
    "determine_label_position",
    "format_coordinate",
]
