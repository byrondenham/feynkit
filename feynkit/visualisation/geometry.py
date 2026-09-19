"""
3D geometry utilities for visualisation.

Provides helper functions for 3D coordinate transformations, projections,
and geometric computations.
"""

import numpy as np
from scipy.spatial import ConvexHull

from ..core.exceptions import ComputationError


def project_to_3d_pca(points: np.ndarray) -> np.ndarray:
    """
    Project high-dimensional points to 3D using PCA.

    Parameters
    ----------
    points : np.ndarray
        Array of points, shape (n, d) where d > 3.

    Returns
    -------
    np.ndarray
        Projected points, shape (n, 3).

    Examples
    --------
    >>> pts = np.random.rand(10, 5)  # 10 points in 5D
    >>> pts_3d = project_to_3d_pca(pts)
    >>> pts_3d.shape
    (10, 3)
    """
    if points.shape[1] <= 3:
        # Already 3D or lower, pad if needed
        if points.shape[1] == 3:
            return points
        elif points.shape[1] == 2:
            return np.column_stack([points, np.zeros(len(points))])
        else:
            raise ComputationError(f"Cannot handle {points.shape[1]}D points")

    # Center the data
    centred = points - points.mean(axis=0, keepdims=True)

    # Compute SVD
    try:
        U, S, Vt = np.linalg.svd(centred, full_matrices=False)
        # Project onto first 3 principal components
        projected = centred @ Vt[:3].T
        return projected
    except np.linalg.LinAlgError as e:
        raise ComputationError(f"PCA projection failed: {e}") from e


def compute_convex_hull_3d(points: np.ndarray) -> ConvexHull:
    """
    Compute convex hull for 3D points.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D points, shape (n, 3).

    Returns
    -------
    ConvexHull
        Scipy ConvexHull object.

    Raises
    ------
    ComputationError
        If convex hull computation fails.
    """
    try:
        hull = ConvexHull(points)
        return hull
    except Exception as e:
        raise ComputationError(f"Convex hull computation failed: {e}") from e


def extract_edges_from_hull(
    hull: ConvexHull, points: np.ndarray, tolerance: float = 1e-8
) -> list[tuple[int, int]]:
    """
    Extract only the 1-skeleton edges of the polytope.

    Reconstructs non-triangulated faces by grouping coplanar triangles,
    then extracts boundary edges of each face.

    Parameters
    ----------
    hull : ConvexHull
        Convex hull object.
    points : np.ndarray
        Original 3D points, shape (n, 3).
    tolerance : float, default 1e-8
        Tolerance for checking if points are coplanar.

    Returns
    -------
    List[Tuple[int, int]]
        List of unique edges as sorted (v1, v2) tuples.
    """
    # Group coplanar facets into faces
    faces = _group_coplanar_facets(hull, points, tolerance)

    # Extract boundary edges of each face
    all_edges: set[tuple[int, int]] = set()
    for face_vertices in faces:
        face_edges = _get_face_boundary_edges(points, face_vertices)
        all_edges.update(face_edges)

    return list(all_edges)


def _group_coplanar_facets(
    hull: ConvexHull, _points: np.ndarray, tolerance: float = 1e-6
) -> list[set[int]]:
    """
    Group triangulated facets that lie in the same plane into single faces.

    Parameters
    ----------
    hull : ConvexHull
        Convex hull object.
    points : np.ndarray
        3D points.
    tolerance : float
        Tolerance for coplanarity.

    Returns
    -------
    List[Set[int]]
        List of faces, where each face is a set of vertex indices.
    """
    # Normalise all facet equations
    equations = hull.equations.copy()
    for i in range(len(equations)):
        norm = np.linalg.norm(equations[i, :3])
        if norm > 1e-10:
            equations[i] /= norm

    # Group facets by their plane equations
    facet_groups: list[list[int]] = []
    used: set[int] = set()

    for i in range(len(hull.simplices)):
        if i in used:
            continue

        # Start a new group with this facet
        group_indices = [i]
        used.add(i)
        eq_i = equations[i]

        # Find all other facets coplanar with this one
        for j in range(i + 1, len(hull.simplices)):
            if j in used:
                continue

            eq_j = equations[j]

            # Check if equations are the same (or opposite - same plane, opposite normal)
            if np.allclose(eq_i, eq_j, atol=tolerance) or np.allclose(eq_i, -eq_j, atol=tolerance):
                group_indices.append(j)
                used.add(j)

        facet_groups.append(group_indices)

    # For each group, collect all unique vertices
    faces: list[set[int]] = []
    for group in facet_groups:
        vertices: set[int] = set()
        for facet_idx in group:
            vertices.update(hull.simplices[facet_idx].tolist())
        faces.append(vertices)

    return faces


def _get_face_boundary_edges(points: np.ndarray, vertex_indices: set[int]) -> set[tuple[int, int]]:
    """
    Get boundary edges of a face by computing its 2D convex hull.

    Parameters
    ----------
    points : np.ndarray
        All 3D points.
    vertex_indices : Set[int]
        Indices of vertices that form this face.

    Returns
    -------
    Set[Tuple[int, int]]
        Set of boundary edges (as sorted tuples).
    """
    if len(vertex_indices) < 3:
        # Degenerate face
        indices = list(vertex_indices)
        if len(indices) == 2:
            v1, v2 = sorted(indices)
            return {(v1, v2)}
        return set()

    indices = list(vertex_indices)
    face_points = points[indices]

    # Find the plane containing these points
    # Use first 3 points to define plane
    p0 = face_points[0]
    v1 = face_points[1] - p0
    v2 = face_points[2] - p0

    # Normal to plane
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        # Degenerate - points are collinear
        # Just return edges between consecutive points
        edges: set[tuple[int, int]] = set()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                v1, v2 = sorted([indices[i], indices[j]])
                edges.add((v1, v2))
        return edges

    normal = normal / norm

    # Create 2D coordinate system in the plane
    u = v1 / (np.linalg.norm(v1) + 1e-10)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-10)

    # Project face points to 2D
    points_2d = np.array([[np.dot(pt - p0, u), np.dot(pt - p0, v)] for pt in face_points])

    # Compute 2D convex hull to get boundary
    try:
        hull_2d = ConvexHull(points_2d)
        boundary_local = hull_2d.vertices

        # Convert back to original indices and create edges
        edges = set()
        n = len(boundary_local)
        for i in range(n):
            v1_local = boundary_local[i]
            v2_local = boundary_local[(i + 1) % n]
            v1_orig = indices[v1_local]
            v2_orig = indices[v2_local]
            edge_v1, edge_v2 = sorted([v1_orig, v2_orig])
            edges.add((edge_v1, edge_v2))

        return edges
    except Exception:
        # Fallback for degenerate cases
        edges = set()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edge_v1, edge_v2 = sorted([indices[i], indices[j]])
                edges.add((edge_v1, edge_v2))
        return edges


def classify_edges_by_visibility(
    points: np.ndarray,  # noqa: ARG001 (kept for call-site symmetry with extract_edges_from_hull)
    hull: ConvexHull,
    edges: list[tuple[int, int]],
    camera_direction: np.ndarray = np.array([0, 0, 1]),
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Classify edges as visible or hidden based on face normals.

    Parameters
    ----------
    points : np.ndarray
        3D coordinates of all points.
    hull : ConvexHull
        Convex hull of the points.
    edges : List[Tuple[int, int]]
        List of edges as (vertex1_idx, vertex2_idx) pairs.
    camera_direction : np.ndarray, default [0, 0, 1]
        Direction from which we're viewing (for determining visibility).

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        Tuple of (visible_edges, hidden_edges).

    Notes
    -----
    An edge is visible if at least one of its adjacent faces has a normal
    pointing toward the camera.

    """
    # Identify front-facing facets
    front_facets: set[int] = set()
    for i, equation in enumerate(hull.equations):
        normal = equation[:3]
        # Check if normal points toward camera
        if np.dot(normal, camera_direction) > 0:
            front_facets.add(i)

    # Map edges to their adjacent facets
    edge_to_facets: dict[tuple[int, int], list[int]] = {edge: [] for edge in edges}

    for facet_idx, simplex in enumerate(hull.simplices):
        n = len(simplex)
        for i in range(n):
            v1 = int(simplex[i])
            v2 = int(simplex[(i + 1) % n])
            edge_v1, edge_v2 = sorted([v1, v2])
            edge = (edge_v1, edge_v2)
            if edge in edge_to_facets:
                edge_to_facets[edge].append(facet_idx)

    # Classify edges
    visible_edges: list[tuple[int, int]] = []
    hidden_edges: list[tuple[int, int]] = []

    for edge, facet_indices in edge_to_facets.items():
        # If any adjacent facet is front-facing, edge is visible
        if any(f in front_facets for f in facet_indices):
            visible_edges.append(edge)
        else:
            hidden_edges.append(edge)

    return visible_edges, hidden_edges


def determine_label_position(point: np.ndarray) -> str:
    """
    Determine appropriate TikZ label position based on point location.

    Parameters
    ----------
    point : np.ndarray
        3D point coordinates [x, y, z].

    Returns
    -------
    str
        TikZ position specifier (e.g., "right", "above", "left").

    Notes
    -----
    Uses a simple heuristic based on which coordinate has the largest magnitude.
    """
    x, y, z = point

    # Simple heuristic based on octant
    if abs(x) > abs(y) and abs(x) > abs(z):
        return "right" if x > 0 else "left"
    elif abs(y) > abs(z):
        return "above" if y > 0 else "below"
    else:
        return "above right" if z > 0 else "below left"


def format_coordinate(value: float, precision: int = 2) -> str:
    """
    Format a coordinate value for display.

    Parameters
    ----------
    value : float
        Coordinate value.
    precision : int, default 2
        Number of decimal places.

    Returns
    -------
    str
        Formatted string (integer if whole number, else float).
    """
    # Check if value is close to an integer
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    else:
        return f"{value:.{precision}f}"
