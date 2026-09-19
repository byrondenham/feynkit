"""
Plain text export utilities for Feynman integral analysis.

Provides functions to generate human-readable text reports.
"""

import sympy as sp

from ..core.graph import Graph
from ..parametrisations.base import ParametrisationResult
from ..systems.complete import GKZSystem


def graph_to_text(graph: Graph) -> str:
    """
    Generate a text summary of graph properties.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        Text summary.

    Examples
    --------
    >>> print(graph_to_text(my_graph))
    """
    lines = [
        "=" * 70,
        "FEYNMAN GRAPH PROPERTIES",
        "=" * 70,
        f"Internal vertices:     {graph.internal_vertices}",
        f"External legs:         {graph.external_legs}",
        f"Internal propagators:  {len(graph.get_internal_edges())}",
        f"Loop number:           {graph.get_loop_count()}",
        "",
    ]
    return "\n".join(lines)


def edges_to_text(graph: Graph) -> str:
    """
    Generate a text list of all edges.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        Text edge list.
    """
    lines = [
        "=" * 70,
        "EDGE LIST",
        "=" * 70,
        f"{'Idx':<5} {'Type':<10} {'From':<5} {'To':<5} {'Mass':<20} {'Name':<15}",
        "-" * 70,
    ]

    for edge in graph.edges:
        edge_type = "Internal" if edge.is_internal else "External"
        mass_str = str(edge.get_mass()) if edge.mass else "---"
        name_str = edge.name if edge.name else "---"
        lines.append(
            f"{edge.idx:<5} {edge_type:<10} {edge.v1:<5} {edge.v2:<5} {mass_str:<20} {name_str:<15}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def parametrisation_to_text(result: ParametrisationResult) -> str:
    """
    Generate a text representation of a parametrisation result.

    Parameters
    ----------
    result : ParametrisationResult
        Parametrisation result.

    Returns
    -------
    str
        Text summary.
    """
    lines = [
        "=" * 70,
        f"{result.name.upper()} PARAMETRISATION",
        "=" * 70,
        "",
        f"Description: {result.description}",
        "",
        f"Parameters ({len(result.parameters)}):",
        "  " + ", ".join(str(p) for p in result.parameters),
        "",
        "Prefactor:",
        f"  {result.prefactor}",
        "",
        "Measure:",
        f"  {result.measure}",
        "",
        "Integrand:",
        f"  {result.integrand}",
        "",
    ]

    if result.constraints:
        lines.append(f"Constraints ({len(result.constraints)}):")
        for i, constraint in enumerate(result.constraints):
            lines.append(f"  [{i}] {constraint}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def gkz_system_to_text(gkz: GKZSystem) -> str:
    """
    Generate a text representation of a GKZ system.

    Parameters
    ----------
    gkz : GKZSystem
        GKZ system.

    Returns
    -------
    str
        Text summary.
    """
    lines = [
        "=" * 70,
        "GKZ HYPERGEOMETRIC SYSTEM",
        "=" * 70,
        "",
        f"Number of variables:   {len(gkz.z_variables)}",
        f"Number of equations:   {len(gkz.euler_equations)}",
        f"Number of monomials:   {len(gkz.support)}",
        f"A-matrix shape:        {gkz.a_matrix.rows} x {gkz.a_matrix.cols}",
        "",
        "A-matrix:",
        str(gkz.a_matrix),
        "",
        f"Parameter vector beta ({len(gkz.beta_parameters)} components):",
        "  " + ", ".join(str(b) for b in gkz.beta_parameters),
        "",
        f"Euler equations ({len(gkz.euler_equations)}):",
    ]

    for i, eq in enumerate(gkz.euler_equations):
        lines.append(f"  [{i}] {eq}")

    lines.append("")
    lines.append("Monomial support:")
    for i, (exp_vec, coeff) in enumerate(gkz.support):
        lines.append(f"  [{i}] u^{exp_vec}: {coeff}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def toric_ideal_to_text(toric_generators: list[sp.Expr]) -> str:
    """
    Generate a text representation of toric ideal generators.

    Parameters
    ----------
    toric_generators : List[sp.Expr]
        List of toric ideal generators.

    Returns
    -------
    str
        Text summary.

    Examples
    --------
    >>> from feynkit.algebra import compute_toric_ideal_generators
    >>> generators = compute_toric_ideal_generators(A_matrix)
    >>> print(toric_ideal_to_text(generators))
    """
    lines = [
        "=" * 70,
        "TORIC IDEAL AND IBP RELATIONS",
        "=" * 70,
        "",
    ]

    if not toric_generators:
        lines.extend(
            [
                "The toric ideal is TRIVIAL (no non-trivial generators).",
                "",
                "This indicates that:",
                "  - All monomial coefficients are algebraically independent",
                "  - The integral is a MASTER INTEGRAL",
                "  - Cannot be reduced further using IBP identities",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Number of generators: {len(toric_generators)}",
                "",
                "These generators encode integration-by-parts (IBP) relations among",
                "the monomial coefficients of the Lee-Pomeransky polynomial.",
                "",
                "Generators (each equation = 0):",
                "",
            ]
        )

        for i, gen in enumerate(toric_generators):
            lines.append(f"  [{i}] {gen} = 0")

        lines.extend(
            [
                "",
                "Each generator represents a syzygy (polynomial relation) among the",
                "monomials, which corresponds to an IBP identity for the Feynman integral.",
                "",
            ]
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def _create_analysis_report(
    graph: Graph,
    u_polynomial: sp.Expr,
    f_polynomial: sp.Expr,
    gkz_system: GKZSystem | None = None,
    parametrisation_results: dict[str, ParametrisationResult] | None = None,
    toric_generators: list[sp.Expr] | None = None,
    title: str = "Feynman Integral Analysis Report",
) -> str:
    """
    Generate a complete plain text analysis report.

    Parameters
    ----------
    graph : Graph
        Feynman graph.
    u_polynomial : sp.Expr
        Symanzik U polynomial.
    f_polynomial : sp.Expr
        Symanzik F polynomial.
    gkz_system : Optional[GKZSystem], default None
        GKZ hypergeometric system.
    parametrisation_results : Optional[Dict[str, ParametrisationResult]], default None
        Dictionary of parametrisation results.
    toric_generators : Optional[List[sp.Expr]], default None
        Toric ideal generators.
    title : str, default "Feynman Integral Analysis Report"
        Report title.

    Returns
    -------
    str
        Complete text report.

    Examples
    --------
    >>> report = create_analysis_report(
    ...     graph=my_graph,
    ...     u_polynomial=U,
    ...     f_polynomial=F,
    ...     toric_generators=toric_gens,
    ... )
    >>> print(report)
    """
    lines = [
        "=" * 70,
        title.center(70),
        "=" * 70,
        "",
        graph_to_text(graph),
        "",
        edges_to_text(graph),
        "",
        "=" * 70,
        "SYMANZIK POLYNOMIALS",
        "=" * 70,
        "",
        "First Symanzik Polynomial (U):",
        f"  U = {u_polynomial}",
        "",
        "Second Symanzik Polynomial (F):",
        f"  F = {f_polynomial}",
        "",
    ]

    if parametrisation_results:
        for _name, result in parametrisation_results.items():
            lines.append("")
            lines.append(parametrisation_to_text(result))

    if gkz_system:
        lines.append("")
        lines.append(gkz_system_to_text(gkz_system))

    if toric_generators is not None:
        lines.append("")
        lines.append(toric_ideal_to_text(toric_generators))

    return "\n".join(lines)


def save_text_report(content: str, filename: str) -> str:
    """
    Save text report to a file.

    Parameters
    ----------
    content : str
        Report content.
    filename : str
        Output filename.

    Returns
    -------
    str
        The filename where content was saved.

    Examples
    --------
    >>> report = create_analysis_report(...)
    >>> save_text_report(report, "analysis.txt")
    'analysis.txt'
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename
