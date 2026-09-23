"""
LaTeX export utilities for Feynman integral analysis.

Provides functions to generate LaTeX documents and code snippets for
mathematical expressions, tables, and complete analysis reports.
"""

from typing import Any

import sympy as sp

from ..core.exceptions import ValidationError
from ..core.graph import Graph
from ..parametrisations.base import ParametrisationResult
from ..systems.complete import GKZSystem
from ..visualisation.tikz import graph_to_tikz as _graph_to_tikz


def to_latex(expr: sp.Expr, **kwargs: Any) -> str:
    """
    Convert a SymPy expression to LaTeX.

    Parameters
    ----------
    expr : sp.Expr
        SymPy expression to convert.
    **kwargs
        Additional arguments passed to sympy.latex().

    Returns
    -------
    str
        LaTeX representation of the expression.

    Examples
    --------
    >>> import sympy as sp
    >>> x, y = sp.symbols('x y')
    >>> expr = x**2 + sp.sqrt(y)
    >>> print(to_latex(expr))
    x^{2} + \\sqrt{y}
    """
    # Set default options for better formatting
    if "fold_short_frac" not in kwargs:
        kwargs["fold_short_frac"] = False
    if "long_frac_ratio" not in kwargs:
        kwargs["long_frac_ratio"] = 2

    return str(sp.latex(expr, **kwargs))


def factor_energy_scale(expr: sp.Expr, scale: sp.Symbol) -> tuple[sp.Expr, int]:
    """
    Write an expression as ``numerator / scale**power``.

    The second Symanzik polynomial carries a factor ``1/mu**2`` on every term,
    where ``mu`` is the energy scale; a report shows that factor once in front
    of the polynomial rather than on each monomial.

    Parameters
    ----------
    expr : sp.Expr
        Expression to factorise.
    scale : sp.Symbol
        Energy scale to pull out.

    Returns
    -------
    tuple[sp.Expr, int]
        The numerator, which is free of ``scale``, and the power. The input and
        zero are returned when ``scale`` does not occur in the expression.

    Raises
    ------
    ValidationError
        If ``scale`` occurs in a form that is not an overall power of a term,
        so that no single power factors out.

    Examples
    --------
    >>> import sympy as sp
    >>> a, mu = sp.symbols('a mu')
    >>> factor_energy_scale(a / mu**2 + 1 / mu**2, mu)
    (a + 1, 2)
    """
    expr = sp.expand(expr)
    if scale not in expr.free_symbols:
        return expr, 0

    complaint = f"Cannot factor a power of {scale} out of {expr}"
    exponents = [term.as_coeff_exponent(scale)[1] for term in sp.Add.make_args(expr)]
    if not all(exponent.is_Integer for exponent in exponents):
        raise ValidationError(complaint)

    power = -int(min(exponents))
    numerator = sp.expand(expr * scale**power)
    if scale in numerator.free_symbols:
        raise ValidationError(complaint)

    return numerator, power


def euler_equation_to_latex(equation: sp.Equality) -> str:
    """
    Convert an Euler equation to LaTeX with simplified notation.

    Replaces Phi(z_1, z_2, ..., z_n) with just Phi to save space.

    Parameters
    ----------
    equation : sp.Equality
        Euler equation (typically lhs = rhs with derivatives).

    Returns
    -------
    str
        LaTeX string with simplified Phi notation.
    """
    # Convert to LaTeX
    latex_str = to_latex(equation)

    # Find and replace Phi(z_1, z_2, ..., z_n) with just \Phi
    # Pattern: \Phi{\left(z_{...} \right)} or similar
    import re

    # Match Phi with any arguments in parentheses
    # This handles patterns like: \Phi{\left(z_{1}, z_{2}, \ldots, z_{n} \right)}
    pattern = r"\\Phi\{?\\left\([^)]+\\right\)\}?"
    latex_str = re.sub(pattern, r"\\Phi", latex_str)

    # Also handle simpler patterns without \left \right
    pattern2 = r"\\Phi\{?\([^)]+\)\}?"
    latex_str = re.sub(pattern2, r"\\Phi", latex_str)

    # Handle operatorname{Phi}(...) if SymPy uses that
    pattern3 = r"\\operatorname\{Phi\}\{?\\left\([^)]+\\right\)\}?"
    latex_str = re.sub(pattern3, r"\\Phi", latex_str)

    pattern4 = r"\\operatorname\{Phi\}\{?\([^)]+\)\}?"
    latex_str = re.sub(pattern4, r"\\Phi", latex_str)

    return latex_str


def to_latex_lines(expr: sp.Expr, max_length: int = 80) -> list[str]:
    """
    Break the LaTeX of a long sum into lines at its top-level signs.

    A line break falls only before a ``+`` or ``-`` outside every brace group
    and every ``\\left ... \\right`` pair, so fractions, exponents, function
    arguments and parenthesised factors are never split apart and every line
    balances its delimiters. Terms are packed greedily into lines of about
    ``max_length`` characters.

    Parameters
    ----------
    expr : sp.Expr
        Expression to convert.
    max_length : int, default 80
        Approximate maximum characters per line.

    Returns
    -------
    list[str]
        The lines; a single line when the expression is short or has no
        top-level sign to break at. Every line after the first starts with its
        sign, ``+ `` or ``- ``.
    """
    latex_str = to_latex(expr)
    if len(latex_str) <= max_length:
        return [latex_str]

    chunks: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(latex_str):
        if latex_str.startswith("\\left", i) and not latex_str[i + 5 : i + 6].isalpha():
            depth += 1
            i += 5
            continue
        if latex_str.startswith("\\right", i) and not latex_str[i + 6 : i + 7].isalpha():
            depth -= 1
            i += 6
            continue
        ch = latex_str[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0 and i > start and latex_str.startswith((" + ", " - "), i):
            chunks.append(latex_str[start:i])
            start = i
        i += 1
    chunks.append(latex_str[start:])

    lines: list[str] = []
    current = chunks[0]
    for chunk in chunks[1:]:
        if len(current) + len(chunk) > max_length:
            lines.append(current)
            current = chunk.strip()
        else:
            current += chunk
    lines.append(current)
    return lines


def to_latex_split(expr: sp.Expr, max_length: int = 80) -> str:
    """
    Convert expression to LaTeX with manual line breaking for long expressions.

    The lines are those of :func:`to_latex_lines`, joined into a ``split``
    environment, which must sit inside ``equation*`` or ``align*``.

    Parameters
    ----------
    expr : sp.Expr
        Expression to convert.
    max_length : int, default 80
        Approximate maximum characters per line.

    Returns
    -------
    str
        LaTeX with split environment if needed.
    """
    lines = to_latex_lines(expr, max_length)
    if len(lines) == 1:
        return lines[0]
    body = " \\\\\n& ".join(lines)
    return "\\begin{split}\n" + body + "\n\\end{split}"


def graph_to_latex_table(graph: Graph) -> str:
    """
    Generate a LaTeX table summarising graph properties.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        LaTeX table code.

    Examples
    --------
    >>> table = graph_to_latex_table(my_graph)
    >>> print(table)
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{@{}ll@{}}",
        "\\toprule",
        "\\textbf{Property} & \\textbf{Value} \\\\",
        "\\midrule",
        f"Internal vertices & {graph.internal_vertices} \\\\",
        f"External legs & {graph.external_legs} \\\\",
        f"Internal propagators & {len(graph.get_internal_edges())} \\\\",
        f"Loop number & {graph.get_loop_count()} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Feynman graph topological properties.}",
        "\\label{tab:graph_properties}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def edges_to_latex_table(graph: Graph) -> str:
    """
    Generate a LaTeX table listing all edges.

    Parameters
    ----------
    graph : Graph
        Feynman graph.

    Returns
    -------
    str
        LaTeX table code.
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{@{}ccccc@{}}",
        "\\toprule",
        "\\textbf{Index} & \\textbf{Type} & \\textbf{From} & \\textbf{To} & \\textbf{Mass} \\\\",
        "\\midrule",
    ]

    for edge in graph.edges:
        edge_type = "Internal" if edge.is_internal else "External"
        mass_str = "$" + to_latex(edge.get_mass()) + "$" if edge.mass else "---"
        lines.append(f"{edge.idx} & {edge_type} & {edge.v1} & {edge.v2} & {mass_str} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Complete edge list for the Feynman graph.}",
            "\\label{tab:edges}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def parametrisation_to_latex(result: ParametrisationResult) -> str:
    """
    Generate LaTeX code for a parametrisation result.

    Parameters
    ----------
    result : ParametrisationResult
        Parametrisation result.

    Returns
    -------
    str
        LaTeX code displaying the parametrisation.

    Examples
    --------
    >>> latex_code = parametrisation_to_latex(schwinger_result)
    """
    lines = [
        "\\subsection{" + result.name + " Parametrisation}",
        "\\label{sec:" + result.name.lower().replace("-", "_") + "}",
        "",
        result.description,
        "",
        "\\paragraph{Parameters.}",
        "The integration parameters are:",
        "\\begin{equation}",
        ", \\quad ".join(to_latex(p) for p in result.parameters),
        "\\end{equation}",
        "",
        "\\paragraph{Prefactor.}",
        "The overall prefactor is:",
        "\\begin{equation}",
        to_latex(result.prefactor),
        "\\end{equation}",
        "",
        "\\paragraph{Measure.}",
        "The integration measure is:",
        "\\begin{equation}",
        to_latex(result.measure),
        "\\end{equation}",
        "",
        "\\paragraph{Integrand.}",
        "The integrand is:",
    ]

    # Use dmath for automatic breaking of long integrands
    integrand_latex = to_latex(result.integrand)
    if len(integrand_latex) > 100:
        lines.extend(
            [
                "\\begin{dmath}",
                integrand_latex,
                "\\end{dmath}",
            ]
        )
    else:
        lines.extend(
            [
                "\\begin{equation}",
                integrand_latex,
                "\\end{equation}",
            ]
        )

    if result.constraints:
        lines.extend(
            [
                "",
                "\\paragraph{Constraints.}",
                "The parameters satisfy the following constraints:",
                "\\begin{align}",
            ]
        )
        for i, constraint in enumerate(result.constraints):
            ending = " \\\\" if i < len(result.constraints) - 1 else ""
            lines.append(to_latex(constraint) + ending)
        lines.append("\\end{align}")

    return "\n".join(lines)


def gkz_system_to_latex(gkz: GKZSystem, include_polytope_tikz: str | None = None) -> str:
    """
    Generate LaTeX code for a GKZ hypergeometric system.

    Parameters
    ----------
    gkz : GKZSystem
        GKZ system.
    include_polytope_tikz : Optional[str], default None
        TikZ code for Newton polytope to include.

    Returns
    -------
    str
        LaTeX code.

    Examples
    --------
    >>> latex_code = gkz_system_to_latex(gkz_system)
    """
    lines = [
        "\\subsection{GKZ $A$-Hypergeometric System}",
        "\\label{sec:gkz_system}",
        "",
        f"The GKZ hypergeometric system has {len(gkz.z_variables)} variables, "
        f"{len(gkz.euler_equations)} differential equations, and "
        f"A-matrix of dimension ${gkz.a_matrix.rows} \\times {gkz.a_matrix.cols}$.",
        "",
        "\\paragraph{$A$-matrix.}",
        "The $A$-matrix encoding the monomial exponent structure is:",
        "\\begin{equation}",
        "A = " + to_latex(gkz.a_matrix),
        "\\label{eq:a_matrix}",
        "\\end{equation}",
        "",
    ]

    # Add Newton polytope if provided
    if include_polytope_tikz:
        lines.extend(
            [
                "\\paragraph{Newton polytope.}",
                "The Newton polytope is the convex hull of the monomial support. ",
                "It encodes the combinatorial structure of the polynomial $G(u) = U(u) + F(u)$. ",
                "The polytope is visualised in \\Cref{fig:newton_polytope}.",
                "",
                include_polytope_tikz,
                "",
            ]
        )

    lines.extend(
        [
            "\\paragraph{Parameter vector.}",
            "The parameter vector $\\boldsymbol{\\beta}$ is:",
            "\\begin{equation}",
            "\\boldsymbol{\\beta} = \\begin{pmatrix}",
        ]
    )

    for i, beta in enumerate(gkz.beta_parameters):
        lines.append(to_latex(beta) + (" \\\\" if i < len(gkz.beta_parameters) - 1 else ""))

    lines.extend(
        [
            "\\end{pmatrix}",
            "\\label{eq:beta_vector}",
            "\\end{equation}",
            "",
            "\\paragraph{Euler equations.}",
            f"The system consists of {len(gkz.euler_equations)} Euler-type differential equations. "
            "These are Horn-type hypergeometric differential equations, where $\\Phi = \\Phi(z_1, \\ldots, z_{"
            + str(len(gkz.z_variables))
            + "})$ represents the Feynman integral. ",
            "Each equation has the form:",
        ]
    )

    # Show equations with proper formatting - each in its own equation environment
    max_equations_shown = 5
    equations_to_show = min(len(gkz.euler_equations), max_equations_shown)

    for i in range(equations_to_show):
        eq = gkz.euler_equations[i]
        # Use custom formatter to simplify Phi notation
        eq_latex = euler_equation_to_latex(eq)

        # Use dmath for long equations to allow automatic breaking
        if len(eq_latex) > 120:
            lines.extend(
                [
                    "\\begin{dmath}",
                    eq_latex,
                    "\\label{eq:euler_" + str(i) + "}",
                    "\\end{dmath}",
                ]
            )
        else:
            lines.extend(
                [
                    "\\begin{equation}",
                    eq_latex,
                    "\\label{eq:euler_" + str(i) + "}",
                    "\\end{equation}",
                ]
            )

    if len(gkz.euler_equations) > max_equations_shown:
        lines.append(
            f"\\noindent (Showing {max_equations_shown} of {len(gkz.euler_equations)} equations for brevity.)"
        )

    lines.extend(
        [
            "",
            "Each equation expresses a homogeneity property of the Feynman integral "
            "under rescalings of the Lee-Pomeransky parameters.",
        ]
    )

    return "\n".join(lines)


def toric_ideal_to_latex(toric_generators: list[sp.Expr]) -> str:
    """
    Generate LaTeX code for toric ideal generators.

    Parameters
    ----------
    toric_generators : List[sp.Expr]
        List of toric ideal generators.

    Returns
    -------
    str
        LaTeX code displaying the toric ideal.

    Examples
    --------
    >>> from feynkit.algebra import compute_toric_ideal_generators
    >>> generators = compute_toric_ideal_generators(A_matrix)
    >>> latex_code = toric_ideal_to_latex(generators)
    """
    lines = [
        "\\subsection{Toric Ideal and Integration-by-Parts Relations}",
        "\\label{sec:toric_ideal}",
        "",
    ]

    if not toric_generators:
        lines.extend(
            [
                "The toric ideal $I_A$ is \\textbf{trivial} (contains no non-trivial generators). ",
                "This has important physical implications:",
                "\\begin{itemize}",
                "\\item All monomial coefficients in the Lee-Pomeransky polynomial are algebraically independent.",
                "\\item The Feynman integral is a \\textbf{master integral} and cannot be reduced further using integration-by-parts (IBP) identities.",
                "\\item The integral belongs to a minimal generating set for this topology.",
                "\\end{itemize}",
            ]
        )
    else:
        lines.extend(
            [
                f"The toric ideal $I_A$ is generated by {len(toric_generators)} polynomial{'s' if len(toric_generators) > 1 else ''}. ",
                "These generators encode integration-by-parts (IBP) relations among the ",
                "monomial coefficients of the Lee-Pomeransky polynomial.",
                "",
                "\\paragraph{Generators.}",
                "The toric ideal is generated by:",
            ]
        )

        for i, gen in enumerate(toric_generators):
            gen_latex = to_latex(gen)

            # Use dmath for long generators
            if len(gen_latex) > 100:
                lines.extend(
                    [
                        "\\begin{dmath}",
                        gen_latex + " = 0",
                        "\\label{eq:toric_gen_" + str(i) + "}",
                        "\\end{dmath}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "\\begin{equation}",
                        gen_latex + " = 0",
                        "\\label{eq:toric_gen_" + str(i) + "}",
                        "\\end{equation}",
                    ]
                )

        lines.extend(
            [
                "",
                "Each generator represents a syzygy (polynomial relation) among the monomials, ",
                "which corresponds to an IBP reduction identity for the Feynman integral. ",
                "These relations can be used to express the integral in terms of simpler master integrals.",
            ]
        )

    return "\n".join(lines)


def _create_analysis_document(
    graph: Graph,
    u_polynomial: sp.Expr,
    f_polynomial: sp.Expr,
    gkz_system: GKZSystem | None = None,
    parametrisation_results: dict[str, ParametrisationResult] | None = None,
    toric_generators: list[sp.Expr] | None = None,
    polytope_tikz: str | None = None,
    title: str = "Feynman Integral Analysis",
    author: str = "Generated by Feynkit",
) -> str:
    """
    Generate a complete LaTeX document with full analysis.

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
        Dictionary of parametrisation results (e.g., {'Schwinger': result, ...}).
    toric_generators : Optional[List[sp.Expr]], default None
        Toric ideal generators.
    polytope_tikz : Optional[str], default None
        TikZ code for Newton polytope visualisation.
    title : str, default "Feynman Integral Analysis"
        Document title.
    author : str, default "Generated by Feynkit"
        Document author.

    Returns
    -------
    str
        Complete LaTeX document.

    Examples
    --------
    >>> doc = create_analysis_document(
    ...     graph=my_graph,
    ...     u_polynomial=U,
    ...     f_polynomial=F,
    ...     gkz_system=gkz,
    ...     toric_generators=toric_gens,
    ...     polytope_tikz=tikz_code,
    ...     title="Bubble Diagram Analysis"
    ... )
    >>> with open("analysis.tex", "w") as f:
    ...     f.write(doc)
    """
    lines = [
        "\\documentclass[11pt,a4paper]{article}",
        "",
        "% Package imports",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{amsmath,amssymb,amsthm}",
        "\\usepackage{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{tikz}",
        "\\usetikzlibrary{calc}",
        "\\usepackage{tikz-3dplot}",
        "\\usepackage{hyperref}",
        "\\usepackage{cleveref}",
        "\\usepackage{microtype}",
        "\\usepackage{breqn}  % For automatic equation breaking (dmath environment)",
        "",
        "% Page geometry",
        "\\geometry{",
        "  a4paper,",
        "  margin=1in,",
        "  marginparwidth=0.75in",
        "}",
        "",
        "% Hyperref setup",
        "\\hypersetup{",
        "  colorlinks=true,",
        "  linkcolor=blue,",
        "  citecolor=blue,",
        "  urlcolor=blue,",
        "  pdfauthor={" + author + "},",
        "  pdftitle={" + title + "}",
        "}",
        "",
        "% Better equation breaking",
        "\\allowdisplaybreaks",
        "",
        "% Title and author",
        f"\\title{{{title}}}",
        f"\\author{{{author}}}",
        "\\date{\\today}",
        "",
        "\\begin{document}",
        "",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        f"This document presents a comprehensive analysis of a {graph.get_loop_count()}-loop "
        f"Feynman integral with {len(graph.get_internal_edges())} internal propagators and "
        f"{graph.external_legs} external legs. "
        "We compute the Symanzik polynomials, parametric representations, "
        "GKZ hypergeometric system, and toric ideal structure. "
        "All calculations were performed using Feynkit, a symbolic computation toolkit for Feynman integrals.",
        "\\end{abstract}",
        "",
        "\\tableofcontents",
        "\\clearpage",
        "",
        "\\section{Introduction}",
        "\\label{sec:introduction}",
        "",
        "This report analyses the structure of a Feynman integral using modern algebraic geometry "
        "and differential equations techniques. We employ the parametric representation framework "
        "and study the integral through its associated GKZ $A$-hypergeometric system.",
        "",
        "\\section{Graph Properties}",
        "\\label{sec:graph}",
        "",
        "We begin by describing the topological structure of the Feynman diagram.",
        "",
        _graph_to_tikz(graph),
        "",
        "\\subsection{Topological Invariants}",
        "\\label{sec:topology}",
        "",
        "The graph has the following topological properties, summarised in \\Cref{tab:graph_properties}.",
        "",
        graph_to_latex_table(graph),
        "",
        "\\subsection{Edge Structure}",
        "\\label{sec:edges}",
        "",
        "The complete list of edges is given in \\Cref{tab:edges}.",
        "",
        edges_to_latex_table(graph),
        "",
        "\\clearpage",
        "\\section{Symanzik Polynomials}",
        "\\label{sec:symanzik}",
        "",
        "The Symanzik polynomials $U$ and $F$ encode the topological and kinematic structure "
        "of the Feynman integral. These polynomials are fundamental to the parametric representation.",
        "",
        "\\subsection{First Symanzik Polynomial}",
        "\\label{sec:u_polynomial}",
        "",
        "The first Symanzik polynomial $U$ depends only on the graph topology:",
    ]

    # Use dmath for long U polynomial
    u_latex = to_latex(u_polynomial)
    if len(u_latex) > 100:
        lines.extend(
            [
                "\\begin{dmath}",
                f"U = {u_latex}",
                "\\label{eq:u_polynomial}",
                "\\end{dmath}",
            ]
        )
    else:
        lines.extend(
            [
                "\\begin{equation}",
                f"U = {u_latex}",
                "\\label{eq:u_polynomial}",
                "\\end{equation}",
            ]
        )

    lines.extend(
        [
            "",
            "This polynomial is homogeneous in the Schwinger parameters and encodes "
            "the sum over all spanning trees of the graph.",
            "",
            "\\subsection{Second Symanzik Polynomial}",
            "\\label{sec:f_polynomial}",
            "",
            "The second Symanzik polynomial $F$ includes kinematic information:",
        ]
    )

    # Use dmath for long F polynomial
    f_latex = to_latex(f_polynomial)
    if len(f_latex) > 100:
        lines.extend(
            [
                "\\begin{dmath}",
                f"F = {f_latex}",
                "\\label{eq:f_polynomial}",
                "\\end{dmath}",
            ]
        )
    else:
        lines.extend(
            [
                "\\begin{equation}",
                f"F = {f_latex}",
                "\\label{eq:f_polynomial}",
                "\\end{equation}",
            ]
        )

    lines.extend(
        [
            "",
            "This polynomial encodes the momentum flow and mass structure of the diagram.",
            "",
        ]
    )

    # Add parametrisations if provided
    if parametrisation_results:
        lines.extend(
            [
                "\\clearpage",
                "\\section{Parametric Representations}",
                "\\label{sec:parametrisations}",
                "",
                "We present three standard parametric representations: Schwinger, Feynman, and Lee-Pomeransky. "
                "Each representation has distinct advantages for different computational tasks.",
                "",
            ]
        )

        for _name, result in parametrisation_results.items():
            lines.append(parametrisation_to_latex(result))
            lines.append("")

    # Add GKZ system if provided
    if gkz_system:
        lines.extend(
            [
                "\\clearpage",
                "\\section{GKZ Hypergeometric System}",
                "\\label{sec:gkz}",
                "",
                "The Feynman integral satisfies a system of partial differential equations known as "
                "the GKZ (Gelfand-Kapranov-Zelevinsky) $A$-hypergeometric system. This system "
                "provides a systematic approach to studying the analytic properties of the integral.",
                "",
                gkz_system_to_latex(gkz_system, include_polytope_tikz=polytope_tikz),
                "",
            ]
        )

    # Add toric ideal if provided
    if toric_generators is not None:
        lines.extend(
            [
                "\\clearpage",
                "\\section{Algebraic Structure}",
                "\\label{sec:algebra}",
                "",
                toric_ideal_to_latex(toric_generators),
                "",
            ]
        )

    lines.extend(
        [
            "\\section{Conclusion}",
            "\\label{sec:conclusion}",
            "",
            "This analysis provides a complete mathematical characterisation of the Feynman integral. "
            "The parametric representations enable systematic evaluation, while the GKZ system "
            "and toric ideal structure reveal the underlying algebraic-geometric properties.",
            "",
            "\\appendix",
            "",
            "\\section{Computational Details}",
            "\\label{sec:computational}",
            "",
            "All calculations in this document were performed using Feynkit, "
            "a Python package for symbolic Feynman integral computations. "
            "The source code is available at \\texttt{https://github.com/feynkit/feynkit}.",
            "",
            "\\end{document}",
        ]
    )

    return "\n".join(lines)


def save_latex_document(content: str, filename: str) -> str:
    """
    Save LaTeX content to a file.

    Parameters
    ----------
    content : str
        LaTeX document content.
    filename : str
        Output filename.

    Returns
    -------
    str
        The filename where content was saved.

    Examples
    --------
    >>> doc = create_analysis_document(...)
    >>> save_latex_document(doc, "analysis.tex")
    'analysis.tex'
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename
