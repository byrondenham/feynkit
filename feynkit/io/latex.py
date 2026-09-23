"""
LaTeX export utilities for Feynman integral analysis.

Provides LaTeX for expressions, long sums broken over lines and graph tables,
and a helper that saves a document. The analysis document itself is rendered
by :mod:`feynkit.io.report_latex`.
"""

from typing import Any

import sympy as sp

from ..core.exceptions import ValidationError
from ..core.graph import Graph


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
    >>> save_latex_document(integral.to_latex(), "analysis.tex")
    'analysis.tex'
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename
