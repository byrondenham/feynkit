"""
Helpers shared by the LaTeX renderers of the report sections.

Escaping, maths, displays, long tables, and the document that collects the
citations and the widest matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from ._report_shared import CITATIONS, Citations
from .latex import to_latex

# amsmath matrices stop at this many columns unless MaxMatrixCols is raised.
_AMSMATH_MATRIX_COLUMNS = 10


_ESCAPES: dict[int, str] = {
    ord("\\"): "\\textbackslash{}",
    ord("&"): "\\&",
    ord("%"): "\\%",
    ord("$"): "\\$",
    ord("#"): "\\#",
    ord("_"): "\\_",
    ord("{"): "\\{",
    ord("}"): "\\}",
    ord("~"): "\\textasciitilde{}",
    ord("^"): "\\textasciicircum{}",
}


# The Lee-Pomeransky integral without its prefactor, the function the GKZ
# system annihilates.
_LATEX_EULER_MELLIN = (
    "I_A(\\beta, z) = \\int_{\\mathbb{R}_{+}^{N}} \\prod_{e} \\mathrm{d}u_e\\, "
    "u_e^{\\nu_e - 1}\\, G(z, u)^{-D/2}"
)


class LatexDocument(Citations):
    """What the preamble and the bibliography need to know about the sections.

    The sections cite through :meth:`cite` and print matrices through
    :meth:`matrix`, so that the bibliography lists the works cited, in the
    order of first citation, and the preamble allows the widest matrix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.widest_matrix = 0

    def cite(self, *keys: str) -> str:
        self.add(*keys)
        return "~\\cite{" + ",".join(keys) + "}"

    def matrix(self, matrix: sp.Matrix) -> str:
        self.widest_matrix = max(self.widest_matrix, matrix.cols)
        rows = [
            " & ".join(to_latex(matrix[r, c]) for c in range(matrix.cols))
            for r in range(matrix.rows)
        ]
        return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"

    def preamble(self) -> str:
        columns = max(_AMSMATH_MATRIX_COLUMNS, self.widest_matrix)
        return "\n".join(
            [
                "\\documentclass[11pt,a4paper]{article}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage[T1]{fontenc}",
                "\\usepackage{lmodern}",
                "\\usepackage{amsmath,amssymb}",
                f"\\setcounter{{MaxMatrixCols}}{{{columns}}}",
                "\\usepackage[margin=2.5cm]{geometry}",
                "\\usepackage{booktabs,array,longtable}",
                "\\usepackage{tikz}",
                "\\usetikzlibrary{calc}",
                "\\usepackage{tikz-3dplot}",
                "\\usepackage[hidelinks,pdfusetitle]{hyperref}",
                "\\allowdisplaybreaks",
            ]
        )

    def bibliography(self) -> str:
        items = [f"\\bibitem{{{key}}} {CITATIONS[key]}" for key in self.keys]
        return "\\begin{thebibliography}{99}\n" + "\n".join(items) + "\n\\end{thebibliography}"


def _escape(text: str) -> str:
    """Escape the characters that LaTeX treats specially in running text."""
    return text.translate(_ESCAPES)


def _math(expr: sp.Expr) -> str:
    return "$" + to_latex(expr) + "$"


def _latex_vector(entries: Sequence[sp.Expr]) -> str:
    return "\\left(" + ", ".join(to_latex(e) for e in entries) + "\\right)"


def _latex_equation(lhs: str, lines: Sequence[str]) -> str:
    """Display ``lhs = ...`` over the given lines, aligned after the equals sign."""
    if len(lines) == 1:
        body = f"{lhs} = {lines[0]}"
    else:
        continued = "".join(f" \\\\\n&\\quad {{}}{line}" for line in lines[1:])
        body = f"\\begin{{split}}\n{lhs} &= {lines[0]}{continued}\n\\end{{split}}"
    return "\\begin{equation*}\n" + body + "\n\\end{equation*}"


def _longtable(spec: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A longtable with booktabs rules; header None for a table without one."""
    head = ["\\toprule", " & ".join(header) + " \\\\", "\\midrule"] if header else ["\\toprule"]
    return "\n".join(
        [
            f"\\begin{{longtable}}{{{spec}}}",
            *head,
            "\\endhead",
            "\\bottomrule",
            "\\endlastfoot",
            *(" & ".join(row) + " \\\\" for row in rows),
            "\\end{longtable}",
        ]
    )
