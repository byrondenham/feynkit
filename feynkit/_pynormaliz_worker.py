"""
The PyNormaliz computations of feynkit.lattice_invariants, runnable on their own.

query gives the Hilbert basis of the cone over a lattice polytope and, when
asked, its Hilbert series and Hilbert quasipolynomial, as plain lists.
feynkit calls it in its own process when no timeout is set. With a timeout
it runs this file's source with ``python -c`` instead, passing the question
as JSON on standard input and reading the answer as JSON from standard
output, so that a computation which outlasts the timeout can be ended. The
module therefore imports nothing from feynkit.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def query(rows: list[list[int]], series: bool) -> dict[str, Any]:
    """PyNormaliz's answers for the polytope with vertices rows, in Z^d with the grading last.

    The keys are "hilbert_basis" and, with series, "series" (numerator,
    denominator exponents, shift) and "quasipolynomial" (its rows, then the
    common denominator).
    """
    import PyNormaliz

    cone = PyNormaliz.Cone(polytope=rows)
    answer: dict[str, Any] = {"hilbert_basis": cone.HilbertBasis()}
    if series:
        answer["series"] = cone.HilbertSeries()
        answer["quasipolynomial"] = cone.HilbertQuasiPolynomial()
    return answer


if __name__ == "__main__":
    question = json.load(sys.stdin)
    json.dump(query(question["rows"], question["series"]), sys.stdout)
