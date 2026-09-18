"""
Optional Sage backend for affine-equivalence checks.

The Sage backend is a thin shim: when invoked it ensures Sage is available
(via the ``sageall`` import) and then delegates to the same brute-force
sympy implementation as the default backend. The shim exists to give
users an explicit opt-in path that they can later re-implement against
Sage's native polytope-isomorphism functionality without changing the
public API.
"""

from __future__ import annotations

from collections.abc import Iterable

import sympy as sp

try:
    from sageall import Matrix as SageMatrix
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    _SAGE_IMPORT_ERROR: ImportError | None = exc
    SageMatrix = None
else:  # pragma: no cover - depends on optional dependency
    _SAGE_IMPORT_ERROR = None


def _require_sage() -> None:
    if _SAGE_IMPORT_ERROR is not None:  # pragma: no cover
        raise ImportError(
            "Sage backend requested, but Sage is not available. "
            "Install SageMath and ensure 'sageall' is importable."
        ) from _SAGE_IMPORT_ERROR


def compare_point_configurations(
    points_a: sp.Matrix | Iterable[Iterable[sp.Expr]],
    points_b: sp.Matrix | Iterable[Iterable[sp.Expr]],
) -> bool:
    """Compare unordered point configurations with the Sage backend (when available)."""
    _require_sage()
    SageMatrix(sp.Matrix(points_a).tolist())
    SageMatrix(sp.Matrix(points_b).tolist())

    # Until a native Sage implementation is wired up, fall back to the
    # brute-force sympy backend through the public API.
    from .affine_equivalence import is_point_config_equivalent

    return bool(is_point_config_equivalent(points_a, points_b).equivalent)
