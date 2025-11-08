"""
Edge representation for Feynman graphs.

This module provides the Edge dataclass representing propagators and exteral momenta
in Feynman diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .constants import MASS_ASSUMPTIONS
from .exceptions import ValidationError


@dataclass(frozen=True)
class Edge:
    """
    An edge in a Feynman graph.

    Represents either an internal propagator or an external leg of a Feynman diagram.
    Edges connect vertices and carry quantum numbers like mass and propagator exponents.

    Attributes
    ----------
    idx : int
        Unique identifier for this edge (must be positive).
    v1 : int
        Starting vertex index (must be an internal vertex, >= 1).
    v2 : int
        Ending vertex index. For internal edges: must be internal vertex (>= 1).
    is_internal : bool
        True if this is an internal propagator, False if external leg.
    mass : Optional[sp.Expr], default None
        Mass of the propagator (symbolic expression). If None, will be generated
        automatically as m_{idx} when needed.
    nu : Optional[sp.Expr], default None
        Exponent parameter for the propagator (used in parametric representations).
        Typically represents the power of the propagator in dimensional regularisation.
        If None, defaults to 1.
    name: Optional[str], default None
        Optional human-readable label for this edge

    Examples
    --------
    >>> import sympy as sp
    >>> from feynkit.core import Edge
    >>>
    >>> # Internal propagator between vertices 1 and 2 with mass m and exponent nu
    >>> m = sp.Symbol('m', nonnegative=True)
    >>> nu = sp.Symbol('nu', positive=True)
    >>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m, nu=nu)
    >>>
    >>> # External leg from internal vertex 1 to external vertex 3
    >>> e_ext = Edge(idx=2, v1=1, v2=3, is_internal=False)
    >>>
    >>> # Edge with custom name
    >>> e2 = Edge(idx=3, v1=2, v2=1, is_internal=True, name="photon_propagator")

    Notes
    -----
    Edges are immutable (frozen=True) to ensure graph integrity. Once created,
    their attributes cannot be modified. To "modify" an edge, create a new one
    with updated values using dataclasses.replace().

    Thte vertex index convention follows:
    - Internal vertices: 1 to r_int
    - External vertices: (r_int + 1) to (r_int + n_ext)

    For self-loops (v1 == v2), special hangling may be needed in graph algorithms.

    References
    ----------
    .. [1] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    """

    idx: int
    v1: int
    v2: int
    is_internal: bool
    mass: sp.Expr | None = None
    nu: sp.Expr | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        """
        Validate edge parameters after initialisation.

        Raises
        ------
        ValidationError
            If idx is not positive, or if v1 or v2 are not positive integers.
        """
        if self.idx <= 0:
            raise ValidationError(f"Edge index must be positive, got {self.idx}")
        if self.v1 <= 0:
            raise ValidationError(f"Starting vertex v1 must be positive, got {self.v1}")
        if self.v2 <= 0:
            raise ValidationError(f"Ending vertex v2 must be positive, got {self.v2}")

    def get_mass(self) -> sp.Expr:
        """
        Get the mass of this edge, creating a symbolic mass if needed.

        Returns
        -------
        sp.Expr
            The mass expression. If mass was None, returns a symbolic mass m_{idx}.

        Examples
        --------
        >>> e = Edge(idx=1, v1=1, v2=2, is_internal=True)
        >>> m = e.get_mass()
        >>> print(m)
        m_1
        """
        if self.mass is not None:
            return self.mass
        return sp.Symbol(f"m_{self.idx}", **MASS_ASSUMPTIONS)

    def get_exponent(self) -> sp.Expr:
        """
        Get the propagator exponent, defaulting to 1 if not specified.

        Returns
        -------
        sp.Expr
            The exponent expression. If nu was None, returns 1.

        Examples
        --------
        >>> e = Edge(idx=1, v1=1, v2=2, is_internal=True)
        >>> nu = e.get_exponent()
        >>> print(nu)
        1
        """
        if self.nu is not None:
            return self.nu
        return sp.Integer(1)

    def is_self_loop(self) -> bool:
        """
        Check if this edge is a self-loop (tadpole).

        Returns
        -------
        bool
            True if v1 == v2, False otherwise.

        Examples
        --------
        >>> e_loop = Edge(idx=1, v1=1, v2=1, is_internal=True)
        >>> e_loop.is_self_loop()
        True
        >>> e_normal = Edge(idx=2, v1=1, v2=2, is_internal=True)
        >>> e_normal.is_self_loop()
        False
        """
        return self.v1 == self.v2

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the edge.

        Returns
        -------
        str
            String representation showing all edge attributes.
        """
        edge_type = "internal" if self.is_internal else "external"
        parts = [
            f"Edge(idx={self.idx}",
            f"v1={self.v1}",
            f"v2={self.v2}",
            f"type={edge_type}",
        ]
        if self.mass is not None:
            parts.append(f"mass={self.mass}")
        if self.nu is not None:
            parts.append(f"nu={self.nu}")
        if self.name is not None:
            parts.append(f"name='{self.name}'")
        return ", ".join(parts) + ")"

    def __str__(self) -> str:
        """
        Return a concise string representation of the edge.

        Returns
        -------
        str
            Short string representation.
        """
        if self.name:
            return f"Edge {self.idx} ({self.name}): {self.v1} → {self.v2}"
        return f"Edge {self.idx}: {self.v1} → {self.v2}"
