"""
Base class for parametric representations of Feynman integrals.

Provides an abstract interface that all parametrisation implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import sympy as sp


@dataclass
class ParametrisationResult:
    """
    Result of a parametric representation computation.

    Attributes
    ----------
    name : str
        Name of the parametrisation (e.g., "Schwinger", "Feynman", "Lee-Pomeransky").
    prefactor : sp.Expr
        Overall prefactor including gamma functions and regularisation factors.
    measure : sp.Expr
        Integration measure (product of parameter differentials and weights).
    integrand : sp.Expr
        The integrand expression to be integrated.
    parameters : List[sp.Symbol]
        List of integration parameters.
    constraints : List[sp.Expr]
        List of constraints on parameters (e.g., delta functions, positivity).
    description : str
        Human-readable description of the parametrisation.
    """

    name: str
    prefactor: sp.Expr
    measure: sp.Expr
    integrand: sp.Expr
    parameters: list[sp.Symbol]
    constraints: list[sp.Expr]
    description: str

    def __str__(self) -> str:
        """Return a readable string representation."""
        lines = [
            f"{self.name} Parametrisation",
            "=" * 60,
            f"Description: {self.description}",
            f"\nParameters: {', '.join(str(p) for p in self.parameters)}",
            f"\nPrefactor:\n  {self.prefactor}",
            f"\nMeasure:\n  {self.measure}",
            f"\nIntegrand:\n  {self.integrand}",
        ]
        if self.constraints:
            lines.append(f"\nConstraints:\n  {', '.join(str(c) for c in self.constraints)}")
        return "\n".join(lines)


class Parametrisation(ABC):
    """
    Abstract base class for parametric representations.

    All parametrisation implementations (Schwinger, Feynman, Lee-Pomeransky)
    inherit from this class and implement the `compute()` method.
    """

    @abstractmethod
    def compute(self) -> ParametrisationResult:
        """
        Compute the parametric representation.

        Returns
        -------
        ParametrisationResult
            Complete parametric representation with all components.

        Raises
        ------
        ComputationError
            If the computation fails.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this parametrisation."""
        pass
