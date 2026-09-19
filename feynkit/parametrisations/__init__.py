"""
Parametric representations module for Feynman integrals.

The Schwinger, Feynman, and Lee-Pomeransky parametric representations of an
integral are accessed through :attr:`feynkit.FeynmanIntegral.schwinger`,
``.feynman``, and ``.lee_pomeransky`` respectively. This module exposes the
underlying classes for users that wish to construct or extend the
parametrisation machinery directly.

Classes
-------
Parametrisation
    Abstract base class for all parametrisations.
ParametrisationResult
    Container for parametrisation computation results.
SchwingerParametrisation
    Schwinger parametrisation implementation.
FeynmanParametrisation
    Feynman parametrisation implementation.
LeePomeranskyParametrisation
    Lee-Pomeransky parametrisation implementation.
AllParametrisations
    Container holding all three parametrisations.
"""

from .base import Parametrisation, ParametrisationResult
from .factory import AllParametrisations
from .feynman import FeynmanParametrisation
from .lee_pomeransky import LeePomeranskyParametrisation
from .schwinger import SchwingerParametrisation

__all__ = [
    "Parametrisation",
    "ParametrisationResult",
    "SchwingerParametrisation",
    "FeynmanParametrisation",
    "LeePomeranskyParametrisation",
    "AllParametrisations",
]
