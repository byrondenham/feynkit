"""
Physical constants and package-wide configuration.

This module defines constants used throughout feynkit for consistency
and ease of maintenance.
"""

from typing import Final

# Package metadata
__version__: Final[str] = "0.3.0"
__author__: Final[str] = "Byron Denham"
__email__: Final[str] = "bd567@bath.ac.uk"

# Default symbol names
DEFAULT_ENERGY_SCALE: Final[str] = "mu"
DEFAULT_DIMENSION: Final[str] = "D"
DEFAULT_EPSILON: Final[str] = "epsilon"
DEFAULT_GAMMA_E: Final[str] = "gamma_E"

# Parameter naming conventions
SCHWINGER_PARAM_PREFIX: Final[str] = "a"
EXTERNAL_PARAM_PREFIX: Final[str] = "b"
ALPHA_PARAM_PREFIX: Final[str] = "alpha"
LEE_POMERANSKY_PARAM_PREFIX: Final[str] = "u"

# Default assumptions for SymPy symbols
MASS_ASSUMPTIONS: Final[dict[str, bool]] = {"nonnegative": True, "real": True}
EXPONENT_ASSUMPTIONS: Final[dict[str, bool]] = {"positive": True, "real": True}
SCALE_ASSUMPTIONS: Final[dict[str, bool]] = {"positive": True, "real": True}
DIMENSION_ASSUMPTIONS: Final[dict[str, bool]] = {"positive": True, "real": True}

# Numerical tolerances
NUMERICAL_ZERO_TOLERANCE: Final[float] = 1e-10
COORDINATE_FORMAT_PRECISION: Final[int] = 10

# Vertex indexing conventions
# Internal vertices are indexed from 1 to r_int
# External vertices are indexed from (r_int + 1) to (r_int + n_ext)
MIN_INTERNAL_VERTEX: Final[int] = 1
