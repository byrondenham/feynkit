"""
Custom exception class for feynkit.

Provides a hierarchy of exceptions for different error conditions
that can occur during Feynman integral computations.
"""


class FeynkitError(Exception):
    """Base exception class for all feynkit errors."""

    pass


class ValidationError(FeynkitError):
    """Raised when input validation fails."""

    pass


class GraphTopologyError(FeynkitError):
    """Raised when graph topology is invalid or inconsistent."""

    pass


class VertexIndexError(GraphTopologyError):
    """Raised when vertex indices are out of valid range."""

    def __init__(self, vertex: int, min_val: int, max_val: int, vertex_type: str = "vertex"):
        self.vertex = vertex
        self.min_val = min_val
        self.max_val = max_val
        self.vertex_type = vertex_type
        super().__init__(
            f"Invalid {vertex_type} index {vertex}: must be in range [{min_val}, {max_val}]"
        )


class EdgeConnectivityError(GraphTopologyError):
    """Raised when edge connectivity violates graph constraints."""

    pass


class ParameterError(FeynkitError):
    """Raised when parameters are missing or invalid."""

    pass


class ComputationError(FeynkitError):
    """Raised when a computation fails."""

    pass


class PolynomialError(ComputationError):
    """Raised when polynomial computation fails."""

    pass


class MatrixError(ComputationError):
    """Raised when matrix computation fails."""

    pass
