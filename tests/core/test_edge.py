"""Tests for the Edge class."""

import pytest
import sympy as sp

from feynkit.core import Edge
from feynkit.core.exceptions import ValidationError


class TestEdgeCreation:
    """Test Edge instantiation and validation."""

    def test_create_internal_edge_with_all_parameters(self) -> None:
        """Test creating an internal edge with all parameters specified."""
        m = sp.Symbol("m", nonnegative=True)
        nu = sp.Symbol("nu", positive=True)

        edge = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m, nu=nu, name="photon")

        assert edge.idx == 1
        assert edge.v1 == 1
        assert edge.v2 == 2
        assert edge.is_internal is True
        assert edge.mass == m
        assert edge.nu == nu
        assert edge.name == "photon"

    def test_create_external_edge_minimal(self) -> None:
        """Test creating an external edge with minimal parameters."""
        edge = Edge(idx=2, v1=1, v2=3, is_internal=False)

        assert edge.idx == 2
        assert edge.v1 == 1
        assert edge.v2 == 3
        assert edge.is_internal is False
        assert edge.mass is None
        assert edge.nu is None
        assert edge.name is None

    def test_edge_is_immutable(self) -> None:
        """Test that Edge is immutable (frozen dataclass)."""
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True)

        with pytest.raises(AttributeError):
            edge.idx = 5  # type: ignore

    def test_create_edge_with_invalid_idx(self) -> None:
        """Test that creating an edge with non-positive idx raises error."""
        with pytest.raises(ValidationError, match="Edge index must be positive"):
            Edge(idx=0, v1=1, v2=2, is_internal=True)

        with pytest.raises(ValidationError, match="Edge index must be positive"):
            Edge(idx=-1, v1=1, v2=2, is_internal=True)

    def test_create_edge_with_invalid_v1(self) -> None:
        """Test that creating an edge with non-positive v1 raises error."""
        with pytest.raises(ValidationError, match="Starting vertex v1 must be positive"):
            Edge(idx=1, v1=0, v2=2, is_internal=True)

    def test_create_edge_with_invalid_v2(self) -> None:
        """Test that creating an edge with non-positive v2 raises error."""
        with pytest.raises(ValidationError, match="Ending vertex v2 must be positive"):
            Edge(idx=1, v1=1, v2=0, is_internal=True)


class TestEdgeMethods:
    """Test Edge methods."""

    def test_get_mass_with_specified_mass(self) -> None:
        """Test get_mass() when mass is specified."""
        m = sp.Symbol("m", nonnegative=True)
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m)

        result = edge.get_mass()
        assert result == m

    def test_get_mass_with_default_mass(self) -> None:
        """Test get_mass() when mass is not specified (creates m_idx)."""
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True)

        result = edge.get_mass()
        assert isinstance(result, sp.Symbol)
        assert str(result) == "m_1"

    def test_get_exponent_with_specified_nu(self) -> None:
        """Test get_exponent() when nu is specified."""
        nu = sp.Symbol("nu", positive=True)
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True, nu=nu)

        result = edge.get_exponent()
        assert result == nu

    def test_get_exponent_with_default_nu(self) -> None:
        """Test get_exponent() when nu is not specified (defaults to 1)."""
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True)

        result = edge.get_exponent()
        assert result == sp.Integer(1)

    def test_is_self_loop_true(self) -> None:
        """Test is_self_loop() for a tadpole."""
        edge = Edge(idx=1, v1=1, v2=1, is_internal=True)

        assert edge.is_self_loop() is True

    def test_is_self_loop_false(self) -> None:
        """Test is_self_loop() for a normal edge."""
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True)

        assert edge.is_self_loop() is False


class TestEdgeStringRepresentation:
    """Test Edge string representations."""

    def test_repr_internal_edge_full(self) -> None:
        """Test __repr__ for internal edge with all parameters."""
        m = sp.Symbol("m")
        nu = sp.Symbol("nu")
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m, nu=nu, name="test")

        repr_str = repr(edge)
        assert "idx=1" in repr_str
        assert "v1=1" in repr_str
        assert "v2=2" in repr_str
        assert "type=internal" in repr_str
        assert "mass=m" in repr_str
        assert "nu=nu" in repr_str
        assert "name='test'" in repr_str

    def test_str_with_name(self) -> None:
        """Test __str__ for edge with name."""
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True, name="photon")

        str_repr = str(edge)
        assert str_repr == "Edge 1 (photon): 1 → 2"

    def test_str_without_name(self) -> None:
        """Test __str__ for edge without name."""
        edge = Edge(idx=1, v1=1, v2=2, is_internal=True)

        str_repr = str(edge)
        assert str_repr == "Edge 1: 1 → 2"
