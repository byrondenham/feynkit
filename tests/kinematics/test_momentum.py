"""Tests for momentum product creation."""

import pytest
import sympy as sp

from feynkit.core.exceptions import ValidationError
from feynkit.kinematics import (
    create_momentum_products,
    get_momentum_product,
    validate_momentum_products_complete,
)


class TestCreateMomentumProducts:
    """Test momentum product creation."""

    def test_generic_products_two_legs(self) -> None:
        """Test generic momentum products for 2 external legs."""
        p_dot = create_momentum_products(n_external=2, use_mandelstam=False)

        assert len(p_dot) == 1
        assert (1, 2) in p_dot
        assert str(p_dot[(1, 2)]) == "p1p2"

    def test_generic_products_three_legs(self) -> None:
        """Test generic momentum products for 3 external legs."""
        p_dot = create_momentum_products(n_external=3, use_mandelstam=False)

        assert len(p_dot) == 3
        assert (1, 2) in p_dot
        assert (1, 3) in p_dot
        assert (2, 3) in p_dot

    def test_mandelstam_two_legs(self) -> None:
        """Test Mandelstam variables for 2 external legs."""
        p_dot = create_momentum_products(n_external=2, use_mandelstam=True)

        assert len(p_dot) == 1
        assert (1, 2) in p_dot

        # Should be s/2
        expr = p_dot[(1, 2)]
        expected = sp.Symbol("s", real=True) / 2
        assert sp.simplify(expr - expected) == 0

    def test_mandelstam_three_legs(self) -> None:
        """Test Mandelstam variables for 3 external legs."""
        p_dot = create_momentum_products(n_external=3, use_mandelstam=True)

        assert len(p_dot) == 3
        assert (1, 2) in p_dot
        assert (1, 3) in p_dot
        assert (2, 3) in p_dot

        # Should have s12, s13, s23 divided by 2
        s12_expected = sp.Symbol("s12", real=True) / 2
        s13_expected = sp.Symbol("s13", real=True) / 2
        s23_expected = sp.Symbol("s23", real=True) / 2

        assert sp.simplify(p_dot[(1, 2)] - s12_expected) == 0
        assert sp.simplify(p_dot[(1, 3)] - s13_expected) == 0
        assert sp.simplify(p_dot[(2, 3)] - s23_expected) == 0

    def test_invalid_n_external(self) -> None:
        """Test that invalid n_external raises error."""
        with pytest.raises(ValidationError):
            create_momentum_products(n_external=0, use_mandelstam=False)

        with pytest.raises(ValidationError):
            create_momentum_products(n_external=-1, use_mandelstam=False)


class TestGetMomentumProduct:
    """Test momentum product retrieval."""

    def test_get_existing_product(self) -> None:
        """Test retrieving an existing momentum product."""
        p_dot = create_momentum_products(n_external=3, use_mandelstam=False)
        prod = get_momentum_product(p_dot, 1, 2)

        expected = sp.Symbol("p1p2", real=True)
        assert sp.simplify(prod - expected) == 0

    def test_get_product_reversed_order(self) -> None:
        """Test retrieving product with reversed indices."""
        p_dot = create_momentum_products(n_external=3, use_mandelstam=False)
        prod = get_momentum_product(p_dot, 2, 1)

        # Should find (1, 2)
        expected = sp.Symbol("p1p2", real=True)
        assert sp.simplify(prod - expected) == 0

    def test_get_nonexistent_product(self) -> None:
        """Test that requesting nonexistent product raises error."""
        p_dot = create_momentum_products(n_external=2, use_mandelstam=False)

        with pytest.raises(ValidationError, match="not found"):
            get_momentum_product(p_dot, 1, 3)


class TestValidateMomentumProducts:
    """Test momentum product validation."""

    def test_validate_complete_products(self) -> None:
        """Test validating complete set of momentum products."""
        p_dot = create_momentum_products(n_external=3, use_mandelstam=False)
        # Should not raise
        validate_momentum_products_complete(p_dot, n_external=3)

    def test_validate_incomplete_products(self) -> None:
        """Test validating incomplete set raises error."""
        p_dot = {(1, 2): sp.Symbol("p1p2")}  # Missing (1, 3) and (2, 3)

        with pytest.raises(ValidationError, match="Missing momentum products"):
            validate_momentum_products_complete(p_dot, n_external=3)
