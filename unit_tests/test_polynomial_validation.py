"""Test validation failure scenarios for PolynomialDescriptor."""

import pytest

from pyetsimul.gaze_models.polynomial import PolynomialDescriptor
from pyetsimul.gaze_models.polynomial.polynomials import register_polynomial


def test_invalid_order_format() -> None:
    """Test that invalid order format is caught during validation."""
    bad_descriptor = PolynomialDescriptor(
        name="bad_order",
        description="Polynomial with invalid order",
        terms=["x", "y", "1"],
        orders=[1, "bad", 0],  # "bad" is not int or list
    )
    with pytest.raises((ValueError, TypeError)):
        register_polynomial(bad_descriptor)


def test_mismatched_lengths() -> None:
    """Test that mismatched terms and orders lengths are caught."""
    with pytest.raises((ValueError, IndexError)):
        PolynomialDescriptor(
            name="bad_length",
            description="Mismatched lengths",
            terms=["x", "y"],  # 2 terms
            orders=[1, 1, 0],  # 3 orders
        )


def test_duplicate_registration() -> None:
    """Test that duplicate polynomial names are rejected."""
    # First registration should work
    good_descriptor = PolynomialDescriptor(
        name="duplicate_test",
        description="First registration",
        terms=["x", "y", "1"],
        orders=[1, 1, 0],
    )
    register_polynomial(good_descriptor)

    # Second registration should fail
    duplicate_descriptor = PolynomialDescriptor(
        name="duplicate_test",  # Same name
        description="Second registration",
        terms=["x", "1"],
        orders=[1, 0],
    )
    with pytest.raises(ValueError, match="already registered"):
        register_polynomial(duplicate_descriptor)
