"""Test validation failure scenarios for PolynomialDescriptor."""

import pytest
from pyetsimul.types.algorithms import PolynomialDescriptor
from pyetsimul.gaze_tracking_algorithms.interpolate.polynomials import register_polynomial


def test_invalid_order_format():
    """Test that invalid order format is caught during validation."""
    with pytest.raises((ValueError, TypeError)):
        bad_descriptor = PolynomialDescriptor(
            name="bad_order",
            description="Polynomial with invalid order",
            terms=["x", "y", "1"],
            orders=[1, "bad", 0],  # "bad" is not int or list
        )
        register_polynomial(bad_descriptor)


def test_mismatched_lengths():
    """Test that mismatched terms and orders lengths are caught."""
    with pytest.raises((ValueError, IndexError)):
        PolynomialDescriptor(
            name="bad_length",
            description="Mismatched lengths",
            terms=["x", "y"],  # 2 terms
            orders=[1, 1, 0],  # 3 orders
        )


def test_duplicate_registration():
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
    with pytest.raises(ValueError, match="already registered"):
        duplicate_descriptor = PolynomialDescriptor(
            name="duplicate_test",  # Same name
            description="Second registration",
            terms=["x", "1"],
            orders=[1, 0],
        )
        register_polynomial(duplicate_descriptor)
