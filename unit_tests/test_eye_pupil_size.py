"""Unit tests for Eye pupil radii methods."""

import numpy as np
import pytest
from et_simul.core.eye import Eye
from et_simul.types import Position3D


def test_get_pupil_radii_default():
    """Test get_pupil_radii returns correct default radii."""
    e = Eye()
    x_radius, y_radius = e.get_pupil_radii()

    # Default pupil radius is 3mm for both axes
    expected_radius = 3e-3
    assert np.isclose(x_radius, expected_radius, rtol=1e-12)
    assert np.isclose(y_radius, expected_radius, rtol=1e-12)


def test_set_get_pupil_radii():
    """Test set_pupil_radii and get_pupil_radii work together correctly."""
    e = Eye()

    # Test setting both radii
    e.set_pupil_radii(x_radius=3e-3, y_radius=2e-3)
    x_radius, y_radius = e.get_pupil_radii()
    assert np.isclose(x_radius, 3e-3, rtol=1e-12)
    assert np.isclose(y_radius, 2e-3, rtol=1e-12)

    # Test setting only x_radius
    e.set_pupil_radii(x_radius=4e-3)
    x_radius, y_radius = e.get_pupil_radii()
    assert np.isclose(x_radius, 4e-3, rtol=1e-12)
    assert np.isclose(y_radius, 2e-3, rtol=1e-12)  # Should remain unchanged

    # Test setting only y_radius
    e.set_pupil_radii(y_radius=1e-3)
    x_radius, y_radius = e.get_pupil_radii()
    assert np.isclose(x_radius, 4e-3, rtol=1e-12)  # Should remain unchanged
    assert np.isclose(y_radius, 1e-3, rtol=1e-12)


def test_set_pupil_radii_error_handling():
    """Test that set_pupil_radii raises error when both radii are None."""
    e = Eye()

    with pytest.raises(ValueError, match="At least one radius must be specified"):
        e.set_pupil_radii()


def test_set_pupil_radii_updates_vectors():
    """Test that set_pupil_radii correctly updates x_pupil and y_pupil vectors."""
    e = Eye()
    x_radius = 3e-3  # 3mm
    y_radius = 2e-3  # 2mm

    e.set_pupil_radii(x_radius=x_radius, y_radius=y_radius)

    # Check that vectors have correct magnitude
    x_magnitude = np.linalg.norm(np.array(e.pupil.x_pupil)[:3])
    y_magnitude = np.linalg.norm(np.array(e.pupil.y_pupil)[:3])

    assert np.isclose(x_magnitude, x_radius, rtol=1e-12)
    assert np.isclose(y_magnitude, y_radius, rtol=1e-12)

    # Check vector directions
    expected_x = np.array([x_radius, 0, 0, 0])
    expected_y = np.array([0, y_radius, 0, 0])

    e.pupil.x_pupil.assert_close(Position3D.from_array(expected_x), rtol=1e-12)
    e.pupil.y_pupil.assert_close(Position3D.from_array(expected_y), rtol=1e-12)
