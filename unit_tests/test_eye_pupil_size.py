"""Unit tests for Eye pupil radii methods."""

import numpy as np
import pytest
from et_simul.core.eye import Eye


def test_get_pupil_radii_default():
    """Test get_pupil_radii returns correct default radii."""
    e = Eye()
    across_radius, up_radius = e.get_pupil_radii()
    
    # Default pupil radius is 3mm for both axes
    expected_radius = 3e-3
    assert np.isclose(across_radius, expected_radius, rtol=1e-12)
    assert np.isclose(up_radius, expected_radius, rtol=1e-12)


def test_set_get_pupil_radii():
    """Test set_pupil_radii and get_pupil_radii work together correctly."""
    e = Eye()
    
    # Test setting both radii
    e.set_pupil_radii(across_radius=3e-3, up_radius=2e-3)
    across_radius, up_radius = e.get_pupil_radii()
    assert np.isclose(across_radius, 3e-3, rtol=1e-12)
    assert np.isclose(up_radius, 2e-3, rtol=1e-12)
    
    # Test setting only across_radius
    e.set_pupil_radii(across_radius=4e-3)
    across_radius, up_radius = e.get_pupil_radii()
    assert np.isclose(across_radius, 4e-3, rtol=1e-12)
    assert np.isclose(up_radius, 2e-3, rtol=1e-12)  # Should remain unchanged
    
    # Test setting only up_radius
    e.set_pupil_radii(up_radius=1e-3)
    across_radius, up_radius = e.get_pupil_radii()
    assert np.isclose(across_radius, 4e-3, rtol=1e-12)  # Should remain unchanged
    assert np.isclose(up_radius, 1e-3, rtol=1e-12)


def test_set_pupil_radii_error_handling():
    """Test that set_pupil_radii raises error when both radii are None."""
    e = Eye()
    
    with pytest.raises(ValueError, match="At least one radius must be specified"):
        e.set_pupil_radii()


def test_set_pupil_radii_updates_vectors():
    """Test that set_pupil_radii correctly updates across_pupil and up_pupil vectors."""
    e = Eye()
    across_radius = 3e-3  # 3mm
    up_radius = 2e-3      # 2mm
    
    e.set_pupil_radii(across_radius=across_radius, up_radius=up_radius)
    
    # Check that vectors have correct magnitude
    across_magnitude = np.linalg.norm(e.across_pupil[:3])
    up_magnitude = np.linalg.norm(e.up_pupil[:3])
    
    assert np.isclose(across_magnitude, across_radius, rtol=1e-12)
    assert np.isclose(up_magnitude, up_radius, rtol=1e-12)
    
    # Check vector directions
    expected_across = np.array([across_radius, 0, 0, 0])
    expected_up = np.array([0, up_radius, 0, 0])
    
    np.testing.assert_allclose(e.across_pupil, expected_across, rtol=1e-12)
    np.testing.assert_allclose(e.up_pupil, expected_up, rtol=1e-12)