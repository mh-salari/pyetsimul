"""Unit tests for Camera.project method."""

import numpy as np
from et_simul.core.camera import Camera


def test_mixed_points_in_out_bounds():
    """Test mixed points with some in bounds and some out of bounds."""
    # Create test camera matching parameters
    c = Camera(focal_length=500, resolution=[640, 480], err_type='uniform', err=0)
    
    # Define test points (homogeneous coordinates format)
    pos = np.array([
        [0, 0, 50],          # x coordinates
        [100, -100, 50],     # y coordinates  
        [-200, -200, -300],  # z coordinates
        [1, 1, 1]            # Homogeneous coordinates
    ])
    
    x, dist, valid_condition = c.project(pos)
    
    # Expected distances
    expected_dist = np.array([200.000000, 200.000000, 300.000000])
    
    # Check that invalid points are set to NaN (points 1 and 2 are out of bounds)
    assert np.isnan(x[0, 0]) and np.isnan(x[1, 0])
    assert np.isnan(x[0, 1]) and np.isnan(x[1, 1])
    
    # Check valid point projection (point 3)
    np.testing.assert_allclose(x[0, 2], 83.333333, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1, 2], 83.333333, rtol=1e-6, atol=1e-6)
    
    # Check distances
    np.testing.assert_allclose(dist, expected_dist, rtol=1e-6, atol=1e-6)
    
    # Check valid condition (only point 3 should be valid)
    valid_indices = np.where(valid_condition)[0]
    expected_valid_indices = [2]
    np.testing.assert_array_equal(valid_indices, expected_valid_indices)


def test_all_points_outside_bounds():
    """Test points that are all outside image bounds."""
    c = Camera(focal_length=500, resolution=[640, 480], err_type='uniform', err=0)
    
    # Define test points that are clearly out of bounds
    pos = np.array([
        [1000, 0, -1000],    # x coordinates  
        [0, 1000, -1000],    # y coordinates
        [-200, -200, -200],  # z coordinates
        [1, 1, 1]            # Homogeneous coordinates
    ])
    
    x, dist, valid_condition = c.project(pos)
    
    expected_dist = np.array([200.000000, 200.000000, 200.000000])
    
    # Check that all points are set to NaN
    assert np.isnan(x).all()
    
    # Check distances
    np.testing.assert_allclose(dist, expected_dist, rtol=1e-6, atol=1e-6)
    
    # Check valid condition (no points should be valid)
    valid_indices = np.where(valid_condition)[0]
    assert len(valid_indices) == 0


def test_boundary_points():
    """Test points exactly on image boundary."""
    c = Camera(focal_length=500, resolution=[640, 480], err_type='uniform', err=0)
    
    # Calculate boundary points
    boundary_x = c.resolution[0] / 2  # 320
    boundary_y = c.resolution[1] / 2  # 240
    z_depth = -200
    
    x_right = boundary_x * z_depth / c.focal_length
    y_top = boundary_y * z_depth / c.focal_length
    x_left = -boundary_x * z_depth / c.focal_length
    y_bottom = -boundary_y * z_depth / c.focal_length
    
    pos = np.array([
        [x_right, 0, x_left, 0],           # x coordinates
        [0, y_top, 0, y_bottom],           # y coordinates  
        [z_depth, z_depth, z_depth, z_depth], # z coordinates
        [1, 1, 1, 1]                       # Homogeneous coordinates
    ])
    
    x, dist, valid_condition = c.project(pos)
    
    # Expected projection values
    expected_x = np.array([
        [-320.0, 0.0, 320.0, 0.0],
        [0.0, -240.0, 0.0, 240.0]
    ])
    
    # Check projected points
    np.testing.assert_allclose(x, expected_x, rtol=1e-6, atol=1e-6)
    
    # Check that all boundary points are valid
    valid_indices = np.where(valid_condition)[0]
    expected_valid_indices = [0, 1, 2, 3]
    np.testing.assert_array_equal(valid_indices, expected_valid_indices)


def test_center_point():
    """Test point at image center."""
    c = Camera(focal_length=500, resolution=[640, 480], err_type='uniform', err=0)
    
    # Point at center
    pos = np.array([
        [0],    # x coordinate
        [0],    # y coordinate  
        [-200], # z coordinate
        [1]     # Homogeneous coordinate
    ])
    
    x, dist, valid_condition = c.project(pos)
    
    # Should project to (0, 0)
    np.testing.assert_allclose(x[0, 0], 0.0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x[1, 0], 0.0, rtol=1e-12, atol=1e-15)
    
    # Distance should be 200
    np.testing.assert_allclose(dist[0], 200.0, rtol=1e-12, atol=1e-15)
    
    # Should be valid
    assert valid_condition[0] == True


def test_output_properties():
    """Test that output has correct properties."""
    c = Camera(focal_length=500, resolution=[640, 480], err_type='uniform', err=0)
    
    # Simple test points
    pos = np.array([
        [0, 100],     # x coordinates
        [0, 0],       # y coordinates  
        [-200, -300], # z coordinates
        [1, 1]        # Homogeneous coordinates
    ])
    
    x, dist, valid_condition = c.project(pos)
    
    # Check types and shapes
    assert isinstance(x, np.ndarray)
    assert isinstance(dist, np.ndarray)
    assert isinstance(valid_condition, np.ndarray)
    assert x.shape == (2, 2)
    assert dist.shape == (2,)
    assert valid_condition.shape == (2,)
    assert x.dtype == np.float64
    assert dist.dtype == np.float64
    assert valid_condition.dtype == bool