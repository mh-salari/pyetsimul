"""Unit tests for find_refraction_sphere function."""

import numpy as np

from pyetsimul.optics.refractions import find_refraction_sphere
from pyetsimul.types import Point3D, Position3D


def test_basic_refraction() -> None:
    """Test basic refraction scenario with MATLAB reference values."""
    # Define sphere
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 10000.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    # Object inside sphere, camera outside
    object_pos = Position3D(2000.0, 1000.0, 3000.0)
    camera_pos = Position3D(15000.0, 8000.0, 12000.0)

    intersection_point = find_refraction_sphere(
        camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
    )

    # MATLAB reference values
    expected_point = Point3D(6693.6966464330183, 3513.2501989855691, 6546.1055784992670)

    assert intersection_point is not None
    intersection_point.assert_close(expected_point, rtol=1e-10, atol=1e-10)

    # Verify point is on sphere surface
    dist_from_center = np.linalg.norm(np.array(intersection_point) - np.array(sphere_center)[:3])
    np.testing.assert_allclose(dist_from_center, sphere_radius, rtol=1e-15, atol=1e-15)


def test_different_refractive_indices() -> None:
    """Test refraction with different refractive indices and MATLAB reference values."""
    # Define sphere
    sphere_center = Position3D(0.0, 0.0, 0.0)
    sphere_radius = 10000.0
    n_outside = 1.33  # Water
    n_sphere = 1.4  # Different glass

    # Different positions
    object_pos = Position3D(-1000.0, 2000.0, -2000.0)
    camera_pos = Position3D(-8000.0, 10000.0, -15000.0)

    intersection_point = find_refraction_sphere(
        camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
    )

    # MATLAB reference values
    expected_point = Point3D(-3936.9221058636397, 5374.8224021012501, -7457.3405767896075)

    assert intersection_point is not None
    intersection_point.assert_close(expected_point, rtol=1e-10, atol=1e-10)

    # Verify point is on sphere surface
    dist_from_center = np.linalg.norm(np.array(intersection_point) - np.array(sphere_center)[:3])
    np.testing.assert_allclose(dist_from_center, sphere_radius, rtol=1e-15, atol=1e-15)


def test_large_sphere() -> None:
    """Test refraction with large sphere and MATLAB reference values."""
    # Define large sphere
    sphere_center = Position3D(0.0, 0.0, 0.0)
    sphere_radius = 50000.0  # Large radius
    n_outside = 1.0
    n_sphere = 1.5

    # Scaled positions
    object_pos = Position3D(5000.0, -3000.0, 8000.0)
    camera_pos = Position3D(80000.0, -20000.0, 60000.0)

    intersection_point = find_refraction_sphere(
        camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
    )

    # MATLAB reference values
    expected_point = Point3D(37369.8407698170811, -10744.8391945206190, 31433.1581538096536)

    assert intersection_point is not None
    intersection_point.assert_close(expected_point, rtol=1e-10, atol=1e-10)

    # Verify point is on sphere surface
    dist_from_center = np.linalg.norm(np.array(intersection_point) - np.array(sphere_center)[:3])
    np.testing.assert_allclose(dist_from_center, sphere_radius, rtol=1e-15, atol=1e-15)


def test_snells_law_verification() -> None:
    """Test that solution satisfies Snell's law with MATLAB reference values."""
    # Use same setup as basic test
    sphere_center = Position3D(0.0, 0.0, 0.0)
    sphere_radius = 10000.0
    n_outside = 1.0
    n_sphere = 1.5
    object_pos = Position3D(2000.0, 1000.0, 3000.0)
    camera_pos = Position3D(15000.0, 8000.0, 12000.0)

    intersection_point = find_refraction_sphere(
        camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
    )
    assert intersection_point is not None

    # Compute vectors
    n_surface = (np.array(intersection_point) - np.array(sphere_center)[:3]) / np.linalg.norm(
        np.array(intersection_point) - np.array(sphere_center)[:3]
    )
    ray_incident = (np.array(intersection_point) - np.array(object_pos)[:3]) / np.linalg.norm(
        np.array(intersection_point) - np.array(object_pos)[:3]
    )
    ray_refracted = (np.array(camera_pos)[:3] - np.array(intersection_point)) / np.linalg.norm(
        np.array(camera_pos)[:3] - np.array(intersection_point)
    )

    # MATLAB reference values
    expected_n_surface = np.array([0.669370, 0.351325, 0.654611])
    expected_ray_incident = np.array([0.733730, 0.392877, 0.554336])
    expected_ray_refracted = np.array([0.761852, 0.411524, 0.500230])

    np.testing.assert_allclose(n_surface, expected_n_surface, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ray_incident, expected_ray_incident, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ray_refracted, expected_ray_refracted, rtol=1e-5, atol=1e-5)

    # Verify angles and Snell's law
    cos_theta_i = np.dot(n_surface, ray_incident)
    cos_theta_r = np.dot(n_surface, ray_refracted)
    sin_theta_i = np.sqrt(1 - cos_theta_i**2)
    sin_theta_r = np.sqrt(1 - cos_theta_r**2)

    # Check Snell's law: n1*sin(theta1) = n2*sin(theta2)
    snell_left = n_sphere * sin_theta_i
    snell_right = n_outside * sin_theta_r
    snell_diff = abs(snell_left - snell_right)

    assert snell_diff < 1e-10


def test_output_properties() -> None:
    """Test that output has correct properties."""
    sphere_center = Position3D(0.0, 0.0, 0.0)
    sphere_radius = 10000.0
    n_outside = 1.0
    n_sphere = 1.5
    object_pos = Position3D(2000.0, 1000.0, 3000.0)
    camera_pos = Position3D(15000.0, 8000.0, 12000.0)

    intersection_point = find_refraction_sphere(
        camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
    )
    assert intersection_point is not None, "intersection_point should not be None for these inputs"
    assert not np.any(np.isnan(np.array(intersection_point))), "intersection_point should not contain NaN values"

    # Check types and shapes
    assert isinstance(intersection_point, Point3D)
    intersection_array = np.array(intersection_point)
    assert intersection_array.shape == (3,)
    assert intersection_array.dtype == np.float64
