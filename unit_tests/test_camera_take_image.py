"""Unit tests for Camera.take_image method."""

import numpy as np

from pyetsimul.core.camera import Camera
from pyetsimul.core.eye import Eye
from pyetsimul.core.light import Light
from pyetsimul.types import CameraImage, Point2D, Position3D


def test_camera_take_image_with_refraction() -> None:
    """Test camera take_image with refraction using actual MATLAB reference values."""
    # Create eye and camera setup (matching MATLAB test)
    e = Eye(fovea_displacement=False)
    e.position = Position3D(0, 500e-3, 200e-3)  # Eye at [0, 500mm, 200mm]

    # Camera at origin pointing at eye
    c = Camera()
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.position)  # Point at eye
    c.err = 0
    c.err_type = "uniform"

    # Eye looks back at camera (mutual gaze)
    e.look_at(Position3D(0, 0, 0))

    # Create light sources (typical eye tracker setup)
    lights = []

    # Light 1: right side, low (4D homogeneous)
    light1 = Light(position=Position3D(200e-3, 0, 50e-3))
    lights.append(light1)

    # Light 2: right side, high (4D homogeneous)
    light2 = Light(position=Position3D(200e-3, 0, 300e-3))
    lights.append(light2)

    # Take image with refraction
    camimg = c.take_image(e, lights, use_refraction=True)

    # Correct MATLAB reference values from the provided script output
    expected_cr = [
        Point2D(8.0091797595, 1.8590841799),  # Light 1 CR
        Point2D(8.5048679096, 11.8448574197),  # Light 2 CR
    ]
    expected_pc = Point2D(0.0, 0.0)  # MATLAB: [-0.0000, 0.0000]
    expected_pupil_points = np.array([
        [18.2201323223, 0.0000],
        [17.3283755729, -5.6303305273],
        [14.7403966885, -10.7095250739],
        [10.7095250739, -14.7403966885],
        [5.6303305273, -17.3283755729],
    ]).T  # Transpose to match 2xM format

    # Using a slightly larger tolerance for cross-language floating point comparisons
    tolerance = 1e-5

    # Test corneal reflexes
    assert len(camimg.corneal_reflections) == 2, "Should have 2 corneal reflexes"
    for i, (expected, actual) in enumerate(zip(expected_cr, camimg.corneal_reflections, strict=False)):
        assert actual is not None, f"Light {i + 1} CR should be visible"
        actual.assert_close(expected, atol=tolerance, msg=f"Light {i + 1} CR mismatch")

    # Test pupil center
    assert camimg.pupil_center is not None, "Pupil center should be visible"
    camimg.pupil_center.assert_close(expected_pc, atol=tolerance, msg="Pupil center mismatch")

    # Test pupil boundary points
    assert camimg.pupil_boundary is not None, "Pupil boundary should be visible"
    actual_pupil_boundary_list = camimg.pupil_boundary
    assert isinstance(actual_pupil_boundary_list, list), "Should be List[Point2D]"
    assert len(actual_pupil_boundary_list) == 20, "Should have 20 pupil boundary points"

    # Test first 5 pupil points against MATLAB reference
    for i in range(5):
        actual_point = actual_pupil_boundary_list[i]
        expected_point = expected_pupil_points[:, i]  # [x, y] from 2xM array
        np.testing.assert_allclose(
            [actual_point.x, actual_point.y],
            expected_point,
            atol=tolerance,
            err_msg=f"Pupil point {i} mismatch",
        )


def test_camera_take_image_without_refraction() -> None:
    """Test camera take_image without refraction using actual MATLAB reference values."""
    # Same setup as refraction test
    e = Eye(fovea_displacement=False)
    e.position = Position3D(0, 500e-3, 200e-3)

    c = Camera()
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(Position3D.from_array(e.trans[:, 3]))
    c.err = 0
    c.err_type = "uniform"

    e.look_at(Position3D(0, 0, 0))

    # Same light sources (4D homogeneous)
    lights = []
    light1 = Light(position=Position3D(200e-3, 0, 50e-3))
    lights.append(light1)

    light2 = Light(position=Position3D(200e-3, 0, 300e-3))
    lights.append(light2)

    # Take image without refraction
    camimg = c.take_image(e, lights, use_refraction=False)

    # Correct MATLAB reference values from the provided script output
    expected_cr = [Point2D(8.0091797595, 1.8590841799), Point2D(8.5048679096, 11.8448574197)]
    expected_pc_simple = Point2D(0.0, 0.0)
    expected_pupil_points_simple = np.array([
        [16.3103041184, 0.0000],
        [15.5120210145, -5.0401611560],
        [13.1953132152, -9.5869562212],
        [9.5869562212, -13.1953132152],
        [5.0401611560, -15.5120210145],
    ]).T  # Transpose to match 2xM format

    tolerance = 1e-5

    # Test that corneal reflexes are the same
    assert len(camimg.corneal_reflections) == 2
    for i, (expected, actual) in enumerate(zip(expected_cr, camimg.corneal_reflections, strict=False)):
        assert actual is not None, f"Light {i + 1} CR should be visible"
        actual.assert_close(expected, atol=tolerance, msg=f"Light {i + 1} CR mismatch in non-refraction test")

    # Test pupil center
    assert camimg.pupil_center is not None
    camimg.pupil_center.assert_close(
        expected_pc_simple, atol=tolerance, msg="Pupil center mismatch in non-refraction test"
    )

    # Test that we get 20 pupil points
    assert camimg.pupil_boundary is not None
    assert isinstance(camimg.pupil_boundary, list), "Should be List[Point2D]"
    assert len(camimg.pupil_boundary) == 20, "Should have 20 pupil boundary points"

    # Test first 5 pupil points against MATLAB reference
    for i in range(5):
        actual_point = camimg.pupil_boundary[i]
        expected_point = expected_pupil_points_simple[:, i]  # [x, y] from 2xM array
        np.testing.assert_allclose(
            [actual_point.x, actual_point.y],
            expected_point,
            atol=tolerance,
            err_msg=f"Pupil point {i} mismatch in non-refraction test",
        )


def test_camera_take_image_output_structure() -> None:
    """Test that camera take_image returns correct output structure."""
    # Minimal setup
    e = Eye(fovea_displacement=False)
    e.position = Position3D(0, 500e-3, 200e-3)

    c = Camera()
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(Position3D.from_array(e.trans[:, 3]))
    c.err = 0
    c.err_type = "uniform"

    e.look_at(Position3D(0, 0, 0))

    # Single light source (4D homogeneous)
    light = Light(position=Position3D(200e-3, 0, 50e-3))
    lights = [light]

    camimg = c.take_image(e, lights, use_refraction=True)

    # Check output structure
    assert isinstance(camimg, CameraImage)
    assert hasattr(camimg, "corneal_reflections")
    assert hasattr(camimg, "pupil_center")
    assert hasattr(camimg, "pupil_boundary")

    # Check types
    assert isinstance(camimg.corneal_reflections, list)
    assert len(camimg.corneal_reflections) == 1

    assert camimg.pupil_center is not None, "camimg.pupil_center should not be None"
    assert isinstance(camimg.pupil_center, Point2D)

    assert camimg.pupil_boundary is not None
    assert isinstance(camimg.pupil_boundary, list)
    assert all(isinstance(p, Point2D) for p in camimg.pupil_boundary)
