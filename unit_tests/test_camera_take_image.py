"""Unit tests for Camera.take_image method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.core.camera import Camera
from et_simul.core.light import Light


def test_camera_take_image_with_refraction():
    """Test camera take_image with refraction using actual MATLAB reference values."""
    # Create eye and camera setup (matching MATLAB test)
    e = Eye(fovea_displacement=False)
    e.trans[0:3, 3] = [0, 500e-3, 200e-3]  # Eye at [0, 500mm, 200mm]
    
    # Camera at origin pointing at eye
    c = Camera()
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Hennessey-style rotation
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])  # Point at eye
    c.err = 0
    c.err_type = 'uniform'
    
    # Eye looks back at camera (mutual gaze)
    e.look_at(np.array([0, 0, 0, 1]))
    
    # Create light sources (typical eye tracker setup)
    lights = []
    
    # Light 1: right side, low
    light1 = Light()
    light1.position = np.array([200e-3, 0, 50e-3, 1])
    lights.append(light1)
    
    # Light 2: right side, high
    light2 = Light()
    light2.position = np.array([200e-3, 0, 300e-3, 1])
    lights.append(light2)
    
    # Take image with refraction
    camimg = c.take_image(e, lights, use_refraction=True)
    
    # Correct MATLAB reference values from the provided script output
    expected_cr = [
        [8.0091797595, 1.8590841799],   # Light 1 CR
        [8.5048679096, 11.8448574197]   # Light 2 CR
    ]
    expected_pc = [0.0, 0.0] # MATLAB: [-0.0000000000, 0.0000000000]
    expected_pupil_points = [
        [18.2201323223, 0.0000000000],   # pupil[:,0]
        [17.3283755729, -5.6303305273],  # pupil[:,1]
        [14.7403966885, -10.7095250739], # pupil[:,2]
        [10.7095250739, -14.7403966885], # pupil[:,3]
        [5.6303305273, -17.3283755729]  # pupil[:,4]
    ]
    
    # Using a slightly larger tolerance for cross-language floating point comparisons
    tolerance = 1e-5
    
    # Test corneal reflexes
    assert len(camimg['cr']) == 2, "Should have 2 corneal reflexes"
    for i, (expected, actual) in enumerate(zip(expected_cr, camimg['cr'])):
        assert actual is not None, f"Light {i+1} CR should be visible"
        np.testing.assert_allclose(actual, expected, atol=tolerance, err_msg=f"Light {i+1} CR mismatch")
    
    # Test pupil center
    assert camimg['pc'] is not None, "Pupil center should be visible"
    np.testing.assert_allclose(camimg['pc'], expected_pc, atol=tolerance, err_msg="Pupil center mismatch")
    
    # Test pupil boundary points
    assert camimg['pupil'].shape[0] == 2, "Should have 2D pupil coordinates"
    assert camimg['pupil'].shape[1] == 20, "Should have 20 pupil boundary points"
    
    # Test first 5 pupil points against MATLAB reference
    for i, expected_point in enumerate(expected_pupil_points):
        actual_point = camimg['pupil'][:, i]
        np.testing.assert_allclose(actual_point, expected_point, atol=tolerance, err_msg=f"Pupil point {i} mismatch")


def test_camera_take_image_without_refraction():
    """Test camera take_image without refraction using actual MATLAB reference values."""
    # Same setup as refraction test
    e = Eye(fovea_displacement=False)
    e.trans[0:3, 3] = [0, 500e-3, 200e-3]
    
    c = Camera()
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])
    c.err = 0
    c.err_type = 'uniform'
    
    e.look_at(np.array([0, 0, 0, 1]))
    
    # Same light sources
    lights = []
    light1 = Light()
    light1.position = np.array([200e-3, 0, 50e-3, 1])
    lights.append(light1)
    
    light2 = Light()
    light2.position = np.array([200e-3, 0, 300e-3, 1])
    lights.append(light2)
    
    # Take image without refraction
    camimg = c.take_image(e, lights, use_refraction=False)
    
    # Correct MATLAB reference values from the provided script output
    expected_cr = [
        [8.0091797595, 1.8590841799],
        [8.5048679096, 11.8448574197]
    ]
    expected_pc_simple = [0.0, 0.0]
    expected_pupil_points_simple = [
        [16.3103041184, 0.0000000000],
        [15.5120210145, -5.0401611560],
        [13.1953132152, -9.5869562212],
        [9.5869562212, -13.1953132152],
        [5.0401611560, -15.5120210145]
    ]

    tolerance = 1e-5
    
    # Test that corneal reflexes are the same
    assert len(camimg['cr']) == 2
    for i, (expected, actual) in enumerate(zip(expected_cr, camimg['cr'])):
        assert actual is not None, f"Light {i+1} CR should be visible"
        np.testing.assert_allclose(actual, expected, atol=tolerance, err_msg=f"Light {i+1} CR mismatch in non-refraction test")

    # Test pupil center
    assert camimg['pc'] is not None
    np.testing.assert_allclose(camimg['pc'], expected_pc_simple, atol=tolerance, err_msg="Pupil center mismatch in non-refraction test")
    
    # Test that we get 20 pupil points
    assert camimg['pupil'].shape[0] == 2
    assert camimg['pupil'].shape[1] == 20
    
    # Test first 5 pupil points against MATLAB reference
    for i, expected_point in enumerate(expected_pupil_points_simple):
        actual_point = camimg['pupil'][:, i]
        np.testing.assert_allclose(actual_point, expected_point, atol=tolerance, err_msg=f"Pupil point {i} mismatch in non-refraction test")


def test_camera_take_image_output_structure():
    """Test that camera take_image returns correct output structure."""
    # Minimal setup
    e = Eye(fovea_displacement=False)
    e.trans[0:3, 3] = [0, 500e-3, 200e-3]
    
    c = Camera()
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])
    c.err = 0
    c.err_type = 'uniform'
    
    e.look_at(np.array([0, 0, 0, 1]))
    
    # Single light source
    light = Light()
    light.position = np.array([200e-3, 0, 50e-3, 1])
    lights = [light]
    
    camimg = c.take_image(e, lights, use_refraction=True)
    
    # Check output structure
    assert isinstance(camimg, dict)
    assert 'cr' in camimg
    assert 'pc' in camimg
    assert 'pupil' in camimg
    
    # Check types
    assert isinstance(camimg['cr'], list)
    assert len(camimg['cr']) == 1
    
    if camimg['pc'] is not None:
        assert isinstance(camimg['pc'], (np.ndarray, tuple, list))
        if isinstance(camimg['pc'], np.ndarray):
            assert camimg['pc'].shape == (2,)
        else:
            assert len(camimg['pc']) == 2
    
    assert isinstance(camimg['pupil'], np.ndarray)
    assert camimg['pupil'].shape[0] == 2