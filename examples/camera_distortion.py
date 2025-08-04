#!/usr/bin/env python3
"""
Camera distortion comparison example.

Compares pinhole vs actual camera distortion for pupil detection.

Camera parameters from Pupil Labs camera models:
https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/camera_models.py
"""

import numpy as np

from et_simul.core import Eye, Camera
from et_simul.types import Position3D, RotationMatrix, CameraMatrix
from et_simul.visualization import plot_interactive_cameras


# Camera configurations from Pupil Labs
CAMERA_CONFIGS = [
    # Pupil Cam1 ID2 configurations
    {
        "name": "Pupil Cam1 ID2",
        "camera_matrix": CameraMatrix(
            np.array(
                [
                    [395.60662814306596, 0.0, 316.72212558212516],
                    [0.0, 395.56975615889445, 259.206579702132],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
        "dist_coefs": np.array(
            [
                -0.2430487205352619,
                0.1623502095383119,
                0.0001632500987373085,
                8.322130878440475e-05,
                0.017859803336754784,
                0.1969284124154412,
                0.00577741263771627,
                0.09892258337410824,
            ]
        ),
        "cam_type": "radial",
    },
    {
        "name": "Pupil Cam1 ID2",
        "camera_matrix": CameraMatrix(
            np.array(
                [
                    [794.3311439869655, 0.0, 633.0104437728625],
                    [0.0, 793.5290139393004, 397.36927353414865],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
        "dist_coefs": np.array(
            [
                -0.3758628065070806,
                0.1643326166951343,
                0.00012182540692089567,
                0.00013422608638039466,
                0.03343691733865076,
                0.08235235770849726,
                -0.08225804883227375,
                0.14463365333602152,
            ]
        ),
        "cam_type": "radial",
    },
    {
        "name": "Pupil Cam1 ID2",
        "camera_matrix": CameraMatrix(
            np.array(
                [
                    [793.8052697386686, 0.0, 953.2237035923064],
                    [0.0, 792.3104221704713, 572.5036513432223],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
        "dist_coefs": np.array(
            [
                -0.13648546769272826,
                -0.0033787366635030644,
                -0.002343859061730869,
                0.001926274947199097,
            ]
        ),
        "cam_type": "fisheye",
    },
    # Neon Sensor Module v1
    {
        "name": "Neon Sensor Module v1",
        "camera_matrix": CameraMatrix(
            np.array(
                [
                    [140.68445787837342, 0.0, 99.42393317744813],
                    [0.0, 140.67571954970256, 96.235134525304],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
        "dist_coefs": np.array(
            [
                0.05449484235207129,
                -0.14013187141454536,
                0.0006598061556076783,
                5.0572400552608696e-05,
                -0.6158040573125376,
                -0.048953803434398195,
                0.04521347340211147,
                -0.7004955138758611,
            ]
        ),
        "cam_type": "radial",
    },
    # Pupil Cam2 ID0 configurations
    {
        "name": "Pupil Cam2 ID0",
        "camera_matrix": CameraMatrix(
            np.array(
                [
                    [282.976877, 0.0, 96],
                    [0.0, 283.561467, 96],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
        "dist_coefs": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "cam_type": "radial",
    },
]


def main():
    """Main function to select camera and create comparison."""
    # Camera selection
    print("Available Camera Configurations:")
    print("=" * 50)
    for i, config in enumerate(CAMERA_CONFIGS, 1):
        res = config["camera_matrix"].resolution
        print(f"{i}. {config['name']} {res.x}x{res.y} ({config['cam_type']})")

    print(f"\nSelect camera configuration (1-{len(CAMERA_CONFIGS)}): ", end="")
    try:
        choice = int(input()) - 1
        if choice < 0 or choice >= len(CAMERA_CONFIGS):
            raise ValueError()
        selected_config = CAMERA_CONFIGS[choice]
    except (ValueError, KeyboardInterrupt):
        print("Using fallback: Pupil Cam1 ID2 (1920, 1080)")
        selected_config = CAMERA_CONFIGS[2]  # Fallback to Pupil Cam1 ID2 1920x1080

    # Create pinhole camera with same focal length and resolution for fair comparison
    c_pinhole = Camera(name="Pinhole")
    c_pinhole.camera_matrix.focal_length = selected_config["camera_matrix"].focal_length
    c_pinhole.camera_matrix.resolution = selected_config["camera_matrix"].resolution

    # Create camera using selected configuration
    c_camera = Camera(
        name=selected_config["name"],
        camera_matrix=selected_config["camera_matrix"],
        dist_coeffs=selected_config["dist_coefs"],
    )

    # Create eye with proper orientation for realistic movement
    eye = Eye(pupil_boundary_points=100)
    rest_orientation = RotationMatrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), validate_handedness=False)
    eye.set_rest_orientation(rest_orientation)

    target_point = Position3D(0, 0, 0)
    eye.position = Position3D(0, 30e-3, 0)
    eye.look_at(target_point)

    # Point cameras at eye
    c_pinhole.point_at(eye.position)
    c_camera.point_at(eye.position)

    print("\nSetup:")
    print(f"- Camera: {selected_config['name']}")
    print(
        f"- Resolution: {selected_config['camera_matrix'].resolution.x}x{selected_config['camera_matrix'].resolution.y}"
    )
    print(f"- Distortion type: {selected_config['cam_type']}")
    print("\nCamera Parameters Comparison:")
    print("  Pinhole Camera:")
    print(f"- Focal length: {c_pinhole.camera_matrix.focal_length:.1f} px")
    print(
        f"- Principal point: ({c_pinhole.camera_matrix.matrix[0, 2]:.1f}, {c_pinhole.camera_matrix.matrix[1, 2]:.1f})"
    )
    print("- Distortion: None (k1=k2=k3=p1=p2=0)")
    print(f"  {selected_config['cam_type'].title()} Camera:")
    print(f"- Focal length: {c_camera.camera_matrix.focal_length:.1f} px")
    print(f"- Principal point: ({c_camera.camera_matrix.matrix[0, 2]:.1f}, {c_camera.camera_matrix.matrix[1, 2]:.1f})")
    print(f"- Distortion coefficients: k1={c_camera.dist_coeffs[0]:.3f}, k2={c_camera.dist_coeffs[1]:.3f}")
    if len(c_camera.dist_coeffs) > 2:
        print(
            f"- Additional coefficients: k3={c_camera.dist_coeffs[2]:.3f}, p1={c_camera.dist_coeffs[3]:.3f}, p2={c_camera.dist_coeffs[4]:.3f}"
        )
    print("\nControls:")
    print("- Arrow keys: Move target")
    print("- I/K/J/L/./,: Move eye")
    print("Move the eye and target around to see distortion effects!")

    # Run the camera comparison
    plot_interactive_cameras([c_pinhole, c_camera], eye, target_point)


if __name__ == "__main__":
    main()
