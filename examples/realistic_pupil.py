#!/usr/bin/env python3
"""
Realistic pupil example.

Compares elliptical vs realistic pupil shapes using pinhole camera.
"""

from et_simul.core import Eye, Camera
from et_simul.types import Position3D, RotationMatrix
from et_simul.visualization import plot_interactive_pupil_comparison


def main():
    # Create two eyes with different pupil types
    eye_elliptical = Eye(pupil_type="elliptical", pupil_boundary_points=300)
    eye_realistic = Eye(pupil_type="realistic", pupil_boundary_points=300, pupil_random_seed=0)

    rest_orientation = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], validate_handedness=False)

    eye_elliptical.set_rest_orientation(rest_orientation)
    eye_realistic.set_rest_orientation(rest_orientation)

    target_point = Position3D(0, 0, 0)
    eye_position = Position3D(0, 50e-3, 0)

    eye_elliptical.position = eye_position
    eye_realistic.position = eye_position

    eye_elliptical.look_at(target_point)
    eye_realistic.look_at(target_point)

    # Create pinhole camera
    camera = Camera()
    camera.point_at(eye_position)

    print("Setup:")
    print(f"- Eye position: {eye_position}")
    print(f"- Camera position: {camera.position}")
    print(f"- Target point: {target_point}")

    # Run the pupil comparison
    plot_interactive_pupil_comparison(eye_elliptical, eye_realistic, camera, target_point)


if __name__ == "__main__":
    main()
