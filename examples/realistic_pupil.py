#!/usr/bin/env python3
"""Realistic pupil example.

Compares elliptical vs realistic pupil shapes using pinhole camera.
"""

from tabulate import tabulate

from pyetsimul.core import Camera, Eye
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.visualization import plot_interactive_pupil_comparison


def main() -> None:
    """Compare elliptical vs realistic pupil shapes using pinhole camera."""
    # Create two eyes with different pupil types
    eye_elliptical = Eye(pupil_type="elliptical", pupil_boundary_points=300)
    eye_realistic = Eye(pupil_type="realistic", pupil_boundary_points=300, pupil_random_seed=0)

    rest_orientation = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], validate_handedness=False)

    eye_elliptical.set_rest_orientation(rest_orientation)
    eye_realistic.set_rest_orientation(rest_orientation)

    target_point = Position3D(0, 0, 0)
    eye_position = Position3D(0, 50, 0)

    eye_elliptical.position = eye_position
    eye_realistic.position = eye_position

    eye_elliptical.look_at(target_point)
    eye_realistic.look_at(target_point)

    # Create pinhole camera
    camera = Camera()
    camera.point_at(eye_position)

    print("\nPupil Comparison Setup")
    headers = ["Component", "Position (x, y, z)", "Unit"]
    data = [
        ["Eye", f"({eye_position.x:.1f}, {eye_position.y:.1f}, {eye_position.z:.1f})", "mm"],
        ["Camera", f"({camera.position.x:.1f}, {camera.position.y:.1f}, {camera.position.z:.1f})", "mm"],
        ["Target", f"({target_point.x:.1f}, {target_point.y:.1f}, {target_point.z:.1f})", "mm"],
    ]
    print(tabulate(data, headers=headers, tablefmt="grid"))

    # Run the pupil comparison
    plot_interactive_pupil_comparison(eye_elliptical, eye_realistic, camera, target_point)


if __name__ == "__main__":
    main()
