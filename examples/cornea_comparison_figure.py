#!/usr/bin/env python3
"""
Cornea comparison figure.

Compares conic vs spherical cornea shapes using camera view visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyetsimul.core import Eye, Camera
from pyetsimul.core.cornea import SphericalCornea, ConicCornea
from pyetsimul.types import Position3D, RotationMatrix
from tabulate import tabulate


def main():
    # Create two eyes with different cornea types
    eye_spherical = Eye(cornea=SphericalCornea(), pupil_boundary_points=300)
    eye_conic = Eye(cornea=ConicCornea(), pupil_boundary_points=300)

    rest_orientation = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], validate_handedness=False)

    eye_spherical.set_rest_orientation(rest_orientation)
    eye_conic.set_rest_orientation(rest_orientation)

    # Setup: Eye looking slightly left of camera (off-axis)
    target_point = Position3D(-20e-3, 0, 0)  # 20mm left of camera
    eye_position = Position3D(0, 50e-3, 0)

    eye_spherical.position = eye_position
    eye_conic.position = eye_position

    eye_spherical.look_at(target_point)
    eye_conic.look_at(target_point)

    # Create pinhole camera
    camera = Camera()
    camera.point_at(eye_position)

    print("\nCornea Comparison Setup")
    headers = ["Component", "Position (x, y, z)", "Unit"]
    data = [
        ["Eye", f"({eye_position.x:.3f}, {eye_position.y:.3f}, {eye_position.z:.3f})", "meters"],
        ["Camera", f"({camera.position.x:.3f}, {camera.position.y:.3f}, {camera.position.z:.3f})", "meters"],
        [
            "Target",
            f"({target_point.x:.3f}, {target_point.y:.3f}, {target_point.z:.3f})",
            "meters",
        ],
    ]
    print(tabulate(data, headers=headers, tablefmt="grid"))

    # Take camera images of both eyes
    image_spherical = camera.take_image(eye_spherical, [])
    image_conic = camera.take_image(eye_conic, [])

    # Create single figure comparing both cornea types
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot pupil boundaries - convert structured types to numpy
    pupil_boundary = image_spherical.pupil_boundary
    if pupil_boundary:
        pupil_points = np.array([[p.x for p in pupil_boundary], [p.y for p in pupil_boundary]])
        closed_points = np.hstack((pupil_points, pupil_points[:, 0:1]))
    else:
        closed_points = None

    if closed_points is not None:
        ax.plot(
            closed_points[0, :],
            closed_points[1, :],
            color="blue",
            linewidth=1,
            label="Spherical Cornea",
        )

    # Plot conic cornea pupil boundary
    pupil_boundary_conic = image_conic.pupil_boundary
    if pupil_boundary_conic:
        pupil_points_conic = np.array([[p.x for p in pupil_boundary_conic], [p.y for p in pupil_boundary_conic]])
        closed_points_conic = np.hstack((pupil_points_conic, pupil_points_conic[:, 0:1]))
        ax.plot(
            closed_points_conic[0, :],
            closed_points_conic[1, :],
            color="red",
            linewidth=1,
            label="Conic Cornea",
        )

    # Plot pupil centers
    center = image_spherical.pupil_center.to_array()
    ax.scatter(
        center[0],
        center[1],
        color="blue",
        s=50,
        marker="+",
        linewidth=1,
        label="Spherical Center",
    )

    center = image_conic.pupil_center.to_array()
    ax.scatter(
        center[0],
        center[1],
        color="red",
        s=50,
        marker="+",
        linewidth=1,
        label="Conic Center",
    )

    # Set camera image limits and formatting
    resolution = camera.camera_matrix.resolution
    ax.set_xlim(-resolution.x / 2, resolution.x / 2)
    ax.set_ylim(-resolution.y / 2, resolution.y / 2)
    ax.invert_yaxis()
    ax.invert_xaxis()

    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.set_title("Cornea Comparison - Camera View", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    # Create figures directory if it doesn't exist
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/cornea_camera_view_comparison.pdf", transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
