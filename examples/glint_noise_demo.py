#!/usr/bin/env python3
"""Example script demonstrating glint noise in PyEtSimul.

Demonstrates how to add random noise to corneal reflection (glint) positions
to simulate realistic eye tracking measurement errors. Based on the basic example.py structure.
"""

import numpy as np
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D
from pyetsimul.camera_noise import GlintNoiseConfig
import matplotlib.pyplot as plt
from tabulate import tabulate


def main():
    """Run glint noise demonstration: simulate eye, camera, and light with different noise levels."""
    # Create an eye
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    e = Eye(fovea_displacement=False, pupil_boundary_points=100)
    e.set_rest_orientation(rotation_matrix)
    e.position = Position3D(0, 500e-3, 200e-3)

    # Create a light
    l = Light(position=Position3D(200e-3, 0, 0))

    # Create cameras with different noise levels
    cameras = {
        "No noise": Camera(),
        "Gaussian 0.5px": Camera(glint_noise_config=GlintNoiseConfig(noise_type="gaussian", std=0.5, seed=71)),
        "Gaussian 1.0px": Camera(glint_noise_config=GlintNoiseConfig(noise_type="gaussian", std=1.0, seed=71)),
        "Uniform 1.0px": Camera(glint_noise_config=GlintNoiseConfig(noise_type="uniform", std=1.0, seed=71)),
        "Constant +1px X": Camera(
            glint_noise_config=GlintNoiseConfig(noise_type="constant_offset", offset_x=1.0, offset_y=0.0)
        ),
    }

    # Point all cameras at the eye position
    for camera in cameras.values():
        camera.point_at(e.position)

    # Get pupil from reference camera (no noise)
    ref_camera = Camera()
    ref_camera.point_at(e.position)
    ref_image = ref_camera.take_image(e, [l])
    pupil_boundary = ref_image.pupil_boundary
    pupil_center = ref_image.pupil_center

    # Take images from each camera to get glints with different noise levels
    plt.figure(figsize=(10, 8))

    # Plot pupil boundary once - closed loop
    if pupil_boundary is not None:
        pupil_x = [p.x for p in pupil_boundary] + [pupil_boundary[0].x]
        pupil_y = [p.y for p in pupil_boundary] + [pupil_boundary[0].y]
        plt.plot(pupil_x[:-1], pupil_y[:-1], "b.", markersize=2, alpha=0.7)  # Small dots (without closing point)
        plt.plot(pupil_x, pupil_y, "b-", linewidth=1, label="Pupil")  # Connected line

    # Plot pupil center once
    if pupil_center is not None:
        plt.plot(pupil_center.x, pupil_center.y, "b+", markersize=8, label="Pupil center")

    # Plot glints from each camera with different colors and markers
    colors = ["red", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "D", "x"]
    results_data = []

    for i, (name, camera) in enumerate(cameras.items()):
        # Take image to get glint position
        image = camera.take_image(e, [l])
        corneal_reflection = image.corneal_reflections[0]

        if corneal_reflection is not None:
            plt.scatter(
                corneal_reflection.x,
                corneal_reflection.y,
                color=colors[i],
                marker=markers[i],
                s=50,
                label=f"Glint: {name}",
            )
            results_data.append([name, f"({corneal_reflection.x:.2f}, {corneal_reflection.y:.2f})", "pixels"])
        else:
            results_data.append([name, "Not detected", "-"])

    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Eye Image: Pupil with Glints at Different Noise Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()

    # Create summary table
    print("\nGlint Position Results")
    headers = ["Camera Setup", "Glint Position", "Unit"]
    print(tabulate(results_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
