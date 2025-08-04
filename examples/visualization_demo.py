#!/usr/bin/env python3
"""
Comprehensive eye tracking visualization demo.

Demonstrates 3D eye anatomy, optical/visual axes, corneal geometry, pupil structure, and camera view with dual-light setup.
"""

import numpy as np
import matplotlib.pyplot as plt

from et_simul.core import Eye, Camera, Light
from et_simul.types import Position3D, RotationMatrix
from et_simul.visualization import plot_setup_and_camera_view


def main():
    """Run a comprehensive demo of eye tracking visualization with dual-light and camera setup."""
    print("Eye Anatomy Visualization Demo")
    print("=" * 50)
    print("Using exact positioning values from MATLAB example.m")

    e = Eye()

    # Looking out along negative y-axis - use validate_handedness=False for legacy MATLAB compatibility
    rest_orientation = RotationMatrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), validate_handedness=False)
    e.set_rest_orientation(rest_orientation)

    eye_position = Position3D(0, 250e-3, 100e-3)
    e.position = eye_position

    # Create two light sources for dual-light eye tracking
    l1 = Light(position=Position3D(100e-3, 0, 0))  # Right side light

    l2 = Light(position=Position3D(-100e-3, 0, 0))  # Left side light (symmetric)

    lights = [l1, l2]

    # Create camera and point it at the eye
    c = Camera()
    c.point_at(eye_position)

    # Set gaze target
    target_point = Position3D(50e-3, 0, 50e-3)  # 5cm left, at screen plane, 5cm up

    print(f"- Eye position: ({e.position.x * 1000:.1f}, {e.position.y * 1000:.1f}, {e.position.z * 1000:.1f}) mm")
    print(
        f"- Light 1 position: ({l1.position.x * 1000:.1f}, {l1.position.y * 1000:.1f}, {l1.position.z * 1000:.1f}) mm"
    )
    print(
        f"- Light 2 position: ({l2.position.x * 1000:.1f}, {l2.position.y * 1000:.1f}, {l2.position.z * 1000:.1f}) mm"
    )
    print(f"- Camera position: ({c.position.x * 1000:.1f}, {c.position.y * 1000:.1f}, {c.position.z * 1000:.1f}) mm")
    print(
        f"- Target position: ({target_point.x * 1000:.1f}, {target_point.y * 1000:.1f}, {target_point.z * 1000:.1f}) mm"
    )

    # Print eye transformation matrix before look_at
    print("\nEye transformation matrix BEFORE look_at:")
    print(e.trans)

    # Make eye look at target
    print("\nMaking eye look at target...")
    e.look_at(target_point)

    # Print eye transformation matrix after look_at
    print("Eye transformation matrix AFTER look_at:")
    print(e.trans)

    # Find corneal reflections for both lights
    print("Finding corneal reflections...")
    cr_3d_list = []
    for i, light in enumerate(lights):
        cr_3d = e.find_cr(light, c)
        cr_3d_list.append(cr_3d)

        if cr_3d is not None:
            print(f"CR successfully found at: ({cr_3d.x * 1000:.1f}, {cr_3d.y * 1000:.1f}, {cr_3d.z * 1000:.1f}) mm")
        else:
            print("CR not found - may be outside corneal boundaries")

    # Create the visualization
    plot_setup_and_camera_view(e, target_point, c, lights=lights)

    print("\nDisplaying eye tracking setup...")
    print("Close the plot window when done.")
    plt.show()


if __name__ == "__main__":
    main()
