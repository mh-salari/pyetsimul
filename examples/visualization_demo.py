#!/usr/bin/env python3
"""
Eye Visualization Demo

This script demonstrates comprehensive eye anatomy visualization capabilities
including optical and visual axes, corneal geometry, pupil structure, and
key anatomical parameters with camera view.
"""

import numpy as np
import matplotlib.pyplot as plt

from et_simul.core import Eye, Camera, Light
from et_simul.visualization import plot_setup_and_camera_view


def main():
    """Main demo function."""
    print("Eye Anatomy Visualization Demo")
    print("=" * 50)
    print("Using exact positioning values from MATLAB example.m")

    # Create eye with MATLAB example.m positioning
    rest_orientation = np.array(
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    )  # Looking out along negative y-axis
    e = Eye(r_cornea=7.98e-3, fovea_displacement=True)
    e.rest_orientation = rest_orientation
    eye_position = [0, 250e-3, 100e-3]
    e.position = eye_position  # MATLAB example.m position

    # Create two light sources for dual-light eye tracking
    l1 = Light(position=np.array([100e-3, 0, 0, 1]))  # Right side light
    l2 = Light(position=np.array([-100e-3, 0, 0, 1]))  # Left side light (symmetric)
    lights = [l1, l2]

    # Create camera and point it at the eye
    c = Camera()
    c.point_at(eye_position)

    # Set gaze target
    target_point = np.array([50e-3, 0, 50e-3, 1])  # 5cm left, at screen plane, 5cm up

    print(
        f"- Eye position: ({e.position[0]*1000:.1f}, {e.position[1]*1000:.1f}, {e.position[2]*1000:.1f}) mm"
    )
    print(
        f"- Light 1 position: ({l1.position[0]*1000:.1f}, {l1.position[1]*1000:.1f}, {l1.position[2]*1000:.1f}) mm"
    )
    print(
        f"- Light 2 position: ({l2.position[0]*1000:.1f}, {l2.position[1]*1000:.1f}, {l2.position[2]*1000:.1f}) mm"
    )
    print(
        f"- Camera position: ({c.position[0]*1000:.1f}, {c.position[1]*1000:.1f}, {c.position[2]*1000:.1f}) mm"
    )
    print(
        f"- Target position: ({target_point[0]*1000:.1f}, {target_point[1]*1000:.1f}, {target_point[2]*1000:.1f}) mm"
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
            print(
                f" CR successfully found at: ({cr_3d[0]*1000:.1f}, {cr_3d[1]*1000:.1f}, {cr_3d[2]*1000:.1f}) mm"
            )
        else:
            print(" CR not found - may be outside corneal boundaries")

    # Create the visualization
    plot_setup_and_camera_view(e, target_point, lights, c, cr_3d_list)

    print("\nDisplaying eye tracking setup...")
    print("Close the plot window when done.")
    plt.show()


if __name__ == "__main__":
    main()
