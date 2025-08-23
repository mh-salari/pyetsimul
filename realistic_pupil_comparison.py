#!/usr/bin/env python3
"""
Realistic pupil comparison figure.

Compares realistic vs circular pupil shapes across different pupil sizes.
Shows three subplots side by side: small, default, and large pupil sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyetsimul.core import Eye
from pyetsimul.types import Position3D, RotationMatrix, Direction3D


def main():
    # Create three different pupil sizes (diameters in mm)
    # Default comes from our code's default pupil radius (3mm radius = 6mm diameter)
    # Research-based extremes: ~2mm (bright light) to ~8mm (dark adaptation)
    small_diameter = 2.5  # Research: bright light condition (constricted)
    default_diameter = 6.0  # Our code's default: 3mm radius = 6mm diameter
    large_diameter = 7.5  # Research: dark adaptation (dilated)

    pupil_sizes = [("Small", small_diameter), ("Default", default_diameter), ("Large", large_diameter)]

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (size_name, diameter_mm) in enumerate(pupil_sizes):
        ax = axes[i]

        # Create eye with realistic pupil
        eye_realistic = Eye(pupil_type="realistic", pupil_boundary_points=100)  # pupil_random_seed=0

        # Create eye with elliptical (circular) pupil for comparison
        eye_circular = Eye(pupil_type="elliptical", pupil_boundary_points=100)

        # Update realistic pupil size
        eye_realistic.pupil.set_diameter(diameter_mm)

        # Update circular pupil size by scaling the radius vectors
        target_radius = (diameter_mm / 2) * 1e-3  # Convert mm to meters
        eye_circular.pupil.x_pupil = Direction3D(target_radius, 0, 0)
        eye_circular.pupil.y_pupil = Direction3D(0, target_radius, 0)

        # Set same orientation for both eyes
        rest_orientation = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], validate_handedness=False)
        eye_realistic.set_rest_orientation(rest_orientation)
        eye_circular.set_rest_orientation(rest_orientation)

        # Position eyes and look at target
        target_point = Position3D(0, 0, 0)
        eye_position = Position3D(0, 50e-3, 0)

        eye_realistic.position = eye_position
        eye_circular.position = eye_position

        eye_realistic.look_at(target_point)
        eye_circular.look_at(target_point)

        # Get pupil boundary points
        realistic_boundary = eye_realistic.pupil.get_boundary_points()
        circular_boundary = eye_circular.pupil.get_boundary_points()

        # Convert to mm for plotting (4×N matrix, take x and y rows)
        realistic_x = realistic_boundary[0, :] * 1000  # X coordinates of all points
        realistic_y = realistic_boundary[1, :] * 1000  # Y coordinates of all points
        circular_x = circular_boundary[0, :] * 1000
        circular_y = circular_boundary[1, :] * 1000

        # Close the boundaries for plotting
        realistic_x = np.append(realistic_x, realistic_x[0])
        realistic_y = np.append(realistic_y, realistic_y[0])
        circular_x = np.append(circular_x, circular_x[0])
        circular_y = np.append(circular_y, circular_y[0])

        # Plot both pupil types
        ax.plot(circular_x, circular_y, "b-", linewidth=2, label="Circular pupil", alpha=0.7)
        ax.plot(realistic_x, realistic_y, "r-", linewidth=2, label="Realistic pupil")

        # Set equal aspect ratio and formatting
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (mm)", fontsize=10)
        ax.set_ylabel("y (mm)", fontsize=10)
        ax.set_title(f"{size_name} pupil\n(d = {diameter_mm:.1f} mm)", fontsize=12, fontweight="bold")

        # Set consistent axis limits based on the largest pupil
        max_radius_mm = (large_diameter / 2) * 1.2  # Add some margin
        ax.set_xlim(-max_radius_mm, max_radius_mm)
        ax.set_ylim(-max_radius_mm, max_radius_mm)

        # Reduce number of ticks for cleaner look
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(labelsize=9)

        # Add legend only to the first subplot
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Create figures directory if it doesn't exist
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/realistic_pupil_comparison.pdf", transparent=True, bbox_inches="tight")
    plt.show()

    print("\nRealistic Pupil Comparison")
    print("Generated comparison showing realistic vs circular pupil shapes")
    print("across pupil sizes:")
    print(f"  Small (research - bright light): {small_diameter:.1f}mm")
    print(f"  Default (code default): {default_diameter:.1f}mm")
    print(f"  Large (research - dark adaptation): {large_diameter:.1f}mm")


if __name__ == "__main__":
    main()
