#!/usr/bin/env python3
"""
Pupil decentration comparison figure.

Shows pupil decentration across different pupil sizes and individual variation profiles.
Three decentration types: no decentration, random individual variation, and fixed seed individual.
Five pupil sizes from constricted to dilated conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyetsimul.core import Eye
from pyetsimul.core.pupil_decentration import PupilDecentrationConfig
from pyetsimul.types import Position3D, RotationMatrix


def main():
    # Five different pupil sizes (diameters in meters)
    # Range from bright light constriction to dark adaptation dilation
    pupil_sizes = [
        ("Very Small", 2.5e-3),  # Bright light (constricted)
        ("Small", 3.5e-3),  # Indoor lighting
        ("Baseline", 4.75e-3),  # Wildenmann baseline (800 lux)
        ("Large", 6.0e-3),  # Dim lighting
        ("Very Large", 7.5e-3),  # Dark adaptation (dilated)
    ]

    baseline_diameter = 4.75e-3  # Baseline from Wildenmann & Schaeffel (2013)

    # Three decentration configurations
    eye_configs = [
        (
            "Standard Model",
            PupilDecentrationConfig(
                enabled=True,
                model_name="wildenmann_2013",
                baseline_diameter=baseline_diameter,
                # Uses default 0.05 mm/mm coefficients from paper
            ),
        ),
        (
            "Random Individual",
            PupilDecentrationConfig(
                enabled=True,
                model_name="wildenmann_2013",
                baseline_diameter=baseline_diameter,
                use_individual_variation=True,  # Random coefficients from 0.044-0.179 range
            ),
        ),
        (
            "Fixed Seed (42)",
            PupilDecentrationConfig(
                enabled=True,
                model_name="wildenmann_2013",
                baseline_diameter=baseline_diameter,
                use_individual_variation=True,
                individual_seed=42,  # Reproducible individual profile
            ),
        ),
    ]

    # Create figure with 3 rows (eye types) × 5 columns (pupil sizes)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Print individual coefficients for reference
    print("Individual variation coefficients:")
    for config_name, config in eye_configs:
        if config.enabled and hasattr(config, "x_coeff") and config.x_coeff is not None:
            print(f"  {config_name}: x_coeff={config.x_coeff:.6f}, y_coeff={config.y_coeff:.6f}")
        else:
            print(f"  {config_name}: No individual coefficients (disabled or using defaults)")
    print()

    for row, (config_name, config) in enumerate(eye_configs):
        for col, (size_name, diameter) in enumerate(pupil_sizes):
            ax = axes[row, col]

            # Create eye with current decentration config
            eye = Eye(pupil_type="elliptical", pupil_boundary_points=100, decentration_config=config)

            # Create reference eye without decentration for comparison
            eye_centered = Eye(pupil_type="elliptical", pupil_boundary_points=100)

            # Set pupil diameter (this will trigger decentration if enabled)
            eye.set_pupil_diameter(diameter)
            eye_centered.set_pupil_diameter(diameter)

            # Set same orientation for both eyes
            rest_orientation = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], validate_handedness=False)
            eye.set_rest_orientation(rest_orientation)
            eye_centered.set_rest_orientation(rest_orientation)

            # Position eyes and look at target
            target_point = Position3D(0, 0, 0)
            eye_position = Position3D(0, 50e-3, 0)

            eye.position = eye_position
            eye_centered.position = eye_position

            eye.look_at(target_point)
            eye_centered.look_at(target_point)

            # Get pupil boundary points
            eye_boundary = eye.pupil.get_boundary_points()
            centered_boundary = eye_centered.pupil.get_boundary_points()

            # Convert to mm for plotting (4×N matrix, take x and y rows)
            eye_x = eye_boundary[0, :] * 1000  # X coordinates
            eye_y = eye_boundary[1, :] * 1000  # Y coordinates
            centered_x = centered_boundary[0, :] * 1000
            centered_y = centered_boundary[1, :] * 1000

            # Close the boundaries for plotting
            eye_x = np.append(eye_x, eye_x[0])
            eye_y = np.append(eye_y, eye_y[0])
            centered_x = np.append(centered_x, centered_x[0])
            centered_y = np.append(centered_y, centered_y[0])

            # Get pupil centers for comparison
            eye_center = eye.pupil.pos_pupil
            centered_center = eye_centered.pupil.pos_pupil

            # Plot both pupil positions
            ax.plot(
                centered_x,
                centered_y,
                "b--",
                linewidth=1.5,
                alpha=0.5,
                label="Centered" if row == 0 and col == 0 else "",
            )
            ax.plot(eye_x, eye_y, "r-", linewidth=2, label="Decentered" if row == 0 and col == 0 else "")

            # Mark pupil centers
            ax.plot(centered_center.x * 1000, centered_center.y * 1000, "bo", markersize=4, alpha=0.7)
            ax.plot(eye_center.x * 1000, eye_center.y * 1000, "ro", markersize=4)

            # Draw arrow showing decentration vector
            dx = (eye_center.x - centered_center.x) * 1000
            dy = (eye_center.y - centered_center.y) * 1000
            if abs(dx) > 0.001 or abs(dy) > 0.001:  # Only draw if significant offset
                ax.arrow(
                    centered_center.x * 1000,
                    centered_center.y * 1000,
                    dx,
                    dy,
                    head_width=0.15,
                    head_length=0.08,
                    fc="red",
                    ec="red",
                    alpha=0.8,
                )

            # Set equal aspect ratio and formatting
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Calculate decentration info for title
            diameter_change = diameter - baseline_diameter
            expected_x_offset = config.x_coeff * diameter_change * 1000
            expected_y_offset = config.y_coeff * diameter_change * 1000
            offset_text = f"Δx={expected_x_offset:.2f}, Δy={expected_y_offset:.2f} mm"

            # Create title
            title = f"{size_name}\n(d = {diameter * 1000:.1f} mm)\n{offset_text}"
            ax.set_title(title, fontsize=9, fontweight="bold")

            # Set consistent axis limits
            max_radius_mm = 5.0  # Fixed limits for consistency
            ax.set_xlim(-max_radius_mm, max_radius_mm)
            ax.set_ylim(-max_radius_mm, max_radius_mm)

            # Reduce number of ticks for cleaner look
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.tick_params(labelsize=8)

            # Label axes only on left and bottom
            if col == 0:
                ax.set_ylabel("y (mm)", fontsize=9)
            if row == 2:
                ax.set_xlabel("x (mm)", fontsize=9)

        # Add row labels on the right
        axes[row, -1].text(
            1.05,
            0.5,
            config_name,
            transform=axes[row, -1].transAxes,
            rotation=90,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    # Add legend to the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)

    # Add main title
    fig.suptitle(
        "Pupil Decentration: Individual Variation Profiles\nWildenmann & Schaeffel (2013) Model",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.show()


if __name__ == "__main__":
    main()
