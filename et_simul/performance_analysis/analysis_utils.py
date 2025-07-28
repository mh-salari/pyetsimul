"""Utility functions for performance analysis.

This module provides helper functions used across performance analysis modules
to ensure consistency and reduce code duplication.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_error_vectors(
    X,
    Y,
    U,
    V,
    errors,
    title_prefix="",
    convert_to_mm=True,
    width=0.001,
    max_arrow_ratio=0.3,
    mark_target_positions=False,
    mark_predicted_positions=False,
    show_grid=True,
    auto_adjust_limits=True,
    figure_size=(10, 8),
    xlabel="Observer X position (mm)",
    ylabel="Observer Y position (mm)",
):
    """Plot error vectors with adaptive scaling .

    Features:
    - Adaptive vector scaling to prevent arrows from going off-screen
    - Color-coded arrows by magnitude with colorbar
    - Clear scaling information displayed in plot

    Args:
        X, Y: Grid coordinates for vector positions
        U, V: Error components in X and Y directions
        errors: Dictionary with error statistics (from calculate_error_statistics)
        title_prefix: Prefix text for plot title
        convert_to_mm: Convert coordinates and vectors to mm
        width: Arrow width
        max_arrow_ratio: Maximum arrow length as fraction of plot range
        mark_target_positions: Show blue crosses at target positions
        mark_predicted_positions: Show red dots at predicted positions
        show_grid: Show grid lines
        auto_adjust_limits: Automatically adjust plot limits
        figure_size: Figure size tuple
        xlabel, ylabel: Axis labels
    """
    # Apply unit conversion if requested
    if convert_to_mm:
        X_plot, Y_plot = X * 1000, Y * 1000
        U_plot, V_plot = U * 1000, V * 1000
        unit_str = "mm"
    else:
        X_plot, Y_plot = X, Y
        U_plot, V_plot = U, V
        unit_str = "m"

    # Calculate adaptive scaling, filtering out NaN/Inf values
    U_flat = U_plot.flatten()
    V_flat = V_plot.flatten()
    magnitudes = np.sqrt(U_flat**2 + V_flat**2)

    # Use only finite magnitudes for scaling
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    if len(finite_magnitudes) == 0:
        print("Warning: All error magnitudes are NaN/Inf, skipping plot")
        return

    max_magnitude = np.max(finite_magnitudes)
    plot_range_x = np.max(X_plot) - np.min(X_plot)
    plot_range_y = np.max(Y_plot) - np.min(Y_plot)
    plot_range = max(plot_range_x, plot_range_y)

    # Target arrow length: configurable % of plot range
    target_arrow_length = plot_range * max_arrow_ratio

    if max_magnitude > target_arrow_length:
        scale_factor = target_arrow_length / max_magnitude
        U_scaled = U_plot * scale_factor
        V_scaled = V_plot * scale_factor
        scaling_applied = True
    else:
        scale_factor = 1.0
        U_scaled = U_plot
        V_scaled = V_plot
        scaling_applied = False

    # Create figure
    plt.style.use("default")  # Reset to default style
    _, ax = plt.subplots(figsize=figure_size, facecolor="white")
    ax.set_facecolor("white")

    # Handle both gridded data (2D arrays) and scattered points (1D arrays)
    if U_scaled.ndim == 2:
        # Gridded data - create meshgrid
        XX, YY = np.meshgrid(X_plot, Y_plot)
        X_pos, Y_pos = XX, YY
    else:
        # Scattered points - use coordinates directly
        X_pos, Y_pos = X_plot, Y_plot

    # Create solid color quiver plot
    ax.quiver(
        X_pos, Y_pos, U_scaled, V_scaled, scale=1, scale_units="xy", angles="xy", alpha=1, width=width, color="black"
    )

    # Add markers for target and predicted positions
    if mark_target_positions:
        ax.scatter(X_pos, Y_pos, marker="+", s=50, c="blue", linewidths=2, alpha=0.8, label="Target")

    if mark_predicted_positions:
        # Calculate predicted positions (arrow tips)
        X_pred = X_pos + U_scaled
        Y_pred = Y_pos + V_scaled
        ax.scatter(X_pred, Y_pred, marker="o", s=30, c="red", alpha=0.8, label="Predicted")

        # Add legend if either marker type is shown
        if mark_target_positions:
            ax.legend()

    # Grid styling
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Auto-adjust limits with margin
    if auto_adjust_limits:
        arrow_tips_x = X_pos.flatten() + U_scaled.flatten()
        arrow_tips_y = Y_pos.flatten() + V_scaled.flatten()

        all_x = np.concatenate([X_pos.flatten(), arrow_tips_x])
        all_y = np.concatenate([Y_pos.flatten(), arrow_tips_y])

        # Filter out NaN values for limit calculations
        valid_x = all_x[~np.isnan(all_x)]
        valid_y = all_y[~np.isnan(all_y)]

        if len(valid_x) > 0 and len(valid_y) > 0:
            x_range = np.max(valid_x) - np.min(valid_x)
            y_range = np.max(valid_y) - np.min(valid_y)
            margin_x = x_range * 0.05
            margin_y = y_range * 0.05

            ax.set_xlim(np.min(valid_x) - margin_x, np.max(valid_x) + margin_x)
            ax.set_ylim(np.min(valid_y) - margin_y, np.max(valid_y) + margin_y)

    # Create comprehensive title with scaling info
    error_stats = (
        f"Max: {errors['mtr']['max'] * 1e3:.2f} {unit_str}, "
        f"Mean: {errors['mtr']['mean'] * 1e3:.2f} {unit_str}, "
        f"Std: {errors['mtr']['std'] * 1e3:.2f} {unit_str}"
    )

    if scaling_applied:
        scale_info = f" (arrows scaled {scale_factor:.2f}×)"
    else:
        scale_info = " (arrows at full scale)"

    if title_prefix:
        title = f"{title_prefix}\n{error_stats}{scale_info}"
    else:
        title = f"Gaze Error Vectors\n{error_stats}{scale_info}"

    ax.set_title(title, pad=20)

    # Improve layout
    plt.tight_layout()
    plt.show()


def calculate_error_statistics(U, V, errs_deg):
    """Calculate error statistics for plotting and analysis.

    Args:
        U, V: Error arrays in X and Y directions (in meters)
        errs_deg: Angular error array (in degrees)

    Returns:
        dict: Error statistics with 'mtr' and 'deg' keys
    """
    errs_mtr = np.sqrt(U**2 + V**2).flatten()
    errs_deg_flat = errs_deg.flatten()

    # Filter out NaN values (use same mask for both since they should correspond)
    valid_mask = ~(np.isnan(errs_mtr) | np.isnan(errs_deg_flat))

    if not np.any(valid_mask):
        return {
            "mtr": {"mean": np.nan, "max": np.nan, "std": np.nan, "median": np.nan},
            "deg": {"mean": np.nan, "max": np.nan, "std": np.nan, "median": np.nan},
        }

    valid_mtr = errs_mtr[valid_mask]
    valid_deg = errs_deg_flat[valid_mask]

    return {
        "mtr": {
            "mean": np.mean(valid_mtr),
            "max": np.max(valid_mtr),
            "std": np.std(valid_mtr),
            "median": np.median(valid_mtr),
        },
        "deg": {
            "mean": np.mean(valid_deg),
            "max": np.max(valid_deg),
            "std": np.std(valid_deg),
            "median": np.median(valid_deg),
        },
    }


def print_error_summary(errors, title="Error Summary"):
    """Print formatted error statistics summary.

    Args:
        errors: Dictionary with error statistics from calculate_error_statistics
        title: Title for the summary section
    """
    print(f"\n{title}:")
    print("  Error statistics (mm):")
    print(f"    Max:    {errors['mtr']['max'] * 1e3:.4f}")
    print(f"    Mean:   {errors['mtr']['mean'] * 1e3:.4f}")
    print(f"    Std:    {errors['mtr']['std'] * 1e3:.4f}")
    print(f"    Median: {errors['mtr']['median'] * 1e3:.4f}")
    print("  Error statistics (degrees):")
    print(f"    Max:    {errors['deg']['max']:.4f}")
    print(f"    Mean:   {errors['deg']['mean']:.4f}")
    print(f"    Std:    {errors['deg']['std']:.4f}")
    print(f"    Median: {errors['deg']['median']:.4f}")
