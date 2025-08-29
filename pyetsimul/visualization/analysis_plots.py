"""Analysis visualization utilities.

This module provides plotting functions for gaze tracking analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def _format_error_statistics(errors: Dict[str, Dict[str, float]], unit_str: str) -> str:
    """Format error statistics for plot titles."""
    return (
        f"Max: {errors['mtr']['max'] * 1e3:.2f} {unit_str} ({errors['deg']['max']:.3f}°), "
        f"Mean: {errors['mtr']['mean'] * 1e3:.2f} {unit_str} ({errors['deg']['mean']:.3f}°), "
        f"Std: {errors['mtr']['std'] * 1e3:.2f} {unit_str} ({errors['deg']['std']:.3f}°)"
    )


def plot_error_vectors_2d(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    errors: Dict[str, Dict[str, float]],
    angular_errors: np.ndarray,
    title_prefix: str = "",
    convert_to_mm: bool = True,
    width: float = 0.002,
    max_arrow_ratio: float = 0.3,
    mark_target_positions: bool = False,
    mark_predicted_positions: bool = False,
    show_grid: bool = True,
    auto_adjust_limits: bool = True,
    figure_size: Tuple[int, int] = (10, 8),
    xlabel: str = "Observer X position (mm)",
    ylabel: str = "Observer Y position (mm)",
) -> None:
    """Plot gaze tracking error vectors with adaptive scaling.

    Creates quiver plot showing error vectors at measurement points.

    Args:
        X, Y: Grid coordinates for vector positions
        U, V: Error components in X and Y directions
        errors: Dictionary with error statistics (from calculate_error_statistics)
        angular_errors: Angular errors (degrees) - used in error statistics
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

    # Compute scaling factor for arrows, filtering out NaN/Inf values
    U_flat = U_plot.flatten()
    V_flat = V_plot.flatten()
    magnitudes = np.sqrt(U_flat**2 + V_flat**2)

    # Extract finite values for scaling calculations
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    if len(finite_magnitudes) == 0:
        print("Warning: All error magnitudes are NaN/Inf, skipping plot")
        return

    max_magnitude = np.max(finite_magnitudes)
    plot_range_x = np.max(X_plot) - np.min(X_plot)
    plot_range_y = np.max(Y_plot) - np.min(Y_plot)
    plot_range = max(plot_range_x, plot_range_y)

    # Set target arrow length as fraction of plot range
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

    # Create quiver plot
    ax.quiver(
        X_pos,
        Y_pos,
        U_scaled,
        V_scaled,
        scale=1,
        scale_units="xy",
        angles="xy",
        alpha=1.0,
        width=width,
        color="darkblue",
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
    error_stats = _format_error_statistics(errors, unit_str)

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


def plot_error_vectors_3d(
    positions: np.ndarray,
    error_vectors: np.ndarray,
    angular_errors: np.ndarray,
    errors: Dict[str, Dict[str, float]],
    title_prefix: str = "",
    convert_to_mm: bool = True,
    max_arrow_ratio: float = 0.2,
    show_grid: bool = True,
    figure_size: Tuple[int, int] = (12, 10),
    position_labels: Tuple[str, str, str] = ("X position", "Y position", "Z position"),
) -> None:
    """Plot 3D gaze tracking error vectors with adaptive scaling.

    Creates 3D quiver plot showing error vectors in 3D space.

    Args:
        positions: Array of shape (N, 3) with [x, y, z] positions
        error_vectors: Array of shape (N, 3) with [dx, dy, dz] error vectors
        angular_errors: Array of shape (N,) with angular errors in degrees
        errors: Dictionary with error statistics (from calculate_error_statistics)
        title_prefix: Prefix text for plot title
        convert_to_mm: Convert coordinates and vectors to mm
        max_arrow_ratio: Maximum arrow length as fraction of plot range
        show_grid: Show grid lines
        figure_size: Figure size tuple
        position_labels: Labels for X, Y, Z axes
    """

    # Filter out invalid entries
    valid_mask = ~(np.isnan(positions).any(axis=1) | np.isnan(error_vectors).any(axis=1) | np.isnan(angular_errors))
    if not np.any(valid_mask):
        print("Warning: No valid data points for 3D plotting")
        return

    positions_valid = positions[valid_mask]
    error_vectors_valid = error_vectors[valid_mask]

    # Apply unit conversion if requested
    if convert_to_mm:
        positions_plot = positions_valid * 1000
        error_vectors_plot = error_vectors_valid * 1000
        unit_str = "mm"
    else:
        positions_plot = positions_valid
        error_vectors_plot = error_vectors_valid
        unit_str = "m"

    # Compute scaling factor for arrows
    magnitudes = np.linalg.norm(error_vectors_plot, axis=1)
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(finite_magnitudes) == 0:
        print("Warning: All error magnitudes are NaN/Inf, skipping plot")
        return

    max_magnitude = np.max(finite_magnitudes)

    # Calculate plot range in 3D
    plot_ranges = np.ptp(positions_plot, axis=0)  # range in each dimension
    plot_range = np.max(plot_ranges)

    # Set target arrow length as fraction of plot range
    target_arrow_length = plot_range * max_arrow_ratio

    if max_magnitude > target_arrow_length:
        scale_factor = target_arrow_length / max_magnitude
        error_vectors_scaled = error_vectors_plot * scale_factor
        scaling_applied = True
    else:
        scale_factor = 1.0
        error_vectors_scaled = error_vectors_plot
        scaling_applied = False

    # Create 3D figure
    plt.style.use("default")
    fig = plt.figure(figsize=figure_size, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Plot error vectors
    ax.quiver(
        positions_plot[:, 0],
        positions_plot[:, 1],
        positions_plot[:, 2],
        error_vectors_scaled[:, 0],
        error_vectors_scaled[:, 1],
        error_vectors_scaled[:, 2],
        color="darkblue",
        alpha=1.0,
        linewidth=1.0,
        arrow_length_ratio=0.05,
    )

    # Grid styling
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Set labels
    ax.set_xlabel(f"{position_labels[0]} ({unit_str})")
    ax.set_ylabel(f"{position_labels[1]} ({unit_str})")
    ax.set_zlabel(f"{position_labels[2]} ({unit_str})")

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    # Create comprehensive title with scaling info
    error_stats = _format_error_statistics(errors, unit_str)

    if scaling_applied:
        scale_info = f" (arrows scaled {scale_factor:.2f}×)"
    else:
        scale_info = " (arrows at full scale)"

    if title_prefix:
        title = f"{title_prefix}\n{error_stats}{scale_info}"
    else:
        title = f"3D Gaze Error Vectors\n{error_stats}{scale_info}"

    ax.set_title(title, pad=20)

    # Improve layout and show
    plt.tight_layout()
    plt.show()
