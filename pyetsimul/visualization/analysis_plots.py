"""Analysis visualization utilities.

This module provides plotting functions for gaze tracking analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np


def _format_error_statistics(errors: dict[str, dict[str, float]], unit_str: str) -> str:
    """Format error statistics for plot titles."""
    return (
        f"Max: {errors['deg']['max']:.3f}° ({errors['mtr']['max'] * 1e3:.2f} {unit_str}), "
        f"Mean: {errors['deg']['mean']:.3f}° ({errors['mtr']['mean'] * 1e3:.2f} {unit_str}), "
        f"Std: {errors['deg']['std']:.3f}° ({errors['mtr']['std'] * 1e3:.2f} {unit_str})"
    )


def plot_error_vectors_2d(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    errors: dict[str, dict[str, float]],
    title_prefix: str = "",
    convert_to_mm: bool = True,
    width: float = 0.002,
    max_arrow_ratio: float = 0.3,
    mark_target_positions: bool = False,
    mark_predicted_positions: bool = False,
    show_grid: bool = True,
    auto_adjust_limits: bool = True,
    figure_size: tuple[int, int] = (10, 8),
    xlabel: str = "Observer X position (mm)",
    ylabel: str = "Observer Y position (mm)",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot gaze tracking error vectors with adaptive scaling.

    Creates quiver plot showing error vectors at measurement points.

    Args:
        X: Grid coordinates for vector X positions
        Y: Grid coordinates for vector Y positions
        U: Error components in X direction
        V: Error components in Y direction
        errors: dictionary with error statistics (from calculate_error_statistics)
        title_prefix: Prefix text for plot title
        convert_to_mm: Convert coordinates and vectors to mm
        width: Arrow width
        max_arrow_ratio: Maximum arrow length as fraction of plot range
        mark_target_positions: Show blue crosses at target positions
        mark_predicted_positions: Show red dots at predicted positions
        show_grid: Show grid lines
        auto_adjust_limits: Automatically adjust plot limits
        figure_size: Figure size tuple
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: Optional matplotlib Axes to draw on. If None, a new figure is created.

    Returns:
        The matplotlib Figure.

    """
    # Apply unit conversion if requested
    if convert_to_mm:
        x_plot, y_plot = X * 1000, Y * 1000
        u_plot, v_plot = U * 1000, V * 1000
        unit_str = "mm"
    else:
        x_plot, y_plot = X, Y
        u_plot, v_plot = U, V
        unit_str = "m"

    # Compute scaling factor for arrows, filtering out NaN/Inf values
    u_flat = u_plot.flatten()
    v_flat = v_plot.flatten()
    magnitudes = np.sqrt(u_flat**2 + v_flat**2)

    # Extract finite values for scaling calculations
    finite_magnitudes = magnitudes[np.isfinite(magnitudes)]
    if len(finite_magnitudes) == 0:
        raise ValueError("All error magnitudes are NaN/Inf, cannot create plot")

    max_magnitude = np.max(finite_magnitudes)
    plot_range_x = np.max(x_plot) - np.min(x_plot)
    plot_range_y = np.max(y_plot) - np.min(y_plot)
    plot_range = max(plot_range_x, plot_range_y)

    # Set target arrow length as fraction of plot range
    target_arrow_length = plot_range * max_arrow_ratio

    if max_magnitude > target_arrow_length:
        scale_factor = target_arrow_length / max_magnitude
        u_scaled = u_plot * scale_factor
        v_scaled = v_plot * scale_factor
        scaling_applied = True
    else:
        scale_factor = 1.0
        u_scaled = u_plot
        v_scaled = v_plot
        scaling_applied = False

    # Create figure or use provided axes
    if ax is None:
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=figure_size, facecolor="white")
        ax.set_facecolor("white")
    else:
        fig = ax.get_figure()

    # Handle both gridded data (2D arrays) and scattered points (1D arrays)
    if u_scaled.ndim == 2:
        # Gridded data - create meshgrid
        xx, yy = np.meshgrid(x_plot, y_plot)
        x_pos, y_pos = xx, yy
    else:
        # Scattered points - use coordinates directly
        x_pos, y_pos = x_plot, y_plot

    # Create quiver plot
    ax.quiver(
        x_pos,
        y_pos,
        u_scaled,
        v_scaled,
        scale=1,
        scale_units="xy",
        angles="xy",
        alpha=1.0,
        width=width,
        color="darkblue",
    )

    # Add markers for target and predicted positions
    if mark_target_positions:
        ax.scatter(x_pos, y_pos, marker="+", s=50, c="blue", linewidths=2, alpha=0.8, label="Target")

    if mark_predicted_positions:
        # Calculate predicted positions (arrow tips)
        x_pred = x_pos + u_scaled
        y_pred = y_pos + v_scaled
        ax.scatter(x_pred, y_pred, marker="o", s=30, c="red", alpha=0.8, label="Predicted")

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
        arrow_tips_x = x_pos.flatten() + u_scaled.flatten()
        arrow_tips_y = y_pos.flatten() + v_scaled.flatten()

        all_x = np.concatenate([x_pos.flatten(), arrow_tips_x])
        all_y = np.concatenate([y_pos.flatten(), arrow_tips_y])

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

    scale_info = f" (arrows scaled {scale_factor:.2f}x)" if scaling_applied else " (arrows at full scale)"

    if title_prefix:
        title = f"{title_prefix}\n{error_stats}{scale_info}"
    else:
        title = f"Gaze Error Vectors\n{error_stats}{scale_info}"

    ax.set_title(title, pad=20)

    plt.tight_layout()

    return fig


def plot_error_vectors_3d(
    positions: np.ndarray,
    error_vectors: np.ndarray,
    angular_errors: np.ndarray,
    errors: dict[str, dict[str, float]],
    title_prefix: str = "",
    convert_to_mm: bool = True,
    max_arrow_ratio: float = 0.2,
    show_grid: bool = True,
    figure_size: tuple[int, int] = (12, 10),
    position_labels: tuple[str, str, str] = ("X position", "Y position", "Z position"),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot 3D gaze tracking error vectors with adaptive scaling.

    Creates 3D quiver plot showing error vectors in 3D space.

    Args:
        positions: Array of shape (N, 3) with [x, y, z] positions
        error_vectors: Array of shape (N, 3) with [dx, dy, dz] error vectors
        angular_errors: Array of shape (N,) with angular errors in degrees
        errors: dictionary with error statistics (from calculate_error_statistics)
        title_prefix: Prefix text for plot title
        convert_to_mm: Convert coordinates and vectors to mm
        max_arrow_ratio: Maximum arrow length as fraction of plot range
        show_grid: Show grid lines
        figure_size: Figure size tuple
        position_labels: Labels for X, Y, Z axes
        ax: Optional 3D matplotlib Axes to draw on. If None, a new figure is created.

    Returns:
        The matplotlib Figure.

    """
    # Filter out invalid entries
    valid_mask = ~(np.isnan(positions).any(axis=1) | np.isnan(error_vectors).any(axis=1) | np.isnan(angular_errors))
    if not np.any(valid_mask):
        raise ValueError("No valid data points for 3D plotting")

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
        raise ValueError("All error magnitudes are NaN/Inf, cannot create plot")

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

    # Create 3D figure or use provided axes
    if ax is None:
        plt.style.use("default")
        fig = plt.figure(figsize=figure_size, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
    else:
        fig = ax.get_figure()

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

    scale_info = f" (arrows scaled {scale_factor:.2f}x)" if scaling_applied else " (arrows at full scale)"

    if title_prefix:
        title = f"{title_prefix}\n{error_stats}{scale_info}"
    else:
        title = f"3D Gaze Error Vectors\n{error_stats}{scale_info}"

    ax.set_title(title, pad=20)

    # Improve layout
    plt.tight_layout()

    return fig
