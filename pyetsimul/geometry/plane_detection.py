"""
Automatic 2D plane detection for calibration targets.

This module provides utilities to automatically detect which 2D plane
calibration points lie in and extract appropriate coordinate mappings
for polynomial fitting.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from ..types.geometry import Position3D, Point3D


@dataclass
class PlaneInfo:
    """Information about detected calibration plane.

    Represents a 2D plane in 3D space for coordinate mapping and polynomial fitting.
    Automatically determines which axis is constant and which two axes vary.
    """

    plane_type: str  # "xy", "xz", or "yz"
    primary_axis: str  # First varying coordinate (e.g., "x")
    secondary_axis: str  # Second varying coordinate (e.g., "z")
    constant_axis: str  # Fixed coordinate (e.g., "y")
    constant_value: float  # Value of the constant coordinate

    def extract_2d_coords(self, position: Position3D) -> Tuple[float, float]:
        """Extract 2D coordinates for polynomial fitting.

        Maps 3D position to 2D coordinates based on plane orientation.
        """
        coords = {"x": position.x, "y": position.y, "z": position.z}
        return coords[self.primary_axis], coords[self.secondary_axis]

    def reconstruct_3d_point(self, coord1: float, coord2: float) -> Point3D:
        """Reconstruct 3D point from 2D polynomial prediction.

        Maps 2D polynomial output back to 3D space using plane information.
        """
        coords = {self.constant_axis: self.constant_value}
        coords[self.primary_axis] = coord1
        coords[self.secondary_axis] = coord2
        return Point3D(coords["x"], coords["y"], coords["z"])

    def serialize(self) -> dict:
        """Serialize plane info to dictionary."""
        return {
            "plane_type": self.plane_type,
            "primary_axis": self.primary_axis,
            "secondary_axis": self.secondary_axis,
            "constant_axis": self.constant_axis,
            "constant_value": self.constant_value,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PlaneInfo":
        """Deserialize from dictionary representation."""
        return cls(**data)


def detect_calibration_plane(calib_points: List[Position3D], tolerance: float = 1e-6) -> PlaneInfo:
    """Automatically detect which 2D plane the calibration points lie in.

    Uses variance analysis to determine which axis is constant and which two axes vary.
    Supports standard orthogonal planes (xy, xz, yz) for polynomial interpolation.

    Args:
        calib_points: List of calibration target positions
        tolerance: Variance threshold for considering an axis constant

    Returns:
        PlaneInfo: Information about the detected plane and coordinate mappings

    Raises:
        ValueError: If points don't lie in exactly one of the standard 2D planes
    """
    if len(calib_points) < 3:
        raise ValueError("Need at least 3 calibration points to detect plane")

    # Convert to numpy array for analysis
    points = np.array([[p.x, p.y, p.z] for p in calib_points])

    # Calculate variance along each axis
    variances = np.var(points, axis=0)
    x_var, y_var, z_var = variances

    # Check which axes are constant (low variance)
    constant_axes = []
    axis_names = ["x", "y", "z"]

    for i, var in enumerate(variances):
        if var < tolerance:
            constant_axes.append(axis_names[i])

    # Must have exactly one constant axis for a valid 2D plane
    if len(constant_axes) == 0:
        raise ValueError(
            f"Calibration points don't lie in a 2D plane. "
            f"All axes vary significantly: x_var={x_var:.2e}, y_var={y_var:.2e}, z_var={z_var:.2e}. "
            f"Points must lie in xy, xz, or yz plane for polynomial interpolation."
        )
    elif len(constant_axes) > 1:
        raise ValueError(
            f"Calibration points are too constrained. "
            f"Multiple axes are constant: {constant_axes}. "
            f"Points must vary in exactly 2 dimensions for polynomial interpolation."
        )

    # Determine plane type and coordinate mapping
    constant_axis = constant_axes[0]
    varying_axes = [axis for axis in axis_names if axis != constant_axis]

    # Get the constant value (mean of the constant axis)
    constant_idx = axis_names.index(constant_axis)
    constant_value = float(np.mean(points[:, constant_idx]))

    # Create plane type string (alphabetically sorted for consistency)
    plane_type = "".join(sorted(varying_axes))

    return PlaneInfo(
        plane_type=plane_type,
        primary_axis=varying_axes[0],  # First alphabetically
        secondary_axis=varying_axes[1],  # Second alphabetically
        constant_axis=constant_axis,
        constant_value=constant_value,
    )


def summarize_plane_detection(calib_points: List[Position3D], plane_info: PlaneInfo) -> str:
    """Create a human-readable summary of the plane detection results.

    Generates formatted output showing plane type, coordinate mapping, and coverage area
    for logging and display purposes.

    Args:
        calib_points: List of calibration target positions
        plane_info: Information about the detected plane

    Returns:
        str: Formatted summary for logging/display
    """
    axis_labels = {"x": "Horizontal (X)", "y": "Depth (Y)", "z": "Vertical (Z)"}

    primary_label = axis_labels[plane_info.primary_axis]
    secondary_label = axis_labels[plane_info.secondary_axis]
    constant_label = axis_labels[plane_info.constant_axis]

    # Extract coordinate ranges
    coords_2d = np.array([plane_info.extract_2d_coords(p) for p in calib_points])
    ranges = np.ptp(coords_2d, axis=0) * 1000  # Convert to mm

    summary = [
        f"Calibration plane: {plane_info.plane_type.upper()}",
        f"  Varying: {primary_label}, {secondary_label}",
        f"  Fixed: {constant_label} = {plane_info.constant_value * 1000:.1f}mm",
        f"  Coverage: {ranges[0]:.1f}mm × {ranges[1]:.1f}mm",
    ]

    return "\n".join(summary)
