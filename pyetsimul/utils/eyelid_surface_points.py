"""Eyelid surface point generation utilities.

Provides functions for generating eyelid surface points and opening boundaries
for visualization and analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def generate_eyelid_points_local(eyelid, n_points: int = 500, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate eyelid surface points as (N, 3) array in local coordinates.

    Sample full sphere, keep front hemisphere points
    that are outside the rotated elliptical opening, keep all back hemisphere points.

    Args:
        eyelid: Eyelid object with sphere and ellipse parameters
        n_points: Number of surface points to generate
        rng: Random number generator (None = use default)

    Returns:
        Array of shape (M, 3) where M <= n_points containing eyelid surface points
    """
    if rng is None:
        rng = np.random.default_rng()

    C = np.array([eyelid.center.x, eyelid.center.y, eyelid.center.z], dtype=float)
    S = float(eyelid.sphere_radius)

    phi_vals = rng.uniform(0.0, np.pi, n_points)
    theta_vals = rng.uniform(0.0, 2.0 * np.pi, n_points)

    # No rotation angle - ellipse aligned with coordinate axes
    cos_a = 1.0
    sin_a = 0.0

    # Ellipse parameters used for masking
    width, height = eyelid.ellipse_axes()
    y_center = eyelid.ellipse_center_offset()

    eyelid_points = []
    for i in range(n_points):
        phi = phi_vals[i]
        theta = theta_vals[i]

        x = C[0] + S * np.sin(phi) * np.cos(theta)
        y = C[1] + S * np.cos(phi)
        z = C[2] - S * np.sin(phi) * np.sin(theta)

        if z <= C[2]:
            # Apply in-plane rotation for ellipse test
            x_rot = (x - C[0]) * cos_a - (y - C[1]) * sin_a
            y_rot = (x - C[0]) * sin_a + (y - C[1]) * cos_a

            if width > 0.0 and height > 0.0:
                y_rel = y_rot - y_center
                ellipse_test = (x_rot / (width / 2.0)) ** 2 + (y_rel / (height / 2.0)) ** 2
                if ellipse_test <= 1.0:
                    continue  # Skip points inside opening

        eyelid_points.append([x, y, z])

    return np.array(eyelid_points, dtype=float)


def generate_eyelid_opening_edge_local(eyelid, n_edge_points: int = 100) -> np.ndarray:
    """Generate opening boundary points on the sphere as (M, 3) array.

    Projects a tilted ellipse onto the spherical surface (front hemisphere).

    Args:
        eyelid: Eyelid object with sphere and ellipse parameters
        n_edge_points: Number of edge points to generate

    Returns:
        Array of shape (n_edge_points, 3) containing opening boundary points
    """
    C = np.array([eyelid.center.x, eyelid.center.y, eyelid.center.z], dtype=float)
    S = float(eyelid.sphere_radius)
    width, height = eyelid.ellipse_axes()
    y_center = eyelid.ellipse_center_offset()
    # No rotation angle - ellipse aligned with coordinate axes
    cos_a = 1.0
    sin_a = 0.0

    opening_edge_points = []

    # Anatomical footprint radius in XY (small circle at phi_max)
    r_xy = S * np.sin(float(eyelid.phi_max))

    for i in range(n_edge_points):
        t = (i / n_edge_points) * 2.0 * np.pi
        x_ellipse = (width / 2.0) * np.cos(t)
        y_ellipse = (height / 2.0) * np.sin(t)

        x_rot = x_ellipse * cos_a - y_ellipse * sin_a
        y_rot = x_ellipse * sin_a + y_ellipse * cos_a
        y_rot += y_center  # shift ellipse center to keep lower edge stationary

        # Constrain to footprint disk so edge aligns with filled surface
        r = np.hypot(x_rot, y_rot)
        if r > r_xy and r > 0:
            scale_xy = r_xy / r
            x_rot *= scale_xy
            y_rot *= scale_xy

        r_sq = x_rot * x_rot + y_rot * y_rot
        # Project to front hemisphere of the sphere
        z_sphere = C[2] - np.sqrt(max(S * S - r_sq, 0.0))
        opening_edge_points.append([C[0] + x_rot, C[1] + y_rot, z_sphere])

    return np.array(opening_edge_points, dtype=float)


def transform_eyelid_points_to_world(points_local: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """Transform eyelid points from local to world coordinates.

    Args:
        points_local: Array of shape (N, 3) with local coordinates
        transformation_matrix: 4x4 transformation matrix (local→world)

    Returns:
        Array of shape (N, 3) with world coordinates
    """
    if len(points_local) == 0:
        return points_local

    # Add homogeneous coordinate
    points_h = np.column_stack([points_local, np.ones(len(points_local))])

    # Transform to world coordinates
    world_h = (transformation_matrix @ points_h.T).T

    # Return 3D coordinates
    return world_h[:, :3]
