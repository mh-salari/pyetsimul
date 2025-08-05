"""Eye-specific surface point generation utilities that handle transformations."""

import numpy as np
from typing import Callable
from ..types import Position3D
from . import surface_points


def generate_corneal_surface_points(
    eye, intersection_func: Callable, surface_type: str = "anterior", n_points: int = 50
) -> np.ndarray:
    """Generate corneal surface points with proper eye transformation handling.

    Args:
        eye: Eye object with cornea and transformation matrix
        intersection_func: Ray intersection function (intersect_ray_sphere, intersect_ray_conic, etc.)
        surface_type: "anterior" or "posterior" surface
        n_points: Number of sampling points per dimension

    Returns:
        points: (N, 3) array of surface points in world coordinates
    """
    # Get local corneal center and radius
    if surface_type == "anterior":
        center_local = eye.cornea.center
        radius = eye.cornea.anterior_radius
    elif surface_type == "posterior":
        center_local = eye.cornea.get_posterior_center()
        radius = eye.cornea.posterior_radius
    else:
        raise ValueError(f"Unknown surface_type '{surface_type}'. Use 'anterior' or 'posterior'")

    # Generate surface points in local coordinates first
    if hasattr(eye.cornea, "anterior_k"):  # Conic cornea
        if surface_type == "anterior":
            k_value = eye.cornea.anterior_k
        else:
            k_value = eye.cornea.posterior_k
        local_points = surface_points.generate(intersection_func, center_local, radius, k_value, n_points=n_points)
    else:  # Spherical cornea
        local_points = surface_points.generate(intersection_func, center_local, radius, n_points=n_points)
    
    # Transform all points to world coordinates
    if len(local_points) > 0:
        # Convert to homogeneous coordinates (add w=1)
        local_homogeneous = np.column_stack([local_points, np.ones(len(local_points))])
        # Transform to world coordinates
        world_homogeneous = (eye.trans @ local_homogeneous.T).T
        # Return only x,y,z coordinates
        return world_homogeneous[:, :3]
    else:
        return local_points


def get_transformed_corneal_landmarks(eye):
    """Get corneal landmark positions transformed to world coordinates.

    Args:
        eye: Eye object with cornea and transformation matrix

    Returns:
        dict: Dictionary with world-coordinate positions of corneal landmarks
    """
    # Get local positions
    anterior_center_local = eye.cornea.center
    posterior_center_local = eye.cornea.get_posterior_center()
    apex_local = eye.cornea.get_apex_position()

    landmarks = {}

    # Transform each landmark to world coordinates
    for name, pos_local in [
        ("anterior_center", anterior_center_local),
        ("posterior_center", posterior_center_local),
        ("apex", apex_local),
    ]:
        pos_homogeneous = np.array([pos_local.x, pos_local.y, pos_local.z, 1.0])
        pos_world_homogeneous = eye.trans @ pos_homogeneous
        landmarks[name] = Position3D(pos_world_homogeneous[0], pos_world_homogeneous[1], pos_world_homogeneous[2])

    return landmarks
