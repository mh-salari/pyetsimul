"""Utility functions for performance analysis.

This module provides helper functions used across performance analysis modules
to ensure consistency and reduce code duplication.
"""

import numpy as np


def _compute_stats(values: np.ndarray) -> dict[str, float]:
    """Compute basic statistics for error values."""
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def calculate_error_statistics(
    U: np.ndarray, V: np.ndarray, angular_errors: np.ndarray
) -> dict[str, dict[str, float]]:
    """Calculate gaze tracking error statistics.

    Args:
        U: Error array in X direction (in mm)
        V: Error array in Y direction (in mm)
        angular_errors: Angular error array (in degrees)

    Returns:
        dict: Error statistics with 'mm' and 'deg' keys containing mean, max, std, median

    """
    euclidean_errors = np.sqrt(U**2 + V**2).flatten()
    angular_errors_flat = angular_errors.flatten()

    valid_mask = ~(np.isnan(euclidean_errors) | np.isnan(angular_errors_flat))

    if not np.any(valid_mask):
        nan_stats = {"mean": np.nan, "max": np.nan, "std": np.nan, "median": np.nan}
        return {"mm": nan_stats, "deg": nan_stats}

    valid_euclidean = euclidean_errors[valid_mask]
    valid_angular = angular_errors_flat[valid_mask]

    return {
        "mm": _compute_stats(valid_euclidean),
        "deg": _compute_stats(valid_angular),
    }
