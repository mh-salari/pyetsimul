"""Polynomial feature functions for interpolation eye tracking.

Based on various eye tracking calibration papers with different polynomial formulations.
"""

import numpy as np


def hennessey_2008(x, y):
    """Hennessey et al. (2008) polynomial: [x*y, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₀*x*y + a₁*x + a₂*y + a₃
    gaze_y = b₀*x*y + b₁*x + b₂*y + b₃

    Args:
        x, y: PCR vector components

    Returns:
        Feature vector for calibration/evaluation
    """
    return np.array([x * y, x, y, 1])


def hoorman_2008(x, y):
    """Hoorman et al. (2008) polynomial: [[x, 1], [y, 1]]

    Mathematical model (2D - separate features):
    gaze_x = a₀*x + a₁
    gaze_y = b₀*y + b₁

    Args:
        x, y: PCR vector components

    Returns:
        2x2 feature matrix
    """
    return np.array([[x, 1], [y, 1]])


def cerrolaza_2008(x, y):
    """Cerrolaza et al. (2008) polynomial: [x², y², x*y, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₀*x² + a₁*y² + a₂*x*y + a₃*x + a₄*y + a₅
    gaze_y = b₀*x² + b₁*y² + b₂*x*y + b₃*x + b₄*y + b₅

    Args:
        x, y: PCR vector components

    Returns:
        Feature vector for calibration/evaluation
    """
    return np.array([x**2, y**2, x * y, x, y, 1])


def second_order(x, y):
    """Second-order polynomial: [x²*y², x², y², x*y, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₀*x²*y² + a₁*x² + a₂*y² + a₃*x*y + a₄*x + a₅*y + a₆
    gaze_y = b₀*x²*y² + b₁*x² + b₂*y² + b₃*x*y + b₄*x + b₅*y + b₆

    Args:
        x, y: PCR vector components

    Returns:
        Feature vector for calibration/evaluation
    """
    return np.array([x**2 * y**2, x**2, y**2, x * y, x, y, 1])


def zhu_ji_2005(x, y):
    """Zhu and Ji (2005) polynomial: [[x*y, x, y, 1], [y², x, y, 1]]

    Mathematical model (2D - separate features):
    gaze_x = a₀*x*y + a₁*x + a₂*y + a₃
    gaze_y = b₀*y² + b₁*x + b₂*y + b₃

    Args:
        x, y: PCR vector components

    Returns:
        2x4 feature matrix
    """
    return np.array([[x * y, x, y, 1], [y**2, x, y, 1]])


def cerrolaza_villanueva_2008(x, y):
    """Cerrolaza and Villanueva (2008) polynomial: [[x², x, y, 1, 0], [x²*y, x², x*y, y, 1]]

    Mathematical model (2D - separate features):
    gaze_x = a₀*x² + a₁*x + a₂*y + a₃
    gaze_y = b₀*x²*y + b₁*x² + b₂*x*y + b₃*y + b₄

    Args:
        x, y: PCR vector components

    Returns:
        2x5 feature matrix
    """
    return np.array([[x**2, x, y, 1, 0], [x**2 * y, x**2, x * y, y, 1]])


def blignaut_wium_2013(x, y):
    """Blignaut and Wium (2013) polynomial: [[1, x, x³, y², x*y, 0, 0], [1, x, x², y, y², x*y, x²*y]]

    Mathematical model (2D - separate features):
    gaze_x = a₀ + a₁*x + a₂*x³ + a₃*y² + a₄*x*y
    gaze_y = b₀ + b₁*x + b₂*x² + b₃*y + b₄*y² + b₅*x*y + b₆*x²*y

    Args:
        x, y: PCR vector components

    Returns:
        2x7 feature matrix
    """
    return np.array([[1, x, x**3, y**2, x * y, 0, 0], [1, x, x**2, y, y**2, x * y, x**2 * y]])


# Dictionary mapping polynomial names to functions
POLYNOMIALS = {
    "hennessey_2008": hennessey_2008,
    "hoorman_2008": hoorman_2008,
    "cerrolaza_2008": cerrolaza_2008,
    "second_order": second_order,
    "zhu_ji_2005": zhu_ji_2005,
    "cerrolaza_villanueva_2008": cerrolaza_villanueva_2008,
    "blignaut_wium_2013": blignaut_wium_2013,
}


def get_polynomial(name="cerrolaza_2008"):
    """Get polynomial function by name.

    Args:
        name: Polynomial name (default: 'cerrolaza_2008')

    Returns:
        Polynomial function

    Raises:
        ValueError: If polynomial name is not recognized
    """
    if name not in POLYNOMIALS:
        available = ", ".join(POLYNOMIALS.keys())
        raise ValueError(f"Unknown polynomial '{name}'. Available: {available}")

    return POLYNOMIALS[name]
