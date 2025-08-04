"""Estimate cornea center using Hennessey's method."""

import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Dict, Any, Tuple
from ...types import Position3D, Point2D


def estimate_cc_hennessey(
    c, l: List, cr: List[Optional[Point2D]], r_cornea_assumed: Optional[float] = None
) -> Optional[Position3D]:
    """Estimates position of cornea center using Hennessey's method.

    Based on: Craig Hennessey, Borna Noureddin, Peter Lawrence.
    "A Single Camera Eye-Gaze Tracking System with Free Head Motion." ETRA 2006.

    Args:
        c: Camera object
        l: List of light sources
        cr: List of corneal reflection points (Point2D or None)
        r_cornea_assumed: Assumed corneal radius in meters (default: 7.98mm)

    Returns:
        Estimated corneal center position as Position3D, or None if estimation fails
    """
    if r_cornea_assumed is None:
        r_cornea_assumed = 7.98e-3

    num_lights = len(l)
    params = {}
    params["R"] = []
    params["alpha"] = np.zeros(num_lights)
    params["l"] = np.zeros(num_lights)
    gx_start = np.zeros(num_lights)

    for j in range(num_lights):
        if cr[j] is None:
            return None

        # Compute vector to CR using structured types
        cr_hat = c.unproject(cr[j], 1.0)  # Returns Position3D
        camera_position = Position3D(c.trans[0, 3], c.trans[1, 3], c.trans[2, 3])
        cr_vec = cr_hat - camera_position
        cr_vec = cr_vec.normalize()

        # Compute vector to light using structured types
        light_pos = l[j].position  # Position3D
        l_vec = light_pos - camera_position
        l_vec_norm = l_vec.magnitude()
        X = l_vec / l_vec_norm  # Now we can use proper division

        # Define auxiliary coordinate system using structured types
        dot_product = cr_vec.dot(X)
        Z = cr_vec - X * dot_product
        Z = Z.normalize()

        # Cross product for Y axis
        Y_cross = np.cross(X.to_array()[:3], Z.to_array()[:3])
        Y = Position3D(Y_cross[0], Y_cross[1], Y_cross[2])

        # Build transformation matrix using structured types
        R_matrix = np.eye(4)  # Start with 4x4 identity matrix
        R_matrix[:3, :3] = np.column_stack([X.to_array()[:3], Y.to_array()[:3], Z.to_array()[:3]])
        R_matrix[:3, 3] = camera_position.to_array()[:3]  # Translation part
        params["R"].append(R_matrix)

        # Compute other required quantities using structured types
        alpha_arg = cr_vec.dot(l_vec) / l_vec_norm
        params["alpha"][j] = np.arccos(alpha_arg)
        params["l"][j] = l_vec_norm
        gx_start[j] = l_vec_norm / 2

    params["r_cornea"] = r_cornea_assumed

    result = minimize(
        lambda gx: objective_func(params, gx)[0],
        gx_start,
        method="Nelder-Mead",
        options={"fatol": 1e-8},
    )
    gx_min = result.x

    err, cc = objective_func(params, gx_min)
    cc_triang = np.mean(cc, axis=1)

    # Convert to Position3D for structured type consistency
    result = Position3D(cc_triang[0], cc_triang[1], cc_triang[2])

    return result


def objective_func(params: Dict[str, Any], gx: np.ndarray) -> Tuple[float, np.ndarray]:
    """Objective function for cornea center estimation.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    Args:
        params: Parameters structure
        gx: Optimization variables

    Returns:
        Tuple of (val, cc) where val is the objective value and cc are the cornea centers as numpy array
    """
    num_lights = len(gx)
    cc = np.zeros((4, num_lights))

    for j in range(num_lights):
        beta = np.arctan(gx[j] * np.tan(params["alpha"][j]) / (params["l"][j] - gx[j]))

        cx = gx[j] - params["r_cornea"] * np.sin((params["alpha"][j] - beta) / 2)
        cz = gx[j] * np.tan(params["alpha"][j]) + params["r_cornea"] * np.cos((params["alpha"][j] - beta) / 2)

        cc[:, j] = params["R"][j] @ np.array([cx, 0, cz, 1])

    val = np.linalg.norm(cc[:, 0] - cc[:, 1])

    return val, cc
