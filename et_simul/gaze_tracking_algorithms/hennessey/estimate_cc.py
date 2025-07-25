"""Estimate cornea center using Hennessey's method."""

import numpy as np
from scipy.optimize import minimize


def estimate_cc_hennessey(c, l, cr, r_cornea_assumed=None):
    """Estimates pos. of cornea center (Hennessey's method).

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    cc_triang = estimate_cc_hennessey(c, l, cr, r_cornea_assumed) estimates
    position of the cornea center (returned in 'cc_triang') for a camera
    object 'c', a cell array 'l' of light sources, a cell array 'cr' with the
    positions of the corresponding corneal reflexes observed in the camera
    image and an assumed corneal radius of 'r_cornea_assumed'. If no corneal
    radius is given, a default value of 7.98 mm is used. The function uses
    the algorithm from [1].

    [1] Craig Hennessey, Borna Noureddin, Peter Lawrence. A Single Camera
        Eye-Gaze Tracking System with Free Head Motion. ETRA 2006

    Args:
        c: Camera object
        l: List of light sources
        cr: List with the positions of the corresponding corneal reflexes observed in the camera image
        r_cornea_assumed: Assumed corneal radius (default 7.98 mm)

    Returns:
        cc_triang: Position of the cornea center
    """
    # Line 31-33: if nargin<4 r_cornea_assumed=7.98e-3; end
    if r_cornea_assumed is None:
        r_cornea_assumed = 7.98e-3

    # Line 35: num_lights=length(l);
    num_lights = len(l)

    params = {
        "R": [],
        "alpha": np.zeros(num_lights),
        "l": np.zeros(num_lights),
        "r_cornea": r_cornea_assumed,
    }
    gx_start = np.zeros(num_lights)

    # For all corneal reflexes
    # Line 38: for j=1:num_lights
    for j in range(num_lights):
        # Check if CR was detected
        if cr[j] is None:
            return None

        # Compute vector to CR
        # Line 40: cr_hat=camera_unproject(c, cr{j}, 1);
        cr_hat = c.unproject(cr[j], 1.0)  # Should return 4x1 column vector
        # Line 41: cr_vec=cr_hat-c.trans(:,4);
        cr_vec = cr_hat - c.trans[:, 3].reshape(-1, 1)  # Ensure column vector
        # Line 42: cr_vec=cr_vec/norm(cr_vec);
        cr_vec = cr_vec / np.linalg.norm(cr_vec)

        # Compute vector to light
        # Line 45: l_vec=l{j}.pos-c.trans(:,4);
        light_pos = l[j]._pos_homogeneous
        # Ensure light_pos is column vector
        if light_pos.ndim == 1:
            light_pos = light_pos.reshape(-1, 1)
        l_vec = light_pos - c.trans[:, 3].reshape(-1, 1)

        # Define auxiliary coordinate system
        # Line 48: X=l_vec/norm(l_vec);
        X = l_vec / np.linalg.norm(l_vec)
        # Line 49: Z=cr_vec-X*(cr_vec'*X);
        # In MATLAB: cr_vec'*X is dot product giving scalar
        dot_product = (cr_vec.T @ X).item()  # Extract scalar from 1x1 matrix
        Z = cr_vec - X * dot_product
        # Line 50: Z=Z/norm(Z);
        Z = Z / np.linalg.norm(Z)
        # Line 51: Y=[cross(X(1:3), Z(1:3)); 0];
        Y_cross = np.cross(X[:3].flatten(), Z[:3].flatten())
        Y = np.append(Y_cross, 0).reshape(-1, 1)  # Make column vector
        # Line 52: params.R{j}=[X Y Z c.trans(:,4)];
        params["R"].append(
            np.column_stack([X.flatten(), Y.flatten(), Z.flatten(), c.trans[:, 3]])
        )

        # Compute other required quantities
        # Line 55: params.alpha(j)=acos(l_vec'*cr_vec/norm(l_vec));
        alpha_arg = (l_vec.T @ cr_vec / np.linalg.norm(l_vec)).item()  # Extract scalar
        params["alpha"][j] = np.arccos(alpha_arg)
        # Line 56: params.l(j)=norm(l_vec);
        params["l"][j] = np.linalg.norm(l_vec)

        # Line 58: gx_start(j)=norm(l_vec)/2;
        gx_start[j] = np.linalg.norm(l_vec) / 2

    # Line 61: params.r_cornea=r_cornea_assumed;
    # Already set above

    # Line 63-65: options=optimset('fminsearch'); options.TolFun=1e-8;
    #             gx_min=fminsearch(@(gx) objective_func(params, gx), gx_start, options);
    result = minimize(
        lambda gx: objective_func(params, gx)[0],
        gx_start,
        method="Nelder-Mead",
        options={"fatol": 1e-8},
    )
    gx_min = result.x

    # Line 67: [err, cc]=objective_func(params, gx_min);
    err, cc = objective_func(params, gx_min)

    # Line 69: cc_triang=mean(cc, 2);
    cc_triang = np.mean(cc, axis=1)

    return cc_triang


def objective_func(params, gx):
    """Objective function for cornea center estimation.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    Args:
        params: Parameters structure
        gx: Optimization variables

    Returns:
        Tuple of (val, cc) where val is the objective value and cc are the cornea centers
    """
    # Line 72: num_lights=length(gx);
    num_lights = len(gx)

    # Line 74: cc=zeros(4, num_lights);
    cc = np.zeros((4, num_lights))

    # Line 76: for j=1:num_lights
    for j in range(num_lights):
        # Line 77: beta=atan(gx(j)*tan(params.alpha(j))/(params.l(j)-gx(j)));
        beta = np.arctan(gx[j] * np.tan(params["alpha"][j]) / (params["l"][j] - gx[j]))

        # Line 79: cx=gx(j)-params.r_cornea*sin((params.alpha(j)-beta)/2);
        cx = gx[j] - params["r_cornea"] * np.sin((params["alpha"][j] - beta) / 2)
        # Line 80-81: cz=gx(j)*tan(params.alpha(j)) + ...
        #             params.r_cornea*cos((params.alpha(j)-beta)/2);
        cz = gx[j] * np.tan(params["alpha"][j]) + params["r_cornea"] * np.cos(
            (params["alpha"][j] - beta) / 2
        )

        # Line 83: cc(:,j)=params.R{j} * [cx; 0; cz; 1];
        cc[:, j] = params["R"][j] @ np.array([cx, 0, cz, 1])

    # Line 86: val=norm(cc(:,1)-cc(:,2));
    val = np.linalg.norm(cc[:, 0] - cc[:, 1])

    return val, cc
