"""Hennessey Prediction algorithm.

Prediction function for Hennessey et al.'s method.
Exactly matches MATLAB hennessey_eval() and hennessey_eval_base().
"""

import numpy as np
from skimage.measure import EllipseModel
from et_simul.geometry.intersections import intersect_ray_sphere, intersect_ray_plane
from et_simul.optics.refractions import refract_ray_sphere
from .estimate_cc import estimate_cc_hennessey


def hennessey_predict_base(et, camimg):
    """Prediction function helper for Hennessey et al.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    Args:
        et: Eye tracker structure
        camimg: Camera image data

    Returns:
        Tuple of (gaze, cc_estim, gaze3d)
    """
    # Line 21: r_cornea_assumed=7.98e-3*(1+et.state.parameter_err);
    r_cornea_assumed = 7.98e-3 * (1 + et.state["parameter_err"])
    # Line 22: rpc_assumed=4.44e-3*(1+et.state.parameter_err);
    rpc_assumed = 4.44e-3 * (1 + et.state["parameter_err"])

    # Find CC
    # Line 25-26: cc_estim=estimate_cc_hennessey(et.cameras{1}, et.lights, camimg{1}.cr, ...
    #             r_cornea_assumed);
    cc_estim = estimate_cc_hennessey(
        et.cameras[0], et.lights, camimg[0]["cr"], r_cornea_assumed
    )

    # Check if CR estimation failed
    if cc_estim is None:
        return None, None, None

    # Find position of PC using ray-sphere-intersection
    # Line 29: switch et.state.pupil_alg
    pupil_alg = et.state["pupil_alg"]

    if pupil_alg == "hennessey":
        # Bring CC into camera coordinate system and compute distance
        # Line 32: cc_cam=et.cameras{1}.trans\cc_estim;
        cc_cam = np.linalg.solve(et.cameras[0].trans, cc_estim)
        # Line 33: d_cc=-cc_cam(3);
        d_cc = -cc_cam[2]  # Python uses 0-based indexing

        # Fit ellipse to pupil image and compute radius of pupil
        # Do Hennessey et al. take into account that the cornea magnifies
        # the image of the pupil? At any rate, we apply an empirical
        # correction to try and compensate for this.
        # Line 39: a=fitellipse_hf(camimg{1}.pupil(1,:), camimg{1}.pupil(2,:));
        points = np.column_stack((camimg[0]["pupil"][0, :], camimg[0]["pupil"][1, :]))
        ellipse = EllipseModel()
        if ellipse.estimate(points):
            # Line 40: r_pupil=max(a(3:4));
            # EllipseModel.params = [xc, yc, a, b, theta], so semi-axes are at indices 2:4
            r_pupil = max(ellipse.params[2:4])
        else:
            return None  # Failed to fit ellipse
        # Line 41: r_pupil=r_pupil/et.cameras{1}.focal_length*d_cc;
        r_pupil = r_pupil / et.cameras[0].focal_length * d_cc
        # Empirical correction
        # Line 43: r_pupil=r_pupil/1.165;
        r_pupil = r_pupil / 1.165

        # Unproject pupil contour
        # Line 46-48: pupil_rays= ...
        #             camera_unproject(et.cameras{1}, camimg{1}.pupil, 1.0) - ...
        #             repmat(et.cameras{1}.trans(:,4), 1, size(camimg{1}.pupil, 2));
        unprojected = et.cameras[0].unproject(camimg[0]["pupil"], 1.0)
        camera_pos = np.tile(
            et.cameras[0].trans[:, 3:4], (1, camimg[0]["pupil"].shape[1])
        )
        pupil_rays = unprojected - camera_pos

        # Initialize array of pupil points
        # Line 51: pupil_points=zeros(size(pupil_rays));
        # Line 54: pupil_points=zeros(4,0);
        pupil_points = np.zeros((4, 0))

        # Refract pupil contour rays
        # Line 55: for j=1:size(pupil_rays, 2)
        for j in range(pupil_rays.shape[1]):
            # Refract ray at cornea. If we don't hit the cornea, ignore the ray.
            # Line 58-59: [U0, Ud]=refract_ray_sphere(et.cameras{1}.trans(:,4), ...
            #             pupil_rays(:,j), cc_estim, r_cornea_assumed, 1, 1.376);
            U0, Ud = refract_ray_sphere(
                et.cameras[0].trans[:, 3],
                pupil_rays[:, j],
                cc_estim,
                r_cornea_assumed,
                1,
                1.376,
            )
            # Line 60-62: if isempty(U0) continue; end
            if U0 is None:
                continue

            # Intersect refracted ray with sphere around cornea center to
            # find pupil contour point. If we don't hit the sphere, ignore the ray.
            # Line 67-68: pt=intersect_ray_sphere(U0, Ud, cc_estim, ...
            #             sqrt(rpc_assumed^2+r_pupil^2));
            pt_tuple = intersect_ray_sphere(
                U0, Ud, cc_estim, np.sqrt(rpc_assumed**2 + r_pupil**2)
            )
            # Line 69-71: if ~isempty(pt) pupil_points(:,j)=pt; end
            if pt_tuple[0] is not None:  # Check if intersection exists
                pt = pt_tuple[0]  # Take the closer intersection
                if pupil_points.shape[1] <= j:
                    # Expand pupil_points array as needed
                    new_points = np.zeros((4, j + 1))
                    new_points[:, : pupil_points.shape[1]] = pupil_points
                    pupil_points = new_points
                # Convert 3D point to homogeneous coordinates if needed
                if pt.shape[0] == 3:
                    pt = np.append(pt, 1)
                pupil_points[:, j] = pt

        # Line 74: pc_estim=mean(pupil_points,2);
        # Only use non-zero columns
        valid_cols = np.any(pupil_points != 0, axis=0)
        if np.any(valid_cols):
            pc_estim = np.mean(pupil_points[:, valid_cols], axis=1)
        else:
            pc_estim = np.zeros(4)

    elif pupil_alg == "pupil_center":
        # This doesn't use the actual Hennessey algorithm. Instead, we
        # reproject the pupil center into the eye and intersect with a
        # sphere of radius r_pc around the cornea center
        # Line 79-80: dir=camera_unproject(et.cameras{1}, camimg{1}.pc, 1.0) - ...
        #             et.cameras{1}.trans(:,4);
        unprojected_pc = et.cameras[0].unproject(camimg[0]["pc"], 1.0)
        dir = unprojected_pc - et.cameras[0].trans[:, 3]
        # Line 81-82: [U0, Ud]=refract_ray_sphere(et.cameras{1}.trans(:,4), dir, ...
        #             cc_estim, r_cornea_assumed, 1, 1.376);
        U0, Ud = refract_ray_sphere(
            et.cameras[0].trans[:, 3], dir, cc_estim, r_cornea_assumed, 1, 1.376
        )
        # Line 83: pc_estim=intersect_ray_sphere(U0, Ud, cc_estim, rpc_assumed);
        pc_estim_tuple = intersect_ray_sphere(U0, Ud, cc_estim, rpc_assumed)
        if pc_estim_tuple[0] is not None:
            pc_estim = pc_estim_tuple[0]  # Take closer intersection
            # Convert to homogeneous coordinates if needed
            if pc_estim.shape[0] == 3:
                pc_estim = np.append(pc_estim, 1)
        else:
            pc_estim = np.zeros(4)
    else:
        # Line 85: error('Unknown pupil finder algorithm');
        raise ValueError(f"Unknown pupil finder algorithm: {pupil_alg}")

    # Compute 3D direction of gaze
    # Line 89: gaze3d=pc_estim-cc_estim;
    gaze3d = pc_estim - cc_estim
    # Line 90: gaze3d=gaze3d/norm(gaze3d);
    gaze3d = gaze3d / np.linalg.norm(gaze3d)

    # Line 92-93: x=intersect_ray_plane(cc_estim, gaze3d, [0 0 0 1]', ...
    #             [0 1 0 0]');
    x = intersect_ray_plane(
        cc_estim, gaze3d, np.array([0, 0, 0, 1]), np.array([0, 1, 0, 0])
    )
    # Line 94: gaze=[x(1) x(3)]';
    gaze = np.array([x[0], x[2]])

    return gaze, cc_estim, gaze3d


def hennessey_predict_main(et, camimg):
    """Prediction function for Hennessey et al.'s method.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    Args:
        et: Eye tracker structure
        camimg: Camera image data

    Returns:
        2D gaze position
    """
    # Line 21: [gaze, cc_estim, gaze3d]=hennessey_eval_base(et, camimg);
    gaze, cc_estim, gaze3d = hennessey_predict_base(et, camimg)

    # Check if base Prediction failed
    if gaze is None:
        return None

    # For hennessey config (recalib_type='hennessey'), skip 3D gaze vector recalibration
    # Only other recalib_types ('angle', 'henn_angle', 'henn3d') modify gaze3d here
    # hennessey config applies recalibration to final 2D gaze position instead

    # Intersect the gaze line with the plane to see where it hits
    # Line 35-36: x=intersect_ray_plane(cc_estim, gaze3d, [0 0 0 1]', ...
    #             [0 1 0 0]');
    x = intersect_ray_plane(
        cc_estim, gaze3d, np.array([0, 0, 0, 1]), np.array([0, 1, 0, 0])
    )
    # Line 37: gaze=[x(1) x(3)]';
    gaze = np.array([x[0], x[2]])

    # Apply hennessey recalibration to final 2D gaze position
    # Line 40-42: switch et.state.recalib_type ... case 'hennessey'
    #             gaze=recalib_hennessey_eval(et.state.recalib_hennessey, gaze);
    recalib_type = et.state["recalib_type"]
    if recalib_type == "hennessey":
        gaze = recalib_hennessey_eval(et.state["recalib_hennessey"], gaze)

    return gaze


def recalib_hennessey_eval(state, gaze):
    """Apply Hennessey's recalibration procedure.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    Args:
        state: Recalibration state from recalib_hennessey_calib
        gaze: 2D gaze position to be corrected

    Returns:
        Corrected 2D gaze position
    """
    if not state or "gaze_measured" not in state or "offsets" not in state:
        return gaze

    # Line 25: d=state.gaze_measured-repmat(gaze, 1, size(state.gaze_measured,2));
    gaze_expanded = np.tile(gaze.reshape(-1, 1), (1, state["gaze_measured"].shape[1]))
    d = state["gaze_measured"] - gaze_expanded
    # Line 26: d=sqrt(sum(d.^2,1));
    d = np.sqrt(np.sum(d**2, axis=0))

    # Line 30: I=find(d<1e-8);
    I = np.where(d < 1e-8)[0]
    # Line 31-33: if ~isempty(I) weights=zeros(size(d)); weights(I(1))=1;
    if len(I) > 0:
        weights = np.zeros(d.shape)
        weights[I[0]] = 1
    else:
        # Line 37: weights=1./d;
        weights = 1.0 / d
        # Line 38: weights=weights/sum(weights);
        weights = weights / np.sum(weights)

    # Line 42: gaze=gaze+sum(state.offsets.*repmat(weights,2,1), 2);
    weights_expanded = np.tile(weights.reshape(1, -1), (2, 1))
    weighted_offsets = state["offsets"] * weights_expanded
    gaze = gaze + np.sum(weighted_offsets, axis=1)

    return gaze
