"""Hennessey calibration algorithm.

Calibration function for Hennessey et al.'s method.
"""

import numpy as np
from .prediction import _predict_base


def hennessey_calib(et, calib_data):
    """Calibration function for Hennessey et al.'s method.




    Args:
        et: Eye tracker structure
        calib_data: Calibration data from et_calib
    """
    # Line 21: gaze_measured=zeros(size(et.calib_points));
    gaze_measured = np.zeros(et.calib_points.shape)
    # Line 22: gaze3d_measured=zeros(4, size(et.calib_points, 2));
    gaze3d_measured = np.zeros((4, et.calib_points.shape[1]))
    # Line 23: gaze3d_desired=zeros(4, size(et.calib_points, 2));
    gaze3d_desired = np.zeros((4, et.calib_points.shape[1]))

    # Determine offsets for each of the four calibration points
    # Line 26: for i=1:size(et.calib_points,2)
    for i in range(et.calib_points.shape[1]):
        # Line 27-28: [gaze_measured(:,i), cc_estim, gaze3d_measured(:,i)]= ...
        #             hennessey_eval_base(et, calib_data{i}.camimg);
        gaze_result, cc_estim, gaze3d_result, _, _ = _predict_base(et, calib_data[i]["camimg"])

        # Skip this calibration point if prediction failed
        if gaze_result is None:
            continue

        gaze_measured[:, i] = gaze_result
        gaze3d_measured[:, i] = gaze3d_result

        # Line 29-31: gaze3d_desired(:,i)= ...
        #             [et.calib_points(1,i) 0 et.calib_points(2,i) 1]' - ...
        #             cc_estim;
        gaze3d_desired[:, i] = np.array([et.calib_points[0, i], 0, et.calib_points[1, i], 1]) - cc_estim

    # For hennessey config, only compute the hennessey recalibration that will actually be used
    # Line 34-35: et.state.recalib_hennessey=recalib_hennessey_calib(gaze_measured, ...
    #             et.calib_points);
    et.state["recalib_hennessey"] = recalib_hennessey_calib(gaze_measured, et.calib_points)

    # Note: Other recalibration types are computed in original MATLAB but not used for hennessey config
    # Skipping them to keep implementation focused


def recalib_hennessey_calib(gaze_measured, gaze_desired):
    """Calibrate Hennessey's recalibration procedure.



    Args:
        gaze_measured: 2×n matrix of estimated gaze positions on screen
        gaze_desired: 2×n matrix of corresponding true gaze positions

    Returns:
        Dictionary containing recalibration state
    """
    # Line 29: state.gaze_measured=gaze_measured;
    # Line 30: state.offsets=gaze_desired-gaze_measured;
    return {"gaze_measured": gaze_measured, "offsets": gaze_desired - gaze_measured}
