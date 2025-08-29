"""Utility functions for calibration analysis.

This module provides utility functions for calibration analysis,
including parameter printing and data formatting functions.
"""

from pyetsimul.core import EyeTracker
from tabulate import tabulate


def pprint_polynomial_parameters(et: EyeTracker) -> None:
    """Print polynomial parameters from calibrated eye tracker.

    Displays calibration coefficients and polynomial type for analysis.
    Shows calibration status and coefficient values for debugging.
    """
    # Only print polynomial parameters for interpolation trackers
    if et.algorithm_name != "interpolation":
        print(f"\nCalibration Parameters ({et.algorithm_name}):")
        print("-" * 40)
        if et.algorithm_state.is_calibrated:
            print("Calibration status: Calibrated")
            print("Algorithm: Geometric gaze estimation (non-polynomial)")
        else:
            print("Calibration status: ✗ Not calibrated")
        return

    print("\nPolynomial Parameters:")

    # General info table
    headers = ["Parameter", "Value"]
    general_data = [
        ["Polynomial type", et.polynomial_name],
        ["Calibration status", "Calibrated" if et.algorithm_state.is_calibrated else "✗ Not calibrated"],
    ]

    if et.algorithm_state.is_calibrated:
        state = et.algorithm_state
        if state.x_coefficients is not None and state.y_coefficients is not None:
            general_data.extend(
                [
                    ["X coefficients shape", f"{state.x_coefficients.shape}"],
                    ["Y coefficients shape", f"{state.y_coefficients.shape}"],
                ]
            )

    print(tabulate(general_data, headers=headers, tablefmt="simple"))

    # Coefficients table if available
    if et.algorithm_state.is_calibrated:
        state = et.algorithm_state
        if state.x_coefficients is not None and state.y_coefficients is not None:
            print("\nCoefficients Values:")
            coeff_headers = ["Index", "X Coefficient", "Y Coefficient"]
            coeff_data = []
            for i, (x_val, y_val) in enumerate(zip(state.x_coefficients, state.y_coefficients)):
                coeff_data.append([i, f"{x_val:8.4f}", f"{y_val:8.4f}"])
            print(tabulate(coeff_data, headers=coeff_headers, tablefmt="grid"))
        else:
            print("Coefficients are None")
    else:
        print("No calibration parameters found (tracker not calibrated)")
