"""Utility functions for calibration analysis.

This module provides utility functions for calibration analysis,
including parameter printing and data formatting functions.
"""

from pyetsimul.core import EyeTracker
from pyetsimul.gaze_tracking_algorithms.interpolate.polynomials import get_polynomial_info
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

    # Coefficients table with term descriptions
    if et.algorithm_state.is_calibrated:
        state = et.algorithm_state
        print("\nCoefficients Values:")

        # Get polynomial term descriptions
        poly_info = get_polynomial_info(et.polynomial_name)
        term_descriptions = poly_info.descriptor.get_term_descriptions()

        if poly_info.descriptor.uses_different_xy_features:
            # Different features for x and y: show separate tables for X and Y coordinates
            coeff_headers = ["Term", "Coefficient"]

            print("\nX Coordinate Terms:")
            x_coeff_data = []
            x_terms = term_descriptions[0]
            for term, coeff in zip(x_terms, state.x_coefficients):
                x_coeff_data.append([term, f"{coeff:8.6f}"])
            print(tabulate(x_coeff_data, headers=coeff_headers, tablefmt="grid"))

            print("\nY Coordinate Terms:")
            y_coeff_data = []
            y_terms = term_descriptions[1]
            for term, coeff in zip(y_terms, state.y_coefficients):
                y_coeff_data.append([term, f"{coeff:8.6f}"])
            print(tabulate(y_coeff_data, headers=coeff_headers, tablefmt="grid"))
        else:
            # same features for x and y: show combined table with shared terms
            coeff_headers = ["Term", "X Coefficient", "Y Coefficient"]
            coeff_data = []
            for term, x_val, y_val in zip(term_descriptions, state.x_coefficients, state.y_coefficients):
                coeff_data.append([term, f"{x_val:8.6f}", f"{y_val:8.6f}"])
            print(tabulate(coeff_data, headers=coeff_headers, tablefmt="grid"))
    else:
        print("No calibration parameters found (tracker not calibrated)")
