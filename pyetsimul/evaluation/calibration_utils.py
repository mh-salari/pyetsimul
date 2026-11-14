"""Utility functions for calibration analysis.

This module provides utility functions for calibration analysis,
including parameter printing and data formatting functions.
"""

from pyetsimul.core import EyeTracker
from pyetsimul.gaze_mapping.polynomial.polynomials import get_polynomial_info
from pyetsimul.log import info, table, warning


def pprint_polynomial_parameters(et: EyeTracker) -> None:
    """Print polynomial parameters from calibrated eye tracker.

    Displays calibration coefficients and polynomial type for analysis.
    Shows calibration status and coefficient values for debugging.
    """
    # Only print polynomial parameters for polynomial gaze model trackers
    if et.algorithm_name != "polynomial":
        info(f"\nCalibration Parameters ({et.algorithm_name}):")
        info("-" * 40)
        if et.algorithm_state.is_calibrated:
            info("Calibration status: Calibrated")
        else:
            warning("Calibration status: [X] Not calibrated")
        return

    info("\nPolynomial Parameters:")

    # General info table
    headers = ["Parameter", "Value"]
    general_data = [
        ["Polynomial type", et.polynomial_name],
        ["Calibration status", "Calibrated" if et.algorithm_state.is_calibrated else "[X] Not calibrated"],
    ]

    if et.algorithm_state.is_calibrated:
        state = et.algorithm_state
        if state.x_coefficients is not None and state.y_coefficients is not None:
            general_data.extend([
                ["X coefficients shape", f"{state.x_coefficients.shape}"],
                ["Y coefficients shape", f"{state.y_coefficients.shape}"],
            ])

    table(general_data, headers=headers, tablefmt="simple")

    # Coefficients table with term descriptions
    if et.algorithm_state.is_calibrated:
        state = et.algorithm_state
        info("\nCoefficients Values:")

        # Get polynomial term descriptions
        poly_info = get_polynomial_info(et.polynomial_name)
        term_descriptions = poly_info.descriptor.get_term_descriptions()

        different_xy = poly_info.descriptor.uses_different_xy_features
        x_terms, y_terms = term_descriptions if different_xy else (term_descriptions, term_descriptions)

        # The coefficients hold the terms of one glint after the other, one block per light.
        n_glints = len(state.x_coefficients) // len(x_terms)
        for glint in range(n_glints):
            if n_glints > 1:
                info(f"\nGlint {glint + 1} of {n_glints}:")
            x_values = state.x_coefficients[glint * len(x_terms) : (glint + 1) * len(x_terms)]
            y_values = state.y_coefficients[glint * len(y_terms) : (glint + 1) * len(y_terms)]

            if different_xy:
                # Different features for x and y: show separate tables for X and Y coordinates
                coeff_headers = ["Term", "Coefficient"]
                info("\nX Coordinate Terms:")
                x_coeff_data = [[term, f"{coeff:8.6f}"] for term, coeff in zip(x_terms, x_values, strict=True)]
                table(x_coeff_data, headers=coeff_headers, tablefmt="grid")
                info("\nY Coordinate Terms:")
                y_coeff_data = [[term, f"{coeff:8.6f}"] for term, coeff in zip(y_terms, y_values, strict=True)]
                table(y_coeff_data, headers=coeff_headers, tablefmt="grid")
            else:
                # same features for x and y: show combined table with shared terms
                coeff_headers = ["Term", "X Coefficient", "Y Coefficient"]
                coeff_data = [
                    [term, f"{x_val:8.6f}", f"{y_val:8.6f}"]
                    for term, x_val, y_val in zip(x_terms, x_values, y_values, strict=True)
                ]
                table(coeff_data, headers=coeff_headers, tablefmt="grid")
    else:
        warning("No calibration parameters found (tracker not calibrated)")
