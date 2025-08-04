"""Utility functions for calibration analysis.

This module provides utility functions for calibration analysis,
including parameter printing and data formatting functions.
"""

from et_simul.core import EyeTracker


def print_polynomial_parameters(et: EyeTracker) -> None:
    """Print polynomial parameters from calibrated eye tracker.

    Displays calibration coefficients and polynomial type for analysis.
    Shows calibration status and coefficient values for debugging.
    """
    # Only print polynomial parameters for interpolation trackers
    if et.algorithm_name != "interpolation":
        print(f"\nCalibration Parameters ({et.algorithm_name}):")
        print("-" * 40)
        if et.algorithm_state.is_calibrated:
            print("Calibration status: ✓ Calibrated")
            print("Algorithm: Geometric gaze estimation (non-polynomial)")
        else:
            print("Calibration status: ✗ Not calibrated")
        return

    print("\nPolynomial Parameters:")
    print("-" * 40)

    print(f"Polynomial type: {et.polynomial_name}")

    if et.algorithm_state.is_calibrated:
        state = et.algorithm_state
        print("Calibration status: ✓ Calibrated")

        if state.x_coefficients is not None and state.y_coefficients is not None:
            print(f"X coefficients shape: {state.x_coefficients.shape}")
            print(f"Y coefficients shape: {state.y_coefficients.shape}")
            print("X coefficients:")
            print(f"  [{', '.join(f'{val:8.4f}' for val in state.x_coefficients)}]")
            print("Y coefficients:")
            print(f"  [{', '.join(f'{val:8.4f}' for val in state.y_coefficients)}]")
        else:
            print("Coefficients are None")
    else:
        print("No calibration parameters found (tracker not calibrated)")
