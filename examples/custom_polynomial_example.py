"""Example of registering and using a custom polynomial with PyEtSimul.

Demonstrates how users can define their own polynomial function
and use it for eye tracking calibration.
"""

import numpy as np
from pyetsimul.gaze_tracking_algorithms.interpolate.polynomials import register_polynomial
from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.types.algorithms import PolynomialFeatures
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix


def my_custom_polynomial(x: float, y: float) -> PolynomialFeatures:
    """Custom second-order polynomial with cross-term: [x², y², xy, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₄*x² + a₅*y² + a₆*x*y + a₇*x + a₈*y + a₉
    gaze_y = b₄*x² + b₅*y² + b₆*x*y + b₇*x + b₈*y + b₉
    """
    features = np.array([x**2, y**2, x * y, x, y, 1])
    return PolynomialFeatures(features=features, polynomial_name="my_custom")


def main():
    """Register custom polynomial and run eye tracking demo."""

    # Register the custom polynomial
    register_polynomial(
        name="my_custom",
        function=my_custom_polynomial,
        description="Custom second-order polynomial with cross-term",
        model_type="non-separable",
        feature_count=6,
    )

    print("Custom Polynomial Eye Tracking Demo\n")

    # Create 3x3 calibration grid on XZ plane (from experiments config)
    calibration_points = [
        Position3D(-200e-3, 0.0, 50e-3),
        Position3D(0, 0.0, 50e-3),
        Position3D(200e-3, 0.0, 50e-3),
        Position3D(-200e-3, 0.0, 200e-3),
        Position3D(0, 0.0, 200e-3),
        Position3D(200e-3, 0.0, 200e-3),
        Position3D(-200e-3, 0.0, 350e-3),
        Position3D(0, 0.0, 350e-3),
        Position3D(200e-3, 0.0, 350e-3),
    ]

    # Standard eye configuration
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    eye.position = Position3D(0.0, 550e-3, 350e-3)

    # Standard camera configuration
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    camera.point_at(eye.position)

    # Standard light configuration
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    # Setup tracker using custom polynomial
    method = "my_custom"
    et = InterpolationTracker.create([camera], [light], calibration_points, method)

    # Display configuration summary
    et.pprint(eye)

    # Calibrate the eye tracker
    print("Calibrating eye tracker with custom polynomial...")
    et.run_calibration(eye)

    print("\n1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    calib_results.pprint("Custom Polynomial Calibration Test")


if __name__ == "__main__":
    main()
