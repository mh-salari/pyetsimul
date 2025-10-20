"""Example of registering and using a custom polynomial with PyEtSimul.

Demonstrates how users can define their own polynomial function
and use it for eye tracking calibration.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.gaze_mapping.polynomial import PolynomialDescriptor, PolynomialGazeModel
from pyetsimul.gaze_mapping.polynomial.polynomials import register_polynomial
from pyetsimul.types import Position3D, RotationMatrix

# Custom third-order polynomial with cross-terms: [x³, y³, x²y, xy², x², y², xy, x, y, 1]
# Mathematical model (same features for both X,Y):
# gaze_x = a₀*x³ + a₁*y³ + a₂*x²*y + a₃*x*y² + a₄*x² + a₅*y² + a₆*x*y + a₇*x + a₈*y + a₉
# gaze_y = b₀*x³ + b₁*y³ + b₂*x²*y + b₃*x*y² + b₄*x² + b₅*y² + b₆*x*y + b₇*x + b₈*y + b₉

MY_CUSTOM_POLYNOMIAL = PolynomialDescriptor(
    name="my_custom",
    description="Custom third-order polynomial with cross-terms",
    terms=["x", "y", "x*y", "x*y", "x", "y", "x*y", "x", "y", "1"],
    orders=[3, 3, [2, 1], [1, 2], 2, 2, [1, 1], 1, 1, 0],
)


def main() -> None:
    """Register custom polynomial and run eye tracking demo."""
    # Register the custom polynomial
    register_polynomial(MY_CUSTOM_POLYNOMIAL)

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

    # Eye configuration
    eye = Eye()
    # Looking along -Y axis (towards camera)
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    eye.position = Position3D(0.0, 550e-3, 350e-3)

    # Camera configuration
    camera = Camera(err=0.0, err_type="gaussian")
    # Facing along +Y axis (towards eye)
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    camera.point_at(eye.position)

    # Standard light configuration
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    # Setup tracker using custom polynomial
    method = "my_custom"
    et = PolynomialGazeModel.create([camera], [light], calibration_points, method)

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
