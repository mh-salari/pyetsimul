"""Algorithm comparison example using parameter variations.

Compares multiple eye tracking algorithms across both eye position and
gaze target variations using the same hardware setup as other examples.
"""

from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.parameter_variations import EyePositionVariation, TargetPositionVariation, AlgorithmComparisonStrategy
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points


def main():
    """Compare interpolation algorithms across eye position and target variations."""

    # Eye configuration - same as other examples
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = Position3D(0, 0.550, 0.350)

    # Camera configuration - same as other examples
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    camera.point_at(eye.position)

    # Light configuration - same as other examples
    light = Light(position=Position3D(0.200, 0, 0.350))

    cameras = [camera]
    lights = [light]

    # Calibration points - exact same as interpolate.py example
    calib_points = [
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

    # Create and calibrate algorithms - same setup as interpolate.py
    print("Setting up algorithms...")

    algorithm_configs = [
        ("cerrolaza_2008", "Cerrolaza 2008"),
        ("second_order", "Second Order"),
    ]

    algorithms = []
    algorithm_names = []

    for method, name in algorithm_configs:
        print(f"\nCalibrating {name}...")
        et = InterpolationTracker.create(cameras, lights, calib_points, method)
        et.use_legacy_look_at = True  # Same as interpolate.py
        et.run_calibration(eye)

        # Test calibration accuracy
        calib_results = accuracy_at_calibration_points(et, eye=eye, interactive_plot=False)
        calib_results.print_summary(f"{name} Calibration Accuracy")

        algorithms.append(et)
        algorithm_names.append(name)

    # Test 1: Eye position variation (fixed gaze target)
    print("\n" + "=" * 60)
    print("TEST 1: EYE POSITION VARIATION (Fixed Gaze Target)")
    print("=" * 60)

    eye_variation = EyePositionVariation(
        center=Position3D(0, 0.550, 0.350),
        dx=[-50e-3, 50e-3],  # X varies: ±50mm
        dy=[-50e-3, 50e-3],  # Y varies: ±50mm
        dz=[0.0, 0.0],  # Z fixed: no variation
        grid_size=[16, 16, 1],  # 16x16x1 = 2D grid in XY plane
    )

    gaze_target = Position3D(0, 0, 0.200)

    comparison_strategy = AlgorithmComparisonStrategy(algorithms=algorithms, algorithm_names=algorithm_names)

    print("Running eye position comparison...")
    eye_results = comparison_strategy.execute(eye, eye_variation, gaze_target)
    comparison_strategy.print_results(eye_results, "Eye Position Variation")

    # Test 2: Target position variation (fixed eye position)
    print("\n" + "=" * 60)
    print("TEST 2: TARGET POSITION VARIATION (Fixed Eye Position)")
    print("=" * 60)

    target_variation = TargetPositionVariation(
        grid_center=Position3D(0, 0, 0.200),
        dx=[-200e-3, 200e-3],
        dy=[0.0, 0.0],
        dz=[-150e-3, 150e-3],
        grid_size=[16, 1, 16],
    )

    print("Running target position comparison...")
    target_results = comparison_strategy.execute(eye, target_variation)
    comparison_strategy.print_results(target_results, "Target Position Variation")


if __name__ == "__main__":
    main()
