"""Minimal custom gaze model example for PyEtSimul.

This example demonstrates how to create a custom gaze estimation model
by implementing the EyeTracker interface. The SimpleLinearGazeModel learns
a simple linear transformation from pupil-CR vectors to gaze positions.
"""

import time
from dataclasses import dataclass

import numpy as np

from pyetsimul.core import Camera, Eye, EyeTracker, Light
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.geometry.plane_detection import PlaneInfo, detect_calibration_plane, summarize_plane_detection
from pyetsimul.types import EyeMeasurement, GazePrediction, Point3D, Position3D, RotationMatrix
from pyetsimul.types.algorithms import AlgorithmState


@dataclass
class SimpleLinearState(AlgorithmState):
    """State for simple linear gaze model."""

    calibration_matrix: np.ndarray | None = None

    def serialize(self) -> dict:
        """Serializes the state of the gaze model."""
        return {
            "is_calibrated": self.is_calibrated,
            "calibration_error": self.calibration_error,
            "last_update": self.last_update,
            "calibration_matrix": self.calibration_matrix.tolist() if self.calibration_matrix is not None else None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SimpleLinearState":
        """Deserializes the state of the gaze model."""
        state = cls(
            is_calibrated=data["is_calibrated"],
            calibration_error=data["calibration_error"],
            last_update=data["last_update"],
        )
        if data["calibration_matrix"] is not None:
            state.calibration_matrix = np.array(data["calibration_matrix"])
        return state


class SimpleLinearGazeModel(EyeTracker):
    """Simple linear mapping from pupil-CR vector to gaze position.

    This is the simplest possible gaze estimation model - it learns
    a linear transformation from the 2D pupil-corneal reflection (P-CR)
    vector to screen coordinates.

    Model: gaze_position = calibration_matrix @ [pcr_x, pcr_y, 1]
    """

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initializes the SimpleLinearGazeModel."""
        super().__init__(**kwargs)
        self.algorithm_state = SimpleLinearState()
        self.plane_info = None

    @property
    def algorithm_name(self) -> str:
        """Returns the name of the gaze algorithm."""
        return "simple_linear"

    @classmethod
    def create(
        cls,
        cameras: list[Camera],
        lights: list[Light],
        calib_points: list[Position3D],
        use_refraction: bool = True,
    ) -> "SimpleLinearGazeModel":
        """Creates an instance of SimpleLinearGazeModel."""
        return cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate using simple linear least squares."""
        self.plane_info = detect_calibration_plane(self.calib_points)
        print(summarize_plane_detection(self.calib_points, self.plane_info))

        features = []
        targets = []

        for measurement, calib_point in zip(calibration_measurements, self.calib_points, strict=False):
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )

            if pc is not None and cr is not None:
                # Extract pupil-CR vector
                pcr = pc - cr
                features.append([pcr.x, pcr.y, 1.0])

                # Map 3D calibration point to 2D plane coordinates
                target_2d = self.plane_info.extract_2d_coords(calib_point)
                targets.append(target_2d)

        # Solve: targets = calibration_matrix @ features
        features_matrix = np.array(features).T
        targets_matrix = np.array(targets).T

        self.algorithm_state.calibration_matrix = targets_matrix @ np.linalg.pinv(features_matrix)
        self.algorithm_state.is_calibrated = True

        print(f"Simple linear calibration complete with {len(features)} points.")

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction:
        """Predict gaze using linear transformation."""
        start_time = time.time()

        pc = measurement.pupil_data.center
        cr = measurement.camera_image.corneal_reflections[0] if measurement.camera_image.corneal_reflections else None

        intermediate_results = {"pc": pc, "cr": cr}

        if pc is not None and cr is not None:
            # Extract pupil-CR vector
            pcr = pc - cr
            feature_vector = np.array([pcr.x, pcr.y, 1.0])

            # Apply calibration
            gaze_2d = self.algorithm_state.calibration_matrix @ feature_vector
            intermediate_results["pcr_vector"] = pcr
            intermediate_results["gaze_2d"] = gaze_2d

            # Convert 2D plane coordinates to 3D position
            gaze_point = self.plane_info.reconstruct_3d_point(float(gaze_2d[0]), float(gaze_2d[1]))
            confidence = 1.0
        else:
            # Missing data
            gaze_point = Point3D(0.0, 0.0, 0.0)
            confidence = 0.0

        processing_time = time.time() - start_time

        return GazePrediction(
            gaze_point=gaze_point,
            confidence=confidence,
            algorithm_name=self.algorithm_name,
            processing_time=processing_time,
            intermediate_results=intermediate_results,
        )

    def serialize(self) -> dict:
        """Serialize for saving to disk."""
        return {
            "algorithm_state": self.algorithm_state.serialize(),
            "plane_info": self.plane_info.serialize() if self.plane_info else None,
            "cameras": [camera.serialize() for camera in self.cameras],
            "lights": [light.serialize() for light in self.lights],
            "calib_points": [point.serialize() for point in self.calib_points],
            "use_refraction": self.use_refraction,
            "use_legacy_look_at": self.use_legacy_look_at,
            "state": self.state,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SimpleLinearGazeModel":
        """Restore from serialized data."""
        cameras = [Camera.deserialize(cam_data) for cam_data in data["cameras"]]
        lights = [Light.deserialize(light_data) for light_data in data["lights"]]
        calib_points = [Position3D.deserialize(pt_data) for pt_data in data["calib_points"]]

        tracker = cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=data["use_refraction"],
        )

        tracker.algorithm_state = SimpleLinearState.deserialize(data["algorithm_state"])
        if data["plane_info"]:
            tracker.plane_info = PlaneInfo.deserialize(data["plane_info"])

        tracker.use_legacy_look_at = data["use_legacy_look_at"]
        tracker.state = data["state"]

        return tracker


def main() -> None:
    """Run example demonstrating custom gaze model."""
    print("Simple Linear Gaze Model Demo\n")

    # Create 3x3 calibration grid on XZ plane
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

    # Eye setup: looking along -Y axis towards camera
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    eye.position = Position3D(0.0, 550e-3, 350e-3)

    # Camera setup: facing along +Y axis towards eye
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    camera.point_at(eye.position)

    # Light setup
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    # Create and configure tracker
    et = SimpleLinearGazeModel.create([camera], [light], calibration_points)

    print("Eye Tracker Configuration:")
    et.pprint(eye)

    print("\nCalibrating...")
    et.run_calibration(eye)

    print("\nTesting calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    calib_results.pprint("Simple Linear Calibration Test")


if __name__ == "__main__":
    main()
