"""Eye tracker module.

This module provides the EyeTracker class that represents a complete eye tracking
system with cameras, lights, calibration points, and algorithm functions.
"""

import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .camera import Camera
from .light import Light
from .eye import Eye
from ..types import Point4D, Point3D


@dataclass
class EyeTracker(ABC):
    """Eye tracker with cameras, lights, calibration, and algorithms.

    Represents a complete eye tracking system including:
    - Cameras for image capture
    - Lights for corneal reflections
    - Calibration points/grid
    - Calibration and evaluation functions
    - Algorithm state/parameters
    """

    # Physical components
    cameras: List[Camera] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)

    # Calibration
    calib_points: Optional[Point4D] = None

    # Algorithm state/parameters
    state: Dict[str, Any] = field(default_factory=dict)

    # Refraction setting
    use_refraction: bool = True

    def add_camera(self, camera: Camera) -> None:
        """Add a camera to the eye tracker."""
        self.cameras.append(camera)

    def add_light(self, light: Light) -> None:
        """Add a light to the eye tracker."""
        self.lights.append(light)

    def run_calibration(self, eye: Eye) -> "EyeTracker":
        """Run the complete calibration workflow.

        Generic calibration process that works for all eye tracker types:
        1. Collect calibration data at multiple target points
        2. Call the eye tracker's specific calibration method

        Args:
            eye: Eye object to calibrate with

        Returns:
            Self for method chaining
        """
        calib_data = self._collect_calibration_data(eye)
        self.calibrate(calib_data)
        return self

    def _collect_calibration_data(self, eye: Eye) -> List[Dict[str, Any]]:
        """Helper to collect data for each calibration point."""
        calib_data = [None] * self.calib_points.shape[1]
        np.random.seed(0)  # For reproducible results

        n_points = self.calib_points.shape[1]
        failed_points = []

        print(f"Collecting calibration data at {n_points} points...")

        for i in range(n_points):
            # Make eye look at calibration point
            target = np.array([self.calib_points[0, i], 0, self.calib_points[1, i], 1])
            eye.look_at(target)

            # Take images from all cameras
            calib_data[i] = {}
            calib_data[i]["camimg"] = [None] * len(self.cameras)
            # Store the target gaze position for this calibration point
            calib_data[i]["gaze"] = np.array([self.calib_points[0, i], self.calib_points[1, i]])

            for iCamera, cam in enumerate(self.cameras):
                camimg = cam.take_image(eye, self.lights, use_refraction=self.use_refraction)
                calib_data[i]["camimg"][iCamera] = camimg

                # Check for detection failures immediately
                pc = camimg["pc"]
                cr = camimg["cr"][0] if camimg["cr"] else None

                if pc is None:
                    failed_points.append((i + 1, self.calib_points[:, i], "PUPIL CENTER not detected"))
                elif cr is None:
                    failed_points.append((i + 1, self.calib_points[:, i], "CR not detected"))

            # Store eye state
            calib_data[i]["e"] = copy.deepcopy(eye)

        # Summary of failed points
        if failed_points:
            print(f"\n⚠️  WARNING: {len(failed_points)}/{n_points} calibration points failed:")
            for point_num, coords, reason in failed_points:
                print(f"  Point {point_num} ({coords[0] * 1000:.0f}mm, {coords[1] * 1000:.0f}mm): {reason}")
            print(f"  Calibration will proceed with {n_points - len(failed_points)} valid points.\n")
        else:
            print(f"✅ All {n_points} calibration points collected successfully.\n")

        return calib_data

    def estimate_gaze_at(self, eye: Eye, look_at_pos: Point3D) -> Optional[Any]:
        """Estimate gaze position when eye looks at a target.

        Generic gaze estimation that works for all eye tracker types.

        Args:
            eye: Eye object
            look_at_pos: 2D position where eye should look [x, y]

        Returns:
            PredictionResult with estimated gaze and intermediate values
        """
        # Make eye look at target position
        target = np.array([look_at_pos[0], 0, look_at_pos[1], 1])
        eye.look_at(target)

        # Take camera images
        camimg = [None] * len(self.cameras)
        for iCamera, cam in enumerate(self.cameras):
            camimg[iCamera] = cam.take_image(eye, self.lights, use_refraction=self.use_refraction)

        # Get gaze prediction - returns None if prediction fails
        return self.predict_gaze(camimg)

    def calculate_gaze_error(self, eye: Eye, look_at_pos: Point3D) -> Tuple[float, float]:
        """Calculate gaze estimation error.

        Generic error calculation that works for all eye tracker types.
        Exactly matches the original gaze_error logic.

        Args:
            eye: Eye object
            look_at_pos: 2D position where eye should look [x, y]

        Returns:
            Tuple of (u, v) gaze error in meters, or (NaN, NaN) if estimation fails
        """
        gaze = self.estimate_gaze_at(eye, look_at_pos)

        if gaze is not None and gaze.gaze_point is not None:
            u = gaze.gaze_point[0] - look_at_pos[0]
            v = gaze.gaze_point[1] - look_at_pos[1]
            return u, v
        else:
            return np.nan, np.nan

    @abstractmethod
    def calibrate(self, calib_data: List[Dict[str, Any]]) -> None:
        """Calibrate the eye tracker using collected data.

        Each eye tracker type must implement its specific calibration algorithm.

        Args:
            calib_data: List of calibration data collected at each calibration point
        """
        pass

    @abstractmethod
    def predict_gaze(self, camimg: List[Dict[str, Any]]) -> Optional[Any]:
        """Predict gaze position from camera images.

        Each eye tracker type must implement its specific gaze prediction algorithm.

        Args:
            camimg: List of camera images containing pupil and corneal reflection data

        Returns:
            2D gaze position [x, y] on screen or None if prediction fails
        """
        pass
