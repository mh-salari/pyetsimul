"""Eye movement experiment data generation."""

import json
import csv
import os
from typing import List, Dict, Any

from ..types import Position3D
from ..experimental_designs import EyeMovement
from ..core import Eye, Camera, Light


class EyeMovementExperiment:
    """Generate synthetic eye tracking data for eye movement experiments.

    Uses EyeMovement experimental design to systematically generate eye tracking
    data with varying eye positions and fixed gaze target, testing robustness
    to observer movement.
    """

    def __init__(
        self,
        movement_pattern: EyeMovement,
        eyes: List[Eye],
        cameras: List[Camera],
        lights: List[Light],
        experiment_name: str,
    ):
        """Initialize eye movement experiment.

        Args:
            movement_pattern: EyeMovement experimental design
            eyes: List of Eye models for simulation
            cameras: List of cameras for data capture
            lights: List of light sources for glint generation
            experiment_name: Name for this experiment
        """
        if not eyes:
            raise ValueError("At least one eye must be provided")
        if not cameras:
            raise ValueError("At least one camera must be provided")
        if not lights:
            raise ValueError("At least one light must be provided")

        self.movement_pattern = movement_pattern
        self.eyes = eyes
        self.cameras = cameras
        self.lights = lights
        self.experiment_name = experiment_name
        self.tracking_data: List[Dict[str, Any]] = []

        # Validate design
        self.movement_pattern.validate_design()

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Run the eye movement experiment and collect data.

        Returns:
            List of tracking data dictionaries with ground truth
        """
        self.tracking_data = []
        eye_positions = self.movement_pattern.generate_eye_positions()

        print(f"Running {self.experiment_name} with {len(eye_positions)} eye positions")

        for i, eye_position in enumerate(eye_positions):
            # Move all eyes to the new position
            for eye in self.eyes:
                eye.position = eye_position

            # Direct all eyes to look at the fixed gaze target
            for eye in self.eyes:
                eye.look_at(self.movement_pattern.gaze_target)

            # Collect measurements from all cameras and eyes
            measurement = self._measure_eye_tracking_data(eye_position, self.movement_pattern.gaze_target)
            self.tracking_data.append(measurement)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(eye_positions)} positions")

        print(f"Experiment completed: {len(self.tracking_data)} measurements collected")
        return self.tracking_data

    def _measure_eye_tracking_data(self, eye_position: Position3D, gaze_target: Position3D) -> Dict[str, Any]:
        """Measure eye tracking data from all cameras and eyes.

        Args:
            eye_position: Current eye position
            gaze_target: Fixed gaze target position

        Returns:
            Dictionary with tracking measurement data for all eyes and cameras
        """
        measurement_data = {
            "eye_position": [eye_position.x, eye_position.y, eye_position.z],
            "gaze_target": [gaze_target.x, gaze_target.y, gaze_target.z],
            "cameras": [],
        }

        # Collect data for each camera
        for camera_idx, camera in enumerate(self.cameras):
            camera_data = {"camera_id": camera_idx, "eye_measurements": []}

            # Collect measurements from each eye for this camera
            for eye_idx, eye in enumerate(self.eyes):
                # Take image with camera
                img = camera.take_image(eye, self.lights)

                # Extract pupil boundary points
                pupil_points = []
                pupil_center = None
                if img.pupil_boundary is not None:
                    pupil_points = [(p.x, p.y) for p in img.pupil_boundary]

                if img.pupil_center is not None:
                    pupil_center = [img.pupil_center.x, img.pupil_center.y]

                # Extract glint positions
                glints = []
                if isinstance(img, dict) and "cr" in img:
                    for cr in img["cr"]:
                        glints.append(cr.tolist() if cr is not None else None)
                else:
                    # Calculate glints for each light
                    for light in self.lights:
                        cr = eye.find_cr(light, camera)
                        glints.append([cr.x, cr.y, cr.z] if cr is not None else None)

                # Warn about missing glints
                if any((g is None or len(g) < 2) for g in glints):
                    print(
                        f"Warning: Some glints missing for camera {camera_idx}, eye {eye_idx}, position {eye_position}"
                    )

                eye_measurement = {
                    "eye_id": eye_idx,
                    "eye_position": [eye_position.x, eye_position.y, eye_position.z],
                    "pupil_center": pupil_center,
                    "pupil_boundary": pupil_points,
                    "corneal_reflections": [(g[0], g[1]) if (g is not None and len(g) >= 2) else None for g in glints],
                }

                camera_data["eye_measurements"].append(eye_measurement)

            measurement_data["cameras"].append(camera_data)

        return measurement_data

    def save_experiment_data(
        self, output_dir: str = "outputs", save_json: bool = True, save_csv: bool = True
    ) -> Dict[str, str]:
        """Save experiment data to files.

        Args:
            output_dir: Directory to save output files
            save_json: Whether to save JSON format
            save_csv: Whether to save CSV format

        Returns:
            Dictionary with paths to saved files
        """
        if not self.tracking_data:
            raise ValueError("No experiment data to save. Run experiment first.")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        base_filename = f"{self.experiment_name}_eye_movement_data"

        # Save JSON format
        if save_json:
            json_path = os.path.join(output_dir, f"{base_filename}.json")
            self._save_json(json_path)
            saved_files["json"] = json_path

        # Save CSV format
        if save_csv:
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            self._save_csv(csv_path)
            saved_files["csv"] = csv_path

        return saved_files

    def _save_json(self, filepath: str):
        """Save tracking data in JSON format."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.tracking_data, f, indent=4)
            print(f"Eye movement data saved to JSON: {filepath}")
        except Exception as e:
            print(f"Error saving JSON file {filepath}: {e}")

    def _save_csv(self, filepath: str):
        """Save tracking data in CSV format."""
        try:
            with open(filepath, "w", newline="") as csvfile:
                fieldnames = ["eye_pos_x", "eye_pos_y", "eye_pos_z", "gaze_target_x", "gaze_target_y", "gaze_target_z"]

                # Determine max dimensions
                max_cameras = max(len(entry["cameras"]) for entry in self.tracking_data)
                max_eyes = max(
                    len(camera_data["eye_measurements"])
                    for entry in self.tracking_data
                    for camera_data in entry["cameras"]
                )
                max_glints = 0
                max_pupil_points = 0

                for entry in self.tracking_data:
                    for camera_data in entry["cameras"]:
                        for eye_measurement in camera_data["eye_measurements"]:
                            if eye_measurement["corneal_reflections"]:
                                max_glints = max(max_glints, len(eye_measurement["corneal_reflections"]))
                            if eye_measurement["pupil_boundary"]:
                                max_pupil_points = max(max_pupil_points, len(eye_measurement["pupil_boundary"]))

                # Add fields for each camera-eye combination
                for camera_idx in range(max_cameras):
                    for eye_idx in range(max_eyes):
                        prefix = f"camera_{camera_idx}_eye_{eye_idx}"
                        fieldnames.extend([f"{prefix}_pupil_center_x", f"{prefix}_pupil_center_y"])

                        for glint_idx in range(max_glints):
                            fieldnames.extend([f"{prefix}_cr_{glint_idx}_x", f"{prefix}_cr_{glint_idx}_y"])

                        for point_idx in range(max_pupil_points):
                            fieldnames.extend(
                                [f"{prefix}_pupil_boundary_{point_idx}_x", f"{prefix}_pupil_boundary_{point_idx}_y"]
                            )

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
                writer.writeheader()

                for entry in self.tracking_data:
                    row = {
                        "eye_pos_x": entry["eye_position"][0],
                        "eye_pos_y": entry["eye_position"][1],
                        "eye_pos_z": entry["eye_position"][2],
                        "gaze_target_x": entry["gaze_target"][0],
                        "gaze_target_y": entry["gaze_target"][1],
                        "gaze_target_z": entry["gaze_target"][2],
                    }

                    for camera_idx, camera_data in enumerate(entry["cameras"]):
                        for eye_idx, eye_measurement in enumerate(camera_data["eye_measurements"]):
                            prefix = f"camera_{camera_idx}_eye_{eye_idx}"

                            pupil_center = eye_measurement["pupil_center"]
                            if pupil_center and len(pupil_center) >= 2:
                                row[f"{prefix}_pupil_center_x"] = pupil_center[0]
                                row[f"{prefix}_pupil_center_y"] = pupil_center[1]
                            else:
                                row[f"{prefix}_pupil_center_x"] = ""
                                row[f"{prefix}_pupil_center_y"] = ""

                            corneal_reflections = eye_measurement["corneal_reflections"] or []
                            for glint_idx in range(max_glints):
                                if glint_idx < len(corneal_reflections) and corneal_reflections[glint_idx] is not None:
                                    row[f"{prefix}_cr_{glint_idx}_x"] = corneal_reflections[glint_idx][0]
                                    row[f"{prefix}_cr_{glint_idx}_y"] = corneal_reflections[glint_idx][1]
                                else:
                                    row[f"{prefix}_cr_{glint_idx}_x"] = ""
                                    row[f"{prefix}_cr_{glint_idx}_y"] = ""

                            pupil_boundary = eye_measurement["pupil_boundary"] or []
                            for point_idx in range(max_pupil_points):
                                if point_idx < len(pupil_boundary) and pupil_boundary[point_idx] is not None:
                                    row[f"{prefix}_pupil_boundary_{point_idx}_x"] = pupil_boundary[point_idx][0]
                                    row[f"{prefix}_pupil_boundary_{point_idx}_y"] = pupil_boundary[point_idx][1]
                                else:
                                    row[f"{prefix}_pupil_boundary_{point_idx}_x"] = ""
                                    row[f"{prefix}_pupil_boundary_{point_idx}_y"] = ""

                    writer.writerow(row)

            print(f"Eye movement data saved to CSV: {filepath}")
        except Exception as e:
            print(f"Error saving CSV file {filepath}: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the experiment."""
        if not self.tracking_data:
            return {"error": "No experiment data available"}

        design_params = self.movement_pattern.get_design_parameters()

        return {
            "experiment_name": self.experiment_name,
            "total_measurements": len(self.tracking_data),
            "design_parameters": design_params,
            "eye_count": len(self.eyes),
            "camera_count": len(self.cameras),
            "light_count": len(self.lights),
        }
