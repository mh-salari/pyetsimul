"""Data generation strategy for parameter variations."""

import copy
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from ...core import Eye
from ...types import Position3D
from ..core import ParameterVariation, VariationStrategy


class DataGenerationStrategy(VariationStrategy):
    """Generates and saves eye tracking data across parameter variations."""

    def __init__(
        self,
        cameras: list,
        lights: list,
        gaze_target: Position3D = None,
        output_format: str = "csv",
        output_dir: str = "output",
        experiment_name: str = None,
    ):
        self.cameras = cameras
        self.lights = lights
        self.gaze_target = gaze_target  # Fixed gaze target for eye position variations
        self.output_format = output_format.lower()
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        if self.output_format not in ["csv", "json", "both"]:
            raise ValueError("output_format must be 'csv', 'json', or 'both'")

    def execute(self, eye: Eye, variation: ParameterVariation) -> Dict[str, Any]:
        """Generate eye tracking data for all parameter variation values."""

        e = copy.deepcopy(eye)
        values = variation.generate_values()
        data = []

        for i, value in enumerate(tqdm(values, desc=f"Generating data for {variation.param_name}")):
            if variation.param_name == "eye_position":
                # Eye position varies, gaze target fixed
                variation.apply_to_eye(e, value)
                if self.gaze_target:
                    e.look_at(self.gaze_target)
                measurement_data = self._generate_measurement_data(e, value, i)
            elif variation.param_name == "target_position":
                # Gaze target varies, eye position fixed
                e.look_at(value)  # value is the target position
                measurement_data = self._generate_measurement_data(e, value, i)
            else:
                raise ValueError(f"Unsupported parameter variation: {variation.param_name}")

            data.append(measurement_data)

        # Save the generated data
        saved_files = self._save_data(data, variation.param_name, self.experiment_name)

        return {
            "total_measurements": len(data),
            "parameter_name": variation.param_name,
            "saved_files": saved_files,
            "data": data,
        }

    def _generate_measurement_data(self, eye: Eye, param_value: Any, index: int) -> Dict[str, Any]:
        """Generate measurement data for a single parameter value."""

        measurement_data = {
            "measurement_id": index,
            "parameter_value": self._serialize_position(param_value)
            if isinstance(param_value, Position3D)
            else param_value,
            "eye_position": self._serialize_position(eye.position) if eye.position else None,
            "cameras": [],
        }

        # Collect data from all cameras (like original implementation)
        for camera_idx, camera in enumerate(self.cameras):
            camera_data = {"camera_id": camera_idx, "eye_measurements": []}

            # Take image with camera
            img = camera.take_image(eye, self.lights)

            # Extract pupil boundary points
            pupil_points = []
            pupil_center = None
            if img.pupil_boundary is not None:
                pupil_points = [(float(p.x), float(p.y)) for p in img.pupil_boundary]

            if img.pupil_center is not None:
                pupil_center = [float(img.pupil_center.x), float(img.pupil_center.y)]

            # Extract corneal reflections (glints)
            glints = []
            for light in self.lights:
                cr = eye.find_cr(light, camera)
                glints.append([float(cr.x), float(cr.y)] if cr is not None else None)

            eye_measurement = {
                "eye_id": 0,  # Single eye for now
                "eye_position": self._serialize_position(eye.position) if eye.position else None,
                "pupil_center": pupil_center,
                "pupil_boundary": pupil_points,
                "corneal_reflections": glints,
            }

            camera_data["eye_measurements"].append(eye_measurement)
            measurement_data["cameras"].append(camera_data)

        return measurement_data

    def _serialize_position(self, position: Position3D) -> Dict[str, float]:
        """Convert Position3D to serializable dict."""
        return {"x": float(position.x), "y": float(position.y), "z": float(position.z)}

    def _save_data(self, data: List[Dict[str, Any]], param_name: str, experiment_name: str = None) -> Dict[str, str]:
        """Save generated data to files."""
        self.output_dir.mkdir(exist_ok=True)
        saved_files = {}

        # Use experiment name for filename if provided
        base_name = experiment_name if experiment_name else param_name

        if self.output_format in ["csv", "both"]:
            csv_file = self.output_dir / f"{base_name}_data.csv"
            self._save_csv(data, csv_file)
            saved_files["csv"] = str(csv_file)

        if self.output_format in ["json", "both"]:
            json_file = self.output_dir / f"{base_name}_data.json"
            self._save_json(data, json_file)
            saved_files["json"] = str(json_file)

        return saved_files

    def _save_csv(self, data: List[Dict[str, Any]], filepath: Path):
        """Save data to CSV file with proper flattening for pandas compatibility."""
        if not data:
            return

        flattened_data = []
        for item in data:
            flat_item = {}

            # Basic fields
            flat_item["measurement_id"] = item["measurement_id"]

            # Parameter value (Position3D)
            if isinstance(item["parameter_value"], dict):
                flat_item["parameter_x"] = item["parameter_value"]["x"]
                flat_item["parameter_y"] = item["parameter_value"]["y"]
                flat_item["parameter_z"] = item["parameter_value"]["z"]
            else:
                flat_item["parameter_value"] = item["parameter_value"]

            # Eye position
            if item.get("eye_position") and isinstance(item["eye_position"], dict):
                flat_item["eye_x"] = item["eye_position"]["x"]
                flat_item["eye_y"] = item["eye_position"]["y"]
                flat_item["eye_z"] = item["eye_position"]["z"]

            # Process each camera
            for cam_idx, camera_data in enumerate(item["cameras"]):
                cam_prefix = f"cam_{cam_idx}"
                flat_item[f"{cam_prefix}_id"] = camera_data["camera_id"]

                # Process each eye measurement
                for eye_idx, eye_measurement in enumerate(camera_data["eye_measurements"]):
                    eye_prefix = f"{cam_prefix}_eye_{eye_idx}"

                    flat_item[f"{eye_prefix}_id"] = eye_measurement["eye_id"]

                    # Eye position for this measurement
                    if eye_measurement.get("eye_position") and isinstance(eye_measurement["eye_position"], dict):
                        flat_item[f"{eye_prefix}_pos_x"] = eye_measurement["eye_position"]["x"]
                        flat_item[f"{eye_prefix}_pos_y"] = eye_measurement["eye_position"]["y"]
                        flat_item[f"{eye_prefix}_pos_z"] = eye_measurement["eye_position"]["z"]

                    # Pupil center
                    if eye_measurement.get("pupil_center") and len(eye_measurement["pupil_center"]) >= 2:
                        flat_item[f"{eye_prefix}_pupil_x"] = eye_measurement["pupil_center"][0]
                        flat_item[f"{eye_prefix}_pupil_y"] = eye_measurement["pupil_center"][1]

                    # Corneal reflections
                    crs = eye_measurement.get("corneal_reflections", [])
                    for cr_idx, cr in enumerate(crs):
                        if cr and len(cr) >= 2:
                            flat_item[f"{eye_prefix}_cr_{cr_idx}_x"] = cr[0]
                            flat_item[f"{eye_prefix}_cr_{cr_idx}_y"] = cr[1]

                    # Pupil boundary points (first few points only for CSV)
                    boundary = eye_measurement.get("pupil_boundary", [])
                    for pt_idx, point in enumerate(boundary[:10]):  # Limit to first 10 points
                        if point and len(point) >= 2:
                            flat_item[f"{eye_prefix}_boundary_{pt_idx}_x"] = point[0]
                            flat_item[f"{eye_prefix}_boundary_{pt_idx}_y"] = point[1]

            flattened_data.append(flat_item)

        with open(filepath, "w", newline="") as csvfile:
            if flattened_data:
                fieldnames = flattened_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)

    def _save_json(self, data: List[Dict[str, Any]], filepath: Path):
        """Save data to JSON file."""
        with open(filepath, "w") as jsonfile:
            json.dump(data, jsonfile, indent=2)
