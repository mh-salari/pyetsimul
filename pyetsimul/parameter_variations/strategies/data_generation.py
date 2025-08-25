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

    def execute(self, eyes: list, variation: ParameterVariation) -> Dict[str, Any]:
        """Generate eye tracking data: camera → eye → parameter variations."""

        values = variation.generate_values()
        all_data = {"cameras": []}
        total_measurements = 0

        # Progress bars: Camera (outer) → Eye (middle) → Variations (inner)
        for camera_idx, camera in enumerate(tqdm(self.cameras, desc="Processing cameras", position=0)):
            camera_data = {
                "camera_id": camera_idx,
                "camera_name": getattr(camera, "name", f"Camera {camera_idx + 1}"),
                "eyes": [],
            }

            for eye_idx, eye in enumerate(tqdm(eyes, desc=f"Camera {camera_idx + 1} eyes", position=1, leave=False)):
                eye_copy = copy.deepcopy(eye)
                eye_data = {"eye_id": eye_idx, "eye_name": f"Eye {eye_idx + 1}", "measurements": []}

                # Process all parameter variations for this camera-eye combination
                for i, value in enumerate(
                    tqdm(values, desc=f"C{camera_idx + 1}E{eye_idx + 1} variations", position=2, leave=False)
                ):
                    if variation.param_name == "eye_position":
                        variation.apply_to_eye(eye_copy, value)
                        if self.gaze_target:
                            eye_copy.look_at(self.gaze_target)
                    elif variation.param_name == "target_position":
                        eye_copy.look_at(value)

                    # Generate measurement for this specific camera-eye-variation
                    measurement = self._generate_single_measurement(eye_copy, camera, value, i)
                    eye_data["measurements"].append(measurement)
                    total_measurements += 1

                camera_data["eyes"].append(eye_data)

            all_data["cameras"].append(camera_data)

        # Save dataset
        saved_files = self._save_data(all_data, variation.param_name, self.experiment_name)

        return {
            "total_measurements": total_measurements,
            "parameter_name": variation.param_name,
            "saved_files": saved_files,
            "data": all_data,
        }

    def _generate_single_measurement(self, eye: Eye, camera, param_value: Any, index: int) -> Dict[str, Any]:
        """Generate measurement data for single camera-eye-parameter combination."""

        # Take image with this specific camera
        img = camera.take_image(eye, self.lights)

        # Extract pupil data
        pupil_points = []
        pupil_center = None
        if img.pupil_boundary is not None:
            pupil_points = [(float(p.x), float(p.y)) for p in img.pupil_boundary]
        if img.pupil_center is not None:
            pupil_center = [float(img.pupil_center.x), float(img.pupil_center.y)]

        # Extract corneal reflections (glints) from all lights
        glints = []
        for light in self.lights:
            cr = eye.find_cr(light, camera)
            glints.append([float(cr.x), float(cr.y)] if cr is not None else None)

        return {
            "measurement_id": index,
            "parameter_value": self._serialize_position(param_value) if hasattr(param_value, "x") else param_value,
            "eye_position": self._serialize_position(eye.position) if eye.position else None,
            "pupil_center": pupil_center,
            "pupil_boundary": pupil_points,
            "corneal_reflections": glints,
        }

    def _save_data(self, data: Dict, param_name: str, experiment_name: str) -> List[str]:
        """Save dataset to files."""
        saved_files = []

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if self.output_format in ["json", "both"]:
            json_file = Path(self.output_dir) / f"{experiment_name}_data.json"
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)
            saved_files.append(str(json_file))

        if self.output_format in ["csv", "both"]:
            csv_file = Path(self.output_dir) / f"{experiment_name}_data.csv"
            # Flatten the nested structure for CSV
            rows = []
            for camera in data["cameras"]:
                for eye in camera["eyes"]:
                    for measurement in eye["measurements"]:
                        row = {
                            "camera_id": camera["camera_id"],
                            "camera_name": camera["camera_name"],
                            "eye_id": eye["eye_id"],
                            "eye_name": eye["eye_name"],
                            **measurement,
                        }
                        rows.append(row)

            if rows:
                fieldnames = rows[0].keys()
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            saved_files.append(str(csv_file))

        return saved_files

    def _serialize_position(self, position: Position3D) -> Dict[str, float]:
        """Convert Position3D to serializable dict."""
        return {"x": float(position.x), "y": float(position.y), "z": float(position.z)}
