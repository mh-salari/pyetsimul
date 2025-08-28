"""Data generation strategy for parameter variations."""

import copy
import json
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
        output_dir: str = "output",
        experiment_name: str = None,
    ):
        self.cameras = cameras
        self.lights = lights
        self.gaze_target = gaze_target  # Fixed gaze target for eye position variations
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

    def execute(self, eyes: list, variation: ParameterVariation) -> Dict[str, Any]:
        """Generate eye tracking data: camera → eye → parameter variations."""

        values = variation.generate_values()
        all_data = {
            "experiment_metadata": self._get_experiment_metadata(variation),
            "setup_configuration": self._get_setup_configuration(eyes),
            "cameras": [],
        }
        total_measurements = 0

        # Progress bars: Camera (outer) → Eye (middle) → Variations (inner)
        for camera_idx, camera in enumerate(tqdm(self.cameras, desc="Processing cameras", position=0)):
            camera_data = {
                "camera_id": camera_idx,
                "camera_name": getattr(camera, "name", f"Camera {camera_idx + 1}"),
                "camera_parameters": camera.serialize(),
                "eyes": [],
            }

            for eye_idx, eye in enumerate(tqdm(eyes, desc=f"Camera {camera_idx + 1} eyes", position=1, leave=False)):
                eye_copy = copy.deepcopy(eye)
                eye_data = {
                    "eye_id": eye_idx,
                    "eye_name": f"Eye {eye_idx + 1}",
                    "initial_eye_parameters": eye.serialize(),
                    "measurements": [],
                }

                # Process all parameter variations for this camera-eye combination
                for i, value in enumerate(
                    tqdm(values, desc=f"C{camera_idx + 1}E{eye_idx + 1} variations", position=2, leave=False)
                ):
                    # Apply parameter variation (eye properties)
                    variation.apply_to_eye(eye_copy, value)

                    # Apply user's gaze target (required except for target variations)
                    if self.gaze_target:
                        eye_copy.look_at(self.gaze_target)

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
            "parameter_value": self._serialize_param_value(param_value),
            "pupil_center": pupil_center,
            "pupil_boundary": pupil_points,
            "corneal_reflections": glints,
            # Save everything so we can recreate any configuration
            "eye_state": eye.serialize(),
            "camera_state": camera.serialize(),
            "lights_state": [light.serialize() for light in self.lights],
            "gaze_target": self.gaze_target.serialize() if self.gaze_target else None,
        }

    def _save_data(self, data: Dict, param_name: str, experiment_name: str) -> List[str]:
        """Save dataset to JSON file."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        json_file = Path(self.output_dir) / f"{experiment_name}_data.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        return [str(json_file)]

    def _serialize_param_value(self, param_value):
        """Serialize parameter values for JSON storage."""
        # Handle None
        if param_value is None:
            return None

        # Handle primitives (already JSON-serializable)
        if isinstance(param_value, (str, int, float, bool)):
            return param_value

        # Handle collections
        if isinstance(param_value, (list, tuple)):
            return [self._serialize_param_value(item) for item in param_value]

        if isinstance(param_value, dict):
            return {key: self._serialize_param_value(val) for key, val in param_value.items()}

        # Handle our structured types (all have serialize method)
        return param_value.serialize()

    def _get_experiment_metadata(self, variation: ParameterVariation) -> Dict[str, Any]:
        """Get experiment metadata and context."""
        return {
            "experiment_name": self.experiment_name,
            "parameter_variation": variation.param_name,
            "total_parameter_values": len(variation.generate_values()),
            "gaze_target": self.gaze_target.serialize() if self.gaze_target else None,
            "num_cameras": len(self.cameras),
            "num_lights": len(self.lights),
        }

    def _get_setup_configuration(self, eyes: list) -> Dict[str, Any]:
        """Get complete setup configuration."""
        return {"num_eyes": len(eyes), "lights": [light.serialize() for light in self.lights]}
