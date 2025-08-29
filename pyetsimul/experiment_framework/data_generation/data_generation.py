"""Data generation strategy for parameter variations."""

import copy
import json
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from ...core import Eye
from ...types import Position3D
from .core import ParameterVariation, EyeParameterVariation, TargetVariation, VariationStrategy
from .composed_variation import ComposedVariation, SequentialVariation


class DataGenerationStrategy(VariationStrategy):
    """Generates and saves eye tracking data across parameter variations."""

    def __init__(
        self,
        cameras: list,
        lights: list,
        gaze_target: Position3D = None,
        output_dir: str = "output",
        experiment_name: str = None,
        save_to_file: bool = True,
        use_legacy_look_at: bool = False,
        use_refraction: bool = True,
    ):
        self.cameras = cameras
        self.lights = lights
        self.gaze_target = gaze_target  # Fixed gaze target for eye position variations
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.save_to_file = save_to_file
        self.use_legacy_look_at = use_legacy_look_at
        self.use_refraction = use_refraction

    def execute(self, eyes: list, variation: ParameterVariation) -> Dict[str, Any]:
        """Generate eye tracking data: camera → eye → parameter variations."""

        values = variation.generate_values()
        all_data = {
            "experiment_metadata": self._get_experiment_metadata(variation),
            "setup_configuration": self._get_setup_configuration(eyes),
            "cameras": [],
        }
        total_measurements = 0

        # The main loop iterates through cameras, then eyes, then parameter variations.
        for camera_idx, camera in enumerate(tqdm(self.cameras, desc="Processing cameras", position=0, leave=False)):
            camera_data = {
                "camera_id": camera_idx,
                "camera_name": getattr(camera, "name", f"Camera {camera_idx + 1}"),
                "camera_parameters": camera.serialize(),
                "eyes": [],
            }

            for eye_idx, eye in enumerate(tqdm(eyes, desc=f"Camera {camera_idx + 1} eyes", position=1, leave=False)):
                eye_data = {
                    "eye_id": eye_idx,
                    "eye_name": f"Eye {eye_idx + 1}",
                    "initial_eye_parameters": eye.serialize(),
                    "measurements": [],
                }

                # Process all parameter variations for this camera-eye combination.
                # For each value in the variation, a new measurement is generated.
                for i, value in enumerate(
                    tqdm(values, desc=f"Camera {camera_idx + 1} Eye {eye_idx + 1} variations", position=2, leave=False)
                ):
                    # A deep copy of the eye is created for each measurement to ensure a stateless starting point.
                    eye_copy = copy.deepcopy(eye)

                    current_gaze_target = self._apply_parameter_variation(eye_copy, variation, value)

                    # Generate and store the measurement for the current configuration.
                    measurement = self._generate_single_measurement(eye_copy, camera, value, i, current_gaze_target)
                    eye_data["measurements"].append(measurement)
                    total_measurements += 1

                camera_data["eyes"].append(eye_data)

            all_data["cameras"].append(camera_data)

        # Save the collected data to a file (if requested).
        saved_files = []
        if self.save_to_file:
            saved_files = self._save_data(all_data, variation.param_name, self.experiment_name)
        else:
            print("Data generated but not saved (save_to_file=False).")

        return {
            "total_measurements": total_measurements,
            "parameter_name": variation.param_name,
            "saved_files": saved_files,
            "data": all_data,
            "parameter_variation": variation,  # Store variation for plotting
        }

    def _generate_single_measurement(
        self, eye: Eye, camera, param_value: Any, index: int, gaze_target: Position3D = None
    ) -> Dict[str, Any]:
        """Generate measurement data for single camera-eye-parameter combination."""

        # Take image with this specific camera using same parameters as estimate_gaze_at
        img = camera.take_image(eye, self.lights, use_refraction=self.use_refraction)

        # Extract pupil data including boundary_points like estimate_gaze_at does
        pupil_points = []
        pupil_center = None
        if img.pupil_boundary is not None:
            pupil_points = [(float(p.x), float(p.y)) for p in img.pupil_boundary]
        if img.pupil_center is not None:
            pupil_center = [float(img.pupil_center.x), float(img.pupil_center.y)]

        # Extract corneal reflections (glints) from camera image (not eye.find_cr!)
        glints = []
        for cr in img.corneal_reflections:
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
            "gaze_target": gaze_target.serialize() if gaze_target else None,
        }

    def _save_data(self, data: Dict, param_name: str, experiment_name: str) -> List[str]:
        """Save dataset to JSON file."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        json_file = Path(self.output_dir) / f"{experiment_name}_data.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Dataset saved to: {json_file}")

        return [str(json_file)]

    def _apply_parameter_variation(self, eye_copy: Eye, variation: ParameterVariation, value: Any) -> Position3D:
        """Apply parameter variation to eye copy and return gaze target.

        Args:
            eye_copy: Eye object to modify
            variation: Parameter variation to apply
            value: Variation value to apply

        Returns:
            Current gaze target position after applying variation
        """
        current_gaze_target = self.gaze_target

        if isinstance(variation, ComposedVariation):
            current_gaze_target = self._handle_composed_variation(eye_copy, variation, value)
        elif isinstance(variation, SequentialVariation):
            current_gaze_target = self._handle_sequential_variation(eye_copy, variation, value)
        elif isinstance(variation, EyeParameterVariation):
            current_gaze_target = self._handle_eye_parameter_variation(eye_copy, variation, value)
        elif isinstance(variation, TargetVariation):
            current_gaze_target = self._handle_target_variation(eye_copy, value)
        else:
            raise ValueError(f"Unknown variation type: {type(variation)}")

        return current_gaze_target

    def _handle_composed_variation(self, eye_copy: Eye, variation: ComposedVariation, value: dict) -> Position3D:
        """Handle ComposedVariation by applying each inner variation."""
        current_gaze_target = self.gaze_target

        # For ComposedVariation, iterate through the inner variations and apply them.
        # This supports combining different types of variations (e.g., eye position and target position).
        for v_inner in variation.variations:
            inner_value = value[v_inner.param_name]
            if isinstance(v_inner, EyeParameterVariation):
                v_inner.apply_to_eye(eye_copy, inner_value)
            elif isinstance(v_inner, TargetVariation):
                eye_copy.look_at(inner_value)
                current_gaze_target = inner_value

        # If no target variation is present in the composition, use the default gaze target.
        if current_gaze_target == self.gaze_target and self.gaze_target:
            eye_copy.look_at(self.gaze_target)

        return current_gaze_target

    def _handle_sequential_variation(self, eye_copy: Eye, variation: SequentialVariation, value: dict) -> Position3D:
        """Handle SequentialVariation by applying one variation from the sequence."""
        current_gaze_target = self.gaze_target

        # For SequentialVariation, apply one variation at a time from the sequence.
        v_inner = variation.variations[value["variation_index"]]
        inner_value = value["value"]
        if isinstance(v_inner, EyeParameterVariation):
            v_inner.apply_to_eye(eye_copy, inner_value)
            if self.gaze_target:
                eye_copy.look_at(self.gaze_target, legacy=self.use_legacy_look_at)
        elif isinstance(v_inner, TargetVariation):
            eye_copy.look_at(inner_value, legacy=self.use_legacy_look_at)
            current_gaze_target = inner_value

        return current_gaze_target

    def _handle_eye_parameter_variation(
        self, eye_copy: Eye, variation: EyeParameterVariation, value: Any
    ) -> Position3D:
        """Handle simple EyeParameterVariation."""
        # For a simple eye parameter variation, apply it and use the default gaze target.
        variation.apply_to_eye(eye_copy, value)
        if self.gaze_target:
            eye_copy.look_at(self.gaze_target, legacy=self.use_legacy_look_at)
        return self.gaze_target

    def _handle_target_variation(self, eye_copy: Eye, value: Any) -> Position3D:
        """Handle TargetVariation."""
        # For a target variation, the value itself is the gaze target.
        eye_copy.look_at(value, legacy=self.use_legacy_look_at)
        return value

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
