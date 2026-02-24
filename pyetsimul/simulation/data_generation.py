"""Data generation strategy for parameter variations."""

import copy
import json
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyetsimul.log import info

from ..core import Eye
from ..types import Position3D
from ..utils.filename import sanitize_filename
from .composed_variation import ComposedVariation, SequentialVariation
from .core import EyeParameterVariation, ParameterVariation, TargetVariation, VariationStrategy

if TYPE_CHECKING:
    from ..core.camera import Camera


def _process_single_variation(args: tuple[Any, ...]) -> dict[str, Any]:
    """Processes a single variation, designed to be called in parallel from DataGenerationStrategy."""
    (
        eye,
        variation,
        camera,
        lights,
        value,
        index,
        gaze_target,
        use_legacy_look_at,
        use_refraction,
        pupil_center_method,
    ) = args

    eye_copy = copy.deepcopy(eye)

    # This re-instantiates a minimal strategy object inside the worker process
    # to avoid pickling the entire parent object.
    strategy = DataGenerationStrategy(
        eyes=[eye_copy],  # Include the eye copy in the strategy
        cameras=[camera],
        lights=lights,
        experiment_name="worker",
        gaze_target=gaze_target,
        use_legacy_look_at=use_legacy_look_at,
        use_refraction=use_refraction,
        pupil_center_method=pupil_center_method,
        save_to_file=False,  # Avoid child processes trying to save
    )

    current_gaze_target = strategy.apply_parameter_variation(eye_copy, variation, value)
    measurement = strategy.generate_single_measurement(eye_copy, camera, value, index, current_gaze_target)
    return measurement


class DataGenerationStrategy(VariationStrategy):
    """Generates eye tracking data across parameter variations.

    This class encapsulates a complete experimental setup (eyes + hardware) and can
    generate datasets by applying different parameter variations to the same setup.

    Design Philosophy:
    - Eyes are part of the experimental setup (biological configuration)
    - Cameras/lights define the hardware configuration
    - Variations are the experimental parameters to test
    - The same setup can be reused to test multiple variations efficiently

    Example Usage::

        # Create strategy with complete setup
        strategy = DataGenerationStrategy(
            eyes=[eye],
            cameras=[camera],
            lights=[light],
            gaze_target=Position3D(0, 0, 200e-3)
        )

        # Test multiple parameter variations on same setup
        pupil_data = strategy.execute(PupilSizeVariation([3e-3, 7e-3], 10))
        kappa_data = strategy.execute(AngleKappaVariation([4, 8], [1, 3], 10))
        radius_data = strategy.execute(CorneaRadiusVariation([7.5e-3, 8.5e-3], 10))
    """

    def __init__(
        self,
        eyes: list,
        cameras: list,
        lights: list,
        experiment_name: str,
        gaze_target: Position3D | None = None,
        output_dir: str = "output",
        save_to_file: bool = True,
        use_legacy_look_at: bool = False,
        use_refraction: bool = True,
        pupil_center_method: str = "ellipse",
    ) -> None:
        """Initialize data generation strategy with complete experimental setup.

        Args:
            eyes: List of Eye objects to use in experiments
            cameras: List of Camera objects for data capture
            lights: List of Light objects for corneal reflections
            gaze_target: Fixed target position (None allows variation-specific targets)
            output_dir: Directory to save generated datasets
            experiment_name: Base name for experiment files
            save_to_file: Whether to save datasets to disk
            use_legacy_look_at: Use legacy eye rotation method for compatibility
            use_refraction: Enable corneal refraction in image capture
            pupil_center_method: Method for pupil center calculation ("ellipse" or "center_of_mass")

        """
        self.eyes = eyes
        self.cameras = cameras
        self.lights = lights
        self.gaze_target = gaze_target
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

        # Auto-generate safe experiment name for file operations
        self.safe_experiment_name = sanitize_filename(experiment_name)

        self.save_to_file = save_to_file
        self.use_legacy_look_at = use_legacy_look_at
        self.use_refraction = use_refraction
        self.pupil_center_method = pupil_center_method

    def set_experiment_name(self, experiment_name: str) -> None:
        """Update the experiment name for subsequent operations.

        Args:
            experiment_name: New experiment name for metadata and file operations

        """
        self.experiment_name = experiment_name
        self.safe_experiment_name = sanitize_filename(experiment_name)

    def execute(self, variation: ParameterVariation) -> dict[str, Any]:
        """Generate eye tracking data using the configured setup and given variation.

        Args:
            variation: Parameter variation to apply across the experimental setup

        Returns:
            Dictionary containing generated dataset with measurements and metadata

        """
        all_data = {
            "experiment_metadata": self._get_experiment_metadata(variation),
            "setup_configuration": self._get_setup_configuration(self.eyes),
            "cameras": [],
        }
        total_measurements = 0

        for camera_idx, camera in enumerate(self.cameras):
            camera_data = {
                "camera_id": camera_idx,
                "camera_name": getattr(camera, "name", f"Camera {camera_idx + 1}"),
                "camera_parameters": camera.serialize(),
                "eyes": [],
            }

            for eye_idx, eye in enumerate(self.eyes):
                info(f"Processing Camera {camera_idx + 1}/{len(self.cameras)}, Eye {eye_idx + 1}/{len(self.eyes)}...")
                eye_data = {
                    "eye_id": eye_idx,
                    "eye_name": f"Eye {eye_idx + 1}",
                    "initial_eye_parameters": eye.serialize(),
                    "measurements": [],
                }

                # Prepare arguments for parallel processing
                tasks = [
                    (
                        eye,
                        variation,
                        camera,
                        self.lights,
                        value,
                        i,
                        self.gaze_target,
                        self.use_legacy_look_at,
                        self.use_refraction,
                        self.pupil_center_method,
                    )
                    for i, value in enumerate(variation.generate_values())
                ]

                # Use multiprocessing Pool to parallelize the generation of measurements
                with multiprocessing.Pool() as pool:
                    results = pool.map(_process_single_variation, tasks)

                eye_data["measurements"] = results
                total_measurements += len(results)
                camera_data["eyes"].append(eye_data)

            all_data["cameras"].append(camera_data)

        # Save the collected data to a file (if requested).
        saved_files = []
        if self.save_to_file:
            saved_files = self._save_data(all_data, self.safe_experiment_name)
        else:
            info("Dataset generated but not saved (save_to_file=False).")

        return {
            "total_measurements": total_measurements,
            "parameter_name": variation.param_name,
            "saved_files": saved_files,
            "data": all_data,
            "parameter_variation": variation,  # Store variation for plotting
        }

    def generate_single_measurement(
        self,
        eye: Eye,
        camera: "Camera",
        param_value: Any,  # noqa: ANN401
        index: int,
        gaze_target: Position3D | None = None,
    ) -> dict[str, Any]:
        """Generate measurement data for single camera-eye-parameter combination."""
        # Take image with this specific camera using same parameters as estimate_gaze_at
        img = camera.take_image(
            eye, self.lights, use_refraction=self.use_refraction, center_method=self.pupil_center_method
        )

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

        measurement = {
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

        # Save glint sizes if available (when lights have physical diameter)
        if img.glint_sizes_px is not None:
            measurement["glint_sizes_px"] = [float(s) if s is not None else None for s in img.glint_sizes_px]

        return measurement

    def _save_data(self, data: dict, experiment_name: str) -> list[str]:
        """Save dataset to JSON file."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        json_file = Path(self.output_dir) / f"{experiment_name}_data.json"
        with Path.open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        info(f"Dataset saved to: {json_file}")

        return [str(json_file)]

    def apply_parameter_variation(
        self,
        eye_copy: Eye,
        variation: ParameterVariation,
        value: Any,  # noqa: ANN401
    ) -> Position3D | None:
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
            raise TypeError(f"Unknown variation type: {type(variation)}")

        return current_gaze_target

    def _handle_composed_variation(
        self, eye_copy: Eye, variation: ComposedVariation, value: dict
    ) -> Position3D | None:
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
            eye_copy.look_at(self.gaze_target, self.use_legacy_look_at)

        return current_gaze_target

    def _handle_sequential_variation(
        self, eye_copy: Eye, variation: SequentialVariation, value: dict
    ) -> Position3D | None:
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
        self,
        eye_copy: Eye,
        variation: EyeParameterVariation,
        value: Any,  # noqa: ANN401
    ) -> Position3D | None:
        """Handle simple EyeParameterVariation."""
        # For a simple eye parameter variation, apply it and use the default gaze target.
        variation.apply_to_eye(eye_copy, value)
        if self.gaze_target:
            eye_copy.look_at(self.gaze_target, legacy=self.use_legacy_look_at)
        return self.gaze_target

    def _handle_target_variation(self, eye_copy: Eye, value: Any) -> Position3D:  # noqa: ANN401
        """Handle TargetVariation."""
        # For a target variation, the value itself is the gaze target.
        eye_copy.look_at(value, legacy=self.use_legacy_look_at)
        return value

    def _serialize_param_value(self, param_value: Any) -> Any:  # noqa: ANN401
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

    def _get_experiment_metadata(self, variation: ParameterVariation) -> dict[str, Any]:
        """Get experiment metadata and context."""
        return {
            "experiment_name": self.experiment_name,
            "parameter_variation": variation.param_name,
            "total_parameter_values": len(variation),
            "gaze_target": self.gaze_target.serialize() if self.gaze_target else None,
            "num_cameras": len(self.cameras),
            "num_lights": len(self.lights),
        }

    def _get_setup_configuration(self, eyes: list) -> dict[str, Any]:
        """Get complete setup configuration."""
        return {"num_eyes": len(eyes), "lights": [light.serialize() for light in self.lights]}
