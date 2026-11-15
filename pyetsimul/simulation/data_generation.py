"""Data generation strategy for parameter variations."""

import copy
import json
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyetsimul.log import configure_worker, get_log_file, get_log_level, info, warning

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
        use_refraction,
        pupil_center_method,
        camera_index,
        eye_index,
    ) = args

    eye_copy = copy.deepcopy(eye)
    # Every task receives the camera with the same generator state, so each measurement starts its own
    # noise stream; the stream numbers make it the same in every run and different between measurements.
    camera.restart_noise(camera_index, eye_index, index)
    if camera.glint_noise_config is not None:
        camera.glint_noise_config.restart(camera_index, eye_index, index)

    # This re-instantiates a minimal strategy object inside the worker process
    # to avoid pickling the entire parent object.
    strategy = DataGenerationStrategy(
        eyes=[eye_copy],  # Include the eye copy in the strategy
        cameras=[camera],
        lights=lights,
        experiment_name="worker",
        gaze_target=gaze_target,
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
            gaze_target=Position3D(0, 0, 200)
        )

        # Test multiple parameter variations on same setup
        pupil_data = strategy.execute(PupilSizeVariation([3, 7], 10))
        kappa_data = strategy.execute(AngleKappaVariation([4, 8], [1, 3], 10))
        radius_data = strategy.execute(CorneaRadiusVariation([7.5, 8.5], 10))
    """

    def __init__(
        self,
        eyes: list,
        cameras: list,
        lights: list,
        experiment_name: str,
        gaze_target: Position3D | None = None,
        output_dir: str = "outputs",
        save_to_file: bool = True,
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
            use_refraction: Enable corneal refraction in image capture
            pupil_center_method: Method for pupil center calculation ("ellipse", "convex_hull" or
                "center_of_mass")

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
                        self.use_refraction,
                        self.pupil_center_method,
                        camera_idx,
                        eye_idx,
                    )
                    for i, value in enumerate(variation.generate_values())
                ]

                # Use multiprocessing Pool to parallelize the generation of measurements. The workers get
                # the log level and the log file of this process, which a spawned worker would not inherit.
                worker_logging = (get_log_level(), get_log_file())
                with multiprocessing.Pool(initializer=configure_worker, initargs=worker_logging) as pool:
                    results = pool.map(_process_single_variation, tasks)

                eye_data["measurements"] = results
                total_measurements += len(results)
                camera_data["eyes"].append(eye_data)

            all_data["cameras"].append(camera_data)

        self._report_missing_features(all_data, total_measurements)

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

    @staticmethod
    def _report_missing_features(all_data: dict[str, Any], total_measurements: int) -> None:
        """Warn once, with counts, when measurements lack a glint or the pupil center.

        A missing feature is stored as None and makes a gaze model return no prediction, so the counts say
        how much of the dataset is affected. The per-measurement reasons are printed at the DEBUG log level.
        """
        measurements = [
            measurement
            for camera_data in all_data["cameras"]
            for eye_data in camera_data["eyes"]
            for measurement in eye_data["measurements"]
        ]
        without_glint = sum(1 for m in measurements if any(glint is None for glint in m["corneal_reflections"]))
        without_pupil = sum(1 for m in measurements if m["pupil_center"] is None)
        if without_glint or without_pupil:
            warning(
                f"Of {total_measurements} measurements, {without_glint} have no glint for at least one light and "
                f"{without_pupil} have no pupil center."
            )

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
        """Apply parameter variation to eye copy, turn the eye to its target and return that target.

        The eye is turned once, after every eye parameter is set, so the order of combined variations does
        not matter. The target is the varied one, else the strategy's gaze target, else the target the eye
        was looking at before the variation. An eye with none of these keeps its rest pose.

        Args:
            eye_copy: Eye object to modify
            variation: Parameter variation to apply
            value: Variation value to apply

        Returns:
            Gaze target the eye looks at after the variation, or None for an eye in its rest pose

        """
        # Read before the variation is applied: placing an eye with hold_fixation off clears its target.
        held_target = eye_copy.current_target_point

        if isinstance(variation, ComposedVariation):
            varied_target = self._apply_composed_variation(eye_copy, variation, value)
        elif isinstance(variation, SequentialVariation):
            varied_target = self._apply_sequential_variation(eye_copy, variation, value)
        elif isinstance(variation, EyeParameterVariation):
            variation.apply_to_eye(eye_copy, value)
            varied_target = None
        elif isinstance(variation, TargetVariation):
            varied_target = value
        else:
            raise TypeError(f"Unknown variation type: {type(variation)}")

        target = varied_target if varied_target is not None else self.gaze_target
        if target is None:
            target = held_target
        if target is not None:
            eye_copy.look_at(target)
        return target

    @staticmethod
    def _apply_composed_variation(eye_copy: Eye, variation: ComposedVariation, value: dict) -> Position3D | None:
        """Apply each eye parameter of a composition and return its varied target, if it has one."""
        varied_target = None

        # This supports combining different types of variations (e.g., eye position and target position).
        for v_inner in variation.variations:
            inner_value = value[v_inner.param_name]
            if isinstance(v_inner, EyeParameterVariation):
                v_inner.apply_to_eye(eye_copy, inner_value)
            elif isinstance(v_inner, TargetVariation):
                varied_target = inner_value
            else:
                raise TypeError(f"A composition cannot hold a variation of type {type(v_inner).__name__}")

        return varied_target

    @staticmethod
    def _apply_sequential_variation(eye_copy: Eye, variation: SequentialVariation, value: dict) -> Position3D | None:
        """Apply the one variation of a sequence that this value belongs to and return its varied target."""
        v_inner = variation.variations[value["variation_index"]]
        inner_value = value["value"]
        if isinstance(v_inner, EyeParameterVariation):
            v_inner.apply_to_eye(eye_copy, inner_value)
            return None
        if isinstance(v_inner, TargetVariation):
            return inner_value
        raise TypeError(f"A sequence cannot hold a variation of type {type(v_inner).__name__}")

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
