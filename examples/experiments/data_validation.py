"""Validate generated eye tracking data."""

import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def find_data_files(outputs_dir: Path) -> List[Path]:
    """Find all JSON data files in the outputs directory."""
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        return []

    json_files = list(outputs_dir.glob("*_data.json"))
    if not json_files:
        print(f"No data files found in {outputs_dir}!")
        return []

    return json_files


def validate_experiment_metadata(data: Dict[str, Any]) -> bool:
    """Validate experiment metadata structure."""
    metadata = data.get("experiment_metadata", {})

    required_fields = ["experiment_name", "parameter_variation", "total_parameter_values"]
    for field in required_fields:
        if field not in metadata:
            print(f"Missing metadata field: {field}")
            return False

    return True


def validate_measurements(data: Dict[str, Any]) -> bool:
    """Validate measurement data by recreating simulator and comparing results."""
    from pyetsimul.core import Eye, Camera, Light

    cameras_data = data.get("cameras", [])
    if not cameras_data:
        print("No camera data found")
        return False

    total_measurements = 0
    valid_measurements = 0

    for camera_data in cameras_data:
        eyes_data = camera_data.get("eyes", [])
        for eye_data in eyes_data:
            measurements = eye_data.get("measurements", [])
            total_measurements += len(measurements)

            for measurement in tqdm(measurements, desc="Validating measurements", leave=False):
                # Check required fields exist
                required_fields = [
                    "measurement_id",
                    "parameter_value",
                    "eye_state",
                    "camera_state",
                    "lights_state",
                    "pupil_center",
                    "pupil_boundary",
                    "corneal_reflections",
                ]
                missing = [f for f in required_fields if f not in measurement]
                if missing:
                    print(f"Measurement missing fields: {missing}")
                    continue

                # Recreate exact simulator state from JSON data
                try:
                    eye = Eye.deserialize(measurement["eye_state"])
                    camera = Camera.deserialize(measurement["camera_state"])
                    lights = [Light.deserialize(light_state) for light_state in measurement["lights_state"]]

                    # Take image with recreated setup
                    img = camera.take_image(eye, lights)

                    # Extract pupil data for comparison
                    expected_pupil_center = measurement["pupil_center"]
                    expected_pupil_boundary = measurement["pupil_boundary"]
                    expected_glints = measurement["corneal_reflections"]

                    # Compare pupil center
                    actual_pupil_center = None
                    if img.pupil_center is not None:
                        actual_pupil_center = [float(img.pupil_center.x), float(img.pupil_center.y)]

                    if not compare_values(actual_pupil_center, expected_pupil_center, "pupil_center"):
                        continue

                    # Compare pupil boundary
                    actual_pupil_boundary = []
                    if img.pupil_boundary is not None:
                        actual_pupil_boundary = [[float(p.x), float(p.y)] for p in img.pupil_boundary]

                    if not compare_values(actual_pupil_boundary, expected_pupil_boundary, "pupil_boundary"):
                        continue

                    # Compare corneal reflections
                    actual_glints = []
                    if img.corneal_reflections is not None:
                        actual_glints = [[float(cr.x), float(cr.y)] for cr in img.corneal_reflections]
                    else:
                        actual_glints = [None] * len(lights)

                    if not compare_values(actual_glints, expected_glints, "corneal_reflections"):
                        continue

                    valid_measurements += 1

                except Exception as e:
                    print(f"Failed to recreate measurement {measurement['measurement_id']}: {e}")
                    continue

    print(f"Valid measurements: {valid_measurements}/{total_measurements}")
    return valid_measurements == total_measurements


def compare_values(actual, expected, field_name, tolerance=1e-10):
    """Compare actual vs expected values with tolerance."""
    if actual is None and expected is None:
        return True

    if (actual is None) != (expected is None):
        print(f"Mismatch in {field_name}: one is None, other is not")
        return False

    if isinstance(expected, list):
        if len(actual) != len(expected):
            print(f"Mismatch in {field_name}: different lengths {len(actual)} vs {len(expected)}")
            return False

        for i, (a, e) in enumerate(zip(actual, expected)):
            if not compare_values(a, e, f"{field_name}[{i}]", tolerance):
                return False
        return True

    if isinstance(expected, (int, float)):
        if abs(actual - expected) > tolerance:
            print(f"Mismatch in {field_name}: {actual} vs {expected} (diff: {abs(actual - expected)})")
            return False
        return True

    if actual != expected:
        print(f"Mismatch in {field_name}: {actual} vs {expected}")
        return False

    return True


def validate_parameter_values(data: Dict[str, Any]) -> bool:
    """Validate parameter values match expected types."""
    experiment_name = data["experiment_metadata"]["experiment_name"]
    cameras = data.get("cameras", [])

    for camera in cameras:
        eyes = camera.get("eyes", [])
        for eye in eyes:
            measurements = eye.get("measurements", [])

            for measurement in measurements:
                param_value = measurement.get("parameter_value")

                # Handle both single and combined parameter experiments
                if "eye_position" in experiment_name:
                    # Extract eye position data (handle both direct and nested formats)
                    eye_pos_data = (
                        param_value.get("eye_position", param_value) if isinstance(param_value, dict) else param_value
                    )
                    if not isinstance(eye_pos_data, dict) or not all(k in eye_pos_data for k in ["x", "y", "z"]):
                        print(f"Invalid eye position parameter: {param_value}")
                        return False

                elif "gaze_movement" in experiment_name:
                    # Extract gaze movement data (handle both direct and nested formats)
                    gaze_data = (
                        param_value.get("target_position", param_value)
                        if isinstance(param_value, dict)
                        else param_value
                    )
                    if not isinstance(gaze_data, dict) or not all(k in gaze_data for k in ["x", "y", "z"]):
                        print(f"Invalid gaze movement parameter: {param_value}")
                        return False

                elif "pupil_size" in experiment_name:
                    # Extract pupil size data (handle both direct and nested formats)
                    pupil_data = (
                        param_value.get("pupil_size", param_value) if isinstance(param_value, dict) else param_value
                    )
                    if not isinstance(pupil_data, (int, float)):
                        print(f"Invalid pupil size parameter: {param_value}")
                        return False
                    # Check reasonable range (1mm to 10mm)
                    if not (0.001 <= pupil_data <= 0.010):
                        print(f"Pupil size out of range: {param_value}")
                        return False

    return True


def validate_coordinate_consistency(data: Dict[str, Any]) -> bool:
    """Validate coordinate system consistency."""
    cameras = data.get("cameras", [])

    for camera in cameras:
        eyes = camera.get("eyes", [])
        for eye in eyes:
            measurements = eye.get("measurements", [])

            for measurement in measurements:
                eye_state = measurement.get("eye_state", {})
                eye_position = eye_state.get("position")

                if eye_position:
                    # Check if coordinates are reasonable (not NaN, not extreme values)
                    coords = [eye_position.get("x"), eye_position.get("y"), eye_position.get("z")]
                    if any(coord is None for coord in coords):
                        print("Eye position has None coordinates")
                        return False

                    # Check for NaN or extreme values
                    for coord in coords:
                        if abs(coord) > 1.0:  # More than 1 meter seems unreasonable
                            print(f"Extreme eye coordinate: {coord}")
                            return False

    return True


def validate_dataset(json_file: Path) -> bool:
    """Validate a complete dataset."""
    print(f"\nValidating: {json_file.name}")

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return False

    # Run all validation checks
    checks = [
        ("Metadata structure", validate_experiment_metadata(data)),
        ("Measurement structure", validate_measurements(data)),
        ("Parameter values", validate_parameter_values(data)),
        ("Coordinate consistency", validate_coordinate_consistency(data)),
    ]

    all_passed = True
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  {check_name}: {status}")
        if not result:
            all_passed = False

    return all_passed


def main():
    """Validate all generated datasets."""
    print("Data Validation")
    print("=" * 30)

    outputs_dir = Path(__file__).parent / "outputs"

    json_files = find_data_files(outputs_dir)
    if not json_files:
        return

    total_files = len(json_files)
    valid_files = 0

    for json_file in tqdm(json_files, desc="Validating files"):
        if validate_dataset(json_file):
            valid_files += 1

    print("\n" + "=" * 30)
    print(f"Validation complete: {valid_files}/{total_files} files valid")

    if valid_files == total_files:
        print("All datasets are valid!")
    else:
        print("Some datasets have issues!")


if __name__ == "__main__":
    main()
