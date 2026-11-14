"""Data loading utilities for cached experiment datasets."""

import json
from pathlib import Path
from typing import Any

from ..utils.filename import sanitize_filename


def load_experiment_data(experiment_name: str, output_dir: str | Path = "outputs") -> dict[str, Any]:
    """Load cached experiment dataset by name.

    Args:
        experiment_name: Original experiment name (will be auto-sanitized for file lookup)
        output_dir: Directory containing the cached data files

    Returns:
        Dictionary containing the cached dataset

    Raises:
        FileNotFoundError: If no cached dataset exists for this experiment name
        ValueError: If the file lacks the metadata or the cameras a generated dataset has

    """
    output_path = Path(output_dir)
    safe_filename = sanitize_filename(experiment_name)
    cache_filename = f"{safe_filename}_data.json"
    cache_path = output_path / cache_filename

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached dataset found for experiment '{experiment_name}' at: {cache_path}")

    with Path.open(cache_path, encoding="utf-8") as f:
        cached_data = json.load(f)

    try:
        parameter_name = cached_data["experiment_metadata"]["parameter_variation"]
        # One measurement per parameter value for every camera and eye, as DataGenerationStrategy.execute counts.
        total_measurements = sum(
            len(eye["measurements"]) for camera in cached_data["cameras"] for eye in camera["eyes"]
        )
    except KeyError as error:
        raise ValueError(f"{cache_path} is not a generated dataset: it has no {error}") from error

    # Return in the format expected by comparison functions
    return {
        "total_measurements": total_measurements,
        "data": cached_data,
        "parameter_name": parameter_name,
        "saved_files": [str(cache_path)],
    }
