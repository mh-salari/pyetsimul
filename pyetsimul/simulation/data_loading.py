"""Data loading utilities for cached experiment datasets."""

import json
from pathlib import Path
from typing import Union, Dict, Any

from ..utils.filename import sanitize_filename


def load_experiment_data(experiment_name: str, output_dir: Union[str, Path] = "outputs") -> Dict[str, Any]:
    """Load cached experiment dataset by name.

    Args:
        experiment_name: Original experiment name (will be auto-sanitized for file lookup)
        output_dir: Directory containing the cached data files

    Returns:
        Dictionary containing the cached dataset

    Raises:
        FileNotFoundError: If no cached dataset exists for this experiment name
    """
    output_path = Path(output_dir)
    safe_filename = sanitize_filename(experiment_name)
    cache_filename = f"{safe_filename}_data.json"
    cache_path = output_path / cache_filename

    if not cache_path.exists():
        raise FileNotFoundError(f"No cached dataset found for experiment '{experiment_name}' at: {cache_path}")

    with open(cache_path, "r") as f:
        cached_data = json.load(f)

    # Return in the format expected by comparison functions
    total_measurements = cached_data.get("experiment_metadata", {}).get("total_parameter_values", 0)
    return {
        "total_measurements": total_measurements,
        "data": cached_data,
        "parameter_name": cached_data.get("experiment_metadata", {}).get("parameter_variation", ""),
        "saved_files": [str(cache_path)],
    }
