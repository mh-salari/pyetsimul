"""Filename utilities for safe file operations."""

import re


def sanitize_filename(name: str) -> str:
    """Convert any string to a safe filename.

    Args:
        name: Original string (e.g., experiment name)

    Returns:
        Safe filename string with only alphanumeric characters and underscores

    Examples:
        >>> sanitize_filename("My Experiment!")
        'my_experiment'
        >>> sanitize_filename("Eye Position Variation")
        'eye_position_variation'
    """
    safe_name = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    return re.sub(r"[-\s]+", "_", safe_name)
