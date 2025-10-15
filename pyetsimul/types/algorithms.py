"""Dataclasses for algorithm configurations and results.

Provides structured dataclasses to replace dictionary-based state management
for eye tracking algorithms and their results.

Note: Algorithm-specific state classes (PolynomialGazeModelState, HomographyNormalizationGazeModelState, etc.)
are defined in their respective algorithm modules, not here.
"""

from dataclasses import dataclass
from typing import Any

from .geometry import Point3D


@dataclass
class GazePrediction:
    """Result of gaze estimation algorithm."""

    gaze_point: Point3D  # Predicted gaze point in world coordinates
    confidence: float  # Confidence score [0, 1]
    algorithm_name: str  # Name of algorithm used
    processing_time: float | None = None  # Processing time in seconds
    intermediate_results: dict[str, Any] | None = None  # Algorithm-specific data

    @property
    def is_reliable(self) -> bool:
        """Check if prediction meets minimum confidence threshold."""
        return self.confidence >= 0.5


@dataclass
class AlgorithmState:
    """Base class for algorithm state management."""

    is_calibrated: bool = False
    calibration_error: float | None = None  # RMS calibration error
    last_update: float | None = None  # Timestamp of last state update

    def reset(self) -> None:
        """Reset algorithm to uncalibrated state."""
        self.is_calibrated = False
        self.calibration_error = None
        self.last_update = None
