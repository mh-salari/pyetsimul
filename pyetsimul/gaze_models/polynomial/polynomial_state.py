"""State management for polynomial gaze model algorithm."""

from dataclasses import dataclass

import numpy as np

from pyetsimul.types.algorithms import AlgorithmState


@dataclass
class PolynomialGazeModelState(AlgorithmState):
    """State for polynomial gaze model algorithm."""

    x_coefficients: np.ndarray | None = None  # Polynomial coefficients for x
    y_coefficients: np.ndarray | None = None  # Polynomial coefficients for y
    input_normalization: dict[str, float] | None = None  # Input scaling parameters

    def serialize(self) -> dict:
        """Serialize polynomial gaze model state to dictionary."""
        return {
            "is_calibrated": self.is_calibrated,
            "calibration_error": self.calibration_error,
            "last_update": self.last_update,
            "x_coefficients": self.x_coefficients.tolist() if self.x_coefficients is not None else None,
            "y_coefficients": self.y_coefficients.tolist() if self.y_coefficients is not None else None,
            "input_normalization": self.input_normalization,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PolynomialGazeModelState":
        """Deserialize from dictionary representation."""
        state = cls(
            is_calibrated=data["is_calibrated"],
            calibration_error=data["calibration_error"],
            last_update=data["last_update"],
            input_normalization=data["input_normalization"],
        )

        # Convert coefficient lists back to numpy arrays
        if data["x_coefficients"] is not None:
            state.x_coefficients = np.array(data["x_coefficients"])
        if data["y_coefficients"] is not None:
            state.y_coefficients = np.array(data["y_coefficients"])

        return state
