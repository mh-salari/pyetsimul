"""State management for homography normalization gaze model."""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import joblib
import numpy as np

from pyetsimul.types.algorithms import AlgorithmState
from pyetsimul.types.geometry import Point2D


@dataclass
class HomographyGazeModelState(AlgorithmState):
    """State for homography normalization gaze model."""

    # Calibration: normalized space → screen space mapping
    H_s_n: np.ndarray | None = None  # 3x3 homography matrix

    # Reference glint pattern in normalized space (unit square corners)
    reference_glints_normalized: list["Point2D"] | None = None  # 4 corners using Point2D

    # RANSAC threshold used during calibration
    ransac_threshold: float = 5.0

    # Optional: Gaussian Process for error correction (Phase 3)
    gp_model: Any | None = None  # GP model object
    calibration_errors: np.ndarray | None = None  # Residual errors at calib points

    def serialize(self) -> dict:
        """Serialize homography state to dictionary.

        Includes GP model serialization using joblib. The GP model is serialized
        to bytes and encoded as base64 for JSON compatibility.
        """
        # Serialize GP model if present
        gp_model_bytes = None
        if self.gp_model is not None:
            # Use joblib to serialize the scikit-learn GP model to bytes
            buffer = BytesIO()
            joblib.dump(self.gp_model, buffer)
            gp_model_bytes = base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "is_calibrated": self.is_calibrated,
            "calibration_error": self.calibration_error,
            "last_update": self.last_update,
            "H_s_n": self.H_s_n.tolist() if self.H_s_n is not None else None,
            "reference_glints_normalized": (
                [pt.serialize() for pt in self.reference_glints_normalized]
                if self.reference_glints_normalized is not None
                else None
            ),
            "ransac_threshold": self.ransac_threshold,
            "gp_model": gp_model_bytes,
            "calibration_errors": (self.calibration_errors.tolist() if self.calibration_errors is not None else None),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "HomographyGazeModelState":
        """Deserialize from dictionary.

        Restores the GP model from base64-encoded joblib bytes if present.
        """
        state = cls(
            is_calibrated=data["is_calibrated"],
            calibration_error=data["calibration_error"],
            last_update=data["last_update"],
            ransac_threshold=data.get("ransac_threshold", 5.0),
        )

        # Convert homography matrix back to numpy
        if data["H_s_n"] is not None:
            state.H_s_n = np.array(data["H_s_n"])

        # Convert reference glints back to Point2D list
        if data["reference_glints_normalized"] is not None:
            state.reference_glints_normalized = [
                Point2D.deserialize(pt_data) for pt_data in data["reference_glints_normalized"]
            ]

        # Deserialize GP model if present
        if data.get("gp_model") is not None:
            # Decode base64 and deserialize with joblib
            gp_model_bytes = base64.b64decode(data["gp_model"].encode("ascii"))
            buffer = BytesIO(gp_model_bytes)
            state.gp_model = joblib.load(buffer)

        # Convert calibration errors back to numpy
        if data.get("calibration_errors") is not None:
            state.calibration_errors = np.array(data["calibration_errors"])

        return state
