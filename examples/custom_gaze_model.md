# Template for Custom Gaze Estimation Models

This guide provides a template for creating a custom gaze estimation model in PyEtSimul. It outlines the essential structure and expected return types, allowing you to fill in your specific implementation details.

## Core Architecture

PyEtSimul gaze estimation models follow a simple pattern:
1. **Inherit from `EyeTracker`** - base class that handles the eye tracking workflow
2. **Store calibration state** - parameters learned during calibration
3. **Implement calibration** - learn the mapping from eye features to gaze positions
4. **Implement prediction** - apply the learned mapping to new measurements

---

## Block 1: Algorithm State

**Purpose**: Store your calibration parameters. This class is responsible for holding any data that your model learns during the calibration phase and needs for prediction.

**What you need**:
- Inherit from `AlgorithmState`.
- Add fields for your calibration parameters (e.g., matrices, coefficients, model weights).
- Implement `serialize()` and `deserialize()` to allow your tracker to be saved to and loaded from disk.

### Placeholder Code

```python
class MyAlgorithmState(AlgorithmState): # Inherit from pyetsimul.types.algorithms.AlgorithmState
    """
    State for your custom gaze model.
    TODO: Add fields to store your calibration parameters.
    """

    def serialize(self) -> dict:
        """Serialize the state to a dictionary for saving."""
        # TODO: Add your custom parameters to the dictionary.
        # Ensure they are JSON-serializable.
        return {} # Return a dictionary of your state

    @classmethod
    def deserialize(cls, data: dict) -> "MyAlgorithmState":
        """Deserialize the dictionary back into a state object."""
        state = cls()
        # TODO: Restore your custom parameters from the dictionary.
        return state
```

---

## Blocks 2, 3, and 4: The Gaze Model Class and its Logic

**Purpose**: The main `EyeTracker` class orchestrates calibration and prediction.

**What you need**:
- Inherit from `EyeTracker` and store your custom `AlgorithmState`.
- Implement the `calibrate` method to learn your model parameters.
- Implement the `predict_gaze` method to apply the learned model.

### Placeholder Code

```python
class MyGazeModel(EyeTracker): # Inherit from pyetsimul.core.EyeTracker
    """
    A template for a custom gaze estimation model.
    TODO: Describe what your model does.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_state = MyAlgorithmState() # Instance of your custom AlgorithmState

    @property
    def algorithm_name(self) -> str:
        """Return a unique identifier for your algorithm."""
        # TODO: Choose a unique name for your model
        return "my_custom_gaze_model"

    @classmethod
    def create(
        cls,
        # TODO: Add any specific parameters your model needs for instantiation
    ) -> "MyGazeModel":
        """A factory method for clean instantiation."""
        return cls() # Return an instance of your model

    # --- Block 3: Calibration Logic ---
    # Purpose: Learn the mapping from eye features to world coordinates.
    # This method modifies the internal state of the tracker, specifically `self.algorithm_state`.
    # It does not return a value.
    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None: # EyeMeasurement from pyetsimul.types
        """Calibrate the model by learning the mapping from eye features to gaze."""
        # TODO: Implement your calibration logic here.
        # This method should update `self.algorithm_state` with learned parameters.
        # Set `self.algorithm_state.is_calibrated = True` when calibration is complete.
        pass

    # --- Block 4: Prediction Logic ---
    # Purpose: Estimate gaze from a single eye measurement using the learned model.
    # This method returns a `GazePrediction` object.
    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction: # GazePrediction from pyetsimul.types
        """Predict gaze position from a single eye measurement."""
        # TODO: Implement your gaze prediction logic here.
        # Extract features from `measurement`.
        # Apply your learned model (from `self.algorithm_state`) to predict gaze.
        # Handle cases where essential data is missing (e.g., return confidence=0.0).

        # Return a GazePrediction object
        return GazePrediction(
            gaze_point=Point3D(0.0, 0.0, 0.0), # Point3D from pyetsimul.types
            confidence=0.0,
            algorithm_name=self.algorithm_name,
            processing_time=0.0,
            intermediate_results={},
        )

    # --- Serialization ---
    def serialize(self) -> dict:
        """Serialize the entire tracker for saving to disk."""
        return {
            "algorithm_state": self.algorithm_state.serialize(),
            # TODO: Add any other tracker-specific state you need to serialize
        }

    @classmethod
    def deserialize(cls, data: dict) -> "MyGazeModel":
        """Restore the tracker from serialized data."""
        tracker = cls()
        tracker.algorithm_state = MyAlgorithmState.deserialize(data["algorithm_state"])
        # TODO: Restore any other tracker-specific state you serialized
        return tracker
```
