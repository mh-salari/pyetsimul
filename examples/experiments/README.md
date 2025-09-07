# Creating Custom Parameter Variations

## Quick Start: Custom Variation

Create a custom parameter variation in 3 steps:

```python
from pyetsimul.simulation import GenericEyeVariation

class MyCustomVariation(GenericEyeVariation):
    """Custom variation that changes any eye parameter."""

    def __init__(self, value_range: list[float], num_steps: int = 10):
        # Parameter path in dot notation (e.g., "cornea.radius", "pupil.diameter")
        super().__init__("parameter.path", value_range, num_steps)

    def describe(self) -> str:
        """Description for this variation."""
        min_val, max_val = self.value_range
        return f"custom parameter {min_val} to {max_val} ({self.num_steps} steps)"

    def apply_to_eye(self, eye, value: float) -> None:
        """Apply the parameter value to the eye."""
        # Example: eye.cornea.radius = value
        # Example: eye.pupil.diameter = value
        pass
```

## Real Example: Corneal K Parameter

```python
from pyetsimul.core.cornea import ConicCornea

class ConicCorneaKVariation(GenericEyeVariation):
    def __init__(self, k_range: list[float], num_steps: int = 10):
        super().__init__("cornea.anterior_k", k_range, num_steps)

    def describe(self) -> str:
        min_val, max_val = self.value_range
        return f"conic cornea K parameter {min_val:.3f} to {max_val:.3f} ({self.num_steps} steps)"

    def apply_to_eye(self, eye, value: float) -> None:
        eye.cornea.anterior_k = value

# Usage
variation = ConicCorneaKVariation([-0.05, -0.15], 5)
```

## Run Your Custom Variation

```python
from pyetsimul.simulation import DataGenerationStrategy, ExperimentConfig
from pyetsimul.core import Eye, Camera, Light

# Create experiment config
config = ExperimentConfig(
    experiment_name="My Custom Experiment",
    eyes=[Eye()],  # Add your custom eye setup if needed
    cameras=[Camera()],
    lights=[Light()],
    output_dir="outputs"
)

# Generate data
data_gen = DataGenerationStrategy(
    eyes=config.eyes,
    cameras=config.cameras,
    lights=config.lights,
    experiment_name="My Custom Experiment",  # Any name - auto-sanitized
    save_to_file=True,
)

dataset = data_gen.execute(variation)
```

That's it! The framework handles the rest automatically.