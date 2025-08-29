"""Pupil size variations for eye tracking simulation."""

import numpy as np
from typing import List, Iterable

from ....core import Eye
from ..core import EyeParameterVariation


class PupilSizeVariation(EyeParameterVariation):
    """Varies pupil diameter to simulate different lighting conditions."""

    def __init__(self, diameter_range: List[float], num_steps: int = 10):
        """Initialize pupil size variation.

        Args:
            diameter_range: [min_diameter, max_diameter] in meters
            num_steps: Number of diameter steps to generate
        """
        super().__init__("pupil_size")
        self.diameter_range = diameter_range
        self.num_steps = num_steps

        if len(diameter_range) != 2:
            raise ValueError("diameter_range must have exactly 2 elements [min, max]")
        if diameter_range[0] >= diameter_range[1]:
            raise ValueError("min_diameter must be less than max_diameter")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

    @property
    def description(self):
        return f"{self.__class__.__name__} ({self.num_steps} steps)"

    def __len__(self) -> int:
        return self.num_steps

    def generate_values(self) -> Iterable[float]:
        """Generate pupil diameter values in meters."""
        if self.num_steps == 1:
            yield (self.diameter_range[0] + self.diameter_range[1]) / 2
            return

        yield from np.linspace(self.diameter_range[0], self.diameter_range[1], self.num_steps)

    def apply_to_eye(self, eye: Eye, value: float) -> None:
        """Apply pupil diameter to the eye."""
        eye.pupil.set_diameter(value)
