"""Light source model for eye tracking simulation.

Defines the Light class for generating corneal reflections (glints) in synthetic eye tracking setups.
"""

from dataclasses import dataclass
from ..types import Position3D


@dataclass
class Light:
    """Light source for generating corneal reflections.

    Represents a point light source positioned in 3D space for eye tracking.
    Used to create corneal reflections (glints).

    Args:
        position: 3D position in world coordinates (meters)
    """

    position: Position3D

    def __repr__(self) -> str:
        """String representation showing the light position."""
        return f"Light(position={self.position})"
