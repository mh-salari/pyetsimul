"""Light source model for eye tracking simulation.

Defines the Light class for generating corneal reflections (glints) in synthetic eye tracking setups.
"""

from dataclasses import dataclass

from tabulate import tabulate

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

    def __str__(self) -> str:
        """Basic string representation of the light."""
        pos = self.position
        return f"Light(pos=({pos.x * 1000:.1f}, {pos.y * 1000:.1f}, {pos.z * 1000:.1f})mm)"

    def pprint(self) -> None:
        """Print detailed light parameters in a formatted table."""
        pos = self.position

        data = [
            ["Position (x,y,z) mm", f"({pos.x * 1000:.3f}, {pos.y * 1000:.3f}, {pos.z * 1000:.3f})"],
            ["Position (x,y,z) m", f"({pos.x:.6f}, {pos.y:.6f}, {pos.z:.6f})"],
            ["Light type", "Point source"],
        ]

        headers = ["Parameter", "Value"]
        print("Light Source Parameters:")
        print(tabulate(data, headers=headers, tablefmt="grid"))

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {"position": self.position.serialize()}

    @classmethod
    def deserialize(cls, data: dict) -> "Light":
        """Deserialize from dictionary representation."""
        return cls(position=Position3D.deserialize(data["position"]))
