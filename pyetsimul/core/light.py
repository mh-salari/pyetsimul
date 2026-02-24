"""Light source model for eye tracking simulation.

Defines the Light class for generating corneal reflections (glints) in synthetic eye tracking setups.
"""

from dataclasses import dataclass

from pyetsimul.log import info, table

from ..types import Position3D


@dataclass
class Light:
    """Light source for generating corneal reflections.

    Represents a point light source positioned in 3D space for eye tracking.
    Used to create corneal reflections (glints).

    Args:
        position: 3D position in world coordinates (meters)
        diameter: Physical diameter of the light source in meters (e.g., 0.005 for a 5mm LED).
            None means point source (default, backward compatible).

    """

    position: Position3D
    diameter: float | None = None

    def __repr__(self) -> str:
        """String representation showing the light position."""
        if self.diameter is not None:
            return f"Light(position={self.position}, diameter={self.diameter})"
        return f"Light(position={self.position})"

    def __str__(self) -> str:
        """Basic string representation of the light."""
        pos = self.position
        base = f"Light(pos=({pos.x * 1000:.1f}, {pos.y * 1000:.1f}, {pos.z * 1000:.1f})mm"
        if self.diameter is not None:
            base += f", d={self.diameter * 1000:.1f}mm"
        return base + ")"

    def pprint(self) -> None:
        """Print detailed light parameters in a formatted table."""
        pos = self.position

        if self.diameter is not None:
            light_type = f"Extended source (diameter: {self.diameter * 1000:.3f} mm)"
        else:
            light_type = "Point source"

        data = [
            ["Position (x,y,z) mm", f"({pos.x * 1000:.3f}, {pos.y * 1000:.3f}, {pos.z * 1000:.3f})"],
            ["Position (x,y,z) m", f"({pos.x:.6f}, {pos.y:.6f}, {pos.z:.6f})"],
            ["Light type", light_type],
        ]

        headers = ["Parameter", "Value"]
        info("Light Source Parameters:")
        table(data, headers=headers, tablefmt="grid")

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        data = {"position": self.position.serialize()}
        if self.diameter is not None:
            data["diameter"] = self.diameter
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "Light":
        """Deserialize from dictionary representation."""
        return cls(
            position=Position3D.deserialize(data["position"]),
            diameter=data.get("diameter"),
        )
