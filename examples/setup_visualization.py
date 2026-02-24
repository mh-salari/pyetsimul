"""Setup visualization example.

Demonstrates real-time eye tracking setup visualization.
Shows 3D view of eye, camera, lights, and target with camera view of corneal reflections.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Point3D, Position3D, RotationMatrix
from pyetsimul.visualization import plot_interactive_setup


def run_setup_demo() -> None:
    """Run setup visualization"""
    # Setup eye
    e_base = Eye(eyelid_enabled=True)
    e_base.eyelid.openness = 0.4

    rest_orientation = RotationMatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    e_base.set_rest_orientation(rest_orientation)

    e_base.position = Position3D(0, 250, 100)

    # Create two light sources
    l1 = Light(position=Position3D(100, 0, 0))
    l2 = Light(position=Position3D(-100, 0, 0))
    lights = [l1, l2]

    # Setup camera pointing at eye
    c = Camera()
    c.point_at(e_base.position)

    # Start target point
    target_point = Point3D(-50, 0, 50)

    # Create and run setup visualization
    plot_interactive_setup(e_base, lights, c, target_point)


if __name__ == "__main__":
    run_setup_demo()
