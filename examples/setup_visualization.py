"""Setup visualization example.

Demonstrates real-time eye tracking setup visualization.
Shows 3D view of eye, camera, lights, and target with camera view of corneal reflections.
"""

import numpy as np

from et_simul.core import Eye, Camera, Light
from et_simul.types import Position3D, RotationMatrix
from et_simul.visualization import plot_interactive_setup


def run_setup_demo():
    """Run setup visualization"""
    # Setup eye
    rest_orientation = RotationMatrix(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), validate_handedness=False)
    e_base = Eye()
    e_base.set_rest_orientation(rest_orientation)
    e_base.position = Position3D(0, 250e-3, 100e-3)

    # Create two light sources
    l1 = Light(position=Position3D(100e-3, 0, 0))
    l2 = Light(position=Position3D(-100e-3, 0, 0))
    lights = [l1, l2]

    # Setup camera pointing at eye
    c = Camera()
    c.point_at(e_base.position)

    # Start target point
    target_point = Position3D(-50e-3, 0, 50e-3)

    # Create and run setup visualization
    plot_interactive_setup(e_base, lights, c, target_point)


if __name__ == "__main__":
    run_setup_demo()
