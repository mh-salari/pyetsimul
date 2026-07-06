"""Visualize the setup interactively: the 3D scene and the camera's view, with live controls.

plot_interactive_setup opens two linked panels (a 3D layout of the eye, camera, lights and gaze target, and the
camera's rendered image of the eye with its pupil and glints) and lets you move the target and the eye with the
keyboard, redrawing both panels live. Watching the camera view change as the eye moves is the quickest way to
build intuition for how the scene maps to what the camera sees.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_interactive_setup

base = get_eye_model("PyEtSimul")
# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
eye = Eye(model=base)
eye.position = Position3D(0, 250, 100)  # place the eye in the scene
target = Position3D(-50, 0, 50)  # the point the eye gazes at (move it with the arrow keys)

camera = Camera()
camera.point_at(eye.position)  # aim the camera at the eye

# Two lights give two glints, so the camera view shows two reflections rather than one.
lights = [Light(position=Position3D(100, 0, 0)), Light(position=Position3D(-100, 0, 0))]

# Opens the interactive view and prints its own keyboard controls (arrow keys move the target, I/K/J/L/.,/ move
# the eye). It applies look_at as the target moves, so the eye does not need to be pointed here.
plot_interactive_setup(eye, lights, camera, target)
