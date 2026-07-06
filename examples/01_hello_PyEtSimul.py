"""Hello, PyEtSimul: build a scene and read the pupil centre and glint from a rendered eye.

Introduces the three objects every scene is made of (Eye, Camera, Light) and Camera.take_image, which
renders the eye and returns the pupil centre and the corneal reflections (glints) in image pixels.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Position3D

# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
eye = Eye()  # a bare Eye() uses the default "PyEtSimul" model
eye.position = Position3D(0, 250, 100)

# look_at turns the eye so its gaze fixates a point in the world.
eye.look_at(Position3D(-50, 0, 50))

camera = Camera()
# point_at aims the camera's optical axis at a location, here the eye, centring it in frame.
camera.point_at(eye.position)

light = Light(position=Position3D(100, 0, 0))  # the glint is this light's reflection on the cornea

image = camera.take_image(eye, [light])

print(f"pupil centre: {image.pupil_center}")
print(f"glint:        {image.corneal_reflections[0]}")
