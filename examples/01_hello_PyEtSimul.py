"""Hello, PyEtSimul: build a scene and read the pupil centre and glint from a rendered eye.

Introduces the three objects every scene is made of (Eye, Camera, Light) and Camera.take_image, which
renders the eye and returns the pupil centre and the corneal reflections (glints) in image pixels.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Position3D

# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
eye = Eye()  # a bare Eye() uses the default "PyEtSimul" model
eye.position = Position3D(0, 250, 100)
# The rest orientation is how the eye sits before it looks at anything: here facing the camera side
# (toward -y), with +z as its up direction.
eye.set_rest_orientation_at_target(Position3D(0, 0, 100), up=Position3D(0, 0, 1))

camera = Camera()
# point_at aims the camera's optical axis at a location, here the eye, centring it in frame. The camera is
# aimed before the eye turns, because eye.position is the corneal apex, which moves when the eye turns.
camera.point_at(eye.position)

light = Light(position=Position3D(100, 0, 0))  # the glint is this light's reflection on the cornea

# look_at turns the eye so its gaze fixates a point in the world.
eye.look_at(Position3D(-50, 0, 50))

image = camera.take_image(eye, [light])

print(f"pupil centre: {image.pupil_center}")
print(f"glint:        {image.corneal_reflections[0]}")
