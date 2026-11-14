Getting Started
===============

Installation
------------

Install the released package from PyPI:

.. code-block:: bash

   pip install pyetsimul

To work on the source, clone the repository and install its dependencies with
`uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/mh-salari/pyetsimul.git
   cd pyetsimul
   uv sync

Basic Usage
-----------

The following walkthrough builds a scene step by step. ``examples/01_hello_PyEtSimul.py`` is the
shortest complete script.

Setting Up the Components
~~~~~~~~~~~~~~~~~~~~~~~~~

Every simulation requires three core components: an **Eye**, a **Camera**, and one or more
**Light** sources. All spatial measurements use **millimeters (mm)**.

.. code-block:: python

   from pyetsimul.core import Eye, Camera, Light
   from pyetsimul.core.eye_model import EyeModel
   from pyetsimul.types import Position3D, RotationMatrix

   # Create an eye with eyelid modeling enabled. The eye model holds the anatomy; the eye places it in the scene.
   eye = Eye(model=EyeModel(eyelid_enabled=True, pupil_boundary_points=50))
   eye.eyelid.openness = 0.65

   # Set the eye orientation and position
   # The rotation matrix maps the eye's local coordinate system to world coordinates.
   # In the eye's local frame, the optical axis points along -z.
   rest_orientation = RotationMatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
   eye.set_rest_orientation(rest_orientation)
   eye.position = Position3D(0, 150, 50)

   # Create a light source
   light = Light(position=Position3D(50, 0, 0))

   # Create a camera and point it at the eye. Do this before the eye turns: eye.position is the
   # corneal apex, which moves when the eye turns (eye.placement is where the eye was placed).
   camera = Camera()
   camera.point_at(eye.position)

Directing Gaze and Taking an Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the setup is in place, direct the eye toward a target and capture an image:

.. code-block:: python

   # Define a gaze target and direct the eye toward it
   target_point = Position3D(0, 0, 50)
   eye.look_at(target_point)

   # Capture camera image of the eye
   image = camera.take_image(eye, [light])

   # Access the projected features
   pupil_center = image.pupil_center
   corneal_reflections = image.corneal_reflections

The ``CameraImage`` returned by ``take_image()`` contains the projected pupil center,
pupil boundary contour, and corneal reflection positions for each light source.

Key Concepts
------------

- **Structured types**: PyEtSimul uses ``Position3D``, ``Direction3D``, ``Point3D`` instead of raw numpy arrays for type safety and clarity.
- **All spatial measurements use millimeters (mm)** as the base unit.
- **Coordinate systems**: Eye and camera optical axes point along :math:`-Z` in their local frames. See :doc:`theory/coordinate_systems` for full details.
- **Corneal models**: Both spherical and conic cornea models are supported (see :doc:`theory/eye_model`).
- **Gaze estimation**: Polynomial and homography normalization methods are available (see :doc:`theory/gaze_estimation_models`).

Examples
--------

The ``examples/`` directory contains ready-to-run scripts covering common use cases:

- ``01_hello_PyEtSimul.py`` — Build a scene and read the pupil center and glint from a rendered eye
- ``02_visualize_setup.py`` — The 3D scene and the camera's view, with live keyboard controls
- ``03_eye_models.py`` — The three named eye models by their parameters, renders and shape
- ``04_cornea_models.py`` — Spherical, conic and toric corneas on one eye
- ``05_pupil_models.py`` — Elliptical and realistic pupils across sizes, with pupil-size decentration
- ``06_eyelid.py`` — How the eyelid occludes the pupil as it closes
- ``07_glint_noise.py`` — Corneal reflection detection noise
- ``08_camera_distortion.py`` — A pinhole camera next to a real eye-tracker lens
- ``09_two_eyes_one_camera.py`` — A binocular pair imaged by one camera
- ``10_two_eyes_two_cameras.py`` — A binocular pair imaged by two cameras
- ``11_generate_dataset.py`` — Render an eye across many gaze targets and save the dataset
- ``12_calibrate.py`` — Calibrate a gaze tracker on an HV9 grid, test it and explore it interactively
- ``13_validate.py`` — Gaze accuracy across the screen and as the head moves
- ``14_custom_eye_model.py`` — Build an eye model of your own
- ``15_custom_gaze_model.py`` — Register a polynomial of your own and calibrate with it
- ``16_custom_variation.py`` — Sweep any eye parameter

