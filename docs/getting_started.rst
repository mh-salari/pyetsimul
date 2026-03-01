Getting Started
===============

Installation
------------

Clone the repository and install dependencies using `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/your-repo/pyetsimul.git
   cd pyetsimul
   uv sync

Basic Usage
-----------

The following walkthrough is based on ``examples/basic_usage_demo.py``.

Setting Up the Components
~~~~~~~~~~~~~~~~~~~~~~~~~

Every simulation requires three core components: an **Eye**, a **Camera**, and one or more
**Light** sources. All spatial measurements use **millimeters (mm)**.

.. code-block:: python

   from pyetsimul.core import Eye, Camera, Light
   from pyetsimul.types import Position3D, RotationMatrix

   # Create an eye with eyelid modeling enabled
   eye = Eye(eyelid_enabled=True, pupil_boundary_points=50)
   eye.eyelid.openness = 0.65

   # Set the eye orientation and position
   # The rotation matrix maps the eye's local coordinate system to world coordinates.
   # In the eye's local frame, the optical axis points along -z.
   rest_orientation = RotationMatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
   eye.set_rest_orientation(rest_orientation)
   eye.position = Position3D(0, 150, 50)

   # Create a light source
   light = Light(position=Position3D(50, 0, 0))

   # Create a camera and point it at the eye
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

- ``basic_usage_demo.py`` — Minimal setup with 3D and camera view visualization
- ``setup_visualization.py`` — 3D setup visualization with multiple eyes, cameras, and lights
- ``eye_anatomy.py`` — Eye anatomy exploration (cornea, pupil, eyelid)
- ``realistic_pupil.py`` — Non-circular pupil shapes based on Wyatt (1995)
- ``cornea_comparison_figure.py`` — Spherical vs. conic cornea comparison
- ``camera_distortion.py`` — Effect of lens distortion on pupil detection
- ``glint_noise_demo.py`` — Corneal reflection detection noise
- ``simple_data_generation.py`` — Generating datasets for gaze estimation evaluation
- ``custom_gaze_model.py`` — Implementing a custom gaze estimation algorithm
- ``experiments/`` — Full evaluation workflows (algorithm comparison, data validation)

