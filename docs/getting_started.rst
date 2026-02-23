Getting Started
===============

Installation
------------

Install PyEtSimul using pip:

.. code-block:: bash

   pip install pyetsimul

Or for development:

.. code-block:: bash

   git clone https://github.com/your-repo/pyetsimul.git
   cd pyetsimul
   uv sync

Quick Example
-------------

.. code-block:: python

   from pyetsimul.core import Eye, Camera, Light, create_cornea, create_pupil
   from pyetsimul.types import Position3D, Direction3D

   # Create an eye at 60cm from the screen
   eye = Eye(
       position=Position3D(0.0, 0.0, 0.6),
       optical_axis_direction=Direction3D(0.0, 0.0, -1.0),
   )

   # Create a camera
   camera = Camera(position=Position3D(0.0, -0.05, 0.0))

   # Create a light source
   light = Light(position=Position3D(0.05, -0.05, 0.0))

Key Concepts
------------

- **Structured Types**: PyEtSimul uses ``Position3D``, ``Direction3D``, ``Point3D`` instead of raw numpy arrays for type safety.
- **All spatial measurements use meters** as the base unit.
- **Eye coordinate system**: Origin at center of rotation, optical axis along negative z-axis.
- **Camera coordinate system**: Pinhole model with optical axis along negative z-axis.
