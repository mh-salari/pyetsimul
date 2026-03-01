Coordinate Systems
==================

PyEtSimul uses four coordinate systems to describe the spatial relationships between
components and to project 3D features onto the camera image plane. All coordinate frames
are **right-handed** by default. All spatial measurements use **millimeters (mm)**.

.. contents:: On this page
   :local:
   :depth: 2

|

Overview
--------

The transformation pipeline from physical 3D space to the final 2D image follows a chain
of coordinate systems:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Coordinate System
     - Origin
     - Optical / Primary Axis
   * - **World**
     - User-defined
     - User-defined
   * - **Eye** (local)
     - Eye rotation center
     - :math:`-Z` (toward cornea)
   * - **Camera** (local)
     - Camera optical center (pinhole)
     - :math:`-Z` (viewing direction)
   * - **Image** (2D)
     - Image center
     - :math:`+X` right, :math:`+Y` down

|

World Coordinate System
-----------------------

The world coordinate system is the fixed global reference frame. All component
positions---eyes, cameras, light sources---and their orientations are specified
relative to this frame.

.. note::

   The world frame itself has no fixed physical meaning; you define what :math:`X`,
   :math:`Y`, and :math:`Z` represent for your setup. For example, you might choose
   :math:`X` = horizontal, :math:`Y` = depth toward the screen, :math:`Z` = vertical.

Each component (``Eye``, ``Camera``, ``Light``) stores its pose as a
:math:`4 \times 4` **homogeneous transformation matrix** that maps from its local
coordinate system to world coordinates:

.. math::

   T = \begin{bmatrix}
   R_{3\times3} & \mathbf{t} \\
   \mathbf{0}^T & 1
   \end{bmatrix}

where :math:`R` is a :math:`3 \times 3` rotation matrix and :math:`\mathbf{t}` is
the translation vector (the component's position in world coordinates).

Eye Coordinate System
---------------------

Each eye has a local coordinate system with the following conventions:

.. list-table::
   :widths: 15 85
   :stub-columns: 1

   * - **Origin**
     - Center of rotation of the eyeball
   * - :math:`+X`
     - Right (temporal direction)
   * - :math:`+Y`
     - Up (superior direction)
   * - :math:`-Z`
     - **Optical axis** --- points anteriorly, toward the cornea

The rest orientation corresponds to the primary gaze position with zero torsion.
It is set via a :math:`3 \times 3` ``RotationMatrix`` that maps the local eye axes
to world axes:

.. code-block:: python

   from pyetsimul.types import RotationMatrix

   # Optical axis (-Z local) will point along -Y in world coordinates
   rest_orientation = RotationMatrix([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, -1, 0]])
   eye.set_rest_orientation(rest_orientation)

Optical Axis vs. Visual Axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 75
   :stub-columns: 1

   * - **Optical axis**
     - The geometric symmetry axis of the cornea, always along :math:`-Z` in eye-local
       coordinates.
   * - **Visual axis**
     - The line connecting the fovea to the fixation point. When foveal displacement is
       enabled, the visual axis is offset from the optical axis by horizontal
       (:math:`\alpha_{\text{fovea}}`) and vertical (:math:`\beta_{\text{fovea}}`) angles.
       When foveal displacement is disabled, the two axes coincide.

The ``eye.look_at(target)`` method aligns the **visual axis** to the target (or the
optical axis, if foveal displacement is disabled). Eye rotations follow **Listing's law**,
which constrains the torsional component so that the rotation axis is perpendicular to
both the initial and final gaze directions.

Camera Coordinate System
------------------------

Each camera has a local coordinate system with the following conventions:

.. list-table::
   :widths: 15 85
   :stub-columns: 1

   * - **Origin**
     - Optical center of the camera (the pinhole)
   * - :math:`+X`
     - Right in the camera's view
   * - :math:`+Y`
     - Up in the camera's view
   * - :math:`-Z`
     - **Viewing direction** --- points toward the scene

.. tip::

   Both the eye and the camera share the same local-axis convention: the optical axis
   points along :math:`-Z`. This symmetry simplifies transformations between the two.

The camera's orientation in the world is set either directly through its transformation
matrix or by using the ``point_at()`` convenience method:

.. code-block:: python

   camera = Camera()
   camera.point_at(eye.position)   # Orient camera to look at the eye

Image Coordinate System
-----------------------

After projection, 2D feature positions are expressed in the **image coordinate system**:

.. list-table::
   :widths: 15 85
   :stub-columns: 1

   * - **Origin**
     - Center of the image (at the principal point :math:`(c_x, c_y)`)
   * - :math:`+X`
     - Right
   * - :math:`+Y`
     - Down

.. important::

   Unlike the typical computer-vision convention where the origin is at the **top-left**
   corner, PyEtSimul places the origin at the **image center**. This means valid image
   coordinates satisfy:

   .. math::

      -\frac{w}{2} \le x \le \frac{w}{2}, \qquad -\frac{h}{2} \le y \le \frac{h}{2}

   where :math:`w` and :math:`h` are the image width and height in pixels.

Points that fall outside the image bounds or behind the camera are marked as invalid
(``NaN``) in the returned ``ProjectionResult``.

Transformation Pipeline
-----------------------

The full transformation from a 3D point in the world to a 2D point on the image plane
proceeds through the following stages:

.. code-block:: text

   World Coordinates
        │
        │  inverse of camera transformation matrix
        ▼
   Camera-Local Coordinates
        │
        │  pinhole projection + optional lens distortion
        ▼
   Image Coordinates (center-origin)

**Step 1 --- World to camera-local:**
A point :math:`\mathbf{p}_w` in world coordinates is transformed to camera-local
coordinates :math:`\mathbf{p}_c` by inverting the camera's transformation matrix:

.. math::

   \mathbf{p}_c = T_{\text{camera}}^{-1} \, \mathbf{p}_w

**Step 2 --- Camera-local to image:**
The camera-local point is projected onto the image plane using the intrinsic matrix
:math:`K` (see :doc:`camera_and_lights`). The depth along the optical axis is
:math:`d = -p_{c,z}` (negative because the camera looks along :math:`-Z`).
Points with :math:`d \le 0` are behind the camera and marked as invalid.

Homogeneous Coordinates
^^^^^^^^^^^^^^^^^^^^^^^

Internally, PyEtSimul uses 4D homogeneous coordinates to unify translations and
rotations into a single matrix multiplication:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Type
     - Homogeneous Representation
   * - ``Position3D`` (point)
     - :math:`[x, \; y, \; z, \; 1]^T`
   * - ``Direction3D`` (vector)
     - :math:`[x, \; y, \; z, \; 0]^T`

The fourth component distinguishes points from directions: points are affected by
translation, directions are not.
