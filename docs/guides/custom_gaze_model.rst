Custom Gaze Models
==================

PyEtSimul includes seven polynomial gaze models, and you can add your own. This guide shows how to
define a polynomial with ``PolynomialDescriptor``, register it with ``register_polynomial()``, and
calibrate a tracker that uses it.

Defining a Polynomial
---------------------

A ``PolynomialDescriptor`` gives the polynomial a name and lists its terms:

.. code-block:: python

   from pyetsimul.gaze_mapping.polynomial import PolynomialDescriptor

   MY_POLYNOMIAL = PolynomialDescriptor(
       name="my_cubic",
       description="third-order polynomial with cross-terms",
       terms=["x", "y", "x*y", "x*y", "x", "y", "x*y", "x", "y", "1"],
       orders=[3, 3, [2, 1], [1, 2], 2, 2, [1, 1], 1, 1, 0],
   )

``x`` and ``y`` are the components of the normalised pupil-glint vector. ``terms`` and ``orders``
must have the same length. Each pair of entries defines one term.

Terms and Orders
----------------

Each term is evaluated as :math:`x^i y^j`, where ``i`` and ``j`` are its entry in ``orders``. The
term string is only a label that makes the list readable. It does not set the exponents. The one
exception is ``"1"``, which always evaluates to the constant 1.

So ``"x*y"`` with order ``[2, 1]`` is :math:`x^2 y`, while the same ``"x*y"`` with order
``[1, 2]`` is :math:`x y^2`. Both appear in the example above. In the same way, ``"x"`` with
order ``3`` is :math:`x^3`.

You can write the order as a plain number for three terms only:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Term
     - Plain order
     - Meaning
   * - ``"x"``
     - ``n``
     - :math:`x^n`
   * - ``"y"``
     - ``n``
     - :math:`y^n`
   * - ``"1"``
     - ``0``
     - the constant term; any other value raises ``ValueError``

All other terms, including cross-terms, need both exponents written out as
``[x_order, y_order]``. Writing ``"x*y"`` with a plain order raises ``ValueError``, because the
descriptor cannot tell which exponent you meant.

Listing the same term twice adds a duplicate column to the design matrix. It does not add a
second independent coefficient.

Different Terms for X and Y
---------------------------

A flat list of terms gives both gaze coordinates the same features. To use different features for
each coordinate, pass a list of two lists, horizontal first, and use the same structure in
``orders``:

.. code-block:: python

   CERROLAZA_LIKE = PolynomialDescriptor(
       name="my_asymmetric",
       description="second-order horizontally, with a cross-term vertically",
       terms=[["x", "x", "y", "1"], ["x*y", "x", "x*y", "y", "1"]],
       orders=[[2, 1, 1, 0], [[2, 1], 2, [1, 1], 1, 0]],
   )

This defines

.. math::

   g_x = a_0 x^2 + a_1 x + a_2 y + a_3

   g_y = b_0 x^2 y + b_1 x^2 + b_2 xy + b_3 y + b_4

The two coordinates do not need the same number of terms.

Registering and Using It
------------------------

``register_polynomial()`` adds the descriptor to the global registry. Once registered, your
polynomial can be used anywhere a built-in name can, including ``PolynomialGazeModel.create``,
data generation, and the evaluation tools:

.. code-block:: python

   from pyetsimul.core import Camera, Eye, Light
   from pyetsimul.core.eye_model import get_eye_model
   from pyetsimul.evaluation import accuracy_at_calibration_points
   from pyetsimul.gaze_mapping.polynomial import PolynomialGazeModel
   from pyetsimul.gaze_mapping.polynomial.polynomials import register_polynomial
   from pyetsimul.types import Position3D

   register_polynomial(MY_POLYNOMIAL)  # "my_cubic" can now be selected by name

   eye = Eye(model=get_eye_model("PyEtSimul"))
   eye.position = Position3D(0.0, 700.0, 50.0)

   camera = Camera()
   camera.position = Position3D(0.0, 350.0, -150.0)
   camera.point_at(eye.position)

   light = Light(position=Position3D(70.0, 350.0, -140.0))

   calib_points = [Position3D(0.0, 0.0, 0.0)]  # use a full HV9 grid in practice

   tracker = PolynomialGazeModel.create(
       cameras=[camera],
       lights=[light],
       calib_points=calib_points,
       polynomial="my_cubic",
   )
   tracker.run_calibration(eye)

   results = accuracy_at_calibration_points(tracker, eye=eye)
   results.pprint("Custom-polynomial calibration accuracy")

Calibration solves for the coefficients by least squares, the same as for the built-in models.

Choosing the Number of Terms
----------------------------

Calibration fits one coefficient per term for each gaze coordinate. You therefore need at least as
many calibration targets as the largest coordinate has terms. The ten-term example above needs
more than the nine targets of an HV9 grid.

The fit uses a pseudo-inverse, so too few targets will not raise an error. Calibration returns
coefficients either way, and the problem shows up as poor accuracy between the calibration
targets.

Built-in Polynomials
--------------------

These names are registered on import and can be passed to ``PolynomialGazeModel.create``:

- ``second_order``
- ``zhu_ji_2005``
- ``cerrolaza_2008_symmetric``
- ``cerrolaza_2008_asymmetric``
- ``hoorman_2008``
- ``hennessey_2008``
- ``blignaut_wium_2013``

:doc:`../theory/gaze_estimation_models` gives their formulations.

Complete Example
----------------

``examples/15_custom_gaze_model.py`` registers a third-order polynomial, calibrates a tracker with
it on an HV9 grid, and reports the accuracy with an interactive per-target plot.
