Moving to Version 6
===================

Version 6.0.0 corrects errors in the simulation and removes parts that had no effect. Some calls have to be
written differently, and some results change. This page lists both, by the version a project is on now.

A project on version 5 needs the first section. A project on version 4 needs the first two sections, and a
project on version 3 needs all three. The changelog gives the details of every change.

From Version 5
--------------

Calls that have to change
~~~~~~~~~~~~~~~~~~~~~~~~~

**Camera matrix.** A ``CameraMatrix`` is created with the image resolution.

.. code-block:: python

   # before
   camera.camera_matrix = CameraMatrix(matrix)
   # now
   camera.camera_matrix = CameraMatrix(matrix, Point2D(640, 480))

**Calibration points in plots.** ``plot_setup`` and ``plot_setup_and_camera_view`` take the calibration points
as 3D positions (``Position3D``), not as 2D points.

**Glint size.** ``compute_glint_diameter_sphere`` and ``compute_glint_diameter_conic`` are replaced by one
function, which takes the camera position as its third argument.

.. code-block:: python

   compute_glint_diameter(light_position, glint_position, camera_position, cornea, eye.trans, led_diameter)

**Names.**

- The polynomial ``"hoorman_2008"`` is ``"hoormann_2008"``.
- The ``algorithm_name`` of a homography prediction is ``"homography_normalization"``, not ``"homography"``.
- Generated datasets are saved to ``outputs/`` by default, not ``output/``. Pass ``output_dir="output"`` to
  keep the old folder.

**Comparing algorithms.** ``cosine_similarity_average`` and ``amplitude_agreement`` take the eye positions as
a third argument.

**Eye model registry.** ``get_eye_model`` returns a copy, so a change to the returned model no longer changes
the registered one. ``register_eye_model`` raises ``ValueError`` for a name that is registered already; pass
``overwrite=True`` to replace it.

**Changing a parameter of an eye that is already in a scene.** The eye model cannot be modified. Give the eye
a model with the new value:

.. code-block:: python

   eye.set_model(eye.model.with_parameter("cornea.anterior_k", -0.3))

**Removed.**

- ``ComposedVariation.description``: use ``describe()``.
- ``PolynomialFeatures.predict``: use ``PolynomialGazeModel.predict_gaze``.
- ``RealisticPupilParams.base_radius`` and ``PupilDefaults.BASE_RADIUS``, ``ELLIPSE_CONTRIBUTION`` and
  ``MAJOR_AXIS_ANGLE``: they had no effect. The size of a pupil is read with ``get_diameter()``.
- ``EyeTracker.state``: the calibration is in ``algorithm_state``.
- The ``cr_3d_lists`` argument of ``plot_camera_view_of_eye``.
- ``order_points_by_angle`` in ``homography_normalization_utils``.

Behaviour that code may rely on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**An eye keeps looking at its target.** After ``eye.look_at(target)``, assigning ``eye.position`` or setting a
rest orientation turns the eye to the same target again. In version 5 the eye kept the rotation it had at its
old position. ``Eye(hold_fixation=False)`` returns the eye to its rest orientation instead.

**A failed gaze prediction is** ``None``. In version 5 it was the point (0, 0, 0) with confidence 0. Code
that reads a prediction has to check for ``None``.

**Calibration leaves the eye as it was.** In version 5 the eye stayed at the last calibration point after
``run_calibration``.

**Gaze error components.** ``EyeTracker.calculate_gaze_error`` returns the error along the two axes of the
calibration plane. In version 5 it returned the error along world x and y.

**Errors instead of silent results.** These cases raise an error now:

- predicting gaze with a tracker that is not calibrated (``ValueError``);
- ``look_at`` with a target exactly behind the eye, or closer than about 20 mm (``ValueError``);
- ``set_rest_orientation_at_target`` with a target at the eye position (``ValueError``);
- ``use_tear_film=True`` on a cornea that cannot model a tear film (``ValueError``);
- a ``GenericEyeVariation`` on a parameter name that does not exist (``AttributeError``);
- two variations with the same ``param_name`` in one ``ComposedVariation`` (``ValueError``);
- evaluating a dataset without measurements, or loading a file that is not a dataset (``ValueError``);
- calibration data that ``Stampe1993GazeModel`` cannot use (``ValueError``);
- a pupil set by its apparent size whose solver does not finish (``RuntimeError``).

Results that change
~~~~~~~~~~~~~~~~~~~

- **Every eye with foveal displacement (angle kappa), which includes the default eye.** ``look_at`` now aims
  the visual axis through the target. Before, the axis passed 0.13 deg beside it. Pupil and glint positions in
  the image move by a small amount for every gaze direction.
- **An eye with an eyelid.** The eyelid stays fixed to the face when the eye turns.
- **An eye with an** ``EyeballCenter``. It rotates about the centre of the eyeball, not about the corneal apex.
- **A non-zero** ``torsion_deg``.
- **Realistic pupils.** The shape stays the same when the size changes. Outlines of pupils with a seed differ
  from those of version 5.
- **The default pupil decentration.** A dilating pupil moves temporally. In version 5 it moved nasally.
- **The glint size of a light source with a diameter.**
- **A camera with a calibrated camera matrix.** Image coordinates are measured from the centre of the image.
- **Datasets.** Sweeps of angle kappa and of other eye-model parameters now change the eye. Sweeps of a cornea
  parameter no longer move the corneal apex. A composed variation generates all its combinations. Glint noise
  with a seed differs from glint to glint.
- **The homography gaze model**, for light arrangements in which glints were matched to the wrong lights.

Saved files
~~~~~~~~~~~

Trackers, cameras, eyes and datasets saved with version 5 cannot be loaded.

- A saved tracker loads after ``"pupil_center_method": "ellipse"`` is added to its dictionary.
- A saved camera loads after the keys ``error_seed``, ``fick_projection``, ``aim_start_orientation`` and
  ``aimed_orientation`` are added with the value ``null``.
- A saved eye cannot be converted. Datasets have to be generated again.

From Version 4
--------------

Version 5 changed the default eye. Everything in the section above applies as well.

**The default eye.** The default cornea is a conic with a back surface (it was a sphere with one surface), and
the pupil sits off the optical axis. Every result that relies on the defaults changes. The eye of version 4
is built like this:

.. code-block:: python

   from pyetsimul.core import EyeModel, SphericalCornea
   from pyetsimul.core.off_axis_pupil import OffAxisPupilConfig

   model = EyeModel(cornea=SphericalCornea(), off_axis_pupil=OffAxisPupilConfig(enabled=False))

**The default rotation centres.** They lie 14.8 mm (horizontal) and 12.2 mm (vertical) behind the corneal
apex. In version 4 they lay at 14.7 mm and 12.0 mm. Eye positions after a rotation shift by up to about
0.1 mm. The depths of version 4 are set like this:

.. code-block:: python

   from dataclasses import replace

   from pyetsimul.core import get_eye_model

   model = get_eye_model("PyEtSimul")
   centre = replace(model.rotation_center, horizontal_depth_mm=14.7, vertical_depth_mm=12.0)
   model = model.copy(rotation_center=centre)

**Variations.** ``GenericEyeVariation`` no longer takes a ``description`` argument.

**Warnings.** A pupil centre or diameter that cannot be fitted prints a warning that names what was reported
instead.

From Version 3
--------------

Version 4 changed how an eye is described and where it sits. Everything in the two sections above applies as
well.

**The eye of version 3.** It is available as the eye model ``"et_simul"``, the eye of Böhme et al. (2008),
which is validated against the MATLAB original. A project that has to keep simulating that eye uses it:

.. code-block:: python

   from pyetsimul.core import Eye, get_eye_model

   eye = Eye(model=get_eye_model("et_simul"))

**Describing an eye.** The parameters of an eye are given in an ``EyeModel``, not as keyword arguments of
``Eye``.

.. code-block:: python

   # before
   eye = Eye(fovea_alpha_deg=5.0, pupil_type="realistic")
   # now
   eye = Eye(model=EyeModel(fovea_alpha_deg=5.0, pupil_type="realistic"), which_eye="right")

**The side of an eye.** It is set with ``Eye(which_eye="left")`` or ``"right"``. ``PupilDecentrationConfig``
no longer takes ``which_eye``.

**The look-at method.** It is set on the model with ``EyeModel(look_at_method=...)``, or for one call with
``look_at(target, method=...)``. This replaces ``Eye.aiming`` and the ``legacy`` argument of ``look_at``.
``look_at_target(eye, target, method)`` replaces ``look_at_target_optical_then_kappa`` and
``look_at_target_line_of_sight``.

**Where the eye sits.** ``Eye.position`` is the position of the corneal apex. In version 3 it was the position
of the eyeball centre, so an eye given the same position now sits further back, by the distance from its
corneal apex to its eyeball centre (12.33 mm for the default eye of version 3.6).

**How the eye turns.** The default eye turns about two rotation centres (Fick). It moves slightly as it turns,
so ``Eye.position`` changes after a ``look_at``. ``Eye.placement`` is the position the eye was given. An eye
with one fixed rotation centre is built with ``EyeModel(rotation_center=EyeballCenter())``.

**The cornea.** Changing the corneal radius no longer moves the corneal apex or rescales the eye. A cornea
that behaves as in version 3 is built with
``SphericalCornea(placement_convention="center", scale_with_radius=True)``.

**Refraction functions.** ``find_refraction_sphere``, ``find_refraction_conic``,
``find_refraction_dual_sphere``, ``find_refraction_dual_conic`` and ``refract_ray_dual_surface`` are removed.
Refraction is solved for the whole pupil boundary at once in ``pyetsimul.optics.refraction_batch``.

**Gaze models.** The EyeLink 1000 Plus gaze model is removed; ``Stampe1993GazeModel`` takes its place.

**Examples.** The example scripts of version 3 are replaced by the numbered examples ``01`` to ``16`` in
``examples/``.
