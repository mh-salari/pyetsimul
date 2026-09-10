Eye Rotation
============

Pointing the eye at a target involves two independent choices: which axis is aimed at the
target, and which point the eye pivots about. The first is the gaze construction, the second is
the rotation center. Both are properties of the eye model and can be changed without touching
the optics. In every case the eye moves as a rigid body, so only its pose changes.

.. contents:: On this page
   :local:
   :depth: 2

|

Listing's Law
-------------

The orientation is the shortest rotation that carries the aimed axis from its rest direction to
the target direction. For a rest direction :math:`\mathbf{u}_r` and a target direction
:math:`\mathbf{u}_n`, both unit vectors, the rotation axis is their cross product:

.. math::

   \mathbf{n} = \frac{\mathbf{u}_n \times \mathbf{u}_r}
                     {\lVert \mathbf{u}_n \times \mathbf{u}_r \rVert}

The rotation matrix is assembled from the orthonormal frame that :math:`\mathbf{n}` spans with
each of the two directions:

.. math::

   A = \bigl[\, \mathbf{n} \;\; \mathbf{u}_n \;\; \mathbf{u}_n \times \mathbf{n} \,\bigr]
       \bigl[\, \mathbf{n} \;\; \mathbf{u}_r \;\; \mathbf{u}_r \times \mathbf{n} \,\bigr]^{T}

The right factor takes :math:`\mathbf{u}_r` to :math:`[0, 1, 0]^{T}` and the left factor takes
that to :math:`\mathbf{u}_n`, while :math:`\mathbf{n}` maps to itself. :math:`A` is therefore a
rotation about :math:`\mathbf{n}` through the angle between the two directions.

The axis :math:`\mathbf{n}` is perpendicular to the rest direction, so it lies in Listing's
plane and the eye picks up no torsion about its own line of sight. This is Listing's law [1].

When the two directions are parallel to floating-point precision, the cross product drops below
:math:`10^{-9}` and the rotation is the identity. Normalizing a near-zero axis there would
produce a matrix that is not orthonormal.

Deliberate Torsion
^^^^^^^^^^^^^^^^^^

Real eyes deviate from Listing's law, so ``torsion_deg`` adds a roll about the eye-local optical
axis on top of the Listing rotation. It applies to the ``visual_axis`` and
``optical_axis_target_direction`` constructions and defaults to zero.

Aiming the Eye
--------------

``look_at_method`` selects which rest-frame axis is aligned with the target. The Listing
rotation is the same in all four cases. They differ in the axis they aim, where the aim
direction is measured from, and the post-rotation they apply.

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Method
     - Axis aimed at the target
   * - ``visual_axis``
     - The visual axis, built from the foveal angles :math:`\alpha` (horizontal) and
       :math:`\beta` (vertical). This is the default.
   * - ``line_of_sight``
     - The axis from the fovea through the current pupil center. That center carries the
       off-axis offset and the size-dependent decentration, so this axis re-aims as the pupil
       decenters.
   * - ``optical_then_kappa``
     - The optical axis, followed by a foveal (kappa) post-rotation, so neither the optical nor
       the visual axis ends up through the target. This is the original et_simul
       construction [2].
   * - ``optical_axis_target_direction``
     - The optical axis, along the apex-to-target direction measured from the fixed rest apex.
       The pose is the azimuth and elevation of that direction, independent of target distance,
       and the axis does not pass through a finite target. This is the eyePose convention of
       gkaModelEye [3].

The horizontal foveal angle is signed by eye side, because the fovea is temporal in both eyes.
The model carries the magnitude and the eye applies the sign.

``look_at`` takes a method argument that overrides the model default for a single call.

Rotation Center
---------------

The human eye has no single center of rotation. It lies about 15 mm behind the cornea in
horizontal gaze and about 12.5 mm in vertical gaze [4]. PyEtSimul provides two models, and both
change only where the rigid eye pivots.

Fixed Center
^^^^^^^^^^^^

``EyeballCenter`` pivots the eye about its own local origin for every gaze direction, and the
eye never translates. The origin is the eyeball center under the ``"center"`` placement
convention, which is where the original et_simul puts it: 12.33 mm behind the corneal apex for
the default 7.98 mm corneal radius. Under the ``"apex"`` convention this model would pivot the
eye about its corneal apex, so an apex-origin eye needs a gaze-dependent center instead.

Gaze-Dependent Center
^^^^^^^^^^^^^^^^^^^^^

``RotationCenter`` has a separate horizontal (azimuth) and vertical (elevation) center. Each
is a corneal-apex-to-pivot depth plus a lateral displacement off the optical axis. The defaults
are the values Fry and Hill measured for azimuth [5] and for elevation [6]:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Center
     - Depth behind the apex
     - Lateral displacement
   * - Azimuth (horizontal gaze)
     - 14.7 mm
     - 0.79 mm nasal
   * - Elevation (vertical gaze)
     - 12.0 mm
     - 0.33 mm superior

The two centers are blended by the horizontal share of the gaze eccentricity. With the target
expressed in the eye's rest frame (:math:`x` right, :math:`y` up, :math:`-z` forward),

.. math::

   f = \frac{x^2}{x^2 + y^2}

so :math:`f = 1` for a purely horizontal target and :math:`f = 0` for a purely vertical one. The
depth and the lateral offset follow:

.. math::

   d(f) = d_v + (d_h - d_v)\, f, \qquad
   (l_x,\, l_y) = \bigl(n f,\; s\,(1 - f)\bigr)

where :math:`d_h` and :math:`d_v` are the horizontal and vertical depths, :math:`n` is the nasal
displacement of the azimuth center, signed by eye side, and :math:`s` is the superior
displacement of the elevation center. Purely horizontal gaze then pivots about the azimuth
center and purely vertical gaze about the elevation center.

The vertical depth can differ between up-gaze and down-gaze through ``vertical_up_depth_mm``
and ``vertical_down_depth_mm``. Left unset, both directions use ``vertical_depth_mm``.

Setting both depths to the model's geometric apex-to-center distance, with zero lateral
magnitudes, reproduces the fixed-center behavior exactly.

Re-pivoting the Eye
-------------------

Once the pivot is not the eye-local origin, the eye has to translate as it turns. Requiring the
pivot point to stay put in world coordinates gives the new position directly:

.. math::

   \mathbf{p} = \mathbf{p}_0 + (R_r - R)\, \mathbf{o}

where :math:`\mathbf{p}_0` is the commanded placement, :math:`R_r` is the rest orientation,
:math:`R` is the new orientation, and :math:`\mathbf{o}` is the pivot in eye-local coordinates:

.. math::

   \mathbf{o} = \bigl(l_x,\; l_y,\; d - a\bigr)

Here :math:`a` is the distance from the eye-local origin to the corneal apex, which makes the
same expression correct under either placement convention. It is zero when the origin is the
apex, and roughly half the axial length when the origin is the eyeball center.

Orientation and position depend on each other, since the aim direction is measured from the
translated origin while the translation needs the orientation that aiming produces. The two are
solved together as a fixed point, to a tolerance of :math:`10^{-9}` mm and at most eight
iterations. The shift is under a millimeter, so it settles in a couple of steps.

Placement versus Position
^^^^^^^^^^^^^^^^^^^^^^^^^

``placement`` is the position the eye was commanded to and the point it re-pivots about.
``position`` is the eye-local origin in world coordinates after the re-pivot, so it moves as the
eye turns to off-axis targets. ``rotation_centre`` reports the azimuth center in world
coordinates.

Fick Centers
------------

With ``fick=True`` the azimuth and elevation rotations are applied about their own centers in
sequence instead of being blended into a single on-axis pivot [3]. The angles are read off the
aimed optical axis :math:`\mathbf{f}`, expressed in the rest frame:

.. math::

   \varepsilon = \arcsin(f_y), \qquad
   \alpha = \operatorname{atan2}(-f_x,\, -f_z)

Elevation is applied about the elevation center :math:`\mathbf{c}_\varepsilon` first and azimuth
about the azimuth center :math:`\mathbf{c}_\alpha` second, which gives the eye-local
translation:

.. math::

   \mathbf{t} = R_\alpha (I - R_\varepsilon)\, \mathbf{c}_\varepsilon
              + (I - R_\alpha)\, \mathbf{c}_\alpha

The orientation is still the Listing rotation described above, and only the translation
changes. With both centers equal, this reduces to the single-pivot expression exactly.

What the Rotation Leaves Alone
------------------------------

The cornea, the pupil, and every optical surface keep their eye-local geometry through any
rotation. The eyelid keeps its own transform, anchored to the rest placement and the rest
orientation, because it is fixed to the face rather than to the globe.

The translation still changes what the camera records. Because the camera views the pupil
obliquely, moving the globe shifts the imaged pupil ellipse and the corneal reflection.

Defaults by Eye Model
---------------------

.. list-table::
   :widths: 25 37 38
   :header-rows: 1

   * - Eye model
     - Gaze construction
     - Rotation center
   * - PyEtSimul default
     - ``visual_axis``
     - Fick centers at 14.7 mm and 12.0 mm
   * - ``et_simul``
     - ``optical_then_kappa``
     - Fixed center at the eyeball center
   * - ``gkaModelEye``
     - ``optical_axis_target_direction``
     - Fick centers at 14.7 mm and 12.0 mm

----

| [1] Haustein, W. (1989). Considerations on Listing's law and the primary position by means of a matrix description of eye position control. *Biological Cybernetics*, 60(6), 411-420.
| [2] Bohme, M., Dorr, M., Graw, M., Martinetz, T., & Barth, E. (2008). A software framework for simulating eye trackers. In *Proceedings of ETRA '08*, pp. 251-258. ACM.
| [3] Aguirre, G. K. (2019). A model of the entrance pupil of the human eye. *Scientific Reports*, 9, 9360.
| [4] Atchison, D. A., & Smith, G. (2023). *Optics of the Human Eye* (2nd ed.), Section 1.7. CRC Press.
| [5] Fry, G. A., & Hill, W. W. (1962). The center of rotation of the eye. *Optometry and Vision Science*, 39(11), 581-595.
| [6] Fry, G. A., & Hill, W. W. (1963). The mechanics of elevating the eye. *Optometry and Vision Science*, 40(12), 707-716.
