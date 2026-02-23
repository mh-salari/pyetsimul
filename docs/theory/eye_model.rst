Eye Model
=========

PyEtSimul adopts Le Grand's two-sphere eye model [1], which represents the eyeball and the
anterior cornea as spheres with fixed radii, assumes a uniform corneal refractive index, and
includes an angular offset between the visual and optical axes. PyEtSimul simplifies the
optical geometry by using a single nodal point located at the center of corneal curvature
and extends the model by representing the cornea as conic. Default anatomical dimensions
(corneal radii, axial length, etc.) follow Boff (1986) [2].

.. image:: /_static/images/eye_model_schematic.png
   :alt: Schematic of the eye model showing cornea surfaces, pupil, fovea, and optical geometry
   :align: center

|

Eye Rotation
------------

The eye's rotational center is defined at the geometric center of the eyeball sphere. All
anatomical structures are specified relative to this reference point.

Eye rotations follow **Listing's law** [3], which governs the torsional component of eye
movements. When the eye rotates from an initial to a final gaze direction, the rotation
occurs around an axis perpendicular to both the initial and final visual axis directions.

Two rotation approaches are available:

1. Align the optical axis using Listing's law, then apply foveal displacement angles
   (:math:`\alpha_{\text{fovea}}`, :math:`\beta_{\text{fovea}}`) as post-rotations
   (default in et_simul [4]).
2. Compute the visual axis direction from foveal displacement and apply Listing's law
   directly to this target direction.

Cornea
------

PyEtSimul provides both **spherical** and **conic** corneal models. In both cases, the cornea
consists of two surfaces: an anterior (outer) surface and a posterior (inner) surface. The
corneal center refers to the center of curvature of the anterior surface, with the posterior
surface positioned relative to it based on corneal thickness.

Cornea Position
^^^^^^^^^^^^^^^

The corneal center is positioned along the optical axis, anterior to the eye's rotation center.
Its location is determined by the eyeball's axial length and the distance from the eyeball's
front surface to the corneal center.

Spherical Cornea
^^^^^^^^^^^^^^^^

The spherical model represents the anterior and posterior surfaces as concentric spheres
with radii :math:`R_{\text{anterior}}` and :math:`R_{\text{posterior}}`.

Conic Cornea
^^^^^^^^^^^^

The conic model provides a more anatomically accurate representation of the human cornea [5].
The anterior and posterior corneal surfaces follow the conic formulation:

.. math::

   r^2 = 2Rz - (1 + k)z^2

where :math:`R` is the radius of curvature at the apex and :math:`k` is the conic constant.
Both surfaces have their own :math:`R` and :math:`k` parameters.

Parameter Scaling
^^^^^^^^^^^^^^^^^

The default anterior corneal radius is 7.98 mm. For the spherical cornea, changing the
anterior radius from this default value proportionally scales related parameters---including
the posterior radius, thickness offset, corneal depth, cornea position, and pupil
radius---to maintain anatomical ratios. This proportional scaling does not apply to the
conic cornea model.

Pupil
-----

The pupil is positioned on the optical axis behind the corneal apex. Beyond circular pupils,
PyEtSimul supports realistic non-circular shapes based on the formulation of Wyatt (1995) [6],
where the pupil boundary points are generated using a Fourier-series representation:

.. math::

   R(\theta) = r_{\text{base}} + r_2 \cos\bigl(2(\theta - \phi_{\text{major}})\bigr)
   + \sum_{n=3}^{N} r_n \cos\bigl(n(\theta - \phi_n)\bigr)

where:

- :math:`r_{\text{base}}` is the base pupil radius
- :math:`r_2` is the elliptical component amplitude
- :math:`\phi_{\text{major}}` is the major-axis angle
- :math:`r_n` are harmonic amplitudes
- :math:`\phi_n` are harmonic phases

In the realistic pupil model, the pupil orientation is determined automatically based on its
diameter. Wyatt (1995) observed that larger pupils in darkness (average diameter 4.93 mm) tend
to have their ellipse major axes oriented vertically (0 deg), while smaller pupils in bright
light (average diameter 3.09 mm) tend to have their major axes oriented horizontally (+/-90 deg).
Based on this observation, 4.0 mm is set as the default threshold to classify pupils as either
dilated (vertical orientation) or constricted (horizontal orientation).

Pupil Decentration
^^^^^^^^^^^^^^^^^^

The pupil center position can shift relative to the limbus center as pupil size changes [7].
Pupil decentration is modeled using the linear decentration model of Wildenmann et al. (2013),
who measured centration changes across illumination-induced pupil size variations.

The decentration offset is calculated as:

.. math::

   \Delta \mathbf{p} = \mathbf{c} \cdot (d - d_0)

where :math:`d` is the current pupil diameter, :math:`d_0` is the baseline diameter, and
:math:`\mathbf{c} = [c_x, c_y]^T` is the coefficient vector. To simulate natural variation
across a population, the coefficient vector :math:`\mathbf{c}` for each simulated eye can be
sampled from a normal distribution.

In addition to this built-in model, PyEtSimul features an extensible registry system that
allows researchers to define and integrate their own custom decentration models.

Eyelid Occlusion
----------------

Eyelid occlusion is modeled using a geometric approach that simulates static eyelid
configurations at different openness levels. Dynamic blinks are not simulated. Following
observations by Nystrom et al. (2024) [8] that spontaneous blinks are dominated by upper
eyelid motion while the lower eyelid remains approximately stationary, the lower edge position
is fixed and only the upper eyelid position varies to control opening height.

The eyelid is modeled as a spherical surface centered at the eye's rotation center with radius
equal to half the eye axial length. The visible aperture is approximated by an ellipse in the
eye's frontal plane, with fixed horizontal width and variable vertical height. The openness
parameter :math:`o \in [0,1]` controls the ellipse height, where :math:`o=0` represents a
closed eyelid and :math:`o=1` represents full opening.

A point :math:`(x,y)` in the eye coordinate system is visible (not occluded) if it falls
inside the elliptical opening:

.. math::

   \left(\frac{x}{w/2}\right)^2 + \left(\frac{y - y_{\text{center}}}{h/2}\right)^2 \le 1

where :math:`w` is the fixed ellipse width, :math:`h = o \cdot h_{\max}` is the
openness-dependent height, and :math:`y_{\text{center}}` is the vertical offset that keeps
the lower edge fixed.

----

| [1] Le Grand, Y. (1957). *Light, Colour and Vision*. Chapman & Hall.
| [2] Boff, K. R., & Lincoln, J. E. (1986). *Engineering Data Compendium: Human Perception and Performance*.
| [3] Haustein, W. (1989). Considerations on Listing's law and the primary position by means of a matrix description of eye position control. *Biological Cybernetics*, 60(6), 411-420.
| [4] Bohme, M., Dorr, M., Graw, M., Martinetz, T., & Barth, E. (2008). A software framework for simulating eye trackers. In *Proceedings of ETRA '08*, pp. 251-258. ACM.
| [5] Goncharov, A. V., & Dainty, C. (2007). Wide-field schematic eye models with gradient-index lens. *Journal of the Optical Society of America A*, 24(8), 2157-2174.
| [6] Wyatt, H. J. (1995). The form of the human pupil. *Vision Research*, 35(14), 2021-2036.
| [7] Wildenmann, U., & Schaeffel, F. (2013). Variations of pupil centration and their effects on video eye tracking. *Ophthalmic and Physiological Optics*, 33(6), 634-641.
| [8] Nystrom, M., et al. (2024). What is a spontaneous blink? *Behavior Research Methods*, 56, 5843-5858.
