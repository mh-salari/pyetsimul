Optical Calculations
====================

The simulator determines the 2D positions of eye features on the camera sensor through two
optical computations: **reflection**, which locates the corneal reflection of each light source
on the corneal surface, and **refraction**, which determines the apparent shape and position of
the pupil as seen through the cornea.

The simulator computes feature coordinates analytically from an explicit geometric eye model.
Pupil contours, corneal reflections, and occlusions are derived directly from the
three-dimensional configuration and projected into the image plane using closed-form geometry
and numerical optimization. No pixel-level rendering is needed.

Corneal Reflection
------------------

A corneal reflection is the virtual image of a light source on the cornea. The corneal
reflection position is computed by finding where on the corneal surface a light ray reflects
into the camera aperture with equal angles of incidence and reflection.

Spherical Cornea Reflection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a spherical cornea, the reflection point must lie in the plane defined by the light source,
the camera, and the center of the corneal sphere. This reduces the search for the corneal
reflection's 3D position to a one-dimensional search for the correct surface normal vector
:math:`\mathbf{n}` along the arc connecting the light and camera directions.

This normal vector is parameterized by a single variable :math:`a \in [0,1]`:

.. math::

   \mathbf{n}(a) = \frac{a\mathbf{d}_c + (1-a)\mathbf{d}_l}{\| a\mathbf{d}_c + (1-a)\mathbf{d}_l \|}

where :math:`\mathbf{d}_c` and :math:`\mathbf{d}_l` are the direction vectors from the corneal
sphere's center to the camera and light source, respectively.

The goal of the optimization is to find an :math:`a` such that the angle between the normal
:math:`\mathbf{n}(a)` and the light ray is equal to the angle between the normal and the camera
ray. This is expressed as finding the root of:

.. math::

   f(a) = \arccos(\mathbf{n}(a) \cdot \mathbf{u}_c) - \arccos(\mathbf{n}(a) \cdot \mathbf{u}_l) = 0

where :math:`\mathbf{u}_c` and :math:`\mathbf{u}_l` are the unit vectors from the reflection
point on the surface to the camera and light. Brent's method is used to compute :math:`a`.
Once determined, the three-dimensional corneal reflection point is known and projected into
the image plane.

Conic Cornea Reflection
^^^^^^^^^^^^^^^^^^^^^^^

For a conic cornea, the geometry is more complex, but the principle is the same. A similar
numerical optimization is used to find the point on the aspherical surface that satisfies
the law of reflection.

Corneal Refraction (Pupil Imaging)
----------------------------------

Because the pupil lies behind the cornea, light rays originating from the pupil's boundary
bend, or refract, as they exit the eye. This is governed by **Snell's Law** [1], which
relates the angles of incidence and refraction to the refractive indices of the different
media (in this case, air and the cornea).

Finding the apparent position of a pupil boundary point requires determining where on the
corneal surface the ray from that point refracts towards the camera. This is also a
root-finding problem. The optimization constraint enforces Snell's law:

.. math::

   n_1 \sin \theta_1 = n_2 \sin \theta_2

The function to solve becomes:

.. math::

   f(a) = n_{\text{air}} \sin(\theta_{\text{camera}}) - n_{\text{cornea}} \sin(\theta_{\text{pupil}}) = 0

where :math:`\theta_{\text{pupil}}` and :math:`\theta_{\text{camera}}` are the angles of the
light ray relative to the normal inside and outside the eye, respectively. This equation is
solved numerically to find the exit point on the cornea.

This process is repeated for multiple points on the pupil boundary to reconstruct its apparent
shape in the camera image.

----

| [1] Forsyth, D. A., & Ponce, J. (2002). *Computer Vision: A Modern Approach*. Prentice Hall.
