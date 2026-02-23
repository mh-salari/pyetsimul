Gaze Estimation Models
======================

PyEtSimul provides three families of gaze estimation algorithms: polynomial models, the
Stampe (1993) biquadratic model, and homography normalization.

Polynomial Models
-----------------

Polynomial models map pupil-corneal reflection (P-CR) vectors to gaze coordinates using
polynomial regression. Given the P-CR vector :math:`(x, y)` in image coordinates, the gaze
position :math:`(g_x, g_y)` is estimated as:

.. math::

   g_x = \sum_{i} a_i \phi_i(x, y), \quad g_y = \sum_{j} b_j \psi_j(x, y)

where :math:`\phi_i` and :math:`\psi_j` are polynomial terms, and :math:`a_i`, :math:`b_j`
are coefficients determined through calibration.

Built-in Polynomial Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

PyEtSimul includes seven built-in polynomial models:

**Hennessey (2008)** [1] --- Polynomial with cross-terms:

.. math::

   g = a_0 xy + a_1 x + a_2 y + a_3

**Hoorman (2008)** [2] --- Linear polynomial (different features for X/Y):

.. math::

   g_x = a_0 x + a_1, \quad g_y = b_0 y + b_1

**Cerrolaza (2008) symmetric** [3] --- Second-order polynomial (same features for X/Y):

.. math::

   g = a_0 x^2 + a_1 y^2 + a_2 xy + a_3 x + a_4 y + a_5

**Cerrolaza (2008) asymmetric** [3] --- Second-order with different features for X/Y:

.. math::

   g_x = a_0 x^2 + a_1 x + a_2 y + a_3

   g_y = b_0 x^2 y + b_1 x^2 + b_2 xy + b_3 y + b_4

**Second Order** --- Full second-order polynomial with all cross-terms:

.. math::

   g = a_0 x^2 y^2 + a_1 x^2 + a_2 y^2 + a_3 xy + a_4 x + a_5 y + a_6

**Zhu Ji (2005)** [4] --- Asymmetric polynomial:

.. math::

   g_x = a_0 xy + a_1 x + a_2 y + a_3

   g_y = b_0 y^2 + b_1 x + b_2 y + b_3

**Blignaut Wium (2013)** [5] --- High-order polynomial:

.. math::

   g_x = a_0 + a_1 x + a_2 x^3 + a_3 y^2 + a_4 xy

   g_y = b_0 + b_1 x + b_2 x^2 + b_3 y + b_4 y^2 + b_5 xy + b_6 x^2 y

Custom Polynomial Registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any custom polynomial formulation can be registered without modifying the framework using
``PolynomialDescriptor`` and ``register_polynomial()``. Once registered, custom models are
fully integrated into the data generation and evaluation pipeline.

See the :doc:`../guides/custom_gaze_model` guide for a complete template.

Stampe (1993) Biquadratic Model
-------------------------------

The Stampe (1993) model [6] uses a two-stage calibration:

**Stage 1 --- Biquadratic polynomial** (5 terms, no cross-term):

.. math::

   g = a + bx + cy + dx^2 + ey^2

where :math:`(x, y)` are the P-CR feature coordinates.

**Stage 2 --- Per-quadrant corner correction** removes residual nonlinearity at the screen
corners. The screen is divided into 4 quadrants relative to the centroid
:math:`(X_c, Y_c)` of the calibration grid. For each quadrant :math:`q`:

.. math::

   X_{\text{final}} = X_{\text{poly}} + c_x^{(q)} (X_{\text{poly}} - X_c)(Y_{\text{poly}} - Y_c)

   Y_{\text{final}} = Y_{\text{poly}} + c_y^{(q)} (X_{\text{poly}} - X_c)(Y_{\text{poly}} - Y_c)

where :math:`c_x^{(q)}` and :math:`c_y^{(q)}` are fit via least-squares over all calibration
points in that quadrant.

Homography Normalization
------------------------

Homography normalization [7] and its degraded variants are gaze estimation methods for
uncalibrated setups (e.g., unknown camera and setup parameters). They use multiple corneal
reflections to estimate a normalizing projective planar transformation that compensates for
head pose.

Given corneal reflection positions :math:`\mathbf{p}_i` in the image and corresponding
reference positions :math:`\mathbf{p}'_i`, the method estimates a normalizing homography
matrix :math:`\mathbf{H}` such that:

.. math::

   \mathbf{p}'_i \sim \mathbf{H} \mathbf{p}_i

where :math:`\sim` denotes equality up to scale. The mapping from normalized pupil coordinates
to gaze can be learned through user calibration.

The implementation includes:

- **RANSAC-based homography estimation** to handle outliers
- **Optional Gaussian Process regression** for residual error correction

This method requires 4 or more light sources to compute the homography.

----

| [1] Hennessey, C., Noureddin, B., & Lawrence, P. (2008). Fixation precision in high-speed noncontact eye-gaze tracking. *IEEE Trans. SMC-B*, 38(2), 289-298.
| [2] Hoormann, J., Jainta, S., & Jaschinski, W. (2007). The effect of calibration errors on the accuracy of eye movement recordings. *Journal of Eye Movement Research*, 1(2).
| [3] Cerrolaza, J. J., Villanueva, A., & Cabeza, R. (2008). Taxonomic study of polynomial regressions applied to the calibration of video-oculographic systems. In *Proceedings of ETRA '08*, pp. 259-266.
| [4] Zhu, Z., & Ji, Q. (2005). Eye gaze tracking under natural head movements. In *CVPR '05*, Vol. 1, pp. 918-923. IEEE.
| [5] Blignaut, P., & Wium, D. (2013). The effect of mapping function on the accuracy of a video-based eye tracker. In *Proceedings of ETSA '13*, pp. 39-46.
| [6] Stampe, D. M. (1993). Heuristic filtering and reliable calibration methods for video-based pupil-tracking systems. *Behavior Research Methods*, 25(2), 137-142.
| [7] Hansen, D. W., Agustin, J. S., & Villanueva, A. (2010). Homography normalization for robust gaze estimation in uncalibrated setups. In *Proceedings of ETRA '10*, pp. 13-20. ACM.
