Evaluation Workflow
===================

PyEtSimul provides an integrated system for generating synthetic datasets with controlled
parameter variations and evaluating gaze estimation algorithms under these conditions.
Researchers define an experimental setup once to generate structured and reproducible test
datasets.

Experimental Setup
------------------

An experimental setup defines the complete configuration of an eye tracking setup, including:

- The eye model with its anatomical parameters (corneal radius, pupil size, angle kappa, etc.)
- Camera parameters (intrinsic calibration matrix, distortion coefficients, position, and
  orientation)
- Light source positions
- A default gaze target position

Parameter Variations
--------------------

Once the experimental setup is defined, researchers specify which parameters to vary during
data generation. PyEtSimul provides built-in variation types for common experimental factors:

**Anatomical variations** modify eye characteristics such as:

- Pupil size
- Pupil decentration
- Pupil size with decentration
- Angle kappa (foveal displacement in horizontal and vertical directions)
- Corneal anterior radius
- Corneal thickness

**Spatial variations** modify positions in 3D space, including:

- Eye position (observer head movement)
- Target position

PyEtSimul supports custom variations beyond the built-in types. Researchers specify which
parameter values to test and how each value modifies the eye model or other simulation
components. See the :doc:`../guides/custom_variations` guide for details.

Gaze Mapping Algorithms
------------------------

PyEtSimul provides two gaze estimation algorithm families: polynomial models and homography
normalization.

Polynomial Models
^^^^^^^^^^^^^^^^^

Polynomial models map pupil-corneal reflection vectors to gaze coordinates using polynomial
regression. Given the pupil-corneal reflection vector :math:`(x, y)` in image coordinates,
gaze position :math:`(g_x, g_y)` is estimated as:

.. math::

   g_x = \sum_{i} a_i \phi_i(x, y), \quad g_y = \sum_{j} b_j \psi_j(x, y)

where :math:`\phi_i` and :math:`\psi_j` are polynomial terms, and :math:`a_i`, :math:`b_j`
are coefficients determined through calibration.

PyEtSimul includes seven built-in polynomial models:

- Second Order
- Zhu Ji (2005)
- Cerrolaza (2008), symmetric and asymmetric variants
- Hoormann (2008)
- Hennessey (2008)
- Blignaut Wium (2013)

Any custom polynomial formulations can be registered without modifying the framework by
specifying the polynomial terms and their respective orders. See the
:doc:`../guides/custom_gaze_model` guide for details.

Homography Normalization
^^^^^^^^^^^^^^^^^^^^^^^^

Homography normalization and its degraded variants [1] are gaze estimation methods for
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

The implementation includes RANSAC-based homography estimation to handle outliers and optional
Gaussian Process regression for residual error correction.

Evaluation Process
------------------

The evaluation process proceeds through several stages:

1. Define experimental setup and parameter variations
2. Generate synthetic dataset
3. Calibrate the gaze estimation algorithm using calibration target measurements
4. Evaluate gaze estimation accuracy across the dataset
5. (Optional) Compare multiple algorithms

At the calibration stage, researchers can assess calibration quality by testing gaze estimation
at the original calibration target points. This verification step ensures the algorithm
correctly fits the calibration data before evaluating generalization performance.

Errors are computed in both **Euclidean distance** (millimeters) and **angular deviation**
(degrees). The evaluation produces error statistics including mean, maximum, and standard
deviation across the tested parameter space, revealing how anatomical variations, spatial
changes, and other experimental factors affect algorithm performance.

When comparing multiple algorithms, PyEtSimul evaluates all methods on identical datasets,
enabling direct performance comparison. Beyond ranking algorithms by error metrics, PyEtSimul
optionally computes pairwise comparisons including angular differences, cosine similarities,
and amplitude differences between algorithm predictions.

----

| [1] Hansen, D. W., & Ji, Q. (2010). In the eye of the beholder: A survey of models for eyes and gaze. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 32(3), 478-500.
