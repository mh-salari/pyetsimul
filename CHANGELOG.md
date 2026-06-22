# Changelog

All notable changes to PyEtSimul are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version 3.0.0 is the release presented in the
ETRA 2026 paper; this log records everything since.

## [Unreleased]

## [3.6.0] - 2026-06-22

### Added

- **Two-surface (posterior) corneal refraction.** The cornea can refract at both its front and back surfaces
  (air to cornea, then cornea to aqueous), not just the front. Enabled per cornea with `use_posterior_surface`.
- **Toric (astigmatic) cornea (`ToricCornea`).** A conic cornea whose anterior surface has two principal-meridian
  radii differing about the conic radius at a configurable axis, defaulting to the young-adult population mean
  (with-the-rule, ~0.1 mm / 0.6 D, axis 90 deg). Select via `create_cornea("toric", ...)`.
- **Optional tear film.** A thin tear film modelled as a third refracting surface in front of the anterior cornea
  (air to tears to cornea), lowering the corneal magnification of the pupil. Off by default; requires the
  two-surface cornea.
- **Corneal surface tilt (`ConicCornea.tilt_x_deg`/`tilt_y_deg`).** Tilts the whole cornea off the eye axis about
  the apex, reorienting both the refraction of the pupil and the reflection of the glint.
- **Elliptical and measured-shape pupils.** `EllipticalPupil` gains an aspect ratio and orientation, and a new
  `ContourPupil` takes an arbitrary measured boundary, so a real pupil contour can be used directly instead of a
  circle.
- **Apparent-size pupil sizing through the cornea.** Axial geometry and corneal-magnification helpers let the pupil
  be set by its apparent (camera) size and shape, converted through the cornea to the physical pupil.
- **Axial pupil-plane shift on dilation.** A `z_coeff` in the decentration model shifts the pupil plane in depth as
  the pupil opens (0 keeps the shift in-plane only).
- **Static off-axis pupil position (`OffAxisPupilConfig` on `Eye`).** The pupil centre can be displaced nasally and
  superiorly from the optical axis by a fixed, size-independent amount (population mean from Wyatt 1995, 0.27 mm
  nasal and 0.20 mm superior, mirrored between the eyes). Off by default. Composed with the size-dependent
  pupil decentration when the pupil centre is positioned.
- **Gaze-direction-dependent rotation centre (`RotationCenter`).** The eye pivots about a gaze-blended rotation
  centre rather than a single fixed globe centre, so the globe centre translates as the eye turns.
- **Separate Fick rotation centres with lateral offsets.** Horizontal (azimuth) and vertical (elevation) rotation
  centres with optional nasal/superior offsets and a gaze-varying vertical depth, plus a `fick` mode that rotates
  azimuth and elevation about their own centres sequentially (Fry & Hill 1962/1963; Aguirre 2019). With both
  centres equal and `fick` off, the single fixed-centre behaviour is reproduced exactly.
- **Line-of-sight gaze aiming (`Eye.aiming`).** With `aiming="line_of_sight"`, `look_at` aligns the fovea-through-
  pupil-centre axis to the target instead of the visual axis, so the eye re-aims as the pupil decentres. Default
  `"visual_axis"` leaves the behaviour unchanged.
- **Pupil-plane tilt and torsion knobs.** `Eye.pupil_tilt_x_deg`/`pupil_tilt_y_deg` apply a z-shear of the pupil
  disc about its centre, and `Eye.torsion_deg` adds a roll about the line of sight on top of Listing's law.
- **`Eye.solve_decentration_from_apparent`.** Inverts a measured image-frame pupil-centre shift (per unit apparent
  pupil-diameter change) to the eye-local decentration coefficients that reproduce it, by Newton-iterating the
  rendered apparent pupil. Adds `calculate_pupil_diameter_from_boundary` in `pupil_imaging`.
- **Convex-hull pupil-centre method and a refraction toggle in pupil imaging.**
- **Single-point projection, pupil-plane reverse-projection, and image-Jacobian helpers.**

### Changed

- **Batch corneal refraction traces an arbitrary stack of refracting surfaces** rather than a fixed two-surface
  cornea; the anterior conic surface accepts toric (astigmatic) coefficients, with rotationally symmetric defaults.
- **Corneal refraction is vectorised over the whole pupil boundary.**
- **`refract_direction` is a public optics helper.**
- **The eye owns its side and resolves per-side geometry.** One `which_eye` flag signs the kappa angle and resolves
  the (already eye-specific) pupil-decentration coefficients.

### Fixed

- **`set_pupil_diameter(apparent=True)` now sets the apparent pupil size exactly.** It previously divided the
  requested apparent diameter by the corneal magnification once; because the magnification varies with pupil size,
  the rendered apparent pupil came out a few percent off. The physical diameter is now found by iterating the
  fixed point `physical = apparent / corneal_magnification(physical)`, matching the request to within 1e-4 mm.
- **Angle kappa is now signed by eye side in `look_at`.** `look_at_target` and `look_at_target_optical_then_kappa`
  built the visual axis from the unsigned `fovea_alpha_deg`; the left eye's horizontal kappa must point the
  opposite way to the right eye's. Both now use the eye's signed kappa.
- **Ellipse and centre-of-mass pupil imaging route through `calculate_pupil_center_from_boundary`.**
- **Adopt the skimage 0.26 `EllipseModel.from_estimate` API.**

### Removed

- **Dead `pupil_offset_from_limbus` field on `RealisticPupilParams`.** Declared but never applied; the off-axis
  pupil offset is now modelled by `OffAxisPupilConfig`.
- **Dead `refract_ray_dual_surface` (spherical two-surface refraction).** Orphaned; the conic two-surface path in
  `refraction_batch` is the live implementation.

## [3.5.0] - 2026-05-30

### Changed

- `compute_calibration_errors` is now public; calibration-view rendering is extracted into `render_calibration_view`.

## [3.4.0] - 2026-05-19

### Changed

- The Stampe (1993) gaze model is delegated to the `stampe1993-gaze-mapping` package; the example ships an HV9 setup.

### Removed

- The EyeLink 1000 Plus gaze model (could not be verified against HREF P-CR data; use the Stampe model instead).

## [3.3.0] - 2026-05-16

### Fixed

- The Stampe (1993) fit is staged: an inner biquadratic polynomial, then an outer corner correction.

## [3.2.5] - 2026-03-31

### Added

- ETRA 2026 DOI in the citation and badges.

## [3.2.1] - [3.2.4] - 2026-03-17

### Added

- `Camera.point_at_binocular`; citation, acknowledgment, and the PyPI publish workflow.

### Fixed

- Conic reflection and refraction solved in eye-local coordinates with a 2D `fsolve`.
- Packaging: dynamic version from metadata, README as the PyPI description, project URLs, funding image URL.

## [3.1.0] - 2026-03-01

### Added

- EyeLink 1000 Plus gaze model with HREF preprocessing.

## [3.0.0] - 2026-02-24

The version presented in the ETRA 2026 paper.
