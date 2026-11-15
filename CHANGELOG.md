# Changelog

All notable changes to PyEtSimul are recorded here. Version 3.0.0 is the release presented in the ETRA 2026 paper; this
log records everything since. Every entry starts with the kind of change (Added, Changed, Fixed or Removed). The
entries of a version that is not released yet are in the order in which the changes were made.

## [Unreleased]

- **Fixed: a copy made with `Eye.copy()` was at a different position than the original eye.**
  `Eye.position` returns the position of the corneal apex, which moves when the eye turns, because the eye turns about
  a point behind the cornea. Before, `Eye.copy()` used this moved position for the copy, so the copy of an eye that had
  turned was displaced. Now it uses `Eye.placement`, the position the eye was given.

- **Fixed: the two eyes in `plot_interactive_pupil_comparison` drifted apart.**
  The plot compares an eye with an elliptical pupil and an eye with a realistic pupil at the same position. Before, at
  every redraw, the second eye was placed at the corneal apex of the first, which moves when the eye turns. After the
  first eye rotation the two eyes were no longer at the same position. Now the two eyes stay at the position they were
  given.

- **Fixed: writing the log to a file raised `AttributeError`.**
  Before, after `set_log_file`, the first log message failed, so no log file could be written.

- **Fixed: `ComposedVariation` generated only a part of its combinations.**
  Before, only the combinations with the first value of its first variation were generated, while `len()` and the
  dataset metadata reported the full number. Now all combinations are generated.

- **Changed: two variations with the same `param_name` in one `ComposedVariation` raise `ValueError`.**
  A `ComposedVariation` combines several parameter variations and stores every combination under the `param_name` of
  its variations. Before, two variations with the same name overwrote each other, and dataset generation stopped
  part-way with `KeyError`. Now creating such a composition raises `ValueError`, which names the repeated `param_name`.

- **Removed: `ComposedVariation.description`.**
  Nothing used it. `describe()` returns the description.

- **Fixed: a camera with glint noise could not be saved.**
  Before, `Camera.serialize` raised `AttributeError` for a camera with a `GlintNoiseConfig`, so no dataset could be
  generated with such a camera.

- **Fixed: a failed gaze prediction was returned as the point (0, 0, 0) and counted as a real prediction.**
  When the pupil centre or a glint is missing, no gaze can be predicted. Before, the polynomial, Stampe and homography
  models returned the point (0, 0, 0) with confidence 0 in this case. `accuracy_at_calibration_points` and the
  interactive gaze plot then used the distance to (0, 0, 0) as the error of that prediction. Now the models return
  `None`.

- **Changed: predicting gaze with a tracker that is not calibrated raises `ValueError`.**
  Before, the polynomial model failed with an error about matrix shapes, and the Stampe and homography models returned
  the point (0, 0, 0) as if it were a prediction.

- **Removed: `PolynomialFeatures.predict`.**
  Nothing called it. `PolynomialGazeModel.predict_gaze` computes the prediction.

- **Fixed: a saved camera was not restored exactly.**
  Before, the orientation matrix was normalised when the camera was saved, which could change its last digits. Gaze
  predictions of a loaded tracker differed from those of the original by about 1e-10 mm. Now the matrix is saved
  unchanged.

- **Fixed: the eyelid moved with the eye when the eye turned.**
  Before, the eyelid was positioned relative to the corneal apex, which moves when the eye turns. After a `look_at`,
  the eyelid in the camera image had moved with the eye, and no longer matched the eyelid drawn by `plot_eye_anatomy`.
  Now the eyelid is positioned relative to `Eye.placement`, the position the eye was given, so it stays fixed to the
  face. Camera images of an eye with an eyelid change for every gaze direction in which the eye is turned away from its
  rest orientation.

- **Fixed: a key press in the interactive plots moved the eye further than one step.**
  In `plot_interactive_cameras`, `plot_interactive_pupil_comparison` and `create_interactive_gaze_plot`, the keyboard
  moves the eye in steps. Before, every step also added the displacement of the corneal apex from the last eye
  rotation, so a 1 mm step could move the eye by about 4 mm, and stepping back did not return it to its start. In
  `create_interactive_gaze_plot`, the Space key reset the eye to a displaced position. Now a step has the stated size,
  and Space returns the eye to its initial position.

- **Fixed: a pupil set from a measured contour was centred on the optical axis.**
  Before, `Eye.set_pupil_from_contour` ignored the off-axis position and the decentration of the pupil that the eye is
  configured with, until the pupil size was set again. Now a contour pupil gets the same position as any other pupil.

- **Fixed: changing the pupil size of one eye changed the realistic pupil of other eyes built from the same eye
  model.**
  Before, when an `EyeModel` had `realistic_pupil_params`, every eye built from it, and every `Eye.copy()`, used the
  same parameter object. Now every pupil has its own copy of the parameters.

- **Fixed: `intersect_ray_conic` returned a wrong point for a ray direction that is not of unit length.**
  Before, for a ray direction of length 2, the returned intersection point was twice as far along the ray as the real
  one. `ConicCornea.intersect`, `ConicCornea.intersect_posterior`, `reflect_ray_conic` and `refract_ray_conic` use this
  function and returned the same wrong point. Ray directions of unit length were not affected.

- **Fixed: a `TransformationMatrix` created from whole numbers dropped decimals.**
  Before, a `TransformationMatrix` built from whole numbers stored integers, so a value such as 0.5 written into it
  later was cut to 0 without a message. A `TransformationMatrix` or `RotationMatrix` also shared its memory with the
  array it was built from, so changing one changed the other. Now each stores its own floating-point copy.

- **Fixed: `look_at` aimed the visual axis about 0.1 deg beside the target (since 4.0.0).**
  With the methods `visual_axis` and `line_of_sight`, `look_at` turns the eye until the visual axis, or the line of
  sight, passes through the target. Before, that axis was only made parallel to the line from the corneal apex to the
  target, so it passed beside the target: by 1.36 mm for the default eye, which is 0.13 deg at a distance of 600 mm.
  Now it passes through the target. The gaze direction of every eye with foveal displacement (angle kappa) changes by
  this small angle, and the simulated images change with it. The methods `optical_then_kappa` and
  `optical_axis_target_direction`, and eyes without foveal displacement, are not affected.

- **Fixed: `plot_eye_cross_section` drew the rotation centre at the wrong depth.**
  Before, the marker was correct only for an eye at the world origin without a rest orientation. For an eye placed 814
  mm from the origin, the rotation centre was drawn 828.8 mm behind the corneal apex. Now it is drawn at its depth in
  the eye model.

- **Fixed: `plot_eye_cross_section` drew a visual axis that differed from the one the simulation uses.**
  Before, the fovea was drawn 0.5 mm from its position in the eye model, and a tilted visual axis was drawn even for an
  eye without foveal displacement. Now the plot draws the axis from the fovea of the eye model through the eyeball
  centre, which is the visual axis that `look_at` uses.

- **Fixed: `len()` of `EyePositionVariation` and `TargetPositionVariation` could report more positions than were
  generated.**
  An axis whose two limits are equal produces a single value, whatever its grid size. Before, `len()` did not account
  for this: with `dy=[0, 0]` and `grid_size=[5, 3, 4]` it reported 60 positions, while 20 were generated. The dataset
  metadata and `describe()` carried the wrong number. Now `len()` returns the number of positions that are generated.

- **Fixed: a saved eye tracker lost its pupil centre method.**
  Before, the method was not stored, so a tracker calibrated with `"center_of_mass"` or `"convex_hull"` used
  `"ellipse"` after loading. Now it is stored. `create` of the three gaze models also accepts `pupil_center_method`;
  before, only the constructor did. A tracker saved with an earlier version has no stored method and does not load
  until `"pupil_center_method": "ellipse"` is added to its dictionary.

- **Fixed: `calculate_pupil_center_methods` rejected the method `"convex_hull"`.**
  Before, it raised `ValueError`, although `take_image` and `Eye.get_pupil_in_camera_image` accept the same three
  methods (`"ellipse"`, `"center_of_mass"` and `"convex_hull"`). Now it accepts all three.

- **Changed: `use_tear_film=True` raises `ValueError` on a cornea that cannot model a tear film.**
  The tear film is modelled only on a `ConicCornea` or `ToricCornea` with its posterior surface. Before, on a
  `SphericalCornea` or with `use_posterior_surface=False`, the option was accepted and had no effect.

- **Fixed: after a failed `look_at`, `current_target_point` named a target the eye was not looking at.**
  Before, when `look_at` raised `ValueError` (unknown method, or a target at the eye position), the eye had not turned,
  but `current_target_point` already held the new target. Now it keeps the previous target.

- **Changed: `set_rest_orientation_at_target` raises `ValueError` when the target is at the eye position.**
  No direction can be computed for such a target. Before, the call returned without doing anything, so the eye kept its
  previous rest orientation and nothing showed that the call had failed.

- **Changed: `look_at` raises `ValueError` for a target exactly behind the eye.**
  A turn of exactly 180 deg from the rest direction has no defined rotation axis under Listing's law. Before, `look_at`
  left the eye facing away from the target, without an error.

- **Changed: a missing coefficient of the linear pupil decentration model raises `ValueError`.**
  Before, when only one of `x_coeff` and `y_coeff` was given, or when decentration was enabled on an eye that was built
  without it, the next pupil size change failed with `TypeError: unsupported operand type(s) for *`. Now the error
  names the missing coefficient.

- **Fixed: printing a cornea that is not part of an eye raised `ValueError`.**
  Before, `str()` and `pprint()` of a cornea that was created on its own failed, because its centre is not set yet. Now
  they show the centre as `unset`.

- **Changed: evaluating a dataset without measurements raises `ValueError`.**
  Before, `evaluate_gaze_accuracy` returned an empty result whose `pprint()` then failed with `ZeroDivisionError`, and
  `compare_algorithms` failed in the same way. Now `evaluate_gaze_accuracy` and `compare_algorithms` report that the
  dataset holds no measurements for the selected camera and eye.

- **Fixed: `load_experiment_data` reported a wrong number of measurements.**
  Before, `total_measurements` was the number of parameter values. For a dataset with two cameras it was half the real
  number. Now it is the number of measurements.

- **Changed: `load_experiment_data` raises `ValueError` for a file that is not a generated dataset.**
  Before, a file without the dataset metadata was loaded as a dataset with 0 measurements and an empty parameter name.

- **Changed: the homography model warns when its Gaussian process correction is not trained.**
  Before, with `use_gp_correction=True` and exactly four valid calibration points, the correction was skipped without a
  message, and the tracker predicted without it. Now a warning is printed.

- **Fixed: `ContourPupil` changed the boundary array that was passed to it.**
  Before, the array was shifted in place so that its mean is zero. An array passed to `ContourPupil`,
  `Eye.set_pupil_from_contour(..., apparent=False)` or `Eye.register_pupil_shape` was therefore changed for the caller.
  Now the pupil works on a copy.

- **Fixed: `Camera.pprint` gave the lens distortion coefficients the wrong names.**
  The camera stores five coefficients in the OpenCV order `k1`, `k2`, `p1`, `p2`, `k3`. Before, the fourth and fifth
  were printed as `p1` and `p2`, and the real `p1` was not printed. Now all five are printed with their own names.

- **Fixed: the algorithm comparison printed its amplitude differences in degrees.**
  The values are differences of distances. Now they are printed in mm.

- **Fixed: the warning about failed calibration points left out the y coordinate.**
  Before, it printed x and z of each failed point, which identifies the point only for a screen in the x-z plane. Now
  all three coordinates are printed.

- **Changed: gaze predictions of the homography model are named `"homography_normalization"` instead of
  `"homography"`.**
  This name is the `algorithm_name` of a prediction. It now matches the name of the tracker. Code that compares
  `algorithm_name` with `"homography"` needs the new name.

- **Fixed: wrong "Corneal radius" and "Pupil radius" in the calibration report.**
  Before, the line "Corneal radius" printed the distance from the corneal apex to the cornea centre (8.62 mm for the
  default cornea, whose radius is 7.76 mm), and the line "Pupil radius" printed the distance from the cornea centre to
  the pupil centre. Now the report prints the anterior corneal radius and the pupil diameter.

- **Fixed: `Eye.pprint` printed the same rotation-centre distance for every eye.**
  Before, the row "Cornea center to rotation center" showed 10.2 mm, whatever the eye model. Now it is replaced by the
  depth of the rotation centre behind the corneal apex, which is 14.8 mm for horizontal and 12.2 mm for vertical
  rotation in the default eye.

- **Changed: generated datasets are saved to `outputs/` by default, instead of `output/`.**
  Before, `DataGenerationStrategy` wrote to `output/`, while `load_experiment_data` and `ExperimentConfig` read from
  `outputs/`. A dataset generated with the defaults was therefore not found by the loader with its defaults.
  `output_dir="output"` keeps the old folder.

- **Fixed: `PupilDecentrationVariation` produced NaN pupil positions when its first range was constant.**
  Before, with a constant first range, for example `dx_range=[0.1, 0.1]` together with a varying `dy_range`, the pupil
  was placed at NaN in every step, without an error. Now any of the ranges may be constant, and `ValueError` is raised
  when all of them are.

- **Fixed: the calibration report showed the coefficients of the first glint only.**
  Before, with several light sources, the coefficient tables of the other glints were missing. Now every glint has its
  own table. For the homography and the Stampe model, the report also printed the line "Geometric gaze estimation
  (non-polynomial)", which describes neither model; that line is removed.

- **Fixed: `plot_setup` drew a line that is not the visual axis, and `plot_eye_anatomy` mislabelled the eyeball
  centre.**
  Before, `plot_setup` drew a line from the pupil centre to the target and labelled it as the visual axis. Now it draws
  the visual axis of the eye model, from the fovea through the eyeball centre. Before, `plot_eye_anatomy` labelled the
  eyeball centre as "Rotation Center". Now the point is labelled "Eyeball Center", and the points the eye rotates about
  are marked as well.

- **Fixed: `plot_eye_anatomy` failed for a toric cornea.**
  Before, it raised `Unknown cornea type: toric`, so the `gkaModelEye` eye model could not be plotted. Now a toric
  cornea is drawn as the conic it is built on.

- **Fixed: `set_rest_orientation_at_target` aimed the visual axis about 0.1 deg beside its target.**
  `set_rest_orientation_at_target` sets the rest orientation of the eye so that the resting eye looks at a given
  target. Before, the visual axis of the resting eye passed 1.36 mm beside that target, and a `look_at` at the same
  target still turned the eye by about 0.1 deg. Now `set_rest_orientation_at_target` and `look_at` use the same
  computation, and looking at the rest target does not turn the eye. The rest orientation of an eye with foveal
  displacement changes by 0.11 deg for a target at 700 mm.

- **Changed: missing glints are reported once per dataset, with a count.**
  A light gives no glint in an image when, for example, its reflection falls outside the cornea or outside the camera
  image. Before, the reporting was uneven: when the reflection solver found no reflection point, a `RuntimeWarning` was
  printed for every such glint, while a glint outside the visible cornea or outside the image was not reported at all.
  Now `DataGenerationStrategy.execute` prints one line with the number of measurements that lack a glint or the pupil
  centre.

- **Added: the reason for every missing glint can be printed, with the new log level `LogLevel.DEBUG`.**
  `set_log_level(LogLevel.DEBUG)` prints one line for every glint that is missing from an image, with the reason. A
  missing glint is not an error. A light gives no glint when its reflection falls outside the cornea or is covered by
  the eyelid, when the reflection falls outside the camera image, or when no reflection point is found for the
  positions of the light, the camera and the eye. `DEBUG` is a fourth log level, more detailed than `SILENT`, `WARNING`
  and `INFO` (the default). Dataset generation passes the log level on to its worker processes.

- **Fixed: in some generated datasets the eye did not look at its target.**
  Before, in a `ComposedVariation` that listed a target variation before an eye variation, the eye was turned to the
  target first and moved or changed afterwards. In every image the eye then looked beside its target, by 48 mm in a
  test with a 40 mm eye shift. Without a `gaze_target` and without a target variation, the eye was not turned at all
  after an eye variation. Now the eye is turned once, after all its parameters are set. A composition that contains an
  unsupported kind of variation raises `TypeError`; before, that variation was skipped without a message.

- **Fixed: `PupilSizeWithDecentrationVariation` produced no decentration with the automatic baseline.**
  The pupil decentration is computed from the difference between the current pupil size and a baseline size. Before,
  the automatic baseline was read after the pupil already had its new size, so the difference, and with it the
  decentration, was zero in every sample. With the population-mean coefficients the sweep did not run at all. Now both
  cases produce the expected decentration. The configuration object passed to the variation is no longer modified.

- **Fixed: glint noise with a seed shifted every glint by the same amount.**
  Before, with a seed in `GlintNoiseConfig`, the same random numbers were drawn for every glint in every image, so all
  glints were shifted identically. Now the shifts differ between glints and are the same in every run, also with
  several worker processes. `examples/07_glint_noise.py` uses a seed and now shows scattered glints.

- **Added: the pixel noise of a camera can be made the same in every run, with `Camera(err_seed=...)`.**
  `Camera(err=...)` adds random noise to every image position the camera reports. Without a seed, this noise is
  different in every run. With `err_seed`, it is the same in every run, also when a dataset is generated with several
  worker processes.

- **Fixed: `look_at` was inexact for a target very close to the eye.**
  With the default rotation centre the eye moves slightly as it turns, so `look_at` finds the position and the
  orientation of the eye together, in repeated steps. Before, it stopped after 8 steps, finished or not. For a target
  40 mm from the eye the aimed axis then missed the target by 0.0034 mm, and for a target at 20 mm by 0.45 mm. Now the
  steps continue until the result no longer changes. For a target closer than about 20 mm, which is about as close as
  the rotation centre of the eye, no result is reached and `look_at` raises `ValueError`. Results at normal viewing
  distances change by about 1e-11 mm.

- **Fixed: setting a pupil by its apparent size could return an unfinished result without a warning.**
  `Eye.set_pupil_diameter(apparent=True)` and `Eye.solve_decentration_from_apparent` find their result in repeated
  steps. Before, when the steps did not finish within the step limit, the value of the last step was returned. Now
  `RuntimeError` is raised in that case. The two functions are also more exact: the physical pupil diameter is solved
  to 1e-10 mm (before 1e-4 mm, which left the apparent diameter off by about 1e-6 mm), and the default tolerance of
  `solve_decentration_from_apparent` is 1e-10 (before 1e-5).

- **Fixed: `reverse_project_to_pupil_plane` returned a wrong point when no solution exists.**
  The function finds the point on the pupil plane that is seen at a given image position. Before, for an image position
  at which no point of the pupil plane is seen, it returned a point anyway. In a test with an image position 400 px
  away from the pupil, the returned point is seen 373 px away from the requested position. Now `ValueError` is raised
  in that case. For positions that have a solution, the result is more exact (within 1e-11 px; before, about 1e-5 px).

- **Fixed: an eye that was moved after `look_at` no longer looked at its target.**
  Before, assigning `Eye.position` after a `look_at` moved the eye but kept the rotation it had at its old position.
  The visual axis then missed the target, by 100 mm for an eye that was moved by 140 mm at about 730 mm from the
  target. `set_rest_orientation` and `set_rest_orientation_at_target` after a `look_at` had a related problem: they
  kept the shift in eye position that the earlier rotation had caused. The same calls in a different order gave eye
  positions that differed by up to 4.4 mm. Now the eye turns to its target again after each of these calls, so the
  result does not depend on their order. If the eye cannot turn to the target from its new position, the call raises
  `ValueError` and leaves the eye in its rest orientation without a target.

- **Added: an eye can be moved without turning back to its target, with `Eye(hold_fixation=False)`.**
  By default, an eye that looks at a target keeps looking at it: after `eye.look_at(target)`, giving the eye a new
  position or a new rest orientation turns it so that it looks at the same target again. With `hold_fixation=False`,
  the eye goes to its rest orientation instead, which is the direction it faces when it looks at no target. It then has
  no target until the next `look_at`.

- **Fixed: variations of angle kappa and of several other eye parameters did not change the eye (since 4.0.0).**
  Before, `AngleKappaVariation`, `AngleKappaAlphaVariation`, `AngleKappaBetaVariation`, and a `GenericEyeVariation` on
  `fovea_alpha_deg`, `fovea_beta_deg`, `axial_length`, `n_aqueous_humor`, `torsion_deg` or the pupil tilt had no effect
  on the eye. Every measurement of such a sweep was made with the same eye, while the dataset recorded the varied
  values. Now these variations change the eye.

- **Fixed: varying a cornea parameter in a dataset moved the corneal apex.**
  This affected `CorneaRadiusVariation`, `CorneaThicknessVariation` on a spherical cornea, and a `GenericEyeVariation`
  on `cornea.anterior_radius` or `cornea.anterior_k`. Before, the new value was applied without positioning the cornea
  again, so the corneal apex moved along the optical axis (by 6.9 mm when the conic constant changed from -0.1 to -0.5)
  while the pupil stayed where it was. Such a sweep mostly changed the distance between cornea and pupil. Now the eye
  is identical to an eye built with the new cornea. Datasets of these sweeps change: in
  `examples/16_custom_variation.py`, the glint moved 42.1 px and the pupil centre 63.3 px over the sweep of the conic
  constant, and with the fix they move 0.7 px and 0.1 px.

- **Fixed: `GenericEyeVariation` accepted a parameter name that does not exist.**
  Before, with a misspelled parameter path, the sweep ran and changed nothing. Now `AttributeError` is raised.

- **Added: one parameter of an eye that is already in a scene can be changed, with `Eye.set_model` and
  `EyeModel.with_parameter`.**
  An `EyeModel` cannot be modified after it is created. `EyeModel.with_parameter(path, value)` returns a copy of the
  model with one parameter changed, and `Eye.set_model(model)` gives that model to an existing eye, for example
  `eye.set_model(eye.model.with_parameter("cornea.anterior_k", -0.3))`. The eye keeps its side (left or right),
  position, pupil size and gaze target. In everything else it is identical to an eye built from the new model.
  `Eye.copy(**overrides)` follows the same rule.

- **Fixed: a pupil position set by hand was lost when the pupil size changed.**
  `Eye.set_pupil_position` and `Eye.move_pupil_position` shift the pupil centre. Before, the next `set_pupil_diameter`
  or `set_pupil_radii` positioned the pupil again without this shift. In a dataset, a `PupilDecentrationVariation`
  followed by a pupil size variation therefore produced no shift, while the opposite order kept it. Now the eye
  remembers the shift and applies it at every pupil size, in addition to the off-axis position and the size-dependent
  decentration of the pupil.

- **Fixed: a realistic pupil got a new random shape every time its size changed.**
  A realistic pupil (`pupil_type="realistic"`) has an irregular outline, built from an ellipse and higher harmonics
  after Wyatt (1995). Before, without a random seed, which is the default, a new random outline was drawn at every
  `set_pupil_diameter`. One eye therefore had a different pupil shape at every size; the normalised outline changed by
  about 5 %. Wyatt (1995) reports that the shape of a pupil is stable over time. Now the random part of the shape is
  drawn once, when the pupil is created. As before, the ellipse has one orientation for a dilated and one for a
  constricted pupil, with the switch at 4 mm. Outlines of pupils with a seed differ from those of 5.1.0.

- **Fixed: a saved realistic pupil was loaded with a different outline.**
  Before, loading drew a new random orientation for the elliptical component of the pupil. Now the saved dictionary
  holds the random part of the shape (`phases`, `dilated_axis_angle` and `constricted_axis_angle`) instead of
  `harmonics` and `r2`.

- **Changed: the ellipse share and the ellipse orientation of a realistic pupil can be set; before, both settings were
  ignored.**
  The two settings are `ellipse_contribution` and `major_axis_angle` of `RealisticPupilParams`. Before,
  `major_axis_angle` was overwritten by the pupil, and `ellipse_contribution` was never read. Now a value for
  `major_axis_angle` sets the orientation of the elliptical component of the pupil at every pupil size. A value for
  `ellipse_contribution`, between 0 and 1, sets the share of the squared noncircularity that the elliptical component
  carries. Both default to `None`, which uses the values of Wyatt (1995), one for a dilated and one for a constricted
  pupil. The orientation that a pupil currently has is `RealisticPupil.major_axis_angle`.

- **Removed: `RealisticPupilParams.base_radius`.**
  The value was overwritten with the pupil size set on the eye, so setting it had no effect. The size of a pupil is
  read with `get_diameter()`.

- **Removed: `PupilDefaults.BASE_RADIUS`, `PupilDefaults.ELLIPSE_CONTRIBUTION` and `PupilDefaults.MAJOR_AXIS_ANGLE`.**
  They were the defaults of parameters that were never used. `pprint_hardware_defaults` no longer prints a "pupil base
  radius" of 2.5 mm, which was not the pupil radius of any eye.

- **Fixed: an eye with `EyeballCenter` rotated about its corneal apex instead of the centre of the eyeball.**
  `EyeballCenter` gives an eye model one fixed rotation centre, which is meant to be the centre of the eyeball. Before,
  with the default corneas, the eye rotated about its corneal apex: a turn of 26 deg swung the centre of the eyeball by
  5.7 mm. `examples/14_custom_eye_model.py` builds such an eye. Now the eye rotates about the centre of the eyeball,
  half the axial length behind the apex, and its images change for every gaze direction. The `et_simul` eye model is
  not affected, and the default eye model does not use `EyeballCenter`.

- **Fixed: `Eye.rotation_centre` and `Camera.point_at(eye, mode="fick")` did not work for an eye with
  `EyeballCenter`.**
  Before, both raised an error for such an eye. Now they work. The new `Eye.rotation_centre_depth_mm` returns how far
  the rotation centre lies behind the corneal apex.

- **Fixed: a non-zero `torsion_deg` turned the eye away from its target.**
  `torsion_deg` rolls the eye about its gaze direction. Before, the roll was made about the optical axis, which differs
  from the visual axis by angle kappa. With the `visual_axis` method, 5 deg of torsion turned the visual axis 0.55 deg
  away from the target (8 mm at 830 mm). The methods `line_of_sight` and `optical_then_kappa` ignored `torsion_deg`
  without a message. Now the eye rolls about the axis that `look_at` aims at the target, with every method, so it keeps
  looking at the target. Results change only for a non-zero `torsion_deg` (the default is 0), and not for
  `optical_axis_target_direction` or for `visual_axis` without foveal displacement.

- **Fixed: a saved eye was loaded as a different eye.**
  Before, `Eye.deserialize` rebuilt every cornea as a `SphericalCornea` and every pupil as an `EllipticalPupil`,
  restored only a part of the eye model, and did not restore the position of the eye correctly. A default eye was
  therefore loaded with a spherical cornea, and its image differed by 1.4 px in a test. An eye with a contour pupil
  raised `KeyError`, and an eye with an eyelid could not be imaged after loading. Now `Eye.serialize` stores the
  complete eye, and a loaded eye gives exactly the same images as the original. This also holds for the eyes stored in
  a generated dataset, whose pupil centre was 0.6 px off after loading. An eye or a dataset saved with an earlier
  version cannot be loaded.

- **Fixed: a saved camera in Fick mode was loaded as a pinhole camera.**
  Before, the mode set with `point_at(eye, mode="fick")` was not stored. Now it is saved with the camera.

- **Fixed: `evaluate_gaze_accuracy` measured the angular error from a displaced eye position.**
  The angular error is the angle, seen from the eye, between the target and the predicted gaze point. Before, the eye
  position was taken at the corneal apex after the eye had turned, which is up to 3.5 mm from the position the eye was
  given. Now the position the eye was given is used. Angular errors change in the fourth digit (a mean of 0.10042 deg
  became 0.10031 deg).

- **Added: an eye model and its parts can be converted to a dictionary and back, with `serialize` and `deserialize`.**
  `EyeModel`, `RotationCenter`, `EyeballCenter`, `PupilDecentrationConfig` and `OffAxisPupilConfig` have both methods.
  Saving an eye to a file uses them. `Cornea.deserialize` and `deserialize_pupil` rebuild a cornea or a pupil of the
  class that the saved dictionary names.

- **Removed: `EyeTracker.state` and the `"state"` key of a saved tracker.**
  The dictionary was saved and loaded but never used. The calibration is stored in `algorithm_state`.

- **Removed: the `"focal_length"` key of a saved camera.**
  The focal length is part of the saved camera matrix.

- **Fixed: wrong image size for a camera with a calibrated camera matrix.**
  Before, `CameraMatrix.resolution` returned twice the principal point. That is the image size only when the principal
  point is exactly at the image centre. For the 640 x 480 camera of `examples/08_camera_distortion.py`, whose principal
  point is at (316.7, 259.2), it returned 633 x 518. As a result, `Camera.project` treated points near the image border
  as outside the image, and the centre-of-mass pupil centre was computed on an image of the wrong size. Now the
  resolution is stored together with the matrix. Cameras whose principal point is at the image centre, which includes
  the default camera, give the same images as before.

- **Changed: a `CameraMatrix` has to be created with the image resolution, as `CameraMatrix(matrix, Point2D(width,
  height))`.**
  `CameraMatrix(matrix)` raises `ValueError`. For a calibrated camera whose principal point is not at the image centre,
  the image coordinates also change: they are now measured from the centre of the image; before, they were measured
  from the principal point. Assigning `camera_matrix.resolution` still sets the image size of a pinhole camera and puts
  the principal point at the image centre.

- **Fixed: no glint for a light source at the camera position on a conic or toric cornea.**
  Before, the reflection solver found no glint for a light at the same position as the camera (coaxial illumination),
  or within about a micrometre of it. The default cornea is conic, so this affected the default eye. Now the solver
  handles every light position. Glints that were found before move by less than 1e-9 mm. A spherical cornea was not
  affected.

- **Fixed: back-projecting a pupil contour through a conic cornea ignored its tilt, astigmatism and tear film.**
  `backproject_contour`, `set_pupil_from_contour` and `register_pupil_shape(apparent=True)` convert a pupil outline
  seen in the camera image into the physical pupil behind the cornea. Before, they traced the rays through a symmetric,
  untilted cornea without a tear film, even when the cornea of the eye was tilted, astigmatic or covered by a tear
  film. An outline that was projected to the image and back therefore did not return to its place. It was off by 0.06
  mm for a cornea tilted by 3 deg, 0.007 mm for 0.3 mm of astigmatism, 0.0005 mm for a tear film, and 0.004 mm for the
  `gkaModelEye` cornea. Now it returns to within 3e-11 mm for every cornea. The default cornea is not affected.

- **Fixed: a toric cornea without posterior surface refracted as if it had no astigmatism.**
  Before, with `use_posterior_surface=False`, the refraction through a `ToricCornea` used only its symmetric conic
  shape, so the pupil was imaged as through a cornea without astigmatism. Now the toric shape is used.

- **Fixed: the result of aiming a camera depended on how the camera was positioned or aimed before.**
  Before, `Camera.pan_tilt` ignored a position or orientation that had been assigned to the camera: a camera that was
  positioned and then turned with `pan_tilt` jumped back to the origin. `Camera.point_at` started from the orientation
  left by the previous `point_at`, so aiming at A and then at B gave a camera that was rolled, by 8.5 deg in a test,
  compared with aiming at B directly. Now `pan_tilt` and `point_at` start from the position and orientation that were
  assigned to the camera. A camera that is positioned and aimed once, as in the examples, is not affected.

- **Changed: trackers, cameras, eyes and datasets saved with an earlier version cannot be loaded.**
  The saved dictionaries have new keys, so `deserialize` raises `KeyError` for older ones. A saved tracker loads again
  after `"pupil_center_method": "ellipse"` is added to its dictionary. A saved camera loads again after the keys
  `error_seed`, `fick_projection`, `aim_start_orientation` and `aimed_orientation` are added with the value `null`. The
  dictionary of an eye has a different layout and cannot be converted. Datasets generated with an earlier version have
  to be generated again before `evaluate_gaze_accuracy` can read them.

- **Fixed: a camera aimed with `point_at(eye, mode="fick")` did not behave like other cameras.**
  This mode reproduces the camera projection of gkaModelEye. Before, in this mode, `project` returned pixel coordinates
  measured from the top-left corner of the image instead of its centre, did not mark points outside the image, and did
  not add the noise set with `err`. `unproject` returned wrong points, `backproject_contour` failed, the centre-of-mass
  pupil centre was wrong (22 px from the ellipse centre in a test), and `Camera.orientation` pointed 115 deg away from
  the eye. Now all of these behave as for a pinhole camera. Lens distortion is not supported in this mode and raises
  `ValueError`. The projection itself is unchanged.

- **Fixed: the homography gaze model could match glints to the wrong light sources.**
  The model has to know which glint belongs to which light. Before, it sorted the glints and the lights by their angle
  around their centre and matched them in that order. For some light arrangements this order changed from one image to
  the next, and the glints were then matched to the wrong lights. For four lights in a diamond rotated by 1 deg, the
  mean gaze error was 462 mm; it is 3.9 mm with the fix. Now glints are matched to lights by their index. For light
  arrangements that were not affected, predictions change by at most 0.0001 mm.

- **Removed: `order_points_by_angle` in `homography_normalization_utils`.**
  Only the homography gaze model used it, and that model no longer sorts the glints.

- **Fixed: `Stampe1993GazeModel` only worked with one order of the calibration points.**
  The model treats the centre and the four edge points of the 3 x 3 grid differently from the four corners. Before, it
  assumed that the first five points in the list are the centre and the edge points. A grid given row by row, or
  corners first, failed with `outer points must span all 4 quadrants`. Now the model tells the points apart by their
  position, and every order gives the same result.

- **Fixed: `Stampe1993GazeModel` could not be saved and loaded.**
  Before, `deserialize` raised `TypeError`, and the fitted mapping was not stored. Now it is stored under
  `stampe_model`, and a loaded tracker gives the same predictions. The fitted mapping of a model is its `stampe_model`
  attribute.

- **Changed: `Stampe1993GazeModel` raises `ValueError` for calibration data it cannot use.**
  The model needs the nine points of a 3 x 3 calibration grid. Before, with more than nine points, the points after the
  ninth were ignored without a message. With a calibration point that had no pupil centre or no glint, calibration
  failed with a `TypeError` about `NoneType`. Now these two cases raise `ValueError` with the reason, and so do nine
  points that do not form a centre, four edge points and four corners.

- **Fixed: `EyeTracker.calculate_gaze_error` lost one component of the error for a screen in the x-z plane.**
  The function returns the gaze error along two axes. Before, these were always world x and y. For a screen in the x-z
  plane, which all examples use, the second value was therefore always 0, and the error along z was not reported. Now
  the function returns the error along the two axes of the calibration plane.

- **Fixed: two measures of `compare_algorithms` depended on where the origin of the world coordinate system is.**
  Before, the cosine similarity and the amplitude difference between the predictions of two algorithms were computed on
  vectors from the world origin to the predicted gaze points. The same predictions therefore gave different values in
  different coordinate systems: a cosine similarity of 0.99652 with the origin at the screen centre, and 0.9999997 with
  the origin 6 m away. Now the vectors run from the eye to the predicted gaze point. `cosine_similarity_average` and
  `amplitude_agreement` take the eye positions as a third argument.

- **Fixed: the camera view showed the glints of the first camera in every camera.**
  Before, in `plot_setup_and_camera_view` with several cameras, the glints were computed for the first camera and drawn
  in all camera views. The position of a glint depends on the camera, so the glints drawn for the other cameras were
  wrong (by 22 px for two cameras 400 mm apart). Now each camera view shows its own glints.

- **Removed: the `cr_3d_lists` argument of `plot_camera_view_of_eye`.**
  The glints are read from the camera images. `plot_setup` still draws the 3D reflection points.

- **Fixed: the interactive gaze plot only worked for a screen in the x-z plane.**
  Before, `render_calibration_view`, `create_interactive_gaze_plot` and `plot_setup` assumed that the screen lies in
  the plane y = 0. For a screen in the x-y plane, the calibration points were drawn up to 124.5 mm from their real
  positions, and the arrow keys moved the target out of the screen plane. Now the plots use the plane of the
  calibration points. A screen in the x-z plane is drawn as before.

- **Added: the keyboard controls of the interactive plots work for a screen in any plane, with
  `InteractiveControls(screen_axes=...)`.**
  The interactive plots move the gaze target and the eyes with the keyboard, along the two world axes of the screen.
  Before, these axes were always x and z. `screen_axes` sets them, for example `("x", "y")` for a screen in the x-y
  plane. The default is `("x", "z")`.

- **Changed: `plot_setup` and `plot_setup_and_camera_view` take the calibration points as 3D positions instead of 2D
  points.**
  Before, `calib_points` was a list of 2D points, which were always drawn in the plane y = 0. Now it is the list of 3D
  calibration positions, which are drawn where they are.

- **Fixed: `plot_axis` raised `TypeError` on every call.**
  The function draws a labelled arrow in a 3D plot. Now it works.

- **Fixed: the eye was upside down in seven examples and in the README quickstart.**
  Examples 01, 02, 11, 12, 13, 15 and 16 and the README quickstart placed the eye without giving it a rest orientation,
  which is the direction the eye faces before it looks at a target. The default rest orientation looks along the world
  -z axis. In these scenes +z is up, so the eye started out looking straight down and had to turn about 90 deg to face
  the screen. That left it upside down, with its corneal apex about 19 mm from the position it was given. Now the
  examples call `set_rest_orientation_at_target` first, so the eye starts out facing the screen. Their printed numbers
  change.

- **Fixed: four examples aimed the camera at a displaced eye position.**
  Before, examples 01, 04, 07 and 14 aimed the camera at `eye.position` after `look_at`. `Eye.position` is the position
  of the corneal apex, which has moved by then. Now they aim the camera before the eye turns, or at `eye.placement`.
  For the same reason, example 13 now centres its range of head positions on `eye.placement`. Example 05 gives its
  realistic pupil a seed, so its output is the same in every run. The printed numbers of these examples change.

- **Fixed: the first code example on the "Getting started" documentation page raised `TypeError`.**
  The example built the eye with `Eye(eyelid_enabled=True, ...)`. Since 4.0.0 these settings are given through an
  `EyeModel`, so the example could not be run. Now it builds the eye from an `EyeModel`. The list of examples on the
  same page named files that no longer exist; now it lists the sixteen current examples.

- **Fixed: the 3D setup plot drew the cornea too deep in the eye.**
  Before, every cornea was drawn as a sphere, around a point that is the centre only for a spherical cornea. The
  default conic cornea appeared 0.86 mm too deep in the eye, and the `gkaModelEye` cornea 6.4 mm. Now the plot draws
  the front surface of the cornea itself.

- **Fixed: plotting an eye turned the eye.**
  Before, `prepare_eye_data_for_plots`, `plot_setup_and_camera_view` and the interactive plots called `look_at` on the
  eyes passed to them, so an eye had a different orientation after it was plotted. Now the plots work on copies.

- **Added: the camera view plot can show the pupil centre as an eye tracker computes it, with `use_refraction` and
  `center_method`.**
  `prepare_eye_data_for_plots` and `plot_setup_and_camera_view` take these two new arguments. Before, the camera view
  always computed the pupil with corneal refraction and marked the centre of the ellipse fitted to it. Now it can, for
  example, leave out the refraction or mark the centre of mass.

- **Fixed: `GazeAccuracyPlotter.plot` failed for datasets that vary an eye parameter.**
  Before, for a dataset in which neither the target position nor the eye position is varied, such as a pupil size or
  angle kappa sweep, it raised `ValueError: Variation object must have grid with dx, dy, dz attributes`. Now it plots
  the angular error against the values of the varied parameter.

- **Added: the gaze error can be plotted against a varied eye parameter, with `plot_error_by_parameter`.**
  The function plots the angular gaze error against the values of the parameter that was varied in a dataset, for
  example the pupil diameter or angle kappa. `GazeAccuracyPlotter` uses it for datasets in which neither the target
  position nor the eye position was varied.

- **Fixed: the legend of `plot_error_vectors_2d` called shortened arrow tips "Predicted".**
  The plot draws an arrow from each target to its predicted gaze point, and shortens the arrows when they do not fit.
  Before, with `mark_predicted_positions`, the markers at the arrow tips were labelled "Predicted" also for shortened
  arrows, whose tips are not at the predicted positions (120 mm away in a test). Now the legend says that the markers
  are arrow tips and gives the scale.

- **Fixed: wrong glint size for a light source with a diameter.**
  When a `Light` has a `diameter`, `Camera.take_image` reports the size of its glint in `glint_sizes_px`. Before, that
  size was computed with a wrong mirror formula and a wrong local corneal radius. It was 3 % too large on a spherical
  cornea and 11 to 65 % too small on the default conic cornea. Now it agrees to 0.02 % with the size found by
  reflecting the two edges of the light off the cornea. Glint positions are not affected.

- **Fixed: a light source with a diameter raised `TypeError` on a toric cornea.**
  Before, `Camera.take_image` failed with `Unsupported cornea type: toric`, so no glint size could be computed for the
  `gkaModelEye` eye model. Now the glint size on a toric cornea is computed on the conic that the toric cornea is built
  on, which gives the mean diameter of the elliptical glint to 0.3 %.

- **Changed: the glint size is computed by one function for every cornea type, `compute_glint_diameter`.**
  It replaces `compute_glint_diameter_sphere` and `compute_glint_diameter_conic`, and takes the camera position as its
  third argument.

- **Fixed: the default pupil decentration moved the pupil centre in the wrong horizontal direction.**
  With `PupilDecentrationConfig(enabled=True)` and no coefficients given, an eye uses the mean coefficients of
  Wildenmann & Schaeffel (2013). In that paper the pupils are decentred nasally and are better centred at larger pupil
  sizes, so a dilating pupil moves temporally in both eyes. Before, the simulated pupil centre moved nasally when the
  pupil dilated: the horizontal coefficients were taken over with the sign of the horizontal axis of the paper, which
  is opposite to the eye-local x axis. Now the horizontal coefficients are +0.03 mm/mm for the right eye and -0.03
  mm/mm for the left eye. The vertical direction was correct. Results change for eyes that use the default decentration
  coefficients. Coefficients that are given explicitly are used as given.

- **Fixed: `CorneaThicknessVariation` on a spherical cornea changed the corneal radius instead of the thickness.**
  Before, for a requested thickness of 0.5 mm, the anterior radius became 3.47 mm (the default is 7.98 mm). With
  `scale_with_radius=False`, which is the default, the thickness stayed at 1.15 mm. With `scale_with_radius=True`, as
  in the `et_simul` eye model, the thickness became 0.5 mm, but the whole cornea shrank with it (a corneal depth of
  1.54 mm instead of 3.54 mm). Now the variation changes only the central thickness, and the radii stay as they are.
  Datasets of this sweep on a spherical cornea change. A conic cornea was not affected.

- **Fixed: `SphericalCornea.thickness` returned the distance between the two centres of curvature (0.61 mm) instead of
  the central thickness (1.15 mm).**
  The central thickness is the distance on the optical axis from the apex of the anterior surface to the apex of the
  posterior surface, as in et_simul (`eye_make.m`). Now the property returns it, and assigning to it sets the
  thickness. The shape of the cornea is unchanged.

- **Changed: the linear polynomial of Hoormann et al. (2008) is registered as `hoormann_2008`; before, it was
  `hoorman_2008`.**
  The name of the first author was misspelled in the registered name, in the constant (`HOORMANN_2008`, before
  `HOORMAN_2008`) and in the description of the polynomial. Code that selects the polynomial by its name needs the new
  spelling; the old name is no longer registered. The reference lists in the code and in the documentation gave the
  year of the paper as 2007. It is 2008: Hoormann, Jainta & Jaschinski (2008), Journal of Eye Movement Research,
  1(2):3.

- **Fixed: messages from the worker processes of a dataset generation were missing from the log file.**
  `DataGenerationStrategy.execute` images the measurements in worker processes. On macOS and Windows a worker process
  starts without the log file of the main process. Before, what a worker logged, such as the warnings from the imaging
  and the lines about missing glints at `LogLevel.DEBUG`, was printed but not written to the file set with
  `set_log_file`: in a test with 5 missing glints, the log file held none of the 5 lines. Now the workers get the log
  file of the process that starts them and append to it.

- **Fixed: an `info()` call without a message wrote an empty entry to the log file.**

- **Fixed: a grid with one point on an axis put that point at the lower end of the range.**
  Before, with `grid_size=[1, 1, 3]` and `dx=[-100, 100]`, `TargetPositionVariation` and `EyePositionVariation` placed
  their single x position 100 mm to one side of the grid centre. Now a single point lies at the middle of the range,
  here at the grid centre, as for a `GenericEyeVariation` with one step. An axis whose two limits are equal is not
  affected.

- **Fixed: `GlintNoiseConfig` could reject a valid covariance matrix.**
  A covariance matrix for perfectly correlated x and y noise is singular: one of its eigenvalues is zero. Before, when
  that eigenvalue came out as a negative number of rounding size (about -1e-16), the matrix was rejected as not
  positive semidefinite. In a test this happened for 54 of 361 such matrices. Now a negative eigenvalue of rounding
  size counts as zero.

- **Fixed: changing an eye model returned by `get_eye_model` changed the registered model for the whole process.**
  Before, `get_eye_model` returned the registered object itself. A line such as
  `get_eye_model("PyEtSimul").cornea.anterior_radius = 5.0` changed the default eye model for every eye built
  afterwards in the same process. Now `get_eye_model` returns a copy, and `register_eye_model` stores a copy of the
  model it is given.

- **Changed: `register_eye_model` raises `ValueError` for a name that is already registered.**
  Before, the new model replaced the registered one without a message, also for the built-in names `PyEtSimul`,
  `et_simul` and `gkaModelEye`. Now a registered model is replaced only with `register_eye_model(name, model,
  overwrite=True)`.

- **Fixed: `Eye.orientation` returned a changed copy of the stored orientation, and failed for a left-handed one.**
  Before, the getter normalised the columns of the stored matrix. The returned matrix could differ from the stored one
  in its last digit (2e-16), a matrix with a scale in it was returned as if it had none, and a left-handed orientation,
  which can be assigned, raised `ValueError` when it was read back. Now the getter returns a copy of the stored matrix
  as it is, and raises `ValueError` when that matrix is not orthonormal.

- **Fixed: `run_calibration` left the eye looking at the last calibration point.**
  Calibration turns the eye to every calibration point. Before, the eye that was passed to `run_calibration` stayed at
  the last point afterwards, with that point as its gaze target, so its position and orientation had changed (by 3 mm
  in a test). Now calibration turns a copy, and the eye that was passed in keeps its pose and its target.

- **Fixed: the calibration warning about failed points looked only at the glint of the first light.**
  Before, with several lights, a calibration point at which the glint of the second or a later light was missing was
  not listed in the warning and not in `failed_calibration_points`. In a test with two lights, the glint of the second
  light was missing at 1 of 9 points and no point was reported. Now a point is reported when the glint of any light is
  missing, and the warning names the light.

## [5.1.0] - 2026-09-10

- **Changed: the default rotation centres lie 14.8 mm (horizontal) and 12.2 mm (vertical) behind the corneal apex.**
  The horizontal (azimuth) rotation centre lies 14.8 mm and the vertical (elevation) rotation centre 12.2 mm behind the
  corneal apex, the values measured by Fry & Hill (1962, 1963). They were 14.7 mm and 12.0 mm. Eye positions after a
  rotation shift by up to about 0.1 mm at large gaze angles. The `gkaModelEye` eye model keeps its own values.

- **Fixed: the Fick rotation centres were credited to the wrong paper.**
  They were credited to Aguirre (2019). That paper, on the entrance pupil, states that it does not study eye rotation.
  They are now credited to Aguirre (2021).

## [5.0.1] - 2026-09-09

- **Fixed: the documentation failed to build on Read the Docs since 4.0.0.**
  The two guide pages in `docs/guides` were symbolic links to example files that were removed in 4.0.0, when the
  examples became the numbered 01 to 16 series. The guides are now documentation pages of their own. The gaze-model
  guide explains how to register a polynomial, which is what the theory pages that link to it refer to.

- **Removed: the `description` argument of `GenericEyeVariation`.**
  It was stored and never used; `describe()` builds its text from the parameter path. Passing the argument raises
  `TypeError`.

## [5.0.0] - 2026-09-09

- **Changed: a warning is issued when a pupil centre or diameter cannot be fitted.**
  When the fit fails, another quantity is reported in its place. Each such case now warns, naming the input that could
  not be fitted and what was reported instead, so that a failed fit cannot be mistaken for a fitted result.

- **Changed: the default cornea is a conic instead of a sphere.**
  `EyeModel` builds a `ConicCornea` instead of a `SphericalCornea`. The front surface of the default cornea is
  aspheric, with a radius of 7.76 mm and a conic constant of -0.10. It was a sphere with a radius of 7.98 mm.

- **Changed: `ConicCornea` models its back surface by default.**
  `use_posterior_surface` defaults to `True`, so light is refracted at both surfaces of the cornea.
  `use_posterior_surface=False` gives a cornea with a single surface.

- **Changed: the pupil is off the optical axis by default.**
  `OffAxisPupilConfig.enabled` defaults to `True`. The pupil centre lies nasally and superiorly of the optical axis
  (Wyatt 1995). `enabled=False` puts the pupil back on the axis.

Together, the three new defaults make the default eye anatomically closer to a real eye. They change every result that
relies on the defaults: the apparent pupil is at its off-axis position and the corneal magnification changes. The
previous behaviour is obtained with `EyeModel(cornea=SphericalCornea(),
off_axis_pupil=OffAxisPupilConfig(enabled=False))`. The `et_simul` and `gkaModelEye` eye models set their own cornea
and pupil position, so their validation against MATLAB is not affected.

- **Fixed: the pupil ellipse fit failed under numpy 2.5.**
  PyEtSimul used `EllipseModel` of scikit-image to fit an ellipse to the pupil boundary. Under numpy 2.5 that fit
  raises `TypeError` for every input. The pupil centre was then replaced, without a message, by the mean of the
  boundary points or of the convex-hull vertices. For a smooth pupil that substitute is close to the ellipse centre;
  for a boundary traced from a real image it differs by several pixels. The ellipse is now fitted inside PyEtSimul
  (`fit_ellipse`). The result agrees with scikit-image 0.25.2 to floating-point precision (centre and axes to 1.4e-13,
  angle to 3.2e-12 deg).

- **Fixed: the Cerrolaza et al. (2008) asymmetric polynomial contained its constant term twice.**
  The horizontal polynomial had five terms where the published model has four. The fitted gaze mapping was still
  correct, because the fit absorbs the duplicate term, but the polynomial did not match the cited model.

- **Fixed: calibration points could fall outside the 3D setup plot.**
  The limits of the plot did not include the calibration points, because the test that should include them never ran.
  Calibration points outside the rest of the scene were cut off. The limits now include them, at their positions on the
  screen plane.

## [4.0.0] - 2026-07-06

- **Added: the specification of an eye is one object, `EyeModel`.**
  An `EyeModel` holds the cornea, angle kappa, the rotation centre, the look-at method and the pupil, and cannot be
  modified after it is created. `EyeModel.copy(**overrides)` returns a model with the given fields changed.
  `Eye.copy(**overrides)` returns a new eye at the same position, built from such a model.

- **Added: eye models can be selected by name, with `get_eye_model`.**
  `get_eye_model`, `register_eye_model` and `list_eye_models` give access to a registry of named eye models.
  `"PyEtSimul"`, `"et_simul"` and `"gkaModelEye"` are built in.

- **Added: the `et_simul` eye model, validated against the MATLAB original.**
  It is the eye of Böhme et al. (2008). A test compares its pupil centre and glint at each gaze target with the output
  of et_simul in MATLAB; they agree to within 0.001 px.

- **Added: the `gkaModelEye` eye model, validated against the MATLAB original.**
  A test compares its pupil centre and glint at each gaze target with the output of gkaModelEye in MATLAB; they agree
  to within 0.05 px.

- **Added: a camera can use the projection of gkaModelEye, with `Camera.point_at(eye, mode="fick")`.**
  The call aims the camera at the horizontal (azimuth) rotation centre of the eye and replaces the pinhole projection
  with the projection of gkaModelEye, in which the rotation is composed of torsion, elevation and azimuth angles. The
  `gkaModelEye` validation uses it. Lens distortion is not applied in this mode.

- **Added: a 2D cross-section of the eye can be plotted, with `plot_eye_cross_section`.**
  The plot shows the cornea, the eyeball, the pupil and the optical and visual axes, computed from the geometry of the
  eye.

- **Added: the roll of the resting eye can be set, with the `up` argument of `set_rest_orientation_at_target`.**
  The optional `up` vector sets the roll of the eye about its gaze direction when the eye is rested facing a target.

- **Added: a new series of examples, numbered 01 to 16.**
  They form one ordered path: scene basics, the eye, cornea, pupil, eyelid, glint and camera models, binocular setups,
  dataset generation and custom parameter variations, calibration and gaze-accuracy evaluation with interactive plots,
  and custom eye and gaze models.

- **Changed: the specification of an eye is given as an `EyeModel`, not as keyword arguments of `Eye`.**
  An eye is created with `Eye(model=..., which_eye=...)`.

- **Changed: the look-at method is set on the eye model.**
  `EyeModel.look_at_method` takes `"visual_axis"` (the default), `"line_of_sight"`, `"optical_then_kappa"` or the new
  `"optical_axis_target_direction"`. `look_at(target, method=...)` overrides it for one call. This replaces
  `Eye.aiming` and the `legacy` argument of `look_at`. `look_at_target(eye, target, method)` covers all four methods;
  `look_at_target_optical_then_kappa` and `look_at_target_line_of_sight` are removed.

- **Changed: the default eye turns about two anatomical rotation centres (Fick) instead of one fixed centre.**
  `EyeModel.rotation_center` takes an `EyeballCenter` (one fixed rotation centre, at the origin of the eye's coordinate
  system) or a `RotationCenter`; `EyeballCenter()` replaces `None`. The default was the fixed centre. It is now a
  `RotationCenter` with the horizontal (azimuth) centre 14.7 mm behind the corneal apex and 0.79 mm nasal, and the
  vertical (elevation) centre 12.0 mm behind the apex and 0.33 mm superior. The eye turns about the two centres one
  after the other (Fick order). With the default model, `look_at` therefore moves the eye as well as turning it.
  Afterwards, `Eye.position` returns the moved position, and `Eye.placement` returns the position the eye was given.
  The `et_simul` model keeps the fixed centre.

- **Changed: changing the corneal radius no longer moves the corneal apex or rescales the eye.**
  `placement_convention` keeps either the corneal apex (`"apex"`, the default) or the centre of curvature (`"center"`)
  at a fixed position in the eye. `scale_with_radius` (default `False`) sets whether the posterior radius, the
  thickness and the corneal depth scale with the anterior radius. In 3.6.0 the cornea was always positioned by its
  centre of curvature, and a spherical cornea always scaled; the `et_simul` model keeps this. With the defaults,
  changing the corneal radius no longer moves the apex or rescales the eye.

- **Changed: `Eye.position` is the position of the corneal apex; before, it was the position of the eyeball centre.**
  `Eye.position` sets and returns the world position of the corneal apex. An eye given the same `Eye.position` as in
  3.6.0 therefore sits further back, by the distance from its corneal apex to its eyeball centre (12.33 mm for the
  default eye of 3.6.0). For a cornea with the `"center"` placement convention, as in the `et_simul` model, the origin
  stays at the eyeball centre.

- **Changed: the module with the default eye model is renamed to `PyEtSimul`.**

- **Changed: `plot_camera_view_of_eye` accepts a rendered image and more styling options.**
  It accepts a plain rendered image, in addition to the data of a 3D setup. It supports styling per camera and per eye
  and zooming to the eye features, and colours each glint by its light source. The default camera colours are a set
  that is easier to tell apart.

- **Changed: an elliptical pupil has 100 boundary points by default, instead of 20.**
  The default was 20. It now matches the default of the `EllipticalPupil` class.

- **Changed: the README has a feature list, a quickstart, an index of the examples and a note on validation.**

- **Fixed: the glint on a `ToricCornea` ignored the astigmatism.**
  The reflection was computed on the rotationally symmetric conic, without the two different radii of the toric
  surface, while the pupil was already refracted through the toric surface. The glint position on a toric cornea
  changes. Spherical and conic corneas are not affected.

- **Fixed: `look_at` could give an invalid orientation for a target almost exactly in the rest gaze direction.**
  When the direction to the target was within about 1e-9 rad of the rest direction, but not identical to it, the
  computed rotation matrix could be non-orthonormal. No rotation is applied for such directions now.

- **Removed: the old example scripts, `examples/experiments/` and `validation_and_scientific_exploitation/`.**
  They are replaced by the numbered examples and the MATLAB tests in `validation/`. The removed files remain in the git
  history and on the ETRA 2026 branch.

## [3.6.0] - 2026-06-22

- **Added: the cornea can refract light at its back surface as well, with `use_posterior_surface`.**
  The cornea can refract light at its front and its back surface (air to cornea, then cornea to aqueous humour), not
  only at the front. It is enabled per cornea with `use_posterior_surface`.

- **Added: a cornea with astigmatism, `ToricCornea`.**
  A conic cornea whose front surface has two different radii along its two principal meridians, at a configurable axis.
  The defaults are the population mean of young adults (with-the-rule, about 0.1 mm or 0.6 D, axis 90 deg). It is
  selected with `create_cornea("toric", ...)`.

- **Added: an optional tear film in front of the cornea.**
  A thin tear film is modelled as a third refracting surface in front of the cornea (air to tears to cornea). It lowers
  the corneal magnification of the pupil. It is off by default and requires the two-surface cornea.

- **Added: the cornea can be tilted, with `ConicCornea.tilt_x_deg` and `tilt_y_deg`.**
  The whole cornea is tilted about its apex, away from the eye axis. This changes both the refraction of the pupil and
  the reflection of the glint.

- **Added: a pupil can be elliptical or have a measured shape.**
  `EllipticalPupil` has an aspect ratio and an orientation. The new `ContourPupil` takes any measured boundary, so the
  contour of a real pupil can be used instead of a circle.

- **Added: the pupil can be set by the size it has in the camera image.**
  The pupil can be set by the size and shape it has in the camera image. The physical pupil behind the cornea is
  computed from it with the corneal magnification.

- **Added: the pupil plane can shift in depth when the pupil size changes, with `z_coeff`.**
  The coefficient `z_coeff` of the decentration model moves the pupil plane along the eye axis as the pupil size
  changes. With 0, the pupil centre shifts only within its plane.

- **Added: the pupil can sit off the optical axis by a fixed amount, with `OffAxisPupilConfig`.**
  The pupil centre can be displaced from the optical axis by a fixed amount that does not depend on the pupil size. The
  defaults are the population mean of Wyatt (1995), 0.27 mm nasal and 0.20 mm superior, mirrored between the two eyes.
  It is off by default, and is added to the size-dependent pupil decentration.

- **Added: the rotation centre of the eye can depend on the gaze direction, with `RotationCenter`.**
  An eye with a `RotationCenter` turns about a point whose depth behind the corneal apex depends on the gaze direction,
  instead of about the eyeball centre. `look_at` then moves the eye as well as turning it, so `Eye.position` changes
  after a `look_at`. The new `Eye.placement` returns the position the eye was given. Off by default.

- **Added: horizontal and vertical eye rotation can have separate rotation centres (Fick).**
  The horizontal (azimuth) and the vertical (elevation) rotation centre can have different depths, optional nasal and
  superior offsets, and a vertical depth that varies with gaze. In the `fick` mode, the eye turns about the two centres
  one after the other (Fry & Hill 1962, 1963; Aguirre 2019). With both centres equal and `fick` off, the result is
  exactly that of a single fixed centre.

- **Added: `look_at` can aim the line of sight at the target instead of the visual axis, with `Eye.aiming`.**
  With `aiming="line_of_sight"`, `look_at` aims the line from the fovea through the pupil centre at the target, instead
  of the visual axis. The eye then re-aims when the pupil centre shifts. The default, `"visual_axis"`, keeps the
  previous behaviour.

- **Added: the pupil plane can be tilted and the eye can have torsion.**
  `Eye.pupil_tilt_x_deg` and `pupil_tilt_y_deg` tilt the pupil disc about its centre, as a shear along z.
  `Eye.torsion_deg` adds a roll about the line of sight on top of Listing's law.

- **Added: decentration coefficients can be computed from a measured pupil-centre shift, with
  `Eye.solve_decentration_from_apparent`.**
  Given a measured shift of the pupil centre in the image per unit change of the apparent pupil diameter, the function
  finds the decentration coefficients of the eye that reproduce this shift. `calculate_pupil_diameter_from_boundary` is
  added to `pupil_imaging`.

- **Added: the pupil centre can be computed from the convex hull of the pupil boundary, with
  `center_method="convex_hull"`.**
  The ellipse is fitted to the convex hull of the pupil boundary. A notch in the boundary, such as one cut by a glint,
  then does not pull the centre toward it.

- **Added: single points inside the eye can be projected to the camera image and back, with functions in
  `optics.pupil_imaging`.**
  `project_point_to_image` projects a point inside the eye to the camera image, with corneal refraction or without it
  (`use_refraction=False`). `reverse_project_to_pupil_plane` finds the points on the pupil plane that are seen at given
  image positions. `image_jacobian` gives the Jacobian of the image position with respect to the position of a point in
  the eye.

- **Changed: batch corneal refraction traces any stack of refracting surfaces.**
  It was limited to a cornea with two surfaces. The front conic surface accepts toric (astigmatic) coefficients; the
  defaults are rotationally symmetric.

- **Changed: corneal refraction is solved for the whole pupil boundary at once.**
  The batch solvers in `optics.refraction_batch` replace the per-point functions `find_refraction_sphere`,
  `find_refraction_conic`, `find_refraction_dual_sphere` and `find_refraction_dual_conic`, which are removed.

- **Changed: `refract_direction` is a public function of the optics module.**

- **Changed: the side of an eye is set with `Eye(which_eye=...)`, `"left"` or `"right"` (the default).**
  It sets the sign of the horizontal angle kappa, whose magnitude is `fovea_alpha_deg`, and selects the default
  pupil-decentration coefficients of that eye. `PupilDecentrationConfig` no longer takes `which_eye`.

- **Fixed: `set_pupil_diameter(apparent=True)` gave an apparent pupil size that was a few percent off.**
  The requested apparent diameter was divided by the corneal magnification once. As the magnification depends on the
  pupil size, the pupil in the image came out a few percent off the request. The physical diameter is now found by
  iteration, and the apparent size matches the request to within 1e-4 mm.

- **Fixed: `look_at` used the same sign of angle kappa for both eyes.**
  The horizontal angle kappa of the left eye points the opposite way to that of the right eye. `look_at_target` and
  `look_at_target_optical_then_kappa` built the visual axis from the unsigned `fovea_alpha_deg`. Both now use the
  signed angle of the eye.

- **Fixed: three pupil-centre functions raised `AttributeError` for every visible pupil.**
  `get_pupil_ellipse_image`, `get_pupil_center_mass_image` and `calculate_pupil_center_methods` passed the pupil
  boundary on in the wrong data type. Now they return the pupil centre.

- **Fixed: the ellipse fit emitted deprecation warnings under scikit-image 0.26.**
  scikit-image 0.26 deprecates `EllipseModel().estimate()` and `EllipseModel.params`, so every fit emitted a
  `FutureWarning`. The fit uses `EllipseModel.from_estimate`, and PyEtSimul requires scikit-image 0.26 or later.

- **Removed: `RealisticPupilParams.pupil_offset_from_limbus`.**
  The field existed but was never applied. `OffAxisPupilConfig` models the off-axis position of the pupil.

- **Removed: `refract_ray_dual_surface`, the spherical two-surface refraction.**
  Nothing called it. Two-surface refraction is done by `refraction_batch`.

## [3.5.0] - 2026-05-30

- **Changed: `compute_calibration_errors` is public, and the calibration view is drawn by the new
  `render_calibration_view`.**

## [3.4.0] - 2026-05-19

- **Changed: the Stampe (1993) gaze model is provided by the `stampe1993-gaze-mapping` package.**
  The example comes with an HV9 (nine-point) setup.

- **Removed: the EyeLink 1000 Plus gaze model.**
  It could not be verified against HREF P-CR data. The Stampe model takes its place.

## [3.3.0] - 2026-05-16

- **Fixed: the Stampe (1993) model fitted its polynomial on all nine calibration points.**
  The biquadratic polynomial was fitted on all nine points, and the corner correction was derived from its residuals.
  The corner points therefore pulled the polynomial, and its coefficients differed from those an EyeLink 1000 Plus
  stores. The polynomial is now fitted on the centre and the four edge points, and the corner correction on the four
  corner points, as in Stampe (1993).

## [3.2.5] - 2026-03-31

- **Added: the DOI of the ETRA 2026 paper in the citation and the badges.**

## [3.2.1] - [3.2.4] - 2026-03-17

- **Added: a camera can be pointed at a pair of eyes, with `Camera.point_at_binocular`.**
  It points the camera at the midpoint between two eyes, rolled so that both eyes are at the same height in the image.

- **Added: citation information, an acknowledgment and the PyPI publish workflow.**

- **Fixed: wrong glint and pupil on a conic cornea for a rotated eye.**
  The solvers for the reflection and the refraction on a conic cornea took the axis of the cornea as the world z-axis,
  which holds only for an eye that is not rotated. They also searched for the reflection and refraction point only in
  the plane through the camera, the centre of the conic and the light or the pupil point. They now work in the
  coordinate system of the eye and also find points outside that plane.

- **Fixed: packaging.**
  The version is read from the package metadata, the README is the description on PyPI, and the project URLs and the
  URL of the funding image are set.

## [3.1.0] - 2026-03-01

- **Added: EyeLink 1000 Plus gaze model with HREF preprocessing.**

## [3.0.0] - 2026-02-24

The version presented in the ETRA 2026 paper.
