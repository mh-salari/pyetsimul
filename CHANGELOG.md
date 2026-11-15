# Changelog

All notable changes to PyEtSimul are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version 3.0.0 is the release presented in the
ETRA 2026 paper; this log records everything since.

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

## [5.1.0] - 2026-09-10

### Changed

- **The default rotation-centre depths are the measured ones.** Azimuth 14.8 mm and elevation 12.2 mm behind
  the corneal apex (Fry & Hill 1962, 1963), replacing 14.7 mm and 12.0 mm. Gaze-dependent eye positions shift
  by up to about 0.1 mm at large eccentricities. `gkaModelEye` keeps the values that model sets.

### Fixed

- **The Fick rotation centres are credited to Aguirre (2021), not (2019).** The 2019 entrance-pupil paper
  states that rotation is not studied there.

## [5.0.1] - 2026-09-09

### Fixed

- **The two guide pages are rebuilt as documentation rather than symlinks.** `docs/guides` held symlinks into
  `examples/`, whose targets were deleted when the examples became the ordered 01-16 series. Sphinx could not read
  them, and with `fail_on_warning` set that failed every Read the Docs build since 4.0.0. The guides are now
  reStructuredText pages in `docs/`, and the gaze-model guide documents polynomial registration, which is what the
  theory pages linking to it describe.

### Removed

- **`GenericEyeVariation` no longer takes a `description` argument.** It was stored and never read; `describe()`
  builds its string from the parameter path. Passing it now raises `TypeError`.

## [5.0.0] - 2026-09-09

### Fixed

- **The pupil centre is fitted in-package rather than by scikit-image.** `fit_ellipse` solves the ellipse
  directly under the conic constraint, on coordinates conditioned before the solve, and `_safe_ellipse_center`,
  `_fit_ellipse_center`, `_fit_convex_hull_center` and `calculate_pupil_diameter_from_boundary` all read from it.
  It reproduces scikit-image 0.25.2 to floating point on synthetic and measured boundaries: centre and axes to
  1.4e-13, angle to 3.2e-12 degrees. Under numpy 2.5 scikit-image's `EllipseModel` raises `TypeError` for every
  input, including a noiseless ellipse, because it applies `%=` to an angle that comes back complex; the fit path
  therefore failed on every call and the centre silently became a boundary or hull-vertex centroid. On a smooth
  pupil that substitute is close enough to pass unnoticed; on a boundary traced from a real image it differs by
  several pixels.
- **The Cerrolaza et al. (2008) asymmetric polynomial listed its constant term twice.** The horizontal row
  carried five terms where the published model has four, so the design matrix held a duplicate column and the
  fitted mapping spent a coefficient on it. The pseudo-inverse absorbed the rank deficiency, leaving the mapping
  itself correct, but the descriptor did not match the cited model.
- **Calibration points are included in the setup plot bounds.** The bounds test asked a two-column array for
  three columns, so it never ran and calibration points outside the rest of the scene fell off the axes. The
  points are now placed on the screen plane, where they are drawn.

### Changed

- **A pupil centre or diameter that falls back now warns.** Each fallback says which input could not be fitted
  and what was reported instead, so a failed fit can no longer be mistaken for a fitted result.
- **The default cornea is a conic.** `EyeModel` builds a `ConicCornea` rather than a `SphericalCornea`, so the
  default eye refracts at an aspheric anterior surface (R 7.76 mm, k -0.10) instead of a sphere at 7.98 mm.
- **`ConicCornea` models its posterior surface by default.** `use_posterior_surface` defaults to `True`, so
  refraction is solved at both corneal surfaces; pass `False` for a single-surface cornea.
- **The static off-axis pupil is applied by default.** `OffAxisPupilConfig.enabled` defaults to `True`, placing
  the pupil centre nasally and superiorly off the optical axis (Wyatt 1995) rather than on it; pass `False` to
  put the pupil back on the axis.

Together these make the default eye anatomically closer to a real one, and they move every result built on the
defaults: the apparent pupil sits at the static offset and the corneal magnification changes. Code that needs the
previous behaviour states it explicitly, as
`EyeModel(cornea=SphericalCornea(), off_axis_pupil=OffAxisPupilConfig(enabled=False))`. The named `et_simul` and
`gkaModelEye` models set their own cornea and off-axis pupil, so their MATLAB validations are unaffected.

## [4.0.0] - 2026-07-06

### Added

- **The eye specification is a first-class `EyeModel`.** An immutable `EyeModel` holds the cornea, angle kappa,
  rotation geometry, gaze convention and pupil; `EyeModel.copy(**overrides)` derives a variant, and `Eye.copy`
  derives a placed eye with model overrides.
- **Named eye-model registry.** `get_eye_model`, `register_eye_model` and `list_eye_models` select models by
  name, with `"PyEtSimul"`, `"et_simul"` and `"gkaModelEye"` built in.
- **`et_simul` eye model with a 1:1 MATLAB validation.** The original Böhme (2008) eye as a named model, with a
  MATLAB reference and a test asserting the pupil centre and glint at each gaze target.
- **`gkaModelEye` eye model with a 1:1 MATLAB validation.** An independently built schematic eye as a named
  model, validated against its MATLAB reference to the iterative-solver floor.
- **`FickRotation` camera projection.** `Camera.point_at(eye, mode="fick")` aims the camera at the eye's
  rotation centre via a Fick-rotation projection.
- **`plot_eye_cross_section` visualization.** A 2D cross-section of the eye (cornea, globe, pupil, optical and
  visual axes), derived from the eye's own geometry.
- **World-up control for `set_rest_orientation_at_target`.** An optional `up` vector sets the eye's roll when it
  is rested facing a target.
- **New examples series (numbered `01` to `16`).** One ordered, self-contained learning path: scene basics, the
  eye, cornea, pupil, eyelid, glint and camera models, binocular setups, dataset generation and custom parameter
  variations, calibration and gaze-accuracy evaluation with interactive plots, and building custom eye and gaze models.

### Changed

- **The eye specification lives in `EyeModel`, not in `Eye` keyword arguments.** An eye is constructed as
  `Eye(model=..., which_eye=...)`.
- **The look-at method is an `EyeModel` property.** `look_at` is unified over `visual_axis`, `line_of_sight`,
  `optical_then_kappa` and `optical_axis_target_direction`, chosen on the model, with a single `look_at_target`.
- **The rotation centre is modelled as data on the eye model,** defaulting to anatomical Fick centres rather than
  a single hard-coded globe centre.
- **Corneal placement and radius-scaling are per-cornea conventions,** carried by each cornea rather than a global
  setting.
- **The eye-local origin is anchored at the corneal apex.**
- **The default eye-model module is renamed to `PyEtSimul`.**
- **`plot_camera_view_of_eye` reworked.** It accepts a plain rendered image (not only 3D-setup data), supports
  per-camera and per-eye styling and zoom-to-features, and colours each glint by its light; the camera palette is
  now distinct.
- **Default elliptical-pupil boundary points raised to 100** (from 20), matching the `EllipticalPupil` class
  default.
- **README** gains a Features list, a Quickstart, an examples index, and a validation note.

### Fixed

- **The glint reflects off the toric anterior surface** rather than the conic base.
- **Near-parallel rotation axes** return the identity rotation instead of an unstable result.

### Removed

- **The ad-hoc example scripts, `examples/experiments/`, and `validation_and_scientific_exploitation/`,**
  superseded by the numbered examples series and the `validation/` MATLAB tests (preserved in git history and the
  ETRA 2026 branch).

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
