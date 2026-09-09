Custom Parameter Variations
===========================

A variation sweeps one parameter over a range of values. ``DataGenerationStrategy`` renders the
eye once per value, so a sweep produces one measurement per step. PyEtSimul ships variations for
the common eye parameters, and ``GenericEyeVariation`` sweeps any other parameter by name, so most
custom sweeps need no new class.

Sweeping Any Parameter
----------------------

For example, to sweep the conic constant of the cornea from -0.1 to -0.5 in five steps:

.. code-block:: python

   from pyetsimul.simulation import GenericEyeVariation

   cornea_k_sweep = GenericEyeVariation("cornea.anterior_k", value_range=[-0.1, -0.5], num_steps=5)

The three arguments are the parameter to vary, the range, and the number of steps. Values are
spaced evenly with ``numpy.linspace``, so this sweep uses -0.1, -0.2, -0.3, -0.4 and -0.5. A
``num_steps`` of 1 is a special case: it uses the midpoint of the range instead of its start.

Parameter Paths
---------------

The first argument names the attribute to set on the ``Eye``. It is resolved in one of three ways:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Effect
   * - ``"pupil_diameter"``
     - calls ``eye.set_pupil_diameter(value)``
   * - ``"cornea.anterior_k"``
     - sets the attribute on the named sub-object, here ``eye.cornea``
   * - ``"fovea_alpha_deg"``
     - sets the attribute directly on the eye

A dotted path goes one level deep. The name before the dot must be an attribute of the ``Eye``,
and the name after it an attribute of that object. ``"cornea.anterior_radius"`` and
``"cornea.anterior_k"`` both work this way.

An unknown path raises ``AttributeError`` when the sweep runs, not when the variation is created.

Running a Sweep
---------------

``DataGenerationStrategy`` holds the scene, and ``execute()`` runs one variation over it:

.. code-block:: python

   from pyetsimul.core import Camera, ConicCornea, Eye, Light
   from pyetsimul.core.eye_model import get_eye_model
   from pyetsimul.simulation import DataGenerationStrategy, GenericEyeVariation
   from pyetsimul.types import Position3D

   if __name__ == "__main__":
       eye = Eye(model=get_eye_model("PyEtSimul").copy(cornea=ConicCornea()))
       eye.position = Position3D(0.0, 700.0, 50.0)

       camera = Camera()
       camera.position = Position3D(0.0, 350.0, -150.0)
       camera.point_at(eye.position)

       light = Light(position=Position3D(70.0, 350.0, -140.0))

       data_gen = DataGenerationStrategy(
           eyes=[eye],
           cameras=[camera],
           lights=[light],
           gaze_target=Position3D(0.0, 0.0, 0.0),  # the eye holds this gaze while the parameter varies
           experiment_name="cornea_k_sweep",
           save_to_file=False,
       )

       sweep = GenericEyeVariation("cornea.anterior_k", value_range=[-0.1, -0.5], num_steps=5)
       result = data_gen.execute(sweep)

Sweeping a parameter only makes sense against a fixed gaze, so pass ``gaze_target`` to hold the
eye still. Leave it as ``None`` when the variation supplies its own targets, as
``TargetPositionVariation`` does.

``save_to_file=False`` keeps the dataset in memory. With the default of ``True`` it is written
under ``output_dir``, named after ``experiment_name``. To reuse one strategy for several sweeps,
call ``data_gen.set_experiment_name()`` between them so the datasets do not overwrite each other.

``execute()`` renders the steps in parallel with a ``multiprocessing.Pool``. On macOS and Windows
the default start method is spawn, which re-imports the main module in every worker, so the call
must sit behind an ``if __name__ == "__main__":`` guard. Without it the workers re-run the script.

Reading the Results
-------------------

``execute()`` returns a nested dictionary, ordered by camera and then by eye:

.. code-block:: python

   measurements = result["data"]["cameras"][0]["eyes"][0]["measurements"]

   for m in measurements:
       print(m["parameter_value"], m["pupil_center"], m["corneal_reflections"])

There is one entry per step, in sweep order. Each carries the ``parameter_value`` that produced it
alongside the rendered ``pupil_center``, ``pupil_boundary`` and ``corneal_reflections``, all in
pixels.

Built-in Variations
-------------------

Import these from ``pyetsimul.simulation``. Reach for ``GenericEyeVariation`` only when the
parameter you want is not covered here.

Eye parameters:

- ``PupilSizeVariation``
- ``PupilSizeWithDecentrationVariation``
- ``PupilDecentrationVariation``
- ``AngleKappaVariation``
- ``CorneaRadiusVariation``
- ``CorneaThicknessVariation``

Positions in space:

- ``EyePositionVariation``
- ``TargetPositionVariation``

``ComposedVariation`` and ``SequentialVariation`` combine several variations into one sweep.

Writing a Variation Class
-------------------------

``GenericEyeVariation`` covers any single parameter, so write a class only when you want a
reusable name, a description of your own, or a value that is not a single number set on one
attribute. Subclass ``GenericEyeVariation`` and override ``describe()``:

.. code-block:: python

   class ConicCorneaKVariation(GenericEyeVariation):
       """Sweep the conic constant of the anterior cornea."""

       def __init__(self, k_range: list[float], num_steps: int = 10):
           super().__init__("cornea.anterior_k", k_range, num_steps)

       def describe(self) -> str:
           min_val, max_val = self.value_range
           return f"conic cornea K parameter {min_val:.3f} to {max_val:.3f} ({self.num_steps} steps)"

``describe()`` is what the sweep prints and what labels the saved dataset. The default reports the
parameter path, the range and the step count.

To vary something that is not one attribute, subclass ``EyeParameterVariation`` instead and
implement four methods: ``generate_values()`` to yield the values, ``apply_to_eye(eye, value)`` to
apply one of them, ``__len__()`` for the step count, and ``describe()``.

.. code-block:: python

   import numpy as np

   from pyetsimul.simulation import EyeParameterVariation


   class PupilAndCorneaVariation(EyeParameterVariation):
       """Widen the pupil and flatten the cornea together."""

       def __init__(self, num_steps: int = 5):
           super().__init__("pupil_and_cornea")
           self.num_steps = num_steps

       def generate_values(self):
           yield from np.linspace(0.0, 1.0, self.num_steps)

       def apply_to_eye(self, eye, value: float) -> None:
           eye.set_pupil_diameter(2.5 + 4.5 * value)
           eye.cornea.anterior_k = -0.1 - 0.4 * value

       def __len__(self) -> int:
           return self.num_steps

       def describe(self) -> str:
           return f"pupil and cornea together ({self.num_steps} steps)"

``apply_to_eye()`` mutates the eye in place and returns nothing. The value it receives is whatever
``generate_values()`` yielded, so a variation is not restricted to floats.

Complete Example
----------------

``examples/16_custom_variation.py`` runs a built-in pupil-size sweep and a custom conic-constant
sweep on the same scene, and reports how far the rendered pupil and glint move across each.
