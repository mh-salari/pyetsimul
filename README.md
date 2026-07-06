# PyEtSimul: An Open-Source Python Software Framework for Eye Tracking Simulation

[![PyPI version](https://img.shields.io/pypi/v/pyetsimul)](https://pypi.org/project/pyetsimul/)
[![Downloads](https://static.pepy.tech/badge/pyetsimul)](https://pepy.tech/project/pyetsimul)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](https://github.com/mh-salari/pyetsimul/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/pyetsimul/badge/?version=latest)](https://pyetsimul.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3806023-blue)](https://doi.org/10.1145/3806023)

PyEtSimul is an open-source Python framework for simulating video-based eye trackers
by generating synthetic eye features through geometric modeling. The framework allows
flexible positioning of eyes, cameras, and light sources in 3D space, with controlled
variation of eye anatomical features and camera properties.
For full documentation, see [pyetsimul.readthedocs.io](https://pyetsimul.readthedocs.io/).

## About

Core functionalities are based on "A Software Framework for Simulating Eye Trackers" by Böhme et al. (2008), ported from the original MATLAB codebase. This Python implementation includes enhancements, improvements, bug fixes, and additional features.

> [!WARNING]
> **Note**: There is no guarantee that this code works exactly like the original MATLAB implementation.

## Features

PyEtSimul represents a video-based eye tracker as a 3D scene of eyes, cameras and lights, and renders the pupil and corneal reflections (glints) each camera would see.

- **Eye models** — named models (`"PyEtSimul"`, `"et_simul"`, `"gkaModelEye"`) selected by name, or your own via the immutable `EyeModel`.
- **Cornea** — spherical, conic and toric surfaces, with an optional posterior surface.
- **Pupil** — elliptical and realistic (Wyatt, 1995) shapes, plus pupil-size decentration (the pupil centre shifts as pupil size changes).
- **Eyelids** — an adjustable lid that occludes the pupil.
- **Glints** — corneal reflections of one or more lights, with optional detection noise.
- **Camera** — a pinhole model, or a full OpenCV camera with intrinsics and lens distortion.
- **Gaze models** — polynomial, Stampe (1993), homography-normalization, and custom mappings.
- **Datasets and evaluation** — generate labelled pupil/glint observations across gaze targets and parameter variations, and score calibration and gaze accuracy.

## Installation

### Requirements
- Python ≥3.11

### From PyPI

```bash
pip install pyetsimul
```

### Using [uv](https://docs.astral.sh/uv/)

```bash
uv pip install pyetsimul
```

### From source

```bash
git clone https://github.com/mh-salari/pyetsimul.git
cd pyetsimul
uv sync
```

Or with pip:
```bash
git clone https://github.com/mh-salari/pyetsimul.git
cd pyetsimul
python3 -m pip install .
```

**For development:**
```bash
git clone https://github.com/mh-salari/pyetsimul.git
cd pyetsimul
python3 -m pip install -e .
```

## Quickstart

Distances are in millimetres, in a right-handed frame: +x right, +y depth (away from the camera), +z up.

```python
from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Position3D

eye = Eye()  # a bare Eye() uses the default "PyEtSimul" model
eye.position = Position3D(0, 250, 100)
eye.look_at(Position3D(-50, 0, 50))  # fixate a point in the world

camera = Camera()
camera.point_at(eye.position)  # aim the camera at the eye

light = Light(position=Position3D(100, 0, 0))  # the glint is this light's reflection on the cornea

image = camera.take_image(eye, [light])
print(image.pupil_center)             # pupil centre in image pixels
print(image.corneal_reflections[0])   # glint in image pixels
```

## Examples

The `examples/` directory is an ordered, self-contained series. Start at `01`, or open the one you need — each script runs top to bottom.

| Example | What you learn |
| --- | --- |
| `01_hello_PyEtSimul` | build a scene (Eye, Camera, Light) and read the pupil centre + glint |
| `02_visualize_setup` | interactive 3D scene + camera view (move the eye and target live) |
| `03_eye_models` | the named models "PyEtSimul" / "et_simul" / "gkaModelEye" side by side |
| `04_cornea_models` | spherical vs conic vs toric cornea |
| `05_pupil_models` | elliptical vs realistic pupil across sizes, with pupil-size decentration |
| `06_eyelid` | eyelid openness and its occlusion of the pupil |
| `07_glint_noise` | glint detection-noise models |
| `08_camera_distortion` | pinhole vs a real OpenCV lens (intrinsics + distortion) |
| `09_two_eyes_one_camera` | a binocular pair imaged by one camera |
| `10_two_eyes_two_cameras` | a binocular pair imaged by two cameras |
| `11_generate_dataset` | render a gaze grid and save/load a labelled dataset |
| `12_calibrate` | calibrate a gaze model on an HV9 grid and measure accuracy (interactive) |
| `13_validate` | gaze accuracy across the screen and as the head moves |
| `14_custom_eye_model` | build a custom `EyeModel` (reproducing et_simul), then tweak it |
| `15_custom_gaze_model` | register a custom polynomial gaze model |
| `16_custom_variation` | sweep any eye parameter (built-in or a custom `GenericEyeVariation`) |

## Validation

PyEtSimul reproduces two independent eye models 1:1: the original et_simul (this is a Python port of it) and the gkaModelEye schematic eye. The `validation/` directory holds the MATLAB references and the tests that assert the simulated pupil centre and glint match them at each gaze target.

## Citation

**Cite as:**
```
Salari, M., Niehorster, D. C., Hansen, D. W., & Bednarik, R. (2026).
PyEtSimul: An Open-Source Python Framework for Eye-Tracking Simulation.
Proceedings of the ACM on Human-Computer Interaction (PACMHCI), ETRA 2026. ACM.
DOI: 10.1145/3806023
```

**Also cite the original work:**
```
Martin Böhme, Michael Dorr, Mathis Graw, Thomas Martinetz, and Erhardt Barth.
"A software framework for simulating eye trackers."
In Proceedings of the 2008 Symposium on Eye Tracking Research & Applications (ETRA '08),
pages 251-258, ACM, 2008.
```

## Original MATLAB Implementation

For the original MATLAB version and detailed background, visit:
https://github.com/mh-salari/et_simul-1.01

## Acknowledgments

This project has received funding from the European Union's Horizon Europe research and innovation funding program under grant agreement No 101072410, Eyes4ICU project.

<p align="center">
<img src="https://raw.githubusercontent.com/mh-salari/pyetsimul/main/resources/Funded_by_EU_Eyes4ICU.png" alt="Funded by EU Eyes4ICU" width="500">
</p>

## License

GPL-3.0-or-later