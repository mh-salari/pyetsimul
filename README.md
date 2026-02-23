# PyEtSimul: An Open-Source Python Software Framework for Eye Tracking Simulation

[![PyPI version](https://img.shields.io/pypi/v/pyetsimul)](https://pypi.org/project/pyetsimul/)
[![Downloads](https://static.pepy.tech/badge/pyetsimul)](https://pepy.tech/project/pyetsimul)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](https://github.com/mh-salari/pyetsimul/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/pyetsimul/badge/?version=latest)](https://pyetsimul.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-TODO-blue)](https://doi.org/TODO)

PyEtSimul is an open-source Python framework for simulating video-based eye trackers
by generating synthetic eye features through geometric modeling. The framework allows
flexible positioning of eyes, cameras, and light sources in 3D space, with controlled
variation of eye anatomical features and camera properties.
For full documentation, see [pyetsimul.readthedocs.io](https://pyetsimul.readthedocs.io/).

## About

Core functionalities are based on "A Software Framework for Simulating Eye Trackers" by Böhme et al. (2008), ported from the original MATLAB codebase. This Python implementation includes enhancements, improvements, bug fixes, and additional features.

> [!WARNING]
> **Note**: There is no guarantee that this code works exactly like the original MATLAB implementation.

## Installation

### Requirements
- Python ≥3.11

### Using [uv](https://docs.astral.sh/uv/) (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/mh-salari/pyetsimul.git
cd pyetsimul
```

2. Install dependencies and run in development mode:
```bash
uv sync
```


### Using pip (Alternative)

**Note:** Create a virtual environment first:
```bash
python3 -m venv pyetsimul_env
source pyetsimul_env/bin/activate      # On Linux/macOS
# or: pyetsimul_env\Scripts\activate   # On Windows
```

**Direct from GitHub (easiest):**
```bash
python3 -m pip install git+https://github.com/mh-salari/pyetsimul.git
```

**From local clone:**
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

## Examples

After installation, explore the various examples in the `examples/` directory to see PyEtSimul in action.

## Citation

If you use this software in your research, please cite both:

1. **This Python implementation:**
```
[To be filled]
```

2. **Original work:**
```
Martin Böhme, Michael Dorr, Mathis Graw, Thomas Martinetz, and Erhardt Barth.
"A software framework for simulating eye trackers."
In Proceedings of the 2008 Symposium on Eye Tracking Research & Applications (ETRA '08),
pages 251-258, ACM, 2008.
```

## Original MATLAB Implementation

For the original MATLAB version and detailed background, visit:
https://github.com/mh-salari/et_simul-1.01

## License

GPL-3.0-or-later