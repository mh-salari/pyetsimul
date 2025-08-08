# PyEtSimul: An Open-Source Python Software Framework for Eye Tracking Simulation

PyEtSimul is a framework that simulates measurements made by video-oculographic eye trackers, enabling objective comparison of different eye tracking setups and gaze estimation algorithms.

## About

Core functionalities are based on "A Software Framework for Simulating Eye Trackers" by Böhme et al. (2008), ported from the original MATLAB codebase. This Python implementation includes enhancements, improvements, bug fixes, and additional features. 

> [!WARNING]  
> **Note**: There is no guarantee that this code works exactly like the original MATLAB implementation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mh-salari/pyetsimul.git
cd pyetsimul
```

2. Install in development mode:
```bash
pip install -e .
```

Or install from PyPI (when published):
```bash
pip install pyetsimul
```

3. Run examples:
```bash
python examples/example.py
python examples/interpolate.py
python examples/eye_anatomy.py
python examples/setup_visualization.py
python examples/camera_distortion.py
python examples/realistic_pupil.py
```

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