# et_simul validation

A 1:1 check that the `et_simul` named model reproduces the original Böhme 2008 et_simul
(https://github.com/mh-salari/et_simul-1.01).

## Scene

An `Eye(model="et_simul")` at (30, 835, 85) mm, a camera at (-100, 420, -125) mm pointed at the eye, and a
light at (95, 435, -115) mm. The eye looks at five targets -- the origin, (±233.8, 0, 0) and
(0, 0, ±124.08) mm -- and at each, the refracted pupil centre and the glint are measured in the camera image.

## Files

- `matlab/run_reference.m` — runs et_simul on the scene and writes `reference.json`.
- `reference.json` — the et_simul reference: pupil centre and glint per target (pixels).
- `python/test_et_simul.py` — rebuilds the scene with `Eye(model="et_simul")` and asserts the match.

## Regenerating the reference

With the et_simul MATLAB on the path (https://github.com/mh-salari/et_simul-1.01):

```
matlab -batch "addpath('validation/et_simul/matlab'); run_reference"
```

## Running the test

```
uv run pytest validation/et_simul/python/test_et_simul.py
```

The pupil centre and glint match the reference to within 1e-3 px at every target.
