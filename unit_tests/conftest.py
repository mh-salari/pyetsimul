"""Shared pytest fixtures and setup for the unit tests.

Importing the eye-model package registers the named models (PyEtSimul, et_simul, ...) so tests can
build them by name, e.g. ``Eye(model="et_simul")``.
"""

import pyetsimul.core.eye_models  # noqa: F401  (import registers the named eye models)
