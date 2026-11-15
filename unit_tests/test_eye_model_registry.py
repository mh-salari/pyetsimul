"""Unit tests for the named eye-model registry and ``Eye(model=...)`` name resolution."""

import pytest

from pyetsimul.core.eye import Eye
from pyetsimul.core.eye_model import EyeModel, get_eye_model, list_eye_models, register_eye_model


def test_pyetsimul_is_registered() -> None:
    """The default PyEtSimul model is available by name and is an EyeModel."""
    assert "PyEtSimul" in list_eye_models()
    assert isinstance(get_eye_model("PyEtSimul"), EyeModel)


def test_lookup_is_case_insensitive() -> None:
    """A model name resolves regardless of casing, to the same registered model."""
    canonical = get_eye_model("PyEtSimul")
    assert get_eye_model("pyetsimul") == canonical
    assert get_eye_model("PYETSIMUL") == canonical


def test_fetched_model_is_a_copy() -> None:
    """Changing a fetched model leaves the registered model as it is."""
    fetched = get_eye_model("PyEtSimul")
    registered_radius = fetched.cornea.anterior_radius
    fetched.cornea.anterior_radius = registered_radius + 1.0
    assert get_eye_model("PyEtSimul").cornea.anterior_radius == registered_radius


def test_unknown_model_raises() -> None:
    """Looking up an unregistered name raises KeyError."""
    with pytest.raises(KeyError):
        get_eye_model("no_such_model")


def test_register_and_retrieve() -> None:
    """A registered model is retrievable by name, case-insensitively."""
    custom = EyeModel(fovea_alpha_deg=3.0)
    register_eye_model("RegistryTestEye", custom)
    assert get_eye_model("registrytesteye") == custom


def test_registering_an_existing_name_needs_overwrite() -> None:
    """A registered name is replaced only with overwrite=True."""
    register_eye_model("RegistryOverwriteEye", EyeModel(fovea_alpha_deg=3.0))
    replacement = EyeModel(fovea_alpha_deg=4.0)
    with pytest.raises(ValueError, match="already registered"):
        register_eye_model("registryoverwriteeye", replacement)
    register_eye_model("registryoverwriteeye", replacement, overwrite=True)
    assert get_eye_model("RegistryOverwriteEye") == replacement


def test_eye_accepts_model_name() -> None:
    """Eye(model="name") resolves the name to its EyeModel (case-insensitive)."""
    eye = Eye(model="pyetsimul")
    assert isinstance(eye.model, EyeModel)
    assert eye.model.look_at_method == "visual_axis"


def test_eye_accepts_eye_model_value() -> None:
    """Eye(model=EyeModel(...)) uses the given model directly."""
    eye = Eye(model=EyeModel(fovea_alpha_deg=4.0))
    assert eye.model.fovea_alpha_deg == pytest.approx(4.0)
