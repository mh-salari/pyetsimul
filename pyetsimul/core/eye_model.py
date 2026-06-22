"""Eye-model specification: the optics and anatomy that define how an eye behaves.

An :class:`EyeModel` is an immutable bundle of the cornea, pupil shape, angle kappa, rotation geometry and
gaze conventions -- the *what kind of eye*, separate from the placement and pose held by
:class:`~pyetsimul.core.eye.Eye`. It is frozen; derive a variant with :meth:`EyeModel.copy`.
"""

from dataclasses import dataclass, field, replace

from .cornea import ConicCornea, SphericalCornea
from .default_configs import EyeAnatomyDefaults
from .off_axis_pupil import OffAxisPupilConfig
from .pupil import RealisticPupilParams
from .pupil_decentration import PupilDecentrationConfig
from .rotation_center import RotationCenter


@dataclass(frozen=True)
class EyeModel:
    """An immutable eye-model specification.

    Holds the cornea, pupil model, angle kappa, rotation geometry and gaze conventions of an eye. Angle
    kappa and pupil-decentration coefficients are stored as *magnitudes*; their sign is applied per eye
    side by the placed :class:`~pyetsimul.core.eye.Eye` via its ``which_eye``.

    Frozen: build a new model rather than mutating fields. :meth:`copy` returns a copy with fields changed.
    """

    # ---------------------------------------------------------------------------
    # Cornea and ocular anatomy
    # ---------------------------------------------------------------------------
    cornea: SphericalCornea | ConicCornea = field(default_factory=SphericalCornea)
    axial_length: float = EyeAnatomyDefaults.AXIAL_LENGTH  # total axial length of the eye (mm)
    n_aqueous_humor: float = EyeAnatomyDefaults.N_AQUEOUS_HUMOR  # index of the medium behind the cornea

    # ---------------------------------------------------------------------------
    # Angle kappa (visual-axis offset from the optical axis; magnitudes, signed per eye side)
    # ---------------------------------------------------------------------------
    fovea_displacement: bool = True  # whether the visual axis is offset from the optical axis
    fovea_alpha_deg: float = EyeAnatomyDefaults.FOVEA_ALPHA_DEG  # horizontal kappa magnitude
    fovea_beta_deg: float = EyeAnatomyDefaults.FOVEA_BETA_DEG  # vertical kappa

    # ---------------------------------------------------------------------------
    # Gaze, rotation and eye-fixed surface orientation
    # ---------------------------------------------------------------------------
    # Default look_at method -- how the eye is pointed at a target (look_at can override it per call):
    # "visual_axis" aligns the fovea-to-eye-centre axis to the target; "line_of_sight" aligns the
    # fovea-to-pupil-centre axis, so the eye re-aims as the pupil decentres; "optical_then_kappa" aims the
    # optical axis at the target then post-rotates by angle kappa (the original Böhme 2008 et_simul construction).
    look_at_method: str = "visual_axis"
    torsion_deg: float = 0.0  # extra roll about the line of sight on top of Listing's law
    # Pupil-plane tilt off the optical axis (deg): a z-shear of the boundary about the pupil centre, so
    # the disc plane tilts instead of lying perpendicular to the optical axis. Eye-fixed (rotates with gaze).
    pupil_tilt_x_deg: float = 0.0
    pupil_tilt_y_deg: float = 0.0
    # Gaze-direction-dependent rotation centre. None keeps the single fixed centre at the geometric centre
    # of the eyeball sphere (the default model behaviour).
    rotation_center: RotationCenter | None = None

    # ---------------------------------------------------------------------------
    # Pupil-centre displacement models
    # ---------------------------------------------------------------------------
    decentration_config: PupilDecentrationConfig = field(default_factory=PupilDecentrationConfig)
    off_axis_pupil: OffAxisPupilConfig = field(default_factory=OffAxisPupilConfig)

    # ---------------------------------------------------------------------------
    # Pupil shape and default size
    # ---------------------------------------------------------------------------
    pupil_type: str = "elliptical"  # "elliptical" (default) or "realistic"
    default_pupil_diameter: float = 2.0 * EyeAnatomyDefaults.PUPIL_RADIUS  # default (nominal) diameter (mm)
    pupil_boundary_points: int | None = None  # boundary point count (uses the pupil's own default if None)
    pupil_random_seed: int | None = None  # seed for the realistic pupil (None = random)
    realistic_pupil_params: RealisticPupilParams | None = None  # full realistic-pupil params (overrides seed)

    # ---------------------------------------------------------------------------
    # Eyelid
    # ---------------------------------------------------------------------------
    eyelid_enabled: bool = False

    def copy(self, **overrides: object) -> "EyeModel":
        """Return a copy of this model with the given fields changed (the model itself is frozen)."""
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# Named eye-model registry
# ---------------------------------------------------------------------------
_EYE_MODELS: dict[str, EyeModel] = {}


def register_eye_model(name: str, model: EyeModel) -> None:
    """Register ``model`` under ``name`` so it can be retrieved by name (matched case-insensitively)."""
    _EYE_MODELS[name] = model


def get_eye_model(name: str) -> EyeModel:
    """Return the eye model registered under ``name`` (matched case-insensitively)."""
    for registered, model in _EYE_MODELS.items():
        if registered.casefold() == name.casefold():
            return model
    available = ", ".join(_EYE_MODELS) or "(none registered)"
    raise KeyError(f"Unknown eye model {name!r}; registered models: {available}")


def list_eye_models() -> list[str]:
    """Return the names of the registered eye models, in registration order."""
    return list(_EYE_MODELS)
