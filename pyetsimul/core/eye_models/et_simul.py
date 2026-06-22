"""The original Böhme 2008 et_simul eye."""

from ..cornea import SphericalCornea
from ..eye_model import EyeModel, register_eye_model
from ..off_axis_pupil import OffAxisPupilConfig
from ..pupil_decentration import PupilDecentrationConfig

ET_SIMUL = EyeModel(
    # Spherical cornea, anterior radius 7.98 mm; single anterior-surface refraction (air -> cornea 1.376).
    cornea=SphericalCornea(anterior_radius=7.98, refractive_index=1.376, use_posterior_surface=False),
    axial_length=24.75,  # mm
    n_aqueous_humor=1.336,
    fovea_displacement=True,
    fovea_alpha_deg=6.0,  # horizontal kappa
    fovea_beta_deg=2.0,  # vertical kappa
    look_at_method="optical_then_kappa",  # the original et_simul look-at
    torsion_deg=0.0,
    pupil_tilt_x_deg=0.0,
    pupil_tilt_y_deg=0.0,
    rotation_center=None,  # single fixed centre at the eyeball-sphere centre
    decentration_config=PupilDecentrationConfig(enabled=False),  # no size-dependent pupil decentration
    off_axis_pupil=OffAxisPupilConfig(enabled=False),  # no static off-axis pupil
    pupil_type="elliptical",
    default_pupil_diameter=6.0,  # 3 mm pupil radius
    pupil_boundary_points=None,
    pupil_random_seed=None,
    realistic_pupil_params=None,
    eyelid_enabled=False,
)

register_eye_model("et_simul", ET_SIMUL)
