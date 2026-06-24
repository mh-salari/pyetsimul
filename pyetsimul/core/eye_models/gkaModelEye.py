"""The gkaModelEye eye, near-infrared variant (https://github.com/gkaguirrelab/gkaModelEye)."""

from ..cornea import ToricCornea
from ..eye_model import EyeModel, register_eye_model
from ..off_axis_pupil import OffAxisPupilConfig
from ..pupil_decentration import PupilDecentrationConfig
from ..rotation_center import RotationCenter

register_eye_model(
    "gkaModelEye",
    EyeModel(
        cornea=ToricCornea(
            use_posterior_surface=True,
            anterior_radius=7.694,  # base radius of the tri-axial anterior surface (7.777 horizontal, 7.612 vertical)
            anterior_k=-0.454,
            anterior_astigmatism_mm=0.1645,  # with-the-rule
            anterior_astigmatism_axis_deg=90.0,
            posterior_radius=6.284,
            posterior_k=-0.5437,
            refractive_index=1.3714,
            use_tear_film=True,
            tear_refractive_index=1.3301,
            _cornea_depth_default=3.9,  # aperture stop 3.9 mm behind the apex
        ),
        axial_length=24.75,
        n_aqueous_humor=1.3315,
        fovea_displacement=True,
        fovea_alpha_deg=5.45,
        fovea_beta_deg=2.5,
        # gkaModelEye poses the eye by the azimuth/elevation of the apex-to-target direction (its eyePose).
        look_at_method="optical_axis_target_direction",
        torsion_deg=0.0,
        pupil_tilt_x_deg=0.0,
        pupil_tilt_y_deg=0.0,
        rotation_center=RotationCenter(
            horizontal_depth_mm=14.7,
            vertical_depth_mm=12.0,
            horizontal_nasal_mm=0.79,
            vertical_superior_mm=0.33,
            fick=True,
        ),
        decentration_config=PupilDecentrationConfig(enabled=False),
        off_axis_pupil=OffAxisPupilConfig(enabled=False),
        pupil_type="elliptical",
        default_pupil_diameter=6.0,
        pupil_boundary_points=None,
        pupil_random_seed=None,
        realistic_pupil_params=None,
        eyelid_enabled=False,
    ),
)
