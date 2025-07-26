from et_simul.core import Eye
from et_simul.visualization import plot_eye_anatomy

e = Eye()

print(f"axial_length: {e.axial_length*1000:.2f} mm")
print(f"r_cornea: {e.r_cornea*1000:.2f} mm")
print(f"r_cornea_inner: {e.r_cornea_inner*1000:.2f} mm")
print(
    f"cornea_center_to_rotation_center: {e.cornea_center_to_rotation_center*1000:.2f} mm"
)
print(f"cornea_thickness_offset: {e.cornea_thickness_offset*1000:.2f} mm")
print(f"depth_cornea: {e.depth_cornea*1000:.2f} mm")
print(f"pupil_radius: {e.pupil_radius_default*1000:.2f} mm")
print(f"n_cornea: {e.n_cornea}")
print(f"n_aqueous_humor: {e.n_aqueous_humor}")
print(f"fovea_alpha_deg: {e.fovea_alpha_deg:.1f}°")
print(f"fovea_beta_deg: {e.fovea_beta_deg:.1f}°")
print(f"angle_kappa: {e.angle_kappa:.3f}°")

plot_eye_anatomy()
