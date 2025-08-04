"""Eye anatomy visualization example.

Demonstrates eye parameter comparison and 3D anatomical visualization for different corneal radii.
"""

from et_simul.core import Eye
from et_simul.core.cornea import SphericalCornea
from et_simul.visualization import plot_eye_anatomy
import matplotlib.pyplot as plt


def print_eye_parameters(eye, title="Eye Parameters"):
    """Display comprehensive eye anatomical parameters for analysis and comparison."""
    print(f"\n=== {title} ===")
    print(f"r_cornea: {eye.cornea.anterior_radius * 1000:.3f} mm")
    print(f"r_cornea_inner: {eye.cornea.posterior_radius * 1000:.3f} mm")
    print(f"axial_length: {eye.axial_length * 1000:.3f} mm")
    print(f"cornea_center_to_rotation_center: {eye.cornea._cornea_center_to_rotation_center_default * 1000:.3f} mm")
    print(f"cornea_thickness_offset: {eye.cornea.thickness_offset * 1000:.3f} mm")
    print(f"depth_cornea: {eye.cornea.get_corneal_depth() * 1000:.3f} mm")
    print(f"n_cornea: {eye.cornea.refractive_index:.3f}")
    print(f"n_aqueous_humor: {eye.n_aqueous_humor:.3f}")
    print(f"fovea_alpha_deg: {eye.fovea_alpha_deg:.1f}°")
    print(f"fovea_beta_deg: {eye.fovea_beta_deg:.1f}°")
    print(f"angle_kappa: {eye.angle_kappa:.3f}°")

    # Position information
    cornea_center = eye.cornea.center
    print(f"pos_cornea: [{cornea_center.x:.6f}, {cornea_center.y:.6f}, {cornea_center.z * 1000:.3f}] (z in mm)")
    apex_pos = eye.cornea.get_apex_position()
    print(f"pos_apex: [{apex_pos.x:.6f}, {apex_pos.y:.6f}, {apex_pos.z * 1000:.3f}] (z in mm)")
    pupil_pos = eye.pupil.pos_pupil
    print(f"pos_pupil: [{pupil_pos.x:.6f}, {pupil_pos.y:.6f}, {pupil_pos.z * 1000:.3f}] (z in mm)")

    # Pupil radii
    x_radius, y_radius = eye.get_pupil_radii()
    print(f"pupil_radius: {x_radius * 1000:.3f} mm")


# Create two eyes with different corneal radii
e_default = Eye()  # Default 7.98mm
e = Eye(cornea=SphericalCornea(anterior_radius=6.98e-3))

# Print parameters
print_eye_parameters(e_default, f"r_cornea = {e_default.cornea.anterior_radius * 1000:.3f} mm)")
print_eye_parameters(e, f"r_cornea = {e.cornea.anterior_radius * 1000:.3f} mm)")


fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(121, projection="3d")
plot_eye_anatomy(e_default, ax=ax1)
ax1.set_title(f"{e_default.cornea.anterior_radius * 1000:.3f} mm", fontsize=16, fontweight="bold")

ax2 = fig.add_subplot(122, projection="3d")
plot_eye_anatomy(e, ax=ax2)
ax2.set_title(f"{e.cornea.anterior_radius * 1000:.3f} mm", fontsize=16, fontweight="bold")

# Set same axis limits for both plots based on the default eye
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()

# Apply the same limits to both plots
for ax in [ax1, ax2]:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

plt.tight_layout()
plt.show()
