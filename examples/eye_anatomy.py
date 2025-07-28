from et_simul.core import Eye
from et_simul.visualization import plot_eye_anatomy
import matplotlib.pyplot as plt
import numpy as np


def print_eye_parameters(eye, title="Eye Parameters"):
    """Print all eye parameters in a formatted way."""
    print(f"\n=== {title} ===")
    print(f"r_cornea: {eye.r_cornea * 1000:.3f} mm")
    print(f"r_cornea_inner: {eye.r_cornea_inner * 1000:.3f} mm")
    print(f"axial_length: {eye.axial_length * 1000:.3f} mm")
    print(f"cornea_center_to_rotation_center: {eye.cornea_center_to_rotation_center * 1000:.3f} mm")
    print(f"cornea_thickness_offset: {eye.cornea_thickness_offset * 1000:.3f} mm")
    print(f"depth_cornea: {eye.depth_cornea * 1000:.3f} mm")
    print(f"n_cornea: {eye.n_cornea:.3f}")
    print(f"n_aqueous_humor: {eye.n_aqueous_humor:.3f}")
    print(f"fovea_alpha_deg: {eye.fovea_alpha_deg:.1f}°")
    print(f"fovea_beta_deg: {eye.fovea_beta_deg:.1f}°")
    print(f"angle_kappa: {eye.angle_kappa:.3f}°")

    # Position information
    print(
        f"pos_cornea: [{eye.pos_cornea[0]:.6f}, {eye.pos_cornea[1]:.6f}, {eye.pos_cornea[2] * 1000:.3f}, {eye.pos_cornea[3]:.1f}] (z in mm)"
    )
    print(
        f"pos_apex: [{eye.pos_apex[0]:.6f}, {eye.pos_apex[1]:.6f}, {eye.pos_apex[2] * 1000:.3f}, {eye.pos_apex[3]:.1f}] (z in mm)"
    )
    print(
        f"pos_pupil: [{eye.pos_pupil[0]:.6f}, {eye.pos_pupil[1]:.6f}, {eye.pos_pupil[2] * 1000:.3f}, {eye.pos_pupil[3]:.1f}] (z in mm)"
    )

    # Pupil radii
    x_radius, y_radius = eye.get_pupil_radii()
    print(f"pupil_radius: {x_radius * 1000:.3f} mm")


# Create two eyes with different corneal radii
e_default = Eye()  # Default 7.98mm
e = Eye(r_cornea=6.98e-3)  # Large 9.0mm

# Print parameters
print_eye_parameters(e_default, f"Default Eye (r_cornea = {e_default.r_cornea * 1000:.1f} mm)")
print_eye_parameters(e, f"Large Cornea Eye (r_cornea = {e.r_cornea * 1000:.1f} mm)")


fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(121, projection="3d")
plot_eye_anatomy(e_default, ax=ax1)
ax1.set_title(f"{e_default.r_cornea * 1000:.3f} mm", fontsize=16, fontweight="bold")

ax2 = fig.add_subplot(122, projection="3d")
plot_eye_anatomy(e, ax=ax2)
ax2.set_title(f"{e.r_cornea * 1000:.3f} mm", fontsize=16, fontweight="bold")

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
