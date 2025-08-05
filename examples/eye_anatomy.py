"""Eye anatomy visualization example.

Demonstrates eye parameter comparison and 3D anatomical visualization for different corneal radii.
"""

from et_simul.core import Eye
from et_simul.core.cornea import SphericalCornea, ConicCornea
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


# Create two eyes: spherical vs conic cornea (both with default parameters)
e_spherical = Eye(cornea=SphericalCornea())  # Default spherical cornea
e_conic = Eye(cornea=ConicCornea())  # Default conic cornea

# Print parameters
print_eye_parameters(e_spherical, "Spherical Cornea (Default)")
print_eye_parameters(e_conic, "Conic Cornea (Default)")


fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(121, projection="3d")
plot_eye_anatomy(e_spherical, ax=ax1)
ax1.set_title("Spherical Cornea", fontsize=14, fontweight="bold")
# Add spherical parameters
spherical_params = f"R_ant = {e_spherical.cornea.anterior_radius*1000:.2f} mm\nR_post = {e_spherical.cornea.posterior_radius*1000:.2f} mm"
ax1.text2D(0.02, 0.95, spherical_params, transform=ax1.transAxes, fontsize=9, verticalalignment='top')

ax2 = fig.add_subplot(122, projection="3d")
plot_eye_anatomy(e_conic, ax=ax2)
ax2.set_title("Conic Cornea", fontsize=14, fontweight="bold")
# Add conic parameters
conic_params = f"R_ant = {e_conic.cornea.anterior_radius*1000:.2f} mm, k = {e_conic.cornea.anterior_k:.2f}\nR_post = {e_conic.cornea.posterior_radius*1000:.2f} mm, k = {e_conic.cornea.posterior_k:.2f}"
ax2.text2D(0.02, 0.95, conic_params, transform=ax2.transAxes, fontsize=9, verticalalignment='top')

# Set same axis limits for both plots based on the default eye
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()

# Apply the same limits to both plots
for ax in [ax1, ax2]:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

# Remove individual legends from subplots
ax1.legend().set_visible(False)
ax2.legend().set_visible(False)

# Create a single legend for the entire figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=10)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.9, right=0.85)  # Make room for legend
plt.show()
