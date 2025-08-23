"""Eye anatomy visualization example.

Demonstrates eye parameter comparison and 3D anatomical visualization for different corneal radii.
"""

from pyetsimul.core import Eye
from pyetsimul.core.cornea import SphericalCornea, ConicCornea
from pyetsimul.visualization import plot_eye_anatomy
from pyetsimul.types import Position3D
import matplotlib.pyplot as plt
from tabulate import tabulate


def get_eye_parameters(eye):
    """Extract eye parameters into a dictionary for table formatting"""
    cornea_center = eye.cornea.center
    apex_pos = eye.cornea.get_apex_position()
    pupil_pos = eye.pupil.pos_pupil
    x_radius, _ = eye.get_pupil_radii()

    return {
        "Anterior corneal radius R_a (mm)": f"{eye.cornea.anterior_radius * 1000:.3f}",
        "Posterior corneal radius R_p (mm)": f"{eye.cornea.posterior_radius * 1000:.3f}",
        "Axial length L (mm)": f"{eye.axial_length * 1000:.3f}",
        "Cornea center to rotation center (mm)": f"{eye.cornea._cornea_center_to_rotation_center_default * 1000:.3f}",
        "Thickness offset t_offset (mm)": f"{eye.cornea.thickness_offset * 1000:.3f}",
        "Corneal depth d_c (mm)": f"{eye.cornea.get_corneal_depth() * 1000:.3f}",
        "Refractive index n_cornea": f"{eye.cornea.refractive_index:.3f}",
        "Refractive index n_aqueous": f"{eye.n_aqueous_humor:.3f}",
        "Fovea α (deg)": f"{eye.fovea_alpha_deg:.1f}",
        "Fovea β (deg)": f"{eye.fovea_beta_deg:.1f}",
        "Angle κ (deg)": f"{eye.angle_kappa:.3f}",
        "Cornea center (x,y,z) mm": f"({cornea_center.x * 1000:.3f}, {cornea_center.y * 1000:.3f}, {cornea_center.z * 1000:.3f})",
        "Anterior apex (x,y,z) mm": f"({apex_pos.x * 1000:.3f}, {apex_pos.y * 1000:.3f}, {apex_pos.z * 1000:.3f})",
        "Pupil center (x,y,z) mm": f"({pupil_pos.x * 1000:.3f}, {pupil_pos.y * 1000:.3f}, {pupil_pos.z * 1000:.3f})",
        "Pupil radius r_p (mm)": f"{x_radius * 1000:.3f}",
    }


def print_eye_comparison(eye1, eye2, title1="Eye 1", title2="Eye 2"):
    """Display comprehensive eye anatomical parameters comparison using tabulate."""
    params1 = get_eye_parameters(eye1)
    params2 = get_eye_parameters(eye2)

    # Create comparison table
    headers = ["Parameter", title1, title2]
    data = []

    for param_name in params1.keys():
        data.append([param_name, params1[param_name], params2[param_name]])

    print("\nEye Parameters Comparison")
    print(tabulate(data, headers=headers, tablefmt="grid"))


# Define target point for consistent gaze direction
# Use Z-forward target to keep eyelid properly oriented in the plot view
target_point = Position3D(10e-3, 10e-3, -10e-3)

# Create two eyes: spherical vs conic cornea (both with default parameters)
e_spherical = Eye(cornea=SphericalCornea(), eyelid_enabled=True)
e_conic = Eye(cornea=ConicCornea(), eyelid_enabled=True)

e_spherical.eyelid.openness = 0.50
e_conic.eyelid.openness = 0.75

# Set rest orientation to face the target (so eyelid aligns with gaze)
e_spherical.set_rest_orientation_at_target(target_point)
e_conic.set_rest_orientation_at_target(target_point)

e_spherical.look_at(target_point)
e_conic.look_at(target_point)


# Print parameter comparison
print_eye_comparison(e_spherical, e_conic, "Spherical Cornea", "Conic Cornea")


fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(121, projection="3d")
plot_eye_anatomy(e_spherical, ax=ax1)
ax1.set_title("Spherical Cornea", fontsize=14, fontweight="bold")
# Add spherical parameters
spherical_params = (
    f"R_a = {e_spherical.cornea.anterior_radius * 1000:.2f} mm\n"
    f"R_p = {e_spherical.cornea.posterior_radius * 1000:.2f} mm\n"
    f"d_c = {e_spherical.cornea.get_corneal_depth() * 1000:.2f} mm"
)
ax1.text2D(0.02, 0.95, spherical_params, transform=ax1.transAxes, fontsize=9, verticalalignment="top")

ax2 = fig.add_subplot(122, projection="3d")
plot_eye_anatomy(e_conic, ax=ax2)
ax2.set_title("Conic Cornea", fontsize=14, fontweight="bold")
# Add conic parameters
conic_params = (
    f"R_a = {e_conic.cornea.anterior_radius * 1000:.2f} mm, k_a = {e_conic.cornea.anterior_k:.2f}\n"
    f"R_p = {e_conic.cornea.posterior_radius * 1000:.2f} mm, k_p = {e_conic.cornea.posterior_k:.2f}"
)
ax2.text2D(0.02, 0.95, conic_params, transform=ax2.transAxes, fontsize=9, verticalalignment="top")

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
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.95), fontsize=10)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.9, right=0.85)  # Make room for legend
plt.show()
