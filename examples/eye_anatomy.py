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
    """Extract eye parameters into a dictionary for table formatting."""
    cornea_center = eye.cornea.center
    apex_pos = eye.cornea.get_apex_position()
    pupil_pos = eye.pupil.pos_pupil
    x_radius, y_radius = eye.get_pupil_radii()

    return {
        "r_cornea (mm)": f"{eye.cornea.anterior_radius * 1000:.3f}",
        "r_cornea_inner (mm)": f"{eye.cornea.posterior_radius * 1000:.3f}",
        "axial_length (mm)": f"{eye.axial_length * 1000:.3f}",
        "cornea_center_to_rotation_center (mm)": f"{eye.cornea._cornea_center_to_rotation_center_default * 1000:.3f}",
        "cornea_thickness_offset (mm)": f"{eye.cornea.thickness_offset * 1000:.3f}",
        "depth_cornea (mm)": f"{eye.cornea.get_corneal_depth() * 1000:.3f}",
        "n_cornea": f"{eye.cornea.refractive_index:.3f}",
        "n_aqueous_humor": f"{eye.n_aqueous_humor:.3f}",
        "fovea_alpha (°)": f"{eye.fovea_alpha_deg:.1f}",
        "fovea_beta (°)": f"{eye.fovea_beta_deg:.1f}",
        "angle_kappa (°)": f"{eye.angle_kappa:.3f}",
        "pos_cornea (x,y,z)": f"({cornea_center.x:.6f}, {cornea_center.y:.6f}, {cornea_center.z * 1000:.3f})",
        "pos_apex (x,y,z)": f"({apex_pos.x:.6f}, {apex_pos.y:.6f}, {apex_pos.z * 1000:.3f})",
        "pos_pupil (x,y,z)": f"({pupil_pos.x:.6f}, {pupil_pos.y:.6f}, {pupil_pos.z * 1000:.3f})",
        "pupil_radius (mm)": f"{x_radius * 1000:.3f}",
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
e_spherical = Eye(cornea=SphericalCornea())  # Default spherical cornea
e_conic = Eye(cornea=ConicCornea())  # Default conic cornea


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
spherical_params = f"R_ant = {e_spherical.cornea.anterior_radius * 1000:.2f} mm\nR_post = {e_spherical.cornea.posterior_radius * 1000:.2f} mm"
ax1.text2D(0.02, 0.95, spherical_params, transform=ax1.transAxes, fontsize=9, verticalalignment="top")

ax2 = fig.add_subplot(122, projection="3d")
plot_eye_anatomy(e_conic, ax=ax2)
ax2.set_title("Conic Cornea", fontsize=14, fontweight="bold")
# Add conic parameters
conic_params = f"R_ant = {e_conic.cornea.anterior_radius * 1000:.2f} mm, k = {e_conic.cornea.anterior_k:.2f}\nR_post = {e_conic.cornea.posterior_radius * 1000:.2f} mm, k = {e_conic.cornea.posterior_k:.2f}"
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
