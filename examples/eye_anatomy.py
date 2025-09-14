"""Eye anatomy visualization example.

Demonstrates eye parameter comparison and 3D anatomical visualization for different corneal radii.
"""

import matplotlib.pyplot as plt

from pyetsimul.core import Eye
from pyetsimul.core.cornea import ConicCornea, SphericalCornea
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_eye_anatomy

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


# Print parameter comparison using new pprint methods
print("\nSpherical Cornea Eye:")
e_spherical.pprint()

print("\nConic Cornea Eye:")
e_conic.pprint()


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
