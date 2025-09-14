"""Eyelid openness comparison example.

Demonstrates eye anatomy visualization with different eyelid openness levels (25%, 50%, 75%).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from pyetsimul.core import Eye
from pyetsimul.core.cornea import SphericalCornea
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_eye_anatomy

# Define target point for consistent gaze direction
target_point = Position3D(10e-3, 10e-3, -10e-3)

# Create three eyes with spherical corneas and different eyelid openness
eye_25 = Eye(cornea=SphericalCornea(), eyelid_enabled=True)
eye_50 = Eye(cornea=SphericalCornea(), eyelid_enabled=True)
eye_75 = Eye(cornea=SphericalCornea(), eyelid_enabled=True)

# Set different eyelid openness levels
eye_25.eyelid.openness = 0.25  # 25% open
eye_50.eyelid.openness = 0.50  # 50% open
eye_75.eyelid.openness = 0.75  # 75% open

# Set rest orientation and gaze direction for all eyes
for eye in [eye_25, eye_50, eye_75]:
    eye.set_rest_orientation_at_target(target_point)
    eye.look_at(target_point)

# Create visualization with 3 subplots
fig = plt.figure(figsize=(18, 6))

# Eye 1: 25% open
ax1 = fig.add_subplot(131, projection="3d")
plot_eye_anatomy(eye_25, ax=ax1)
ax1.set_title("25% Eyelid Openness", fontsize=14, fontweight="bold")
# Remove the eye openness text that gets added automatically
for text in ax1.texts:
    text.remove()

# Eye 2: 50% open
ax2 = fig.add_subplot(132, projection="3d")
plot_eye_anatomy(eye_50, ax=ax2)
ax2.set_title("50% Eyelid Openness", fontsize=14, fontweight="bold")
# Remove the eye openness text that gets added automatically
for text in ax2.texts:
    text.remove()

# Eye 3: 75% open
ax3 = fig.add_subplot(133, projection="3d")
plot_eye_anatomy(eye_75, ax=ax3)
ax3.set_title("75% Eyelid Openness", fontsize=14, fontweight="bold")
# Remove the eye openness text that gets added automatically
for text in ax3.texts:
    text.remove()


# Set same axis limits for all plots
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
zlim = ax1.get_zlim()

# Apply the same limits and viewing angle to all plots
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=-30, azim=45, roll=60)

# Remove individual legends from subplots
ax1.legend().set_visible(False)
ax2.legend().set_visible(False)
ax3.legend().set_visible(False)

plt.tight_layout(pad=2.0)

# Create figures directory if it doesn't exist and save figure
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/eyelid_openness_comparison.pdf", transparent=True, bbox_inches="tight")
plt.show()
