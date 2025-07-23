#!/usr/bin/env python3
"""
Basic Eye Anatomy Demo

This script demonstrates basic eye anatomy visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import modules from the package
from et_simul.core import Eye
from et_simul.visualization.draw import draw_eye_anatomy


def demo_basic_eye_anatomy():
    """Demonstrate basic eye anatomy visualization."""
    print("Demo: Basic Eye Anatomy")
    print("=" * 40)

    # Create a standard eye
    rest_orientation = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    e = Eye(r_cornea=7.98e-3, rest_pos=rest_orientation, fovea_displacement=False)
    e.trans[:3, 3] = [0, 0.55, 0.35]  # Position at 55cm from origin, 35cm high

    print("Eye created with standard parameters:")
    print(f"- Corneal radius: {e.r_cornea*1000:.2f} mm")
    print(
        f"- Position: ({e.trans[0,3]*100:.0f}, {e.trans[1,3]*100:.0f}, {e.trans[2,3]*100:.0f}) cm"
    )

    # Visualize basic anatomy without axes
    draw_eye_anatomy(e, show_axes=False, show_annotations=True)

    print("Displaying basic eye anatomy...")
    print("Close the plot window to continue.\n")
    plt.show()


if __name__ == "__main__":
    demo_basic_eye_anatomy()
