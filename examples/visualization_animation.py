import numpy as np
import matplotlib.pyplot as plt
import copy

# Import modules from the package
from et_simul.core import Eye, Camera, Light
from et_simul.visualization import plot_setup_and_camera_view


def manual_eye_gaze_keyboard_control():
    print("Interactive Eye Tracking Visualization")
    print("=" * 40)
    print("CONTROLS:")
    print("Target Movement (Arrow keys):")
    print("  ↑/↓: Move target up/down")
    print("  ←/→: Move target left/right")
    print()
    print("Eye Movement (I/K/J/L/./):")
    print("  I/K: Move eye up/down")
    print("  J/L: Move eye left/right")
    print("  ./,: Move eye closer/farther from camera")
    print("=" * 40)

    rest_orientation = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    e_base = Eye()
    e_base.set_rest_orientation(rest_orientation)
    e_base.position = [0, 250e-3, 100e-3]

    # Create two light sources
    l1 = Light(position=np.array([100e-3, 0, 0]))  # Right side light

    l2 = Light(position=np.array([-100e-3, 0, 0]))  # Left side light

    lights = [l1, l2]

    c = Camera()
    c.point_at(e_base.position)

    # Start target point
    target_point = np.array([-50e-3, 0, 50e-3, 1])

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Reference bounds from initial view
    e_ref = copy.deepcopy(e_base)
    e_ref.look_at(target_point)
    # Find corneal reflections for both lights

    plot_setup_and_camera_view(
        e_ref, target_point, lights, c, ax1=ax1, ax2=ax2, fig=fig
    )
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    def update():
        nonlocal target_point, e_base
        e = copy.deepcopy(e_base)
        e.look_at(target_point)
        # Find corneal reflections for both lights
        plot_setup_and_camera_view(
            e,
            target_point,
            lights,
            c,
            ax1=ax1,
            ax2=ax2,
            fig=fig,
            ref_bounds=ref_bounds,
        )
        eye_pos = e_base.position
        fig.suptitle(
            f"Target X={target_point[0]*1000:.1f} mm, Y={target_point[1]*1000:.1f} mm, Z={target_point[2]*1000:.1f} mm\n"
            f"Eye X={eye_pos[0]*1000:.1f} mm, Y={eye_pos[1]*1000:.1f} mm, Z={eye_pos[2]*1000:.1f} mm",
            fontsize=14,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal target_point, e_base
        step = 2.5e-3  # 0.25 cm step size

        # TARGET MOVEMENT (Arrow keys only)
        if event.key == "up":
            target_point[2] += step  # Target up (increase Z)
        elif event.key == "down":
            target_point[2] -= step  # Target down (decrease Z)
        elif event.key == "left":
            target_point[0] -= step  # Target left (decrease X)
        elif event.key == "right":
            target_point[0] += step  # Target right (increase X)

        # EYE MOVEMENT (I/M/J/L/./,)
        elif event.key == "j":
            e_base.trans[0, 3] -= step  # Eye left (decrease X)
        elif event.key == "l":
            e_base.trans[0, 3] += step  # Eye right (increase X)
        elif event.key == "i":
            e_base.trans[2, 3] += step  # Eye up (increase Z)
        elif event.key == "k":
            e_base.trans[2, 3] -= step  # Eye down (decrease Z)
        elif event.key == ".":
            e_base.trans[1, 3] -= step  # Eye closer to camera (decrease Y)
        elif event.key == ",":
            e_base.trans[1, 3] += step  # Eye farther from camera (increase Y)

        update()

    fig.canvas.mpl_connect("key_press_event", on_key)

    update()
    plt.show()


if __name__ == "__main__":
    manual_eye_gaze_keyboard_control()
