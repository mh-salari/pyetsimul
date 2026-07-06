"""Camera lens distortion: a pinhole vs a real eye-tracker lens, side by side.

Real camera lenses bend the image (radial and tangential distortion); a pinhole does not. PyEtSimul's Camera
takes OpenCV intrinsics and distortion coefficients and applies them when it images an eye, so the apparent
pupil contour, its centre, and the glints come out warped exactly as that lens would warp them. This picks one
lens from a menu of real eye-tracker cameras, builds a matched pinhole (same focal length and resolution, zero
distortion) and the distorted lens, points both at one eye, and opens an interactive view: move the eye and
target with the keyboard to watch the distorted pupil pull away from the pinhole one, most at the edges of the
frame where distortion is strongest.
"""

import numpy as np
from tabulate import tabulate

from pyetsimul.core import Camera, Eye
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import CameraMatrix, Position3D
from pyetsimul.visualization import plot_interactive_cameras

# Published OpenCV calibrations (camera matrix + radial/tangential distortion) of real eye-tracker cameras, from
# Pupil Labs' camera_models (https://github.com/pupil-labs/pupil). Coefficients are in OpenCV order
# (k1, k2, p1, p2, k3, ...). Each entry is one lens the menu can pick.
CAMERA_CONFIGS = [
    {
        "name": "Pupil Cam1 ID2 (low res)",
        "camera_matrix": CameraMatrix(
            np.array([
                [395.60662814306596, 0.0, 316.72212558212516],
                [0.0, 395.56975615889445, 259.206579702132],
                [0.0, 0.0, 1.0],
            ])
        ),
        "dist_coeffs": np.array([
            -0.2430487205352619,
            0.1623502095383119,
            0.0001632500987373085,
            8.322130878440475e-05,
            0.017859803336754784,
            0.1969284124154412,
            0.00577741263771627,
            0.09892258337410824,
        ]),
    },
    {
        "name": "Pupil Cam1 ID2 (high res)",
        "camera_matrix": CameraMatrix(
            np.array([
                [794.3311439869655, 0.0, 633.0104437728625],
                [0.0, 793.5290139393004, 397.36927353414865],
                [0.0, 0.0, 1.0],
            ])
        ),
        "dist_coeffs": np.array([
            -0.3758628065070806,
            0.1643326166951343,
            0.00012182540692089567,
            0.00013422608638039466,
            0.03343691733865076,
            0.08235235770849726,
            -0.08225804883227375,
            0.14463365333602152,
        ]),
    },
    {
        "name": "Neon Sensor Module v1",
        "camera_matrix": CameraMatrix(
            np.array([
                [140.68445787837342, 0.0, 99.42393317744813],
                [0.0, 140.67571954970256, 96.235134525304],
                [0.0, 0.0, 1.0],
            ])
        ),
        "dist_coeffs": np.array([
            0.05449484235207129,
            -0.14013187141454536,
            0.0006598061556076783,
            5.0572400552608696e-05,
            -0.6158040573125376,
            -0.048953803434398195,
            0.04521347340211147,
            -0.7004955138758611,
        ]),
    },
    {
        "name": "Pupil Cam2 ID0 (no distortion)",
        "camera_matrix": CameraMatrix(
            np.array([
                [282.976877, 0.0, 96.0],
                [0.0, 283.561467, 96.0],
                [0.0, 0.0, 1.0],
            ])
        ),
        "dist_coeffs": np.zeros(5),
    },
]

# Pick a lens from the menu.
print("Available camera lenses:")
for i, config in enumerate(CAMERA_CONFIGS, 1):
    res = config["camera_matrix"].resolution
    print(f"  {i}. {config['name']} ({res.x}x{res.y})")
try:
    choice = int(input(f"Select a lens (1-{len(CAMERA_CONFIGS)}): ")) - 1
except (ValueError, KeyboardInterrupt):
    choice = -1
if not 0 <= choice < len(CAMERA_CONFIGS):
    choice = 0  # an out-of-range or empty entry falls to the first lens, announced rather than silent
    print(f"Out of range; using the first lens: {CAMERA_CONFIGS[0]['name']}")
lens = CAMERA_CONFIGS[choice]

# A matched pinhole: same focal length and resolution as the lens but zero distortion, so the only difference
# between the two views is the lens distortion itself.
pinhole = Camera(name="pinhole")
pinhole.camera_matrix.focal_length = lens["camera_matrix"].focal_length
pinhole.camera_matrix.resolution = lens["camera_matrix"].resolution
# The real lens: its OpenCV intrinsics and distortion coefficients, which project/take_image apply.
distorted = Camera(name=lens["name"], camera_matrix=lens["camera_matrix"], dist_coeffs=lens["dist_coeffs"])

base = get_eye_model("PyEtSimul")
eye = Eye(model=base)
# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
target = Position3D(0, 0, 0)  # the point the eye gazes at (move it with the arrow keys)
eye.position = Position3D(0, 30, 0)  # ~30 mm in front, like a head-mounted eye camera
eye.set_rest_orientation_at_target(target, up=Position3D(0, 0, 1))  # rest it facing the camera
eye.look_at(target)  # rotate the eye to gaze at the target
pinhole.point_at(eye.position)  # aim both cameras at the eye
distorted.point_at(eye.position)

# Compare the two cameras' parameters: same intrinsics, distortion only on the lens.
coeffs = list(distorted.dist_coeffs) + [0.0] * 5  # pad so the first five are always present
rows = [
    ["focal length (px)", f"{pinhole.camera_matrix.focal_length:.1f}", f"{distorted.camera_matrix.focal_length:.1f}"],
    [
        "principal point (px)",
        f"({pinhole.camera_matrix.matrix[0, 2]:.1f}, {pinhole.camera_matrix.matrix[1, 2]:.1f})",
        f"({distorted.camera_matrix.matrix[0, 2]:.1f}, {distorted.camera_matrix.matrix[1, 2]:.1f})",
    ],
    *[[label, "0.000000", f"{coeffs[i]:.6f}"] for i, label in enumerate(["k1", "k2", "p1", "p2", "k3"])],
]
print()
print(tabulate(rows, headers=["parameter", "pinhole", lens["name"]], tablefmt="grid"))

print("\nMove the eye toward a frame edge to see the lens distortion grow; the viewer lists the keyboard controls.")
plot_interactive_cameras([pinhole, distorted], eye, target)
