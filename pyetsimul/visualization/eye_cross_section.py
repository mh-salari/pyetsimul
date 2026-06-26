"""Side-on cross-section of a model eye: corneal profile, optical and visual axes, and the key centres.

A 2D schematic of an eye's optical and geometric parameters in the eye's own frame, with the corneal apex at
the origin and the optical axis along x. It draws the corneal profile (anterior, plus the posterior surface and
tear film for two-surface corneas), the pupil aperture, the sclera meeting the cornea at the limbus, and the
apex, pupil, corneal-centre and rotation-centre positions along the optical axis, with the visual axis tilted
from the optical axis by the horizontal angle kappa toward the fovea. All geometry is read from the eye, so the
figure adapts when its parameters change.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..core import Eye

# Default color per drawn element; the apex, pupil, corneal centre and rotation centre each get their own hue
# so they read as separate dots alongside the axes. Pass colors={...} to override any subset; "visual_axis"
# also colors the fovea it ends at.
DEFAULT_COLORS = {
    "cornea": "black",
    "posterior_cornea": "0.45",
    "tear_film": "#5DADE2",
    "sclera": "0.45",
    "iris": "#7B4AA0",
    "optical_axis": "black",
    "visual_axis": "#16A085",
    "apex": "#E67E22",
    "pupil": "#2980B9",
    "corneal_centre": "#C0392B",
    "rotation_centre": "#27AE60",
}


def _conic_sag(r: np.ndarray, radius: float, k: float) -> np.ndarray:
    """Depth into the eye of a conic surface (apex curvature ``radius``, conic ``k``) at radial distance r."""
    return r**2 / (radius * (1 + np.sqrt(1 - (1 + k) * (r / radius) ** 2)))


def plot_eye_cross_section(
    eye: "Eye", ax: "Axes | None" = None, *, colors: "dict[str, str] | None" = None, legend: bool = True
) -> "Axes":
    """Draw a horizontal cross-section of the eye along the optical axis.

    The optical axis runs along x with the corneal apex at 0; the visual axis is tilted by the horizontal angle
    kappa (alpha) and ends at the fovea on the retina, so the vertical kappa (beta) is out of this plane. The
    anterior cornea is always drawn; the posterior cornea and tear film appear for two-surface corneas. The
    apex, pupil, corneal centre and rotation centre are marked along the optical axis. The eye's current pose is
    not used: the schematic is in the eye's own frame, so it needs no prior ``look_at``.

    Args:
        eye: Eye whose model geometry is drawn.
        ax: Axis to draw on; a new figure and axis are created if None.
        colors: Override the color of any subset of elements, keyed by the names in ``DEFAULT_COLORS``;
            unknown keys raise ``ValueError``.
        legend: Whether to label and show the optical/visual-axis legend.

    Returns:
        The axis drawn on.

    """
    overrides = colors if colors is not None else {}
    unknown = set(overrides) - set(DEFAULT_COLORS)
    if unknown:
        raise ValueError(f"Unknown color keys {sorted(unknown)}; valid keys are {sorted(DEFAULT_COLORS)}")
    color = {**DEFAULT_COLORS, **overrides}

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    cornea = eye.cornea
    radius = cornea.anterior_radius
    k = getattr(cornea, "anterior_k", None) or 0.0
    apex_z = cornea.get_apex_position().z
    alpha = np.radians(eye.model.fovea_alpha_deg)
    axial = eye.model.axial_length
    nodal = 0.29 * axial  # approximate second nodal point, where the optical and visual axes cross

    def depth(z: float) -> float:
        return abs(z - apex_z)

    rc = eye.model.rotation_center
    rot_z = eye.rotation_centre.z if hasattr(rc, "horizontal_depth_mm") else 0.0  # EyeballCenter pivots at origin
    acd = depth(eye.pupil.pos_pupil.z)
    rot_depth = depth(rot_z)
    pupil_d = eye.model.default_pupil_diameter

    # The cornea is a steep cap that ends at the limbus, one corneal depth back; its half-width there is the
    # conic's radial extent at that depth. The globe is the sphere through that limbus ring and the retina (one
    # axial length back), so its anterior pole sits behind the apex, the cornea protrudes, and the two join.
    limbus_depth = cornea.get_corneal_depth()
    limbus_r = np.sqrt(max(0.0, 2 * radius * limbus_depth - (1 + k) * limbus_depth**2))
    if 1 + k > 0:
        limbus_r = min(limbus_r, 0.999 * radius / np.sqrt(1 + k))  # stay within the conic's valid radius
    globe_c = (axial**2 - limbus_depth**2 - limbus_r**2) / (2 * (axial - limbus_depth))
    globe_r = axial - globe_c
    phi_limbus = np.arctan2(limbus_r, limbus_depth - globe_c)

    r = np.linspace(-limbus_r, limbus_r, 220)
    ax.plot(_conic_sag(r, radius, k), r, color=color["cornea"], lw=2.0)
    if cornea.use_posterior_surface:  # back of the cornea (two-surface models only)
        pr, pk = cornea.posterior_radius, getattr(cornea, "posterior_k", 0.0) or 0.0
        pa = depth(cornea.get_posterior_center().z - pr / (1 + pk))
        rp = np.linspace(-limbus_r * 0.85, limbus_r * 0.85, 200)
        ax.plot(pa + _conic_sag(rp, pr, pk), rp, color=color["posterior_cornea"], lw=1.2, ls=":")
        ax.annotate(
            "posterior cornea",
            (pa + 0.8, limbus_r * 0.55),
            xytext=(pa + 4.0, limbus_r + 0.6),
            fontsize=7,
            color=color["posterior_cornea"],
            arrowprops={"arrowstyle": "-", "color": color["posterior_cornea"], "lw": 0.6},
        )
    if getattr(cornea, "use_tear_film", False):  # thin film on the front (thickness exaggerated to be visible)
        ax.plot(_conic_sag(r, radius, k) - 0.18, r, color=color["tear_film"], lw=1.0)
        ax.annotate(
            "tear film",
            (-0.18, limbus_r * 0.72),
            xytext=(-2.6, limbus_r - 0.2),
            fontsize=7,
            color=color["tear_film"],
            arrowprops={"arrowstyle": "-", "color": color["tear_film"], "lw": 0.6},
        )

    b = np.linspace(phi_limbus, -phi_limbus, 320)  # sclera arc: from the limbus around the back to the limbus
    ax.plot(globe_c + globe_r * np.cos(b), globe_r * np.sin(b), color=color["sclera"], lw=2.0)
    for s in (1, -1):  # iris: the aperture edges from the pupil rim out to the limbus
        ax.plot([acd, acd], [s * pupil_d / 2, s * limbus_r * 0.96], color=color["iris"], lw=3.5)

    ca, sa = np.cos(alpha), np.sin(alpha)
    ax.plot([-3, axial], [0, 0], color=color["optical_axis"], ls="--", lw=0.8, zorder=1, label="optical axis")
    dx0 = nodal - globe_c
    t_back = -dx0 * ca + np.sqrt(globe_r**2 - (dx0 * sa) ** 2)  # visual axis meets the retina (globe sphere)
    fovea = (nodal + t_back * ca, t_back * sa)
    ax.plot(
        [-3, fovea[0]],
        [np.tan(alpha) * (-3 - nodal), fovea[1]],
        color=color["visual_axis"],
        ls="-.",
        lw=1.2,
        zorder=1,
        label="visual axis (angle kappa)",
    )
    ax.plot(*fovea, "o", color=color["visual_axis"], ms=4, zorder=5)
    ax.annotate("fovea", fovea, xytext=(fovea[0], fovea[1] + 1.8), ha="center", fontsize=8, color=color["visual_axis"])

    label_y = -(limbus_r + 1.6)
    for x, lbl, key in [
        (0.0, "apex", "apex"),
        (acd, "pupil", "pupil"),
        (radius, "corneal\ncentre", "corneal_centre"),
        (rot_depth, "rotation\ncentre", "rotation_centre"),
    ]:
        c = color[key]
        ax.plot(x, 0, "o", color=c, ms=5, zorder=5)
        ax.annotate(
            lbl,
            (x, 0),
            xytext=(x, label_y - (1.7 if lbl == "pupil" else 0)),
            ha="center",
            fontsize=8,
            color=c,
            arrowprops={"arrowstyle": "-", "color": c, "lw": 0.7},
        )

    ax.set_aspect("equal")
    ax.set_xlim(-3, axial + 1.5)
    ax.set_ylim(-(limbus_r + 5.0), limbus_r + 3.7)
    ax.set_yticks([])
    ax.set_xlabel("mm from corneal apex")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    return ax
