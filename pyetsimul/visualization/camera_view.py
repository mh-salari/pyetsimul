"""Draw what a camera sees of one or more eyes.

Shows the pupil outline, the pupil centre, and the corneal reflections (glints) in image-pixel coordinates.
"""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from ..core import Camera
from ..types import CameraImage
from .plot_config import create_plot_config


def _draw_glint(ax: "Axes", x: float, y: float, color: str, label: str, alpha: float, size: float) -> None:
    """Draw one glint at (x, y) as a star in the given color."""
    ax.scatter(x, y, color=color, s=size, marker="*", edgecolor="none", zorder=3, label=label, alpha=alpha)


def plot_camera_view_of_eye(
    camera_images: list[CameraImage] | CameraImage,
    cameras: list[Camera] | Camera,
    cr_3d_lists: list[list[Any]] | list[Any] | None = None,
    ax: "Axes | None" = None,
    eye_colors: list[str] | None = None,
    camera_colors: list[str] | None = None,
    *,
    zoom: bool = False,
    margin: float = 0.35,
    pupil_fill_color: str | None = None,
    pupil_edge_color: str | None = None,
    glint_colors: list[str] | None = None,
    pupil_linestyles: list[str] | None = None,
    center_markers: list[str] | None = None,
    alpha: float = 1.0,
    linewidth: float = 0.9,
    marker_size: float = 30,
    legend: bool = True,
) -> None:
    """Plot camera views of eyes.

    Shows what each camera sees - the pupil, its centre, and the glints (corneal reflections) - in image
    pixels. Works with a plain image from ``camera.take_image(eye, lights)`` (one pupil, glints from the
    image's own ``corneal_reflections``) and with the 3D-setup path, which attaches several eyes' pupils to one
    image as ``pupil_boundaries`` / ``pupil_centers`` and passes the glints as 3D reflection points in
    ``cr_3d_lists``.

    Args:
        camera_images: CameraImage object or list of CameraImage objects (one per camera)
        cameras: Camera object or list of Camera objects
        cr_3d_lists: list of lists of corneal reflection 3D positions for each eye, or single list. When given,
            glints are projected from these; otherwise they are taken from each image's corneal_reflections.
        ax: Optional matplotlib axis
        eye_colors: Optional list of colors for the eyes.
        camera_colors: Per-camera color for the pupil and centre (distinguishes the models when overlaid).
        zoom: Fit the axes to the drawn pupil and glints with a margin, instead of the full sensor.
        margin: Fraction of the feature span to pad around it when zoom is True.
        pupil_fill_color: Fill color inside the pupil contour; None (the default) draws an outline only.
        pupil_edge_color: Pupil border color; when None the border identifies the camera if several share the
            frame, otherwise the eye.
        glint_colors: Per-camera glint color (one per image); defaults to the reflection gold for all.
        pupil_linestyles: Per-camera pupil-outline line style (e.g. "-", ":", "--"); helps tell apart pupils
            that overlap. Defaults to solid for all.
        center_markers: Per-camera centre marker (e.g. "+", "x", "o"); helps tell apart centres that overlap.
            Defaults to "+" for all.
        alpha: Opacity of the drawn features; below 1 lets overlapping pupils/centres/glints show through.
        linewidth: Line width for the pupil outline and centre marker (kept thin by default).
        marker_size: Scatter size for the centre marker and glint star (kept small by default).
        legend: Whether to add a legend. Turn off when several views are overlaid and the caller labels them.

    """
    if not isinstance(camera_images, list):
        camera_images = [camera_images]
    if not isinstance(cameras, list):
        cameras = [cameras]
    if cr_3d_lists is not None and not isinstance(cr_3d_lists, list):
        cr_3d_lists = [cr_3d_lists]

    if len(camera_images) != len(cameras):
        raise ValueError("Number of camera images must match number of cameras")

    ax2 = ax
    if ax2 is None:
        _, ax2 = plt.subplots()
    else:
        ax2.cla()

    config = create_plot_config()

    if eye_colors is None:
        eye_colors = config.colors.eyes
    if camera_colors is None:
        camera_colors = config.colors.cameras

    multi_camera = len(camera_images) > 1
    all_resolutions = []
    feature_x: list[float] = []
    feature_y: list[float] = []
    has_valid_data = False

    for cam_idx, (camera_image, camera) in enumerate(zip(camera_images, cameras, strict=False)):
        if camera_image is None:
            continue

        cam_color = camera_colors[cam_idx % len(camera_colors)]
        pupil_ls = pupil_linestyles[cam_idx % len(pupil_linestyles)] if pupil_linestyles is not None else "-"
        center_marker = center_markers[cam_idx % len(center_markers)] if center_markers is not None else "+"
        all_resolutions.append(camera.camera_matrix.resolution)

        # An image may carry several eyes as plural fields (the 3D-setup path) or one pupil (a plain rendered
        # image). Read the plural fields when present, else the single pupil; the two run in parallel per eye.
        boundaries = getattr(camera_image, "pupil_boundaries", None)
        centers = getattr(camera_image, "pupil_centers", None)
        if boundaries is None:
            boundaries = [camera_image.pupil_boundary]
            centers = [camera_image.pupil_center]

        for eye_idx, (boundary, center) in enumerate(zip(boundaries, centers, strict=False)):
            # Colour by camera when several share the frame, by eye when one camera shows several, else neutral.
            # The pupil and centre share this colour so a model reads as one set.
            if pupil_edge_color is not None:
                identity_color = pupil_edge_color
            elif multi_camera:
                identity_color = cam_color
            elif len(boundaries) > 1:
                identity_color = eye_colors[eye_idx % len(eye_colors)]
            else:
                identity_color = config.colors.pupil

            if boundary is not None and len(boundary) > 2:
                has_valid_data = True
                pupil_x = [p.x for p in boundary] + [boundary[0].x]
                pupil_y = [p.y for p in boundary] + [boundary[0].y]
                ax2.fill(
                    pupil_x,
                    pupil_y,
                    facecolor=pupil_fill_color if pupil_fill_color is not None else "none",
                    edgecolor=identity_color,
                    linewidth=linewidth,
                    linestyle=pupil_ls,
                    alpha=alpha,
                    zorder=2,
                    label="Pupil",
                )
                feature_x.extend(pupil_x)
                feature_y.extend(pupil_y)

            if center is not None:
                has_valid_data = True
                center_img = center.to_array()
                ax2.scatter(
                    center_img[0],
                    center_img[1],
                    color=identity_color,
                    s=marker_size,
                    marker=center_marker,
                    linewidth=linewidth,
                    alpha=alpha,
                    label="Centre",
                    zorder=4,
                )
                feature_x.append(center_img[0])
                feature_y.append(center_img[1])

        # Glints come from the projected 3D reflection points when given, else the image's own corneal
        # reflections. They keep their own colour: the reflection gold by default, or per camera via glint_colors.
        if glint_colors is not None:
            glint_color = glint_colors[cam_idx % len(glint_colors)]
        else:
            glint_color = config.colors.corneal_reflection
        if cr_3d_lists:
            for cr_3d_list in cr_3d_lists:
                for cr_idx, cr_3d in enumerate(cr_3d_list):
                    if cr_3d is None:
                        continue
                    has_valid_data = True
                    cr_img = camera.project(cr_3d).image_points
                    x, y = float(cr_img[0, 0]), float(cr_img[1, 0])
                    _draw_glint(ax2, x, y, glint_color, f"Glint {cr_idx + 1}", alpha, marker_size)
                    feature_x.append(x)
                    feature_y.append(y)
        else:
            for cr_idx, reflection in enumerate(camera_image.corneal_reflections):
                if reflection is None:
                    continue
                has_valid_data = True
                _draw_glint(ax2, reflection.x, reflection.y, glint_color, f"Glint {cr_idx + 1}", alpha, marker_size)
                feature_x.append(reflection.x)
                feature_y.append(reflection.y)

    if not has_valid_data:
        ax2.text(0, 0, "No camera data to display", ha="center", va="center", fontsize=config.fonts.subtitle)
        return

    if zoom and feature_x:
        cx = (min(feature_x) + max(feature_x)) / 2
        cy = (min(feature_y) + max(feature_y)) / 2
        half = max(max(feature_x) - min(feature_x), max(feature_y) - min(feature_y)) / 2 * (1 + margin)
        half = max(half, 1.0)  # keep a sensible window if the features collapse to a point
        ax2.set_xlim(cx - half, cx + half)
        ax2.set_ylim(cy - half, cy + half)
    elif all_resolutions:
        max_res_x = max(res.x for res in all_resolutions)
        max_res_y = max(res.y for res in all_resolutions)
        ax2.set_xlim(-max_res_x / 2, max_res_x / 2)
        ax2.set_ylim(-max_res_y / 2, max_res_y / 2)
    ax2.invert_yaxis()
    ax2.invert_xaxis()

    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_title("Camera View")
    ax2.grid(config.elements.grid_enabled, alpha=config.lines.grid_alpha)
    if config.elements.equal_aspect:
        ax2.set_aspect("equal")

    if legend:
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=False))  # one legend entry per feature label
            ax2.legend(
                unique.values(), unique.keys(), fontsize=config.fonts.annotation, **config.layout.legend_outside_right
            )
