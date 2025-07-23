"""Eye tracking setup visualization module.

This module provides comprehensive visualization functions for eye tracking
setups, including 3D scene views and camera image views.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_eye_tracking_visualization(eye, target_point, lights, camera, cr_3d_list=None, 
                                    ax1=None, ax2=None, fig=None, ref_bounds=None):
    """Create comprehensive eye tracking visualization with 3D setup and camera view.
    
    Args:
        eye: Eye object with transformation matrix and anatomy
        target_point: Target point [x, y, z, 1] or [x, y, z]
        lights: List of Light objects with positions
        camera: Camera object with transformation and parameters
        cr_3d_list: List of corneal reflection 3D positions
        ax1, ax2: Optional matplotlib axes for reuse
        fig: Optional matplotlib figure for reuse  
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys
        
    Returns:
        fig: Matplotlib figure object
    """
    # Create figure and axes if not provided
    if ax1 is None or ax2 is None:
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        ax1.cla()
        ax2.cla()

    # Transform eye coordinates to world coordinates
    def transform_point(point):
        return eye.trans @ point

    # Get eye anatomy points
    cornea_center = eye.pos_cornea
    apex_point = eye.pos_apex
    pupil_center = eye.pos_pupil
    r_cornea = eye.r_cornea
    depth_cornea = eye.depth_cornea

    # Transform anatomical points to world coordinates
    cornea_center_world = transform_point(cornea_center)
    apex_world = transform_point(apex_point)
    pupil_world = transform_point(pupil_center)

    # Draw corneal surface
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    X_cornea = r_cornea * np.outer(np.cos(u), np.sin(v))
    Y_cornea = r_cornea * np.outer(np.sin(u), np.sin(v))
    Z_cornea = r_cornea * np.outer(np.ones(np.size(u)), np.cos(v))

    # Only show anterior surface (cap)
    mask = Z_cornea > -r_cornea + depth_cornea
    X_cornea[mask] = np.nan
    Y_cornea[mask] = np.nan
    Z_cornea[mask] = np.nan

    # Transform cornea surface points to world coordinates
    for i in range(X_cornea.shape[0]):
        for j in range(X_cornea.shape[1]):
            if not np.isnan(X_cornea[i, j]):
                point = cornea_center[:3] + np.array([X_cornea[i, j], Y_cornea[i, j], Z_cornea[i, j]])
                point_world = transform_point(np.append(point, 1))
                X_cornea[i, j] = point_world[0]
                Y_cornea[i, j] = point_world[1]
                Z_cornea[i, j] = point_world[2]

    # Plot corneal surface
    ax1.plot_surface(X_cornea, Y_cornea, Z_cornea, alpha=0.6, color='lightblue', 
                     edgecolor='blue', linewidth=0.1)

    # Mark key anatomical points
    ax1.scatter(*cornea_center_world[:3], color='green', s=100, label='Cornea Center')
    ax1.scatter(*pupil_world[:3], color='cornflowerblue', s=100, label='Pupil Center')

    # Draw optical axis - points along negative z-axis in eye coordinates
    # Transform the negative z-axis direction from eye to world coordinates
    optical_axis_direction_local = np.array([0, 0, -1, 0])  # negative z in homogeneous coordinates
    optical_axis_direction_world = transform_point(optical_axis_direction_local)[:3]
    
    # Extend optical axis from cornea center outward
    optical_axis_length = 0.1  # 100mm extension
    optical_axis_end = cornea_center_world[:3] + optical_axis_direction_world * optical_axis_length

    # Draw optical axis 
    ax1.plot([cornea_center_world[0], optical_axis_end[0]],
             [cornea_center_world[1], optical_axis_end[1]],
             [cornea_center_world[2], optical_axis_end[2]],
             'g--', linewidth=3, label='Optical Axis')

    # Draw visual axis to target
    if len(target_point) == 3:
        target_point = np.append(target_point, 1)
    target_world = target_point

    ax1.plot([pupil_world[0], target_world[0]],
             [pupil_world[1], target_world[1]],
             [pupil_world[2], target_world[2]],
             'r--', linewidth=3, label='Visual Axis')

    # Add scene elements - multiple lights
    light_colors = ['yellow', 'orange', 'gold', 'khaki']  # Colors for different lights
    for i, light in enumerate(lights):
        light_pos = light.position[:3]
        color = light_colors[i % len(light_colors)]
        ax1.scatter(*light_pos, color=color, s=200, marker='*', 
                   label=f'Light Source {i+1}')

    camera_pos = camera.trans[:3, 3]
    ax1.scatter(*camera_pos, color='black', s=200, marker='s', label='Camera')

    ax1.scatter(*target_world[:3], color='magenta', s=150, marker='D', label='Gaze Target')

    # Add corneal reflections if provided
    if cr_3d_list is not None:
        cr_colors = ['#FFE171', '#F9F871', '#FFD67C', '#C9AF41']  # Custom colors for different CRs
        for i, cr_3d in enumerate(cr_3d_list):
            if cr_3d is not None:
                color = cr_colors[i % len(cr_colors)]
                ax1.scatter(*cr_3d[:3], color=color, s=80, marker='o', 
                           edgecolor='black', linewidth=1, label=f'CR {i+1}')
                
                # Get corresponding light position
                light = lights[i]
                light_pos = light.position[:3]
                
                # Draw light ray paths (only from light to CR, not to camera)
                ax1.plot([light_pos[0], cr_3d[0]], [light_pos[1], cr_3d[1]], [light_pos[2], cr_3d[2]], 
                        color=color, linestyle='-', linewidth=2, alpha=0.7)

    # 3D plot formatting
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Eye Tracking Setup')
    ax1.legend(loc='upper left')
    
    # Convert axes to mm for better readability
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.0f}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.0f}'))
    ax1.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.0f}'))

    # Apply reference bounds if provided
    if ref_bounds:
        ax1.set_xlim(ref_bounds['x'])
        ax1.set_ylim(ref_bounds['y'])
        ax1.set_zlim(ref_bounds['z'])

    # Camera view
    camera_image = camera.take_image(eye, lights)

    # Draw pupil in camera image
    if camera_image['pupil'] is not None and camera_image['pupil'].shape[1] > 2:
        pupil_points_img = camera_image['pupil']
        closed_pupil_points = np.hstack((pupil_points_img, pupil_points_img[:, 0:1]))
        ax2.plot(closed_pupil_points[0, :], closed_pupil_points[1, :], 
                color='cornflowerblue', linewidth=3, label='Pupil')

    # Draw pupil center in camera image
    if camera_image['pc'] is not None:
        pupil_center_img = camera_image['pc']
        ax2.scatter(pupil_center_img[0], pupil_center_img[1], 
                   color='cornflowerblue', s=100, marker='+', linewidth=3, label='Pupil Center')

    # Draw corneal reflections in camera image
    if camera_image['cr']:
        cr_colors = ['#FFE171', '#F9F871', '#FFD67C', '#C9AF41']  # Same colors as 3D view
        for i, cr_3d in enumerate(cr_3d_list or []):
            if cr_3d is not None and i < len(camera_image['cr']) and camera_image['cr'][i] is not None:
                cr_img, _, _ = camera.project(cr_3d)
                
                color = cr_colors[i % len(cr_colors)]
                ax2.scatter(cr_img[0, 0], cr_img[1, 0], color=color, s=80, marker='o', 
                           edgecolor='black', linewidth=1, label=f'CR {i+1}')

    # Set camera image limits
    resolution = camera.resolution
        
    ax2.set_xlim(-resolution[0]/2, resolution[0]/2)
    ax2.set_ylim(-resolution[1]/2, resolution[1]/2)

    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_title('Camera View of Eye')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend()

    # Add measurement annotations for multiple CRs
    if camera_image['pc'] is not None and camera_image['cr']:
        cr_colors = ['#FFE171', '#F9F871', '#FFD67C', '#C9AF41']
        for i, cr_3d in enumerate(cr_3d_list or []):
            if cr_3d is not None and i < len(camera_image['cr']) and camera_image['cr'][i] is not None:
                cr_img, _, _ = camera.project(cr_3d)
                
                pupil_cr_vector = cr_img.flatten() - pupil_center_img
                pupil_cr_distance_pixels = np.linalg.norm(pupil_cr_vector)
                
                color = cr_colors[i % len(cr_colors)]
                ax2.plot([pupil_center_img[0], cr_img[0, 0]], [pupil_center_img[1], cr_img[1, 0]], 
                        color=color, alpha=0.7, linewidth=2)
                
                mid_point = [(pupil_center_img[0] + cr_img[0, 0])/2, (pupil_center_img[1] + cr_img[1, 0])/2]
                ax2.annotate(f'{pupil_cr_distance_pixels:.1f} px', xy=mid_point, 
                             xytext=(10, 10 + i*15), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                             fontsize=9)

    return fig


def setup_interactive_plot(eye_base, light, camera, target_point):
    """Setup interactive plot with reference bounds for consistent view.
    
    Args:
        eye_base: Base eye object
        light: Light object
        camera: Camera object
        target_point: Initial target point
        
    Returns:
        dict: Contains fig, ax1, ax2, ref_bounds for reuse
    """
    
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    # Create reference bounds from initial view
    e_ref = eye_base.copy()
    e_ref.look_at(target_point)
    cr_ref = e_ref.find_cr(light, camera)
        
    create_eye_tracking_visualization(e_ref, target_point, light, camera, cr_ref, 
                                    ax1=ax1, ax2=ax2, fig=fig)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {'x': xlim, 'y': ylim, 'z': zlim}

    return {
        'fig': fig,
        'ax1': ax1,
        'ax2': ax2,
        'ref_bounds': ref_bounds
    }


def update_interactive_plot(plot_setup, eye_base, light, camera, target_point):
    """Update interactive plot with new target position.
    
    Args:
        plot_setup: Dict returned from setup_interactive_plot
        eye_base: Base eye object
        light: Light object
        camera: Camera object
        target_point: New target point
    """
    e = eye_base.copy()
    e.look_at(target_point)
    cr_3d = e.find_cr(light, camera)
        
    create_eye_tracking_visualization(e, target_point, light, camera, cr_3d, 
                                    ax1=plot_setup['ax1'], ax2=plot_setup['ax2'], 
                                    fig=plot_setup['fig'], ref_bounds=plot_setup['ref_bounds'])
    
    plot_setup['fig'].suptitle(
        f"Target X={target_point[0]*1000:.1f} mm, Y={target_point[1]*1000:.1f} mm, Z={target_point[2]*1000:.1f} mm",
        fontsize=16
    )
    plot_setup['fig'].canvas.draw_idle()