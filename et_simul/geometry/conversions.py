import numpy as np


def gaze2angle(gaze, rest_pos=None):
    """Determines rotation angles of the eye from given gaze direction.



    angles = gaze2angle(gaze, rest_pos) calculates the rotation angles of the
    eye for the gaze direction 'gaze', relative to the rest position of the
    eye 'rest_pos' (specified as a 3x3 rotation matrix). angles(1) is the
    rotation angle in the x-z-plane, and angles(2) is the angle in the
    y-z-plane. If no rest position is specified, uses [1 0 0; 0 0 1; 0 1 0].

    Args:
        gaze: 3D or 4D gaze direction vector
        rest_pos: Optional 3x3 rotation matrix for eye rest position

    Returns:
        2-element array [horizontal_angle, vertical_angle] in radians
    """
    # Convert to numpy array if needed
    gaze = np.asarray(gaze)

    # Default rest position if not provided
    if rest_pos is None:
        rest_pos = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    else:
        rest_pos = np.asarray(rest_pos)

    # Use only first 3 components of gaze vector
    gaze_3d = gaze[:3]
    # Solve rest_pos * x = gaze_3d for x
    gaze_transformed = np.linalg.solve(rest_pos, gaze_3d)

    angles = np.zeros(2)

    # angles(1) = atan2(gaze(1), -gaze(3))
    angles[0] = np.arctan2(gaze_transformed[0], -gaze_transformed[2])

    # angles(2) = atan2(gaze(2), norm(gaze([1,3])))
    angles[1] = np.arctan2(gaze_transformed[1], np.linalg.norm([gaze_transformed[0], gaze_transformed[2]]))

    return angles


def angle2gaze(angles, rest_pos=None):
    """Calculates gaze direction from rotation angles.



    gaze = angle2gaze(angles, rest_pos) calculates a gaze direction from
    Euler rotation angles. angles(1) is the rotation angle in the
    x-z-plane (applied last), and angles(2) is the rotation angle in the
    y-z-plane (applied first). 'rest_pos' is a 3x3 rotation matrix specifying
    the rest position of the eye; if no rest position is specified, uses
    [1 0 0; 0 0 1; 0 1 0].

    Args:
        angles: 2-element array of rotation angles [x_z_angle, y_z_angle] in radians
        rest_pos: Optional 3x3 rotation matrix for eye rest position

    Returns:
        4D homogeneous gaze direction vector [x, y, z, 0]
    """
    # Convert to numpy array if needed
    angles = np.asarray(angles)

    # Default rest position if not provided
    if rest_pos is None:
        rest_pos = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    else:
        rest_pos = np.asarray(rest_pos)

    # rotat_matrix_x_z
    cos_a1, sin_a1 = np.cos(angles[0]), np.sin(angles[0])
    rotat_matrix_x_z = np.array([[cos_a1, 0, -sin_a1, 0], [0, 1, 0, 0], [sin_a1, 0, cos_a1, 0], [0, 0, 0, 1]])

    # rotat_matrix_y_z
    cos_a2, sin_a2 = np.cos(angles[1]), np.sin(angles[1])
    rotat_matrix_y_z = np.array([[1, 0, 0, 0], [0, cos_a2, -sin_a2, 0], [0, sin_a2, cos_a2, 0], [0, 0, 0, 1]])

    # Convert 3x3 rest_pos to 4x4 homogeneous matrix
    rest_pos_4x4 = np.zeros((4, 4))
    rest_pos_4x4[:3, :3] = rest_pos
    rest_pos_4x4[3, 3] = 1

    # Apply transformations: rest_pos * rotat_matrix_x_z * rotat_matrix_y_z * default_gaze
    default_gaze = np.array([0, 0, -1, 0])
    gaze = rest_pos_4x4 @ rotat_matrix_x_z @ rotat_matrix_y_z @ default_gaze

    return gaze


def calculate_angular_error_degrees(actual_point, predicted_point, observer_pos):
    """Calculate angular error in degrees between two gaze points.

    This function computes the angular error between actual and predicted gaze positions
    by creating 3D gaze vectors from the observer position to each point, normalizing them,
    and calculating the angle between them using the dot product formula.

    Args:
        actual_point: [x, z] actual target position in meters
        predicted_point: [x, z] predicted gaze position in meters
        observer_pos: [x, y, z] or [x, y, z, 1] observer position in meters

    Returns:
        Angular error in degrees
    """
    # Ensure we use only the first 3 components of observer position
    obs_pos_3d = np.asarray(observer_pos)[:3]

    # Create 3D gaze vectors from observer to target points (Y=0 for screen plane)
    gaze3d_real = np.array([actual_point[0], 0, actual_point[1]]) - obs_pos_3d
    gaze3d_measured = np.array([predicted_point[0], 0, predicted_point[1]]) - obs_pos_3d

    # Normalize vectors to unit length
    gaze3d_real = gaze3d_real / np.linalg.norm(gaze3d_real)
    gaze3d_measured = gaze3d_measured / np.linalg.norm(gaze3d_measured)

    # Calculate angle between vectors using dot product
    # Clip to [-1, 1] to handle numerical precision issues
    dot_product = np.clip(np.dot(gaze3d_real, gaze3d_measured), -1, 1)

    # Convert angle from radians to degrees
    angle_deg = 180 / np.pi * np.real(np.arccos(dot_product))

    return angle_deg
