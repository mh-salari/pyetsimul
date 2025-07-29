import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
from skimage.measure import EllipseModel

from ..optics import refractions
from .camera import Camera
from .light import Light
from .coordinate_system import validate_orientation_matrix
from .pupil import Pupil, create_pupil
from .cornea import Cornea, SphericalCornea, SpheroidCornea


@dataclass
class Eye:
    """Creates a structure that represents an eye.

     The center of rotation of the eye lies at the origin of the local eye
     coordinate system. The optical axis of the eye points out along the
     negative z axis. The x axis lies in the horizontal plane, the y axis in
     the sagittal plane.

     The eye object 'e' that is produced contains the following elements:

     - 'trans' is the transformation matrix from eye to world coordinates.

     - 'rest_orientation' is a 3x3 rotation matrix that specifies the rest position of
       the eye -- i.e. a transformation that rotates the local eye coordinate
       system into the rest position of the eye in world coordinates. The rest
       position is important for computing the amount of torsion that occurs
       during eye rotations (as given by Listing's law).

    - 'r_cornea' is the radius of corneal curvature (the cornea is modeled as
      a spherical surface).

    - 'pos_cornea' is the corneal curvature center.

    - 'r_cornea_inner' is the curvature radius of the cornea's inner surface.

    - 'cornea_inner_center' is the curvature center of the cornea's inner
      surface.

    - 'n_cornea' is the refractive index of the cornea.

    - 'n_aqueous_humor' is the refractive index of the aqueous humor.

    - 'pos_apex' is the position of the apex (the frontmost part of the
      cornea).

    - 'depth_cornea' is the distance, measured on the optical axis, between
      the cornea apex and the projection of the limbus (the boundary of the
      cornea) onto the optical axis.

    - 'pos_pupil' is the position of the center of the pupil.

    - 'x_pupil' and 'y_pupil' are two orthogonal vectors that extend
     from the center of the pupil to its border. Any position on the pupil's
     border can thus be obtained by taking

     pos_pupil + cos(alpha)*x_pupil + sin(alpha)*y_pupil

     Eye Model Components (based on Böhme et al. 2008):
     1. Cornea: Spherical cap with radius r_cornea, center pos_cornea on optical axis
     2. Pupil: Circle with radius pupil_radius, center pos_pupil on optical axis
     3. Visual axis displacement: Horizontal angle fovea_alpha_deg and vertical angle
        fovea_beta_deg
     4. Numerical values for these parameters and others that follow are taken
        from the standard eye in Boff and Lincoln [1988, Section 1.210]

     Note: The model assumes a simplified spherical eye with the nodal point at the
     cornea center, which is sufficient for eye tracking simulation purposes.

     This class is based on the original MATLAB implementation from the
     et_simul project — © 2008 Martin Böhme, University of Lübeck.
     Python port © 2025 Mohammadhossein Salari.
     Licensed under the GNU GPL v3.0 or later.
    """

    # Instance parameters
    cornea: Optional[Cornea] = None  # Cornea object (SphericalCornea or SpheroidCornea)
    fovea_displacement: bool = True
    fovea_alpha_deg: float = 6.0  # Horizontal fovea displacement (degrees)
    fovea_beta_deg: float = 2.0  # Vertical fovea displacement (degrees)
    pupil_type: str = "elliptical"  # Pupil type: "elliptical" (default), "realistic"

    # These fields are calculated in __post_init__
    trans: np.ndarray = field(init=False)
    _rest_orientation: np.ndarray = field(init=False)
    r_cornea_inner: float = field(init=False)
    axial_length: float = field(init=False)  # Calculated axial length (m)
    cornea_center_to_rotation_center: float = field(init=False)  # Calculated distance (m)
    cornea_thickness_offset: float = field(init=False)  # Calculated thickness offset (m)
    cornea_inner_center: np.ndarray = field(init=False)
    n_cornea: float = field(init=False)
    n_aqueous_humor: float = field(init=False)
    pos_apex: np.ndarray = field(init=False)
    depth_cornea: float = field(init=False)
    pupil: Pupil = field(init=False)  # Pupil object that handles all pupil calculations

    def __post_init__(self) -> None:
        """Initializes the eye's anatomical properties based on constructor parameters."""
        # Default constants for scaling (meters) from Boff and Lincoln [1988, Section 1.210]
        r_cornea_default = 7.98e-3  # Default outer corneal radius (m)
        r_cornea_inner_default = 6.22e-3  # Default inner corneal radius (m)
        cornea_depth_default = 3.54e-3  # Corneal depth (distance from apex to limbus projection) (m)
        pupil_radius_default = 3e-3  # Default pupil radius (m)
        n_cornea_default = 1.376  # Refractive index of cornea
        n_aqueous_humor_default = 1.336  # Refractive index of aqueous humor
        axial_length_default = 24.75e-3  # Default total axial length of eye (m)
        cornea_center_to_rotation_center_default = (
            10.20e-3  # Default distance from corneal center to rotation center (m)
        )

        cornea_thickness_offset_default = 1.15e-3  # Default corneal thickness offset (m)

        # Create default cornea if none provided
        if self.cornea is None:
            self.cornea = SphericalCornea(radius=r_cornea_default)

        # Calculate scale factor based on corneal radius
        scale = self.cornea.radius / r_cornea_default

        # Calculate anatomical position if center not provided
        if self.cornea.center is None:
            cornea_z_offset = axial_length_default - 2 * cornea_center_to_rotation_center_default
            self.cornea.center = np.array([0, 0, -scale * cornea_z_offset, 1])

        # Initialize transformation matrix (identity at rest position)
        self.trans = np.eye(4)
        self._rest_orientation = np.eye(3)
        self.trans[:3, :3] = self._rest_orientation

        # Set anatomical parameters (original MATLAB scaling)
        self.axial_length = axial_length_default
        self.cornea_center_to_rotation_center = cornea_center_to_rotation_center_default
        self.cornea_thickness_offset = cornea_thickness_offset_default

        # Inner corneal surface radius (scaled)
        self.r_cornea_inner = scale * r_cornea_inner_default

        # Inner corneal surface center
        thickness_term = self.cornea.radius - self.r_cornea_inner - scale * self.cornea_thickness_offset
        self.cornea_inner_center = self.cornea.center - np.array([0, 0, thickness_term, 0])

        # Refractive indices
        self.n_cornea = n_cornea_default
        self.n_aqueous_humor = n_aqueous_humor_default

        # Corneal apex (frontmost point)
        self.pos_apex = self.cornea.center + np.array([0, 0, -self.cornea.radius, 0])

        # Corneal depth (scaled)
        self.depth_cornea = scale * cornea_depth_default

        # Create pupil object using factory pattern
        pos_pupil = self.pos_apex + np.array([0, 0, scale * cornea_depth_default, 0])
        pupil_radius_scaled = scale * pupil_radius_default
        x_pupil = pupil_radius_scaled * np.array([1, 0, 0, 0])
        y_pupil = pupil_radius_scaled * np.array([0, 1, 0, 0])

        self.pupil = create_pupil(pupil_type=self.pupil_type, pos_pupil=pos_pupil, x_pupil=x_pupil, y_pupil=y_pupil)

    @property
    def orientation(self) -> np.ndarray:
        """Get/set the eye's current orientation (3x3 rotation matrix)."""
        return self.trans[:3, :3]

    @orientation.setter
    def orientation(self, value: np.ndarray) -> None:
        """Set the eye's current orientation and update transformation matrix."""
        self.trans[:3, :3] = value

    def set_rest_orientation(self, value: np.ndarray) -> None:
        """Set the rest orientation and initialize current orientation to match.

        Args:
            value: 3x3 rotation matrix (must be right-handed with determinant = +1)

        Raises:
            ValueError: If the matrix is not right-handed (det ≠ +1)
        """
        # Validate orientation matrix
        validate_orientation_matrix(value, "Eye")

        self._rest_orientation = value.copy()
        self.trans[:3, :3] = value

    @property
    def rest_orientation(self) -> np.ndarray:
        """Get the rest orientation (read-only)."""
        return self._rest_orientation.copy()

    @property
    def position(self) -> np.ndarray:
        """Get/set the eye's position in world coordinates (3D vector)."""
        return self.trans[:3, 3]

    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Set the eye's position and update transformation matrix."""
        self.trans[:3, 3] = value

    def point_within_cornea(self, p: np.ndarray) -> bool:
        """Tests whether a point lies within the cornea.

        within=point_within_cornea(e, p) tests whether the point 'p', lying
        on the corneal sphere of the eye 'e', lies within the boundaries of the
        cornea, as defined by e.depth_cornea. This function is used by
        find_cr() and find_refraction_sphere().

        Args:
            p: Point to test (4D homogeneous coordinates)

        Returns:
            bool: True if point lies within cornea boundaries, False otherwise

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Line 25: p=e.trans\p;
        p = np.linalg.solve(self.trans, p)

        # Use cornea object's point_within_cornea method
        return self.cornea.point_within_cornea(p, self)

    def find_cr(self, l: Light, c: Camera) -> Optional[np.ndarray]:
        """Finds the position of a corneal reflex.

        cr = find_cr(e, l, c) finds the position of the corneal reflex
        generated by light 'l' on the eye 'e' as seen by the camera 'c', i.e. it
        determines the point 'cr' on the surface of the cornea where a ray
        emanating from 'l' will be reflected directly onto 'c'. If the reflex did
        not fall within the cornea, [] is returned.

        Args:
            l: Light source structure
            c: Camera structure

        Returns:
            4D homogeneous position of corneal reflex, or None if not within cornea

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Line 26: cr=find_reflection(l.pos, c.trans(:,4), e.trans*e.pos_cornea, e.r_cornea);
        cr = self.cornea.find_reflection(l._pos_homogeneous, c.trans[:, 3], self.trans)

        # Lines 29-31: if ~eye_point_within_cornea(e, cr), cr=[]; end
        if cr is not None and not self.point_within_cornea(cr):
            cr = None

        return cr

    @staticmethod
    def listings_law(out_rest: np.ndarray, out_new: np.ndarray) -> np.ndarray:
        """Uses Listing's law to compute a matrix for eye rotation.

        A = listings_law(out_rest, out_new) uses Listing's law to compute a
        matrix 'A' for the rotation an eye makes when the direction of its
        optical axis moves from the rest position 'out_rest' to the new position
        'out_new'.

        Args:
            out_rest: Direction of optical axis in rest position (3D vector)
            out_new: Direction of optical axis in new position (3D vector)

        Returns:
            3x3 rotation matrix A representing the eye rotation

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Lines 25-27: Normalize out_rest and out_new
        out_rest = out_rest / np.linalg.norm(out_rest)
        out_new = out_new / np.linalg.norm(out_new)

        # Line 30: axis1=cross(out_new, out_rest);
        axis1 = np.cross(out_new, out_rest)

        # Line 32: if norm(axis1)==0
        if np.linalg.norm(axis1) == 0:
            # Line 33: A=eye(3);
            A = np.eye(3)
        else:
            # Line 35: axis=axis1/norm(axis1);
            axis = axis1 / np.linalg.norm(axis1)
            # Line 36: axis=axis/norm(axis);
            axis = axis / np.linalg.norm(axis)

            # Lines 40-41: Calculate third vectors
            third_rest = np.cross(out_rest, axis)
            third_new = np.cross(out_new, axis)

            # Line 43: A=[axis out_new third_new]*[axis'; out_rest'; third_rest'];
            left_matrix = np.column_stack([axis, out_new, third_new])
            right_matrix = np.vstack([axis, out_rest, third_rest])
            A = left_matrix @ right_matrix

        return A

    def look_at(self, pos: Union[np.ndarray, List[float]]) -> None:
        """Rotates an eye to look at a given position in space.

        e=eye_look_at(e, pos) rotates the eye 'e' to look at the
        three-dimensional position 'pos' in world coordinates and returns the
        rotated eye. Listing's law is used to compute the amount of torsion that
        occurs. If the global variable FEAT_FOVEA_DISPLACEMENT is true, the
        displacement of the fovea from the optical axis is taken into account
        (i.e. the line-of-sight will be aligned with the given position, not the
        optical axis).

        Args:
            pos: Three-dimensional position in world coordinates to look at

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Line 31: out=pos(1:3)-e.trans(1:3,4);
        out = pos[:3] - self.position
        # Line 32: out=out/norm(out);
        out = out / np.linalg.norm(out)

        # Line 35-36: out_rest=e.rest_pos*[0 0 -1]'; e.trans(1:3, 1:3)=listings_law(out_rest, out)*e.rest_pos;
        out_rest = self._rest_orientation @ np.array([0, 0, -1])
        self.orientation = self.listings_law(out_rest, out) @ self._rest_orientation

        # Lines 39-46: Compensate for fovea displacement
        if self.fovea_displacement:
            # Use configurable fovea displacement angles
            alpha = self.fovea_alpha_deg / 180 * np.pi  # Horizontal displacement
            beta = self.fovea_beta_deg / 180 * np.pi  # Vertical displacement
            # Line 43: A=[cos(alpha) 0 -sin(alpha); 0 1 0; sin(alpha) 0 cos(alpha)];
            A = np.array(
                [
                    [np.cos(alpha), 0, -np.sin(alpha)],
                    [0, 1, 0],
                    [np.sin(alpha), 0, np.cos(alpha)],
                ]
            )
            # Line 44: B=[1 0 0; 0 cos(beta) sin(beta); 0 -sin(beta) cos(beta)];
            B = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(beta), np.sin(beta)],
                    [0, -np.sin(beta), np.cos(beta)],
                ]
            )
            # Line 45: e.trans(1:3, 1:3)=e.trans(1:3, 1:3)*B*A;
            self.orientation = self.orientation @ B @ A

    def get_pupil(self) -> np.ndarray:
        """Returns an array of points describing the pupil boundary.

        X = get_pupil(e) returns a 4×N matrix of points (in world
        coordinates) on the pupil boundary of the eye 'e', where N is
        determined by the pupil's resolution setting.

        Returns:
            4×N matrix of points in world coordinates on pupil boundary

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Get pupil boundary points from pupil object
        pupil_points = self.pupil.get_boundary_points()

        # Transform to world coordinates
        pupil_world = self.trans @ pupil_points

        return pupil_world

    def get_pupil_in_camera_image(
        self, c: Camera, use_refraction: bool = True, center_method: str = "ellipse"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Gets pupil boundary and center in camera image using specified method.

        Args:
            c: Camera object
            use_refraction: Whether to use refraction model (default True)
            center_method: Method to use for pupil center detection (default "ellipse")
                          Options: "ellipse", "center_of_mass"

        Returns:
            Tuple of (pupil_boundary, pupil_center) where:
            - pupil_boundary: 2×N matrix of pupil boundary points in camera image
            - pupil_center: 2-element vector with pupil center position, or None if not found

        Raises:
            ValueError: If center_method is not recognized
        """
        # Get pupil boundary points in camera image
        pupil_boundary = self.get_pupil_boundary_in_camera_image(c, use_refraction=use_refraction)

        # Calculate center using specified method
        if center_method == "ellipse":
            pupil_center = self._fit_ellipse_center(pupil_boundary)
        elif center_method == "center_of_mass":
            pupil_center = self._calculate_center_of_mass(pupil_boundary, c.resolution)
        else:
            raise ValueError(f"Unknown center_method '{center_method}'. Use 'ellipse' or 'center_of_mass'")

        return pupil_boundary, pupil_center

    def get_pupil_radii(self) -> tuple[float, float]:
        """Returns the current pupil radii from both axes.

        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        return self.pupil.get_radii()

    def set_pupil_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Sets the pupil radii and updates pupil geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)

        Raises:
            ValueError: If both radii are None
        """
        self.pupil.set_radii(x_radius, y_radius)

    def find_cr_simple(self, l: Light, c: Camera) -> Optional[np.ndarray]:
        """Finds the position of a corneal reflex (simplified).

        cr = find_cr_simple(e, l, c) finds the position of the corneal reflex
        generated by light 'l' on the eye 'e' as seen by the camera 'c'. In
        contrast to find_cr(), which computes the position of the CR exactly,
        this routine uses a paraxial approximation proposed by Morimoto, Amir and
        Flicker ('Detecting Eye Position and Gaze from a Single Camera and 2
        Light Sources').

        Args:
            l: Light source structure
            c: Camera structure

        Returns:
            4D homogeneous position of corneal reflex, or None if not found

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Line 27: cc=e.trans*e.pos_cornea;
        cc = self.trans @ self.cornea.center

        # Line 28: to_cam=c.trans(:,4)-cc;
        to_cam = c.trans[:, 3] - cc

        # Line 29: to_cam=to_cam/norm(to_cam);
        to_cam = to_cam / np.linalg.norm(to_cam)

        # Line 30: w=e.r_cornea/(2*(l.pos-cc)'*to_cam);
        light_to_cornea = l._pos_homogeneous - cc
        denominator = 2 * np.dot(light_to_cornea, to_cam)

        if abs(denominator) < 1e-10:  # Avoid division by zero
            return None

        w = self.cornea.radius / denominator

        # Line 31: cr=cc+w*(l.pos-cc);
        cr = cc + w * light_to_cornea

        # Check if point is within cornea boundaries (improvement over original)
        if not self.point_within_cornea(cr):
            return None

        return cr

    def refract_ray(self, R0: np.ndarray, Rd: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Computes refraction of ray at cornea surface.

        [U0, Ud] = refract_ray(e, R0, Rd) takes a ray (specified by its
        origin 'R0' and direction 'Rd') entering the eye 'e' from the outside and
        computes the point 'U0' where the ray strikes the surface of the cornea
        and the direction 'Ud' of the refracted ray.

        Returns [] for 'U0' and 'Ud' if the ray does not strike the eye.

        'R0', 'Rd', 'O0', 'I0' and 'Id' are in world coordinates. 3D or
        homogeneous coordinates may be passed in; the same type of coordinates is
        passed out.

        Args:
            R0: Ray origin (3D or 4D homogeneous coordinates)
            Rd: Ray direction (3D or 4D homogeneous coordinates)

        Returns:
            Tuple of (U0, Ud) where U0 is intersection point, Ud is refracted direction.
            Returns (None, None) if ray doesn't strike eye.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """

        # Line 32-33: Compute refraction at surface of cornea
        # Compute corneal center position (4D homogeneous)
        cornea_center = self.trans @ self.cornea.center

        # Use appropriate refraction method based on cornea type
        if isinstance(self.cornea, SphericalCornea):
            # refract_ray_sphere handles 3D/4D coordinates properly (fixes MATLAB's coordinate bug)
            U0, Ud = refractions.refract_ray_sphere(R0, Rd, cornea_center, self.cornea.radius, 1.0, self.n_cornea)
        elif isinstance(self.cornea, SpheroidCornea):
            # Use spheroid refraction
            U0, Ud = refractions.refract_ray_spheroid(
                R0, Rd, cornea_center, self.cornea.a, self.cornea.b, self.cornea.c, 1.0, self.n_cornea
            )
        else:
            raise NotImplementedError(f"Refraction not implemented for cornea type: {type(self.cornea)}")

        return U0, Ud

    def refract_ray_advanced(
        self, R0: np.ndarray, Rd: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Computes refraction at both surfaces of cornea.

        [O0, I0, Id] = refract_ray_advanced(e, R0, Rd) takes a ray (specified by its
        origin 'R0' and direction 'Rd') entering the eye 'e' from the outside and
        computes refraction at both the outer and inner surfaces of the cornea.

        Returns:
        - O0: Point where ray strikes outer corneal surface
        - I0: Point where ray strikes inner corneal surface
        - Id: Direction of ray exiting inner surface

        Returns (None, None, None) if ray doesn't strike eye.

        'R0', 'Rd', 'O0', 'I0' and 'Id' are in world coordinates. 3D or
        homogeneous coordinates may be passed in; the same type of coordinates is
        passed out.

        Args:
            R0: Ray origin (3D or 4D homogeneous coordinates)
            Rd: Ray direction (3D or 4D homogeneous coordinates)

        Returns:
            Tuple of (O0, I0, Id) where:
            - O0: Point where ray strikes outer corneal surface
            - I0: Point where ray strikes inner corneal surface
            - Id: Direction of ray exiting inner surface
            Returns (None, None, None) if ray doesn't strike eye.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Line 34-35: Compute refraction at outer surface of cornea
        # Compute corneal center positions (4D homogeneous)
        cornea_center = self.trans @ self.cornea.center

        # Use appropriate refraction method based on cornea type
        if isinstance(self.cornea, SphericalCornea):
            # refract_ray_sphere handles 3D/4D coordinates properly (fixes MATLAB's coordinate bug)
            O0, Od = refractions.refract_ray_sphere(R0, Rd, cornea_center, self.cornea.radius, 1.0, self.n_cornea)
        elif isinstance(self.cornea, SpheroidCornea):
            # Use spheroid refraction for outer surface
            O0, Od = refractions.refract_ray_spheroid(
                R0, Rd, cornea_center, self.cornea.a, self.cornea.b, self.cornea.c, 1.0, self.n_cornea
            )
        else:
            raise NotImplementedError(f"Refraction not implemented for cornea type: {type(self.cornea)}")

        if O0 is None or Od is None:
            return None, None, None

        # Line 38-39: Compute refraction at inner surface of cornea
        inner_center = self.trans @ self.cornea_inner_center
        I0, Id = refractions.refract_ray_sphere(
            O0,
            Od,
            inner_center,
            self.r_cornea_inner,
            self.n_cornea,
            self.n_aqueous_humor,
        )

        return O0, I0, Id

    def find_refraction(self, C: np.ndarray, O: np.ndarray) -> Optional[np.ndarray]:
        """Computes observed position of intraocular objects.

        I = find_refraction(e, C, O) computes the position at which a camera
        at position 'C' observes an object at position 'O' inside the eye 'e' due
        to refraction at the surface of the cornea. The position of the image 'I'
        that is returned lies on the surface of the cornea; a ray emanating from
        'O' is refracted at 'I' to pass directly through 'C'. If no suitable
        point can be found within the boundaries of the cornea, [] is returned
        for 'I'.

        Args:
            C: Camera position (3D or 4D homogeneous coordinates)
            O: Object position inside eye (3D or 4D homogeneous coordinates)

        Returns:
            4D homogeneous position on corneal surface where refraction occurs, or None

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """

        I = self.cornea.find_refraction(C, O, 1.0, self.n_cornea, self.trans)

        if I is None:
            return None

        # Ensure result is in 4D homogeneous coordinates for point_within_cornea
        if len(I) == 3:
            I = np.array([I[0], I[1], I[2], 1.0])
        else:
            I = I.copy()  # Already 4D

        # Check if point is within cornea
        if not self.point_within_cornea(I):
            I = None

        return I

    def get_pupil_boundary_in_camera_image(self, c: Camera, use_refraction: bool = True) -> np.ndarray:
        """Computes image of pupil boundary.

        X = get_pupil_boundary_in_camera_image(e, c) returns a 2×M matrix of points
        describing the image of the pupil boundary of the eye 'e' as observed by
        the camera 'c'. M can be less than the pupil's N if some boundary points
        lie outside the camera image or are not visible through the cornea.
        Refraction at the corneal surface and camera error are taken into account.

        Args:
            c: Camera object
            use_refraction: Whether to apply corneal refraction (default True)

        Returns:
            2×M matrix of pupil boundary points in camera image

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Line 31: pupil=eye_get_pupil(e);
        pupil = self.get_pupil()

        if use_refraction:
            # Line 32: X=zeros(4,0);
            X = np.zeros((4, 0))

            # Line 33-38: For each pupil point, find image with refraction
            for i in range(pupil.shape[1]):
                # Line 34: img=eye_find_refraction(e, c.trans(:,4), pupil(:,i));
                img = self.find_refraction(c.trans[:, 3], pupil[:, i])

                # Line 35-37: If point found, add to results
                if img is not None:
                    # Convert to homogeneous coordinates
                    img_homo = np.array([img[0], img[1], img[2], 1.0])
                    if X.shape[1] == 0:
                        X = img_homo.reshape(-1, 1)
                    else:
                        X = np.column_stack([X, img_homo])

            # Line 40-41: Project to camera and filter valid points
            if X.shape[1] > 0:
                X_proj, _, valid = c.project(X)
                X = X_proj[:, valid]
            else:
                X = np.zeros((2, 0))
        else:
            # Direct projection without refraction - use the correct approach
            X_proj, _, _ = c.project(pupil)
            X = X_proj

        return X

    def get_pupil_center_in_world(self) -> np.ndarray:
        """Get the pupil center position in world coordinates.

        Returns:
            4D homogeneous coordinates of the pupil center in world coordinates
        """
        return self.pupil.get_center_world_coords(self.trans)

    def get_pupil_ellipse_in_camera_image(
        self, c: Camera, use_refraction: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Determines pupil ellipse in camera image.

        [pupil, pc] = get_pupil_ellipse_in_camera_image(e, c) finds the image of the pupil border in the camera image
        (returned as 'pupil'), then fits an ellipse to those points and returns
        the center of the ellipse in 'pc'.

        Args:
            c: Camera object
            use_refraction: Whether to use refraction model (default True)

        Returns:
            Tuple of (pupil, pc) where:
            - pupil: 2×N matrix of pupil boundary points in camera image
            - pc: 2-element vector with pupil center position, or None if not found

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.
        """
        # Get pupil image (with or without refraction)
        pupil = self.get_pupil_boundary_in_camera_image(c, use_refraction=use_refraction)

        # Find center of pupil using ellipse fitting
        pc = self._fit_ellipse_center(pupil)

        return pupil, pc

    def get_pupil_center_of_mass_in_camera_image(
        self, c: Camera, use_refraction: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Determines pupil center using center of mass calculation.

        [pupil, pc] = get_pupil_center_of_mass_in_camera_image(e, c) finds the image of the pupil border
        in the camera image (returned as 'pupil'), creates a binary mask from the boundary points,
        and calculates the center of mass of the pupil region.

        Args:
            c: Camera object
            use_refraction: Whether to use refraction model (default True)

        Returns:
            Tuple of (pupil, pc) where:
            - pupil: 2×N matrix of pupil boundary points in camera image
            - pc: 2-element vector with center of mass position, or None if not found
        """
        # Get pupil image (with or without refraction)
        pupil = self.get_pupil_boundary_in_camera_image(c, use_refraction=use_refraction)

        # Find center of pupil using center of mass calculation
        pc = self._calculate_center_of_mass(pupil, c.resolution)

        return pupil, pc

    def _fit_ellipse_center(self, pupil: np.ndarray) -> Optional[np.ndarray]:
        """Fit ellipse to pupil boundary points and return center.

        Args:
            pupil: 2xN matrix of pupil boundary points

        Returns:
            2-element array with center coordinates [xc, yc], or None if fitting fails
        """
        if pupil.shape[1] >= 5:
            # Fit ellipse to pupil boundary points
            points = np.column_stack((pupil[0, :], pupil[1, :]))
            ellipse = EllipseModel()
            if ellipse.estimate(points):
                # Extract center coordinates directly
                return ellipse.params[:2]  # [xc, yc]

        # Not enough points for ellipse fitting or fitting failed
        return None

    def _calculate_center_of_mass(self, pupil: np.ndarray, camera_resolution: np.ndarray) -> Optional[np.ndarray]:
        """Calculate center of mass from pupil boundary points using binary mask.

        Args:
            pupil: 2xN matrix of pupil boundary points
            camera_resolution: 2-element array [width, height] of camera resolution

        Returns:
            2-element array with center of mass coordinates [xc, yc], or None if calculation fails
        """
        if pupil.shape[1] < 3:
            return None

        try:
            from skimage.draw import polygon
            from scipy import ndimage
        except ImportError:
            return None

        # Convert camera coordinates to image array coordinates
        # Camera: (0,0) at center, ranges -res/2 to +res/2
        # Array: (0,0) at top-left, ranges 0 to res
        width, height = int(camera_resolution[0]), int(camera_resolution[1])

        # Convert pupil points to array coordinates
        pupil_array_x = pupil[0, :] + width // 2
        pupil_array_y = pupil[1, :] + height // 2

        # Clip to valid image bounds
        pupil_array_x = np.clip(pupil_array_x, 0, width - 1)
        pupil_array_y = np.clip(pupil_array_y, 0, height - 1)

        # Create binary mask
        mask = np.zeros((height, width), dtype=bool)

        # Fill polygon defined by pupil boundary
        rr, cc = polygon(pupil_array_y, pupil_array_x, shape=(height, width))
        mask[rr, cc] = True

        if not np.any(mask):
            return None

        # Calculate center of mass
        # For binary mask, center of mass is just the centroid of True pixels
        y_center, x_center = ndimage.center_of_mass(mask.astype(float))

        # Convert back to camera coordinates
        x_camera = x_center - width // 2
        y_camera = y_center - height // 2

        return np.array([x_camera, y_camera])

    @property
    def fovea_position(self) -> np.ndarray:
        """Calculate the 3D position of the fovea on the retinal surface.

        Based on our simplified spherical eye model where:
        - The eye is modeled as a sphere with nodal point at cornea center
        - Fovea is displaced from optical axis by fovea_alpha_deg (horizontal)
          and fovea_beta_deg (vertical) angles
        - Retina is positioned at axial_length distance from cornea center

        Returns:
            3-element array with fovea position [x, y, z] in eye coordinate system
        """
        # Convert displacement angles to radians
        alpha = self.fovea_alpha_deg * np.pi / 180.0  # Horizontal (temporal) displacement
        beta = self.fovea_beta_deg * np.pi / 180.0  # Vertical (upward) displacement

        # Retina distance from rotation center (from our eye model)
        retina_distance = self.axial_length / 2

        # Calculate fovea position in eye coordinate system using spherical coordinates
        fovea_x = retina_distance * np.sin(alpha) * np.cos(beta)  # Temporal displacement
        fovea_y = retina_distance * np.sin(beta)  # Vertical displacement
        fovea_z = retina_distance * np.cos(alpha) * np.cos(beta)  # Along optical axis

        # Position in eye coordinate system (relative to eye rotation center)
        fovea_position = np.array([fovea_x, fovea_y, fovea_z])

        return fovea_position

    @property
    def angle_kappa(self) -> float:
        """Calculate angle kappa (degrees) - the angle between optical and visual axes.

        Angle kappa is the angle between:
        - Optical axis: direction the eye is pointing (-Z axis in eye coordinates)
        - Visual axis: direction from rotation center to fovea

        This is calculated using the 3D fovea position for accuracy.

        Returns:
            Angle kappa in degrees
        """
        # Get fovea position (3D coordinates relative to rotation center)
        fovea_pos = self.fovea_position

        # Calculate visual axis direction (normalized)
        visual_axis = fovea_pos / np.linalg.norm(fovea_pos)

        # Optical axis points along -Z direction in eye coordinates
        optical_axis = np.array([0, 0, -1])

        # Calculate angle between visual and optical axes
        dot_product = np.dot(visual_axis, optical_axis)
        # Use abs() to get acute angle and clip to avoid numerical errors
        angle_kappa_rad = np.arccos(np.clip(np.abs(dot_product), 0, 1))

        # Convert to degrees
        return angle_kappa_rad * 180.0 / np.pi
