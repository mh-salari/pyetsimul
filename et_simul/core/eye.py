import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
from skimage.measure import EllipseModel

from ..optics import reflections, refractions
from ..geometry import utils


@dataclass
class Eye:
    """Creates a structure that represents an eye.

     This class is based on the original MATLAB implementation from the
     et_simul project — © 2008 Martin Böhme, University of Lübeck.
     Python port © 2025 Mohammadhossein Salari.
     Licensed under the GNU GPL v3.0 or later.

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
    """

    r_cornea: float = 7.98e-3
    fovea_displacement: bool = True
    fovea_alpha_deg: float = 6.0  # Horizontal fovea displacement in degrees
    fovea_beta_deg: float = 2.0  # Vertical fovea displacement in degrees

    # These fields are calculated in __post_init__
    trans: np.ndarray = field(init=False)
    pos_cornea: np.ndarray = field(init=False)
    r_cornea_inner: float = field(init=False)
    cornea_inner_center: np.ndarray = field(init=False)
    n_cornea: float = field(init=False)
    n_aqueous_humor: float = field(init=False)
    pos_apex: np.ndarray = field(init=False)
    depth_cornea: float = field(init=False)
    pos_pupil: np.ndarray = field(init=False)
    x_pupil: np.ndarray = field(init=False)
    y_pupil: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initializes the eye's anatomical properties based on constructor parameters."""
        # Line 75: r_cornea_default=7.98e-3;
        r_cornea_default = 7.98e-3

        # Line 80: scale=r_cornea/r_cornea_default;
        scale = self.r_cornea / r_cornea_default

        # Line 86: e.trans=eye(4);
        self.trans = np.eye(4)

        # Line 94: e.pos_cornea=[0 0 -scale*(24.75e-3 - 2*10.20e-3) 1]';
        self.pos_cornea = np.array([0, 0, -scale * (24.75e-3 - 2 * 10.20e-3), 1])

        # Line 97: e.r_cornea_inner=scale*6.22e-3;
        self.r_cornea_inner = scale * 6.22e-3

        # Lines 100-101: e.cornea_inner_center=e.pos_cornea-[0 0 e.r_cornea - e.r_cornea_inner - scale*1.15e-3 0]';
        self.cornea_inner_center = self.pos_cornea - np.array(
            [0, 0, self.r_cornea - self.r_cornea_inner - scale * 1.15e-3, 0]
        )

        # Line 104: e.n_cornea=1.376;
        self.n_cornea = 1.376

        # Line 107: e.n_aqueous_humor=1.336;
        self.n_aqueous_humor = 1.336

        # Line 110: e.pos_apex=e.pos_cornea+[0 0 -e.r_cornea 0]';
        self.pos_apex = self.pos_cornea + np.array([0, 0, -self.r_cornea, 0])

        # Line 113: e.depth_cornea=scale*3.54e-3;
        self.depth_cornea = scale * 3.54e-3

        # Line 116: e.pos_pupil=e.pos_apex+[0 0 scale*3.54e-3 0]';
        self.pos_pupil = self.pos_apex + np.array([0, 0, scale * 3.54e-3, 0])

        # Line 119: e.x_pupil=scale*3e-3*[1 0 0 0]';
        self.x_pupil = scale * 3e-3 * np.array([1, 0, 0, 0])

        # Line 120: e.y_pupil=scale*3e-3*[0 1 0 0]';
        self.y_pupil = scale * 3e-3 * np.array([0, 1, 0, 0])

    @property
    def rest_orientation(self) -> np.ndarray:
        """Get/set the eye's orientation (3x3 rotation matrix)."""
        return self.trans[:3, :3]
    
    @rest_orientation.setter
    def rest_orientation(self, value: np.ndarray) -> None:
        """Set the eye's orientation and update transformation matrix."""
        self.trans[:3, :3] = value
    
    @property
    def rotation(self) -> np.ndarray:
        """Get/set the eye's rotation matrix (alias for rest_orientation)."""
        return self.trans[:3, :3]
    
    @rotation.setter
    def rotation(self, value: np.ndarray) -> None:
        """Set the eye's rotation matrix and update transformation matrix."""
        self.trans[:3, :3] = value
    
    @property
    def position(self) -> np.ndarray:
        """Get/set the eye's position in world coordinates (3D vector)."""
        return self.trans[:3, 3]
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Set the eye's position and update transformation matrix."""
        self.trans[:3, 3] = value
    
    @property
    def world_position(self) -> np.ndarray:
        """Get/set the eye's world position (alias for position)."""
        return self.trans[:3, 3]
    
    @world_position.setter
    def world_position(self, value: np.ndarray) -> None:
        """Set the eye's world position and update transformation matrix."""
        self.trans[:3, 3] = value
    
    @property
    def optical_axis(self) -> np.ndarray:
        """Get the direction the eye is looking (read-only)."""
        return -self.trans[:3, 2]

    def point_within_cornea(self, p: np.ndarray) -> bool:
        """Tests whether a point lies within the cornea.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        within=point_within_cornea(e, p) tests whether the point 'p', lying
        on the corneal sphere of the eye 'e', lies within the boundaries of the
        cornea, as defined by e.depth_cornea. This function is used by
        find_cr() and find_refraction().

        Args:
            p: Point to test (4D homogeneous coordinates)

        Returns:
            bool: True if point lies within cornea boundaries, False otherwise
        """
        # Line 25: p=e.trans\p;
        p = np.linalg.solve(self.trans, p)

        # Lines 27-28: within = ((p-e.pos_apex)'*(e.pos_cornea-e.pos_apex) / norm(e.pos_cornea-e.pos_apex) < e.depth_cornea);
        diff = p - self.pos_apex
        direction = self.pos_cornea - self.pos_apex
        within = np.dot(diff, direction) / np.linalg.norm(direction) < self.depth_cornea

        return within

    def find_cr(self, l: "Light", c: "Camera") -> Optional[np.ndarray]:
        """Finds the position of a corneal reflex.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

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
        """
        # Line 26: cr=find_reflection(l.pos, c.trans(:,4), e.trans*e.pos_cornea, e.r_cornea);
        cr = reflections.find_reflection(
            l.position, c.trans[:, 3], self.trans @ self.pos_cornea, self.r_cornea
        )

        # Lines 29-31: if ~eye_point_within_cornea(e, cr), cr=[]; end
        if cr is not None and not self.point_within_cornea(cr):
            cr = None

        return cr

    @staticmethod
    def listings_law(out_rest: np.ndarray, out_new: np.ndarray) -> np.ndarray:
        """Uses Listing's law to compute a matrix for eye rotation.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        A = listings_law(out_rest, out_new) uses Listing's law to compute a
        matrix 'A' for the rotation an eye makes when the direction of its
        optical axis moves from the rest position 'out_rest' to the new position
        'out_new'.

        Args:
            out_rest: Direction of optical axis in rest position (3D vector)
            out_new: Direction of optical axis in new position (3D vector)

        Returns:
            3x3 rotation matrix A representing the eye rotation
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

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        e=eye_look_at(e, pos) rotates the eye 'e' to look at the
        three-dimensional position 'pos' in world coordinates and returns the
        rotated eye. Listing's law is used to compute the amount of torsion that
        occurs. If the global variable FEAT_FOVEA_DISPLACEMENT is true, the
        displacement of the fovea from the optical axis is taken into account
        (i.e. the line-of-sight will be aligned with the given position, not the
        optical axis).

        Args:
            pos: Three-dimensional position in world coordinates to look at
        """
        # Line 31: out=pos(1:3)-e.trans(1:3,4);
        out = pos[:3] - self.position
        # Line 32: out=out/norm(out);
        out = out / np.linalg.norm(out)

        rest_orientation_base = np.eye(3)
        out_rest = rest_orientation_base @ np.array([0, 0, -1])
        self.rest_orientation = self.listings_law(out_rest, out) @ rest_orientation_base

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
            self.rest_orientation = self.rest_orientation @ B @ A

    def get_pupil(self, N: int = 20) -> np.ndarray:
        """Returns an array of points describing the pupil boundary.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        X = get_pupil(e, N) returns a 4xN matrix of points (in world
        coordinates) on the pupil boundary of the eye 'e'.

        Args:
            N: Number of points on pupil boundary (default 20)

        Returns:
            4xN matrix of points in world coordinates on pupil boundary
        """
        # Line 24: if nargin<2, N=20; end (handled by default parameter)

        # Line 27: alpha=2*pi*(0:(N-1))/N;
        alpha = 2 * np.pi * np.arange(N) / N

        # Lines 28-29: X=repmat(e.pos_pupil, 1, N) + e.across_pupil*cos(alpha) + e.up_pupil*sin(alpha);
        pupil_points = (
            np.tile(self.pos_pupil.reshape(-1, 1), (1, N))
            + self.x_pupil.reshape(-1, 1) @ np.cos(alpha).reshape(1, -1)
            + self.y_pupil.reshape(-1, 1) @ np.sin(alpha).reshape(1, -1)
        )

        # Transform to world coordinates
        pupil_world = self.trans @ pupil_points

        return pupil_world

    def get_pupil_radii(self) -> tuple[float, float]:
        """Returns the current pupil radii from both axes.
        
        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        x_radius = np.linalg.norm(self.x_pupil[:3])
        y_radius = np.linalg.norm(self.y_pupil[:3])
        return x_radius, y_radius

    def set_pupil_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Sets the pupil radii and updates pupil geometry.
        
        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)
            
        Raises:
            ValueError: If both radii are None
        """
        if x_radius is None and y_radius is None:
            raise ValueError("At least one radius must be specified")
            
        if x_radius is not None:
            self.x_pupil = x_radius * np.array([1, 0, 0, 0])
        if y_radius is not None:
            self.y_pupil = y_radius * np.array([0, 1, 0, 0])

    def find_cr_simple(self, l: "Light", c: "Camera") -> Optional[np.ndarray]:
        """Finds the position of a corneal reflex (simplified).

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

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
        """
        # Line 27: cc=e.trans*e.pos_cornea;
        cc = self.trans @ self.pos_cornea

        # Line 28: to_cam=c.trans(:,4)-cc;
        to_cam = c.trans[:, 3] - cc

        # Line 29: to_cam=to_cam/norm(to_cam);
        to_cam = to_cam / np.linalg.norm(to_cam)

        # Line 30: w=e.r_cornea/(2*(l.pos-cc)'*to_cam);
        light_to_cornea = l.position - cc
        denominator = 2 * np.dot(light_to_cornea, to_cam)

        if abs(denominator) < 1e-10:  # Avoid division by zero
            return None

        w = self.r_cornea / denominator

        # Line 31: cr=cc+w*(l.pos-cc);
        cr = cc + w * light_to_cornea

        # Check if point is within cornea boundaries (improvement over original)
        if not self.point_within_cornea(cr):
            return None

        return cr

    def refract_ray(
        self, R0: np.ndarray, Rd: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Computes refraction of ray at cornea surface.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

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
        """

        # Line 32-33: Compute refraction at surface of cornea
        # Compute corneal center position (4D homogeneous)
        cornea_center = self.trans @ self.pos_cornea

        # refract_ray_sphere handles 3D/4D coordinates properly (fixes MATLAB's coordinate bug)
        U0, Ud = refractions.refract_ray_sphere(
            R0, Rd, cornea_center, self.r_cornea, 1.0, self.n_cornea
        )

        return U0, Ud

    def refract_ray_advanced(
        self, R0: np.ndarray, Rd: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Computes refraction at both surfaces of cornea.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

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
        """
        # Line 34-35: Compute refraction at outer surface of cornea
        # Compute corneal center positions (4D homogeneous)
        cornea_center = self.trans @ self.pos_cornea

        # refract_ray_sphere handles 3D/4D coordinates properly (fixes MATLAB's coordinate bug)
        O0, Od = refractions.refract_ray_sphere(
            R0, Rd, cornea_center, self.r_cornea, 1.0, self.n_cornea
        )

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

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

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
        """
        # Compute corneal center position (4D homogeneous)
        cornea_center = self.trans @ self.pos_cornea

        # Line 28: I=find_refraction(C, O, e.trans*e.pos_cornea, e.r_cornea, 1, e.n_cornea);
        # find_refraction handles 3D/4D coordinates properly (fixes MATLAB's coordinate bug)
        I = refractions.find_refraction(
            C, O, cornea_center, self.r_cornea, 1.0, self.n_cornea
        )

        if I is None:
            return None

        # Ensure result is in 4D homogeneous coordinates for point_within_cornea
        if len(I) == 3:
            I = np.array([I[0], I[1], I[2], 1.0])
        else:
            I = I.copy()  # Already 4D

        # Line 30-32: Check if point is within cornea
        if not self.point_within_cornea(I):
            I = None

        return I

    def get_pupil_image(self, c: "Camera", N: int = 20, use_refraction: bool = True) -> np.ndarray:
        """Computes image of pupil boundary.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        X = get_pupil_image(e, c, N) returns a 2xM matrix of points
        describing the image of the pupil boundary of the eye 'e' as observed by
        the camera 'c'. Normally, M=N, but M can be less than N if some of the
        boundary points lie outside the camera image or are not visible through
        the cornea. Refraction at the corneal surface and camera error are taken
        into account.

        Args:
            c: Camera object
            N: Number of pupil boundary points (default 20)
            use_refraction: Whether to apply corneal refraction (default True)

        Returns:
            2xM matrix of pupil boundary points in camera image
        """
        # Line 31: pupil=eye_get_pupil(e, N);
        pupil = self.get_pupil(N)

        # Line 32: X=zeros(4,0);
        X = np.zeros((4, 0))

        # Line 33-38: For each pupil point, find image (with or without refraction)
        for i in range(pupil.shape[1]):
            if use_refraction:
                # Line 34: img=eye_find_refraction(e, c.trans(:,4), pupil(:,i));
                img = self.find_refraction(c.trans[:, 3], pupil[:, i])
            else:
                # Direct projection without refraction
                pupil_world = self.trans @ pupil[:, i]
                img = pupil_world

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

        return X

    def get_pupil_center(self) -> np.ndarray:
        """Get the pupil center position in world coordinates.
        
        Returns:
            4D homogeneous coordinates of the pupil center in world coordinates
        """
        return self.trans @ self.pos_pupil

    def get_pc(
        self, c: "Camera", use_refraction: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Determines position of pupil in camera image.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        [pupil, pc] = get_pc(e, c) finds the image of the pupil border in the camera image
        (returned as 'pupil'), then fits an ellipse to those points and returns
        the center of the ellipse in 'pc'.

        Args:
            c: Camera object
            use_refraction: Whether to use refraction model (default True)

        Returns:
            Tuple of (pupil, pc) where:
            - pupil: 2×N matrix of pupil boundary points in camera image
            - pc: 2-element vector with pupil center position, or None if not found
        """
        # Use refraction model for accurate pupil imaging
        if use_refraction:
            # Get pupil image using refraction
            pupil = self.get_pupil_image(c)

            # Find center of pupil using ellipse fitting
            if pupil.shape[1] >= 5:
                # Fit ellipse to pupil boundary points
                points = np.column_stack((pupil[0, :], pupil[1, :]))
                ellipse = EllipseModel()
                if ellipse.estimate(points):
                    # Extract center coordinates directly
                    pc = ellipse.params[:2]  # [xc, yc]
                else:
                    pc = None
            else:
                # Not enough points for ellipse fitting
                pc = None
        else:
            # Simple pupil version without refraction
            pupil = self.get_pupil()
            pupil_proj, _, valid = c.project(pupil)
            pupil = pupil_proj[:, valid]

            # Project pupil center directly
            pc_proj, _, _ = c.project(self.trans @ self.pos_pupil)
            if np.any(np.isnan(pc_proj)):
                pc = None
            else:
                pc = pc_proj.flatten()

        return pupil, pc
