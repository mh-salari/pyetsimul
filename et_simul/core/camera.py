import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any


@dataclass
class Camera:
    """Pinhole camera model for eye tracking.

    This class is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    The camera model is a pinhole model, the center of projection is at the
    origin of the camera coordinate system, and the camera's optical axis
    points out along the negative z axis. The x and y axes of the image plane
    are aligned with the x and y axes of the camera coordinate system.

    Camera object contains the following elements:

    - 'trans' is the transformation matrix from camera to world coordinates.
      The default value for this is the identity matrix.

    - 'rest_trans' is used for pan-tilt cameras to store the transformation
      matrix from camera to world coordinates in the camera's rest position.
      This is needed because the 'trans' matrix is changed when the camera
      pans and tilts out of its rest position. 'rest_trans' need not be set
      for fixed cameras.

    - 'focal_length' is the focal length of the camera in pixels. A point at
      a distance of 1 metre from the camera and offset horizontally from the
      optical axis by 1 metre will appear at an x coordinate of
      'focal_length' pixels in the camera image. The default value for this
      parameter is 2880.

    - 'resolution' is a two-dimensional vector specifying the image
      resolution of the camera (resolution (1) is the horizontal resolution,
      and resolution(2) is the vertical resolution). The point where the
      optical axis intersects the image plane has the image coordinates
      (0,0); hence, valid x-coordinates range from -resolution(1)/2 to
      resolution(1)/2, and valid y-coordinates range from -resolution(2)/2 to
      resolution(2)/2. Points that fall outside this range cannot be "seen"
      by the camera. The default resolution is [1280, 1024].

    - 'err' is the amount of random error in measurements made in the camera
      image. When project() is used to project a point onto the
      camera, a certain amount of random error is added to the position of
      the point in the image. The exact meaning of this parameter depends on
      the type of error distribution selected in 'err_type'. The default
      value is 0.

    - 'err_type' specifies the type of error. This parameter can have the
      following values:

      'gaussian' (default): A bivariate Gaussian distribution with mean 0
      and standard deviation err

      'uniform': A uniform distribution between -err and +err for both the x
      and y coordinate
    """

    focal_length: float = 2880
    resolution: Optional[np.ndarray] = None
    err: float = 0.0
    err_type: str = "gaussian"
    trans: Optional[np.ndarray] = None
    rest_trans: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Initialize camera with default values."""
        # Line 69: c.trans=eye(4);
        if self.trans is None:
            self.trans = np.eye(4)

        # Line 70: c.rest_trans=c.trans;
        if self.rest_trans is None:
            self.rest_trans = self.trans.copy()

        # Line 74: c.resolution=[1280 1024];
        if self.resolution is None:
            self.resolution = np.array([1280, 1024])

    def project(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projects points in space onto the camera's image plane.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        [x, dist] = project(self, pos) transforms a list of points 'pos'
        (given as a 4xn matrix) into the local coordinate system of camera
        and projects them onto the camera's image plane. A certain amount of
        random error is added to the image coordinates. The function returns
        a 2xn matrix 'x' of image points, a row vector 'dist' of length n
        containing the distances of the points from the camera (measured along
        the optical axis), and a boolean row vector 'valid' of length n
        specifying, for each point, whether it fell within the camera image
        (as defined by the resolution parameter). In addition, if a point fell
        outside the camera image, the corresponding entry in 'x' is set to 'nan'.

        Args:
            pos: Points to project (4×n array of homogeneous coordinates)

        Returns:
            Tuple of (x, dist, valid) where:
            - x: 2×n matrix of image coordinates (NaN for invalid points)
            - dist: 1×n array of distances from camera along optical axis
            - valid: 1×n boolean array indicating points within image bounds
        """
        # Handle single point case
        if pos.ndim == 1:
            pos = pos.reshape(-1, 1)

        # Line 38: pos=c.trans\pos;
        pos = np.linalg.solve(self.trans, pos)

        # Line 39: dist=-pos(3,:);
        dist = -pos[2, :]

        # Line 41: x=[c.focal_length*pos(1,:)./dist; c.focal_length*pos(2,:)./dist];
        x = np.array(
            [self.focal_length * pos[0, :] / dist, self.focal_length * pos[1, :] / dist]
        )

        # Lines 43-49: Add error
        if self.err_type == "uniform":
            # Line 44: x=x + c.err*(2*rand(2,size(pos,2))-1);
            x = x + self.err * (2 * np.random.rand(2, pos.shape[1]) - 1)
        elif self.err_type == "gaussian":
            # Line 46: x=x + c.err*normal_deviates(size(pos,2))';
            x = x + self.err * np.random.normal(0, 1, (pos.shape[1], 2)).T
        else:
            raise ValueError("Unknown error type")

        # Lines 51-52: valid=find(x(1,:)>=-c.resolution(1)/2 & x(1,:)<=c.resolution(1)/2 & ...
        #                         x(2,:)>=-c.resolution(2)/2 & x(2,:)<=c.resolution(2)/2);
        condition = (
            (x[0, :] >= -self.resolution[0] / 2)
            & (x[0, :] <= self.resolution[0] / 2)
            & (x[1, :] >= -self.resolution[1] / 2)
            & (x[1, :] <= self.resolution[1] / 2)
        )

        # Line 54: x(:,~valid)=nan;
        # CORRECTED FROM MATLAB: The original MATLAB code has ~valid where valid=find(...) returns indices
        # This means ~valid does NOT create a proper logical mask, so NaN assignment fails in MATLAB
        # The MATLAB code should use ~condition, but due to this bug, MATLAB doesn't set NaN values
        # Here we implement the corrected behavior: properly set out-of-bounds points to NaN
        invalid_mask = ~condition
        x[:, invalid_mask] = np.nan

        return x, dist, condition

    def unproject(self, X: np.ndarray, d: float) -> np.ndarray:
        """Unprojects points on the image plane back into 3D space.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        pos = unproject(self, X, d) unprojects the two-dimensional points
        contained in the columns of the 2xn matrix 'X' from the image plane
        back to a distance 'd' from the camera (measured along the
        optical axis). The points are returned as a 4xn matrix of homogeneous
        column vectors.

        Args:
            X: 2xn matrix of 2D image points
            d: Distance from camera along optical axis

        Returns:
            4xn matrix of homogeneous 3D points
        """
        # Handle single point case
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Line 26: n=size(X,2);
        n = X.shape[1]

        # Line 27-28: pos=c.trans*[X(1,:)/c.focal_length*d; X(2,:)/c.focal_length*d;
        #                         repmat(-d,1,n); ones(1,n)];
        camera_coords = np.array(
            [
                X[0, :] / self.focal_length * d,
                X[1, :] / self.focal_length * d,
                np.full(n, -d),
                np.ones(n),
            ]
        )

        pos = self.trans @ camera_coords

        return pos

    def pan_tilt(self, look_at: Union[np.ndarray, List[float]]) -> None:
        """Pans and tilts a camera towards a certain location.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        c = pan_tilt(c, look_at) pans and tilts the camera 'c' so that it
        is looking directly at the point 'look_at' (given in world coordinates).
        The coordinate transformation 'trans' of the returned camera is modified
        accordingly. The camera is panned and tilted around the origin of its
        coordinate system.

        Args:
            c: Camera structure
            look_at: Point to look at in world coordinates (3D or 4D homogeneous)

        Returns:
            Camera structure with updated transformation matrix
        """
        # Lines 27-29: Extend 'look_at' to homogeneous coordinates if necessary
        if len(look_at) < 4:
            look_at = np.array([look_at[0], look_at[1], look_at[2], 1])

        # Line 32: axis=c.rest_trans\look_at;
        axis = np.linalg.solve(self.rest_trans, look_at)

        # Line 35: axis=axis(1:3)/norm(axis(1:3));
        axis = axis[:3] / np.linalg.norm(axis[:3])

        # Line 38: alpha=pi/2-atan2(-axis(3), axis(1));
        alpha = np.pi / 2 - np.arctan2(-axis[2], axis[0])

        # Line 39: beta=asin(axis(2));
        beta = np.arcsin(axis[1])

        # Lines 43-47: Construct pan matrix
        pan_matrix = np.array(
            [
                [np.cos(alpha), 0, -np.sin(alpha), 0],
                [0, 1, 0, 0],
                [np.sin(alpha), 0, np.cos(alpha), 0],
                [0, 0, 0, 1],
            ]
        )

        # Lines 48-52: Construct tilt matrix
        tilt_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(beta), -np.sin(beta), 0],
                [0, np.sin(beta), np.cos(beta), 0],
                [0, 0, 0, 1],
            ]
        )

        # Line 55: c.trans=c.rest_trans*pan_matrix*tilt_matrix;
        self.trans = self.rest_trans @ pan_matrix @ tilt_matrix

    def point_at(self, point_at: Union[np.ndarray, List[float]]) -> None:
        """Points camera towards a certain location.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        c = point_at(c, point_at) changes the rest position of the
        camera 'c' so that it is pointing at the point 'point_at' (given in world
        coordinates). The elements 'rest_trans' and 'trans' of the returned
        camera are modified accordingly.

        Note: This function differs from pan_tilt() in that the latter
        does not affect the rest position of the camera, while this function
        does.

        Args:
            c: Camera structure
            point_at: Point to point at in world coordinates (3D or 4D homogeneous)

        Returns:
            Camera structure with updated transformation and rest transformation matrices
        """
        # Line 30: c.rest_trans=c.trans;
        self.rest_trans = self.trans.copy()

        # Line 33: c=camera_pan_tilt(c, point_at);
        self.pan_tilt(point_at)

        # Line 36: c.rest_trans=c.trans;
        self.rest_trans = self.trans.copy()

    def take_image(
        self, e: "Eye", lights: List["Light"], use_refraction: bool = True
    ) -> Dict[str, Any]:
        """Computes the image of an eye seen by a camera.

        This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        camimg = take_image(camera, e, lights) computes the image of the
        eye 'e' as seen by the camera 'camera'. 'lights' is a cell array of light
        objects that generate CRs on the cornea. The function returns a structure
        'camimg' containing the following:

        'cr'     An n-element cell array containing the positions of the n
                 corneal reflexes in the camera image. If the reflex did not fall
                 within the cornea, or if the reflex fell outside the camera
                 image, [] is returned for the corresponding CR.

        'pc'     A two-element vector with the position of the pupil center in
                 the camera image. If the pupil fell outside the camera image, []
                 is returned.

        'pupil'  A 2-times-n matrix with the positions of n points on the pupil
                 border in the camera image. The number of points can depend on
                 how much of the pupil is visible in the image; if the pupil is
                 outside the image, a 2-times-0 matrix is returned.

        Args:
            camera: Camera structure
            e: Eye structure
            lights: List of light source structures
            use_refraction: Whether to use refraction model for pupil (default True)

        Returns:
            dict: Camera image structure containing 'cr', 'pc', and 'pupil' fields
        """

        camimg = {}

        # Lines 37-49: Find the CRs
        cr = [None] * len(lights)
        for k in range(len(lights)):
            # Line 39: cr_3d=eye_find_cr(e, lights{k}, camera);
            cr_3d = e.find_cr(lights[k], self)

            # Line 41: if isempty(cr_3d)
            if cr_3d is None:
                # Line 42: cr{k}=[];
                cr[k] = None
            else:
                # Line 44: cr{k}=camera_project(camera, cr_3d);
                cr_proj, _, _ = self.project(cr_3d)
                # Line 45: if any(isnan(cr{k}))
                if np.any(np.isnan(cr_proj)):
                    # Line 46: cr{k}=[];
                    cr[k] = None
                else:
                    cr[k] = cr_proj.flatten()

        # Line 50: camimg.cr=cr;
        camimg["cr"] = cr

        # Line 53: [camimg.pupil, camimg.pc]=get_pc(e, camera);
        pupil, pc = e.get_pc(self, use_refraction=use_refraction)
        camimg["pupil"] = pupil
        camimg["pc"] = pc

        return camimg
