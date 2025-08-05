"""
Yoo and Chung eye tracking algorithm implementation.

This module implements the Yoo and Chung method for gaze estimation using
cross-ratio invariance under large head motion.

Reference: Dong Hyun Yoo and Myung Jin Chung. A novel non-intrusive eye gaze
estimation using cross-ratio under large head motion. Computer Vision and Image
Understanding 98, 25-51, 2005.
"""

import numpy as np
import time
from typing import List, Optional

from et_simul.core import EyeTracker, Camera, Light
from et_simul.types.algorithms import GazePrediction, YooConfig, YooState
from et_simul.types.imaging import EyeMeasurement
from et_simul.types.geometry import Position3D, Point3D, Point2D
from et_simul.geometry.plane_detection import detect_calibration_plane
from et_simul.geometry.utils import line_intersect_2d


class YooTracker(EyeTracker):
    """Yoo and Chung eye tracking algorithm implementation.
    
    Uses cross-ratio invariance under large head motion for gaze estimation.
    Requires 5 lights: 4 around monitor corners and 1 co-located with camera.
    
    Based on: Yoo and Chung (2005) "A novel non-intrusive eye gaze estimation 
    using cross-ratio under large head motion"
    """
    
    def __init__(self, config: YooConfig = None, **kwargs):
        """Initialize Yoo tracker with structured configuration.
        
        Args:
            config: YooConfig instance
            **kwargs: Arguments passed to parent EyeTracker
        """
        super().__init__(**kwargs)
        self.config = config or YooConfig()
        self.algorithm_state = YooState()
        self.plane_info = None
        
    @property
    def algorithm_name(self) -> str:
        """Algorithm name identifier."""
        return "yoo_chung"
        
    @classmethod
    def create(
        cls,
        cameras: List[Camera],
        lights: List[Light], 
        calib_points: List[Position3D],
        config: YooConfig = None
    ) -> "YooTracker":
        """Factory method for creating Yoo tracker with external components.
        
        Args:
            cameras: List of Camera objects (expects exactly 1 camera)
            lights: List of Light objects (expects exactly 5 lights) 
            calib_points: List of calibration points as Position3D (expects 4 points)
            config: Algorithm configuration
            
        Returns:
            Configured YooTracker instance
            
        Raises:
            ValueError: If component counts don't match Yoo algorithm requirements
        """
        tracker = cls(config=config)
        
        # Validate Yoo algorithm requirements
        if len(cameras) != 1:
            raise ValueError(f"Yoo algorithm requires exactly 1 camera, got {len(cameras)}")
        if len(lights) != 5:
            raise ValueError(f"Yoo algorithm requires exactly 5 lights, got {len(lights)}")
        if len(calib_points) < tracker.config.min_calibration_points:
            raise ValueError(f"Yoo algorithm requires at least {tracker.config.min_calibration_points} calibration points, got {len(calib_points)}")
        
        # Add components using structured types
        for camera in cameras:
            tracker.add_camera(camera)
        for light in lights:
            tracker.add_light(light)
        for point in calib_points:
            tracker.add_calibration_point(point)
            
        return tracker
        
    def calibrate(self, calibration_measurements: List[EyeMeasurement]) -> None:
        """Calibrate using collected measurements.
        
        Computes alpha values for each calibration point using cross-ratio calculations.
        Updates algorithm_state with calibration parameters.
        
        Args:
            calibration_measurements: List of EyeMeasurement objects
        """
        if len(calibration_measurements) < self.config.min_calibration_points:
            raise ValueError(f"Need at least {self.config.min_calibration_points} measurements")
            
        print(f"Calibrating Yoo algorithm with {len(calibration_measurements)} points...")
        
        # Detect calibration plane for coordinate system mapping
        self.plane_info = detect_calibration_plane(self.calib_points)
        
        # Initialize alpha values
        alpha_values = np.ones(len(calibration_measurements))
        
        # Process each calibration measurement
        for i, measurement in enumerate(calibration_measurements):
            try:
                # Extract corneal reflections (expect 5: 4 corners + 1 center)
                camera_image = measurement.camera_image
                if not camera_image.corneal_reflections or len(camera_image.corneal_reflections) != 5:
                    print(f"Warning: Expected 5 corneal reflections, got {len(camera_image.corneal_reflections) if camera_image.corneal_reflections else 0}")
                    continue
                    
                # Extract pupil center
                pupil_center = camera_image.pupil_center
                if pupil_center is None:
                    print(f"Warning: No pupil center detected for calibration point {i}")
                    continue
                    
                # Get corner CRs (first 4) and center CR (5th)
                corner_crs = camera_image.corneal_reflections[:4]
                center_cr = camera_image.corneal_reflections[4]
                
                if center_cr is None:
                    print(f"Warning: No center CR detected for calibration point {i}")
                    continue
                    
                # Calculate alpha value using cross-ratio
                # Use the i-th corner CR for the i-th calibration point
                if i < len(corner_crs) and corner_crs[i] is not None:
                    r_i = corner_crs[i]
                    pc_distance = np.linalg.norm([pupil_center.x - center_cr.x, pupil_center.y - center_cr.y])
                    cr_distance = np.linalg.norm([r_i.x - center_cr.x, r_i.y - center_cr.y])
                    
                    if cr_distance > 1e-10:  # Avoid division by zero
                        alpha_values[i] = pc_distance / cr_distance
                    else:
                        alpha_values[i] = 1.0
                else:
                    alpha_values[i] = 1.0
                        
            except Exception as e:
                print(f"Error processing calibration point {i}: {e}")
                alpha_values[i] = 1.0
                
        # Store calibration results
        self.algorithm_state.alpha_values = alpha_values
        self.algorithm_state.is_calibrated = True
        
        print(f"Calibration successful. Alpha values: {alpha_values}")
        
    def predict_gaze(self, measurement: EyeMeasurement) -> Optional[GazePrediction]:
        """Predict gaze direction using calibrated algorithm.
        
        Args:
            measurement: EyeMeasurement with current eye state
            
        Returns:
            GazePrediction with structured Point3D gaze point, or None if failed
        """
        if not self.algorithm_state.is_calibrated:
            return None
            
        start_time = time.time()
        
        try:
            camera_image = measurement.camera_image
            
            # Extract corneal reflections and pupil center
            if not camera_image.corneal_reflections or len(camera_image.corneal_reflections) != 5:
                return None
                
            corner_crs = camera_image.corneal_reflections[:4]
            center_cr = camera_image.corneal_reflections[4]
            pupil_center = camera_image.pupil_center
            
            if center_cr is None or pupil_center is None:
                return None
                
            
            # Convert to numpy arrays for computation
            r = np.array([[cr.x if cr else 0, cr.y if cr else 0] for cr in corner_crs]).T
            c = np.array([center_cr.x, center_cr.y])
            p = np.array([pupil_center.x, pupil_center.y])
            
            # Compute virtual projection points using calibrated alpha values
            v = np.zeros_like(r)
            for i in range(4):
                if i < len(self.algorithm_state.alpha_values):
                    v[:, i] = c + self.algorithm_state.alpha_values[i] * (r[:, i] - c)
                else:
                    v[:, i] = r[:, i]  # Fallback if no alpha value
                    
            # Compute screen center and vanishing points for cross-ratio calculation
            e_point = line_intersect_2d(
                Point2D(v[0, 0], v[1, 0]), Point2D(v[0, 2], v[1, 2]),
                Point2D(v[0, 1], v[1, 1]), Point2D(v[0, 3], v[1, 3])
            )
            
            m = np.zeros((2, 4))
            
            # First pair of cross-ratio points
            vanish = line_intersect_2d(
                Point2D(v[0, 0], v[1, 0]), Point2D(v[0, 3], v[1, 3]),
                Point2D(v[0, 1], v[1, 1]), Point2D(v[0, 2], v[1, 2])
            )
            m_temp = line_intersect_2d(
                vanish, e_point,
                Point2D(v[0, 0], v[1, 0]), Point2D(v[0, 1], v[1, 1])
            )
            m[:, 0] = [m_temp.x, m_temp.y]
            
            m_temp = line_intersect_2d(
                vanish, Point2D(p[0], p[1]),
                Point2D(v[0, 0], v[1, 0]), Point2D(v[0, 1], v[1, 1])
            )
            m[:, 1] = [m_temp.x, m_temp.y]
            
            # Second pair of cross-ratio points  
            vanish = line_intersect_2d(
                Point2D(v[0, 0], v[1, 0]), Point2D(v[0, 1], v[1, 1]),
                Point2D(v[0, 2], v[1, 2]), Point2D(v[0, 3], v[1, 3])
            )
            m_temp = line_intersect_2d(
                vanish, Point2D(p[0], p[1]),
                Point2D(v[0, 1], v[1, 1]), Point2D(v[0, 2], v[1, 2])
            )
            m[:, 2] = [m_temp.x, m_temp.y]
            
            m_temp = line_intersect_2d(
                vanish, e_point,
                Point2D(v[0, 1], v[1, 1]), Point2D(v[0, 2], v[1, 2])
            )
            m[:, 3] = [m_temp.x, m_temp.y]
            
            # Calculate cross ratios for X and Y directions
            vx, vy = v[0, :], v[1, :]
            mx, my = m[0, :], m[1, :]
            
            # Cross ratio calculations
            cross_x = ((vx[0] * my[0] - mx[0] * vy[0]) * (mx[1] * vy[1] - vx[1] * my[1])) / (
                (vx[0] * my[1] - mx[1] * vy[0]) * (mx[0] * vy[1] - vx[1] * my[0])
            )
            
            cross_y = ((vx[1] * my[2] - mx[2] * vy[1]) * (mx[3] * vy[2] - vx[2] * my[3])) / (
                (vx[1] * my[3] - mx[3] * vy[1]) * (mx[2] * vy[2] - vx[2] * my[2])
            )
            
            # Convert cross ratios to screen coordinates using config values
            # Original algorithm used: xg = -200e-3 + cross_x / (1 + cross_x) * 400e-3
            # Original algorithm used: yg = 350e-3 + cross_y / (1 + cross_y) * (-300e-3)
            screen_left = self.config.screen_center_x - self.config.screen_width / 2
            screen_top = self.config.screen_center_y + self.config.screen_height / 2
            
            xg = screen_left + cross_x / (1 + cross_x) * self.config.screen_width
            zg = screen_top + cross_y / (1 + cross_y) * (-self.config.screen_height)
            
            # In our 3D coordinate system, screen is in XZ plane at Y = constant
            gaze_point = Point3D(xg, 0.0, zg)
            
            processing_time = time.time() - start_time
            
            # Return structured GazePrediction
            return GazePrediction(
                gaze_point=gaze_point,
                confidence=self._calculate_confidence(measurement),
                algorithm_name=self.algorithm_name,
                processing_time=processing_time,
                intermediate_results={
                    "cross_ratios": {"x": cross_x, "y": cross_y},
                    "virtual_points": v.tolist(),
                    "alpha_values": self.algorithm_state.alpha_values.tolist() if self.algorithm_state.alpha_values is not None else None
                }
            )
            
        except Exception as e:
            print(f"Gaze prediction failed: {e}")
            return None
            
    def _calculate_confidence(self, measurement: EyeMeasurement) -> float:
        """Calculate prediction confidence score.
        
        Args:
            measurement: EyeMeasurement for confidence assessment
            
        Returns:
            Confidence score [0, 1]
        """
        # Basic confidence based on detection quality
        confidence = 0.5
        
        camera_image = measurement.camera_image
        
        # Boost confidence if pupil detected
        if camera_image.pupil_center is not None:
            confidence += 0.2
            
        # Boost confidence if all CRs detected
        if (camera_image.corneal_reflections and 
            len(camera_image.corneal_reflections) == 5 and
            all(cr is not None for cr in camera_image.corneal_reflections)):
            confidence += 0.3
            
        return min(confidence, 1.0)