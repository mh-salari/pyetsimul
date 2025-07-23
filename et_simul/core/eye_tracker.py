"""Eye tracker module.

This module provides the EyeTracker class that represents a complete eye tracking
system with cameras, lights, calibration points, and algorithm functions.
"""

import numpy as np
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from .camera import Camera
from .light import Light


@dataclass
class EyeTracker(ABC):
    """Eye tracker with cameras, lights, calibration, and algorithms.
    
    Represents a complete eye tracking system including:
    - Cameras for image capture
    - Lights for corneal reflections
    - Calibration points/grid
    - Calibration and evaluation functions
    - Algorithm state/parameters
    """
    
    # Physical components
    cameras: List[Camera] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)
    
    # Calibration
    calib_points: Optional[np.ndarray] = None
    
    # Algorithm state/parameters
    state: Dict[str, Any] = field(default_factory=dict)
    
    def add_camera(self, camera: Camera):
        """Add a camera to the eye tracker."""
        self.cameras.append(camera)
    
    def add_light(self, light: Light):
        """Add a light to the eye tracker."""
        self.lights.append(light)
    
    def run_calibration(self, eye):
        """Run the complete calibration workflow.
        
        Generic calibration process that works for all eye tracker types:
        1. Collect calibration data at multiple target points
        2. Call the eye tracker's specific calibration method
        
        Args:
            eye: Eye object to calibrate with
            
        Returns:
            Self for method chaining
        """
        calib_data = self._collect_calibration_data(eye)
        self.calibrate(calib_data)
        return self
    
    def _collect_calibration_data(self, eye):
        """Collect calibration data at all calibration points.
        
        Generic data collection process that works for all eye trackers.
        Exactly matches the original calibrate_eye_tracker logic.
        """
        if self.calib_points is None:
            raise ValueError("Calibration points not set")
            
        calib_data = [None] * self.calib_points.shape[1]
        np.random.seed(0)  # For reproducible results
        
        for i in range(self.calib_points.shape[1]):
            # Make eye look at calibration point
            target = np.array([self.calib_points[0, i], 0, self.calib_points[1, i], 1])
            eye.look_at(target)
            
            # Take images from all cameras
            calib_data[i] = {}
            calib_data[i]['camimg'] = [None] * len(self.cameras)
            for iCamera, cam in enumerate(self.cameras):
                calib_data[i]['camimg'][iCamera] = cam.take_image(eye, self.lights)
            
            # Store eye state
            calib_data[i]['e'] = copy.deepcopy(eye)
        
        return calib_data
    
    def calculate_gaze_error(self, eye, look_at_pos):
        """Calculate gaze estimation error.
        
        Generic error calculation that works for all eye tracker types.
        Exactly matches the original gaze_error logic.
        
        Args:
            eye: Eye object
            look_at_pos: 2D position where eye should look [x, y]
            
        Returns:
            Tuple of (u, v) gaze error in meters, or (NaN, NaN) if estimation fails
        """
        # Make eye look at target position
        target = np.array([look_at_pos[0], 0, look_at_pos[1], 1])
        eye.look_at(target)
        
        # Take camera images
        camimg = [None] * len(self.cameras)
        for iCamera, cam in enumerate(self.cameras):
            camimg[iCamera] = cam.take_image(eye, self.lights)
        
        # Get gaze prediction
        gaze = self.predict_gaze(camimg)
        
        if gaze is not None:
            u = gaze[0] - look_at_pos[0]
            v = gaze[1] - look_at_pos[1]
            return u, v
        else:
            return np.nan, np.nan
    
    @abstractmethod
    def calibrate(self, calib_data):
        """Calibrate the eye tracker using collected data.
        
        Each eye tracker type must implement its specific calibration algorithm.
        
        Args:
            calib_data: List of calibration data collected at each calibration point
        """
        pass
    
    @abstractmethod
    def predict_gaze(self, camimg):
        """Predict gaze position from camera images.
        
        Each eye tracker type must implement its specific gaze prediction algorithm.
        
        Args:
            camimg: List of camera images containing pupil and corneal reflection data
            
        Returns:
            2D gaze position [x, y] on screen or None if prediction fails
        """
        pass