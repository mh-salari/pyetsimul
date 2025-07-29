import numpy as np
from dataclasses import dataclass


@dataclass
class Pupil:
    """Represents a pupil with parametric elliptical shape.
    
    This class encapsulates all pupil-related calculations using the parametric
    representation: pos_pupil + cos(α)*x_pupil + sin(α)*y_pupil
    
    Args:
        pupil_type: Type of pupil ("elliptical" is currently supported)
        pos_pupil: 4D homogeneous center position
        x_pupil: 4D vector defining X-axis radius/direction  
        y_pupil: 4D vector defining Y-axis radius/direction
    """
    pupil_type: str
    pos_pupil: np.ndarray
    x_pupil: np.ndarray  
    y_pupil: np.ndarray
    
    def __post_init__(self):
        """Validate pupil type."""
        if self.pupil_type != "elliptical":
            raise NotImplementedError(
                f"Pupil type '{self.pupil_type}' is not yet implemented. "
                f"Currently supported: 'elliptical'. "
                f"Integration with other pupil types (e.g., 'human') has not been implemented yet."
            )
    
    def get_boundary_points(self, N: int = 20) -> np.ndarray:
        """Generate pupil boundary points using parametric representation.
        
        Args:
            N: Number of points on pupil boundary
            
        Returns:
            4xN matrix of points on pupil boundary
        """
        alpha = 2 * np.pi * np.arange(N) / N
        
        # Parametric pupil boundary: pos_pupil + cos(α)*x_pupil + sin(α)*y_pupil
        pupil_points = (
            np.tile(self.pos_pupil.reshape(-1, 1), (1, N))
            + self.x_pupil.reshape(-1, 1) @ np.cos(alpha).reshape(1, -1)
            + self.y_pupil.reshape(-1, 1) @ np.sin(alpha).reshape(1, -1)
        )
        
        return pupil_points
    
    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.
        
        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        x_radius = np.linalg.norm(self.x_pupil[:3])
        y_radius = np.linalg.norm(self.y_pupil[:3])
        return x_radius, y_radius
    
    def set_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Set pupil radii and update geometry.
        
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
    
    def get_center_world_coords(self, eye_transform: np.ndarray) -> np.ndarray:
        """Get pupil center in world coordinates.
        
        Args:
            eye_transform: 4x4 transformation matrix from eye to world coordinates
            
        Returns:
            4D homogeneous coordinates of pupil center in world coordinates  
        """
        return eye_transform @ self.pos_pupil