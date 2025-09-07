"""
This module provides dataclasses to replace raw numpy arrays,
improving type safety and code readability.
"""

from dataclasses import dataclass
from typing import Optional, overload, Union
import numpy as np


@dataclass(frozen=True)
class Point2D:
    """A 2D point with validation."""

    x: float
    y: float

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Enable numpy array operations."""
        arr = np.array([self.x, self.y])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        return arr

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point2D":
        """Create Point2D from numpy array."""
        if arr.shape != (2,):
            raise ValueError(f"Expected array shape (2,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]))

    def isclose(self, other: "Point2D", rtol=1e-9, atol=1e-12) -> bool:
        """Compare with tolerance."""
        return bool(
            np.isclose(self.x, other.x, rtol=rtol, atol=atol) and np.isclose(self.y, other.y, rtol=rtol, atol=atol)
        )

    def assert_close(self, other: "Point2D", rtol=1e-9, atol=1e-12, msg="") -> None:
        """Assert close with custom error message."""
        if not self.isclose(other, rtol=rtol, atol=atol):
            error_msg = f"{self} != {other} (rtol={rtol}, atol={atol})"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def __sub__(self, other):
        """Subtract point or scalar from point."""
        if isinstance(other, Point2D):
            return Point2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, (int, float)):
            return Point2D(self.x - other, self.y - other)
        else:
            return NotImplemented

    def __add__(self, other) -> "Point2D":
        """Add two points/vectors or add scalar to point."""
        if isinstance(other, Point2D):
            return Point2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, (int, float)):
            return Point2D(self.x + other, self.y + other)
        else:
            return NotImplemented

    def __mul__(self, scalar: float) -> "Point2D":
        """Multiply point by a scalar."""
        return Point2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Point2D":
        """Multiply point by a scalar (reverse order)."""
        return self.__mul__(scalar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain Point2D type."""
        if ufunc == np.multiply and method == "__call__":
            # Handle scalar * point multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], (int, float, np.number)) and isinstance(inputs[1], Point2D):
                    return inputs[1] * float(inputs[0])  # point * scalar
                elif isinstance(inputs[0], Point2D) and isinstance(inputs[1], (int, float, np.number)):
                    return inputs[0] * float(inputs[1])  # point * scalar

        # For other operations, defer to numpy
        return NotImplemented

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {"x": float(self.x), "y": float(self.y)}

    @classmethod
    def deserialize(cls, data: dict) -> "Point2D":
        """Deserialize from dictionary representation."""
        return cls(data["x"], data["y"])


@dataclass(frozen=True)
class Point3D:
    """A 3D point with validation."""

    x: float
    y: float
    z: float

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Enable numpy array operations."""
        arr = np.array([self.x, self.y, self.z])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        return arr

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    def to_homogeneous(self) -> np.ndarray:
        """Convert to homogeneous coordinates [x, y, z, 1]."""
        return np.array([self.x, self.y, self.z, 1.0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point3D":
        """Create Point3D from numpy array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected array shape (3,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    def distance_to(self, other: "Point3D") -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def isclose(self, other: "Point3D", rtol=1e-9, atol=1e-12) -> bool:
        """Compare with tolerance."""
        return bool(
            np.isclose(self.x, other.x, rtol=rtol, atol=atol)
            and np.isclose(self.y, other.y, rtol=rtol, atol=atol)
            and np.isclose(self.z, other.z, rtol=rtol, atol=atol)
        )

    def assert_close(self, other: "Point3D", rtol=1e-9, atol=1e-12, msg="") -> None:
        """Assert close with custom error message."""
        if not self.isclose(other, rtol=rtol, atol=atol):
            error_msg = f"{self} != {other} (rtol={rtol}, atol={atol})"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    @overload
    def __sub__(self, other: Union["Point3D", "Position3D"]) -> "Vector3D": ...
    @overload
    def __sub__(self, other: Union["Vector3D", "Direction3D"]) -> "Point3D": ...
    @overload
    def __sub__(self, other: int | float) -> "Point3D": ...
    def __sub__(self, other):
        """Subtract point, position, vector, or scalar from point."""
        if isinstance(other, (Point3D, Position3D)):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (Vector3D, Direction3D)):
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return Point3D(self.x - other, self.y - other, self.z - other)
        else:
            return NotImplemented

    def __add__(self, other) -> "Point3D":
        """Add a vector or scalar to a point to get a new point."""
        if isinstance(other, Vector3D):
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (int, float)):
            return Point3D(self.x + other, self.y + other, self.z + other)
        else:
            return NotImplemented

    def __mul__(self, scalar: float) -> "Point3D":
        """Multiply point by a scalar."""
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Point3D":
        """Multiply point by a scalar (reverse order)."""
        return self.__mul__(scalar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain Point3D type."""
        if ufunc == np.multiply and method == "__call__":
            # Handle scalar * point multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], (int, float, np.number)) and isinstance(inputs[1], Point3D):
                    return inputs[1] * float(inputs[0])  # point * scalar
                elif isinstance(inputs[0], Point3D) and isinstance(inputs[1], (int, float, np.number)):
                    return inputs[0] * float(inputs[1])  # point * scalar

        # For other operations, defer to numpy
        return NotImplemented

    def to_position3d(self) -> "Position3D":
        """Convert to Position3D (homogeneous coordinates with w=1)."""
        return Position3D(x=self.x, y=self.y, z=self.z)


@dataclass(frozen=True)
class Vector3D:
    """A 3D vector with validation and common operations."""

    x: float
    y: float
    z: float

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Enable numpy array operations."""
        arr = np.array([self.x, self.y, self.z])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        return arr

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for compatibility."""
        return np.array([self.x, self.y, self.z])

    def to_homogeneous(self) -> np.ndarray:
        """Convert to homogeneous coordinates [x, y, z, 0]."""
        return np.array([self.x, self.y, self.z, 0.0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vector3D":
        """Create Vector3D from numpy array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected array shape (3,), got {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector3D":
        """Return normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)

    @overload
    def dot(self, other: "Direction3D") -> float: ...
    @overload
    def dot(self, other: "Vector3D") -> float: ...
    def dot(self, other) -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Calculate cross product with another vector."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def to_direction3d(self) -> "Direction3D":
        """Convert to Direction3D."""
        return Direction3D(self.x, self.y, self.z)

    def isclose(self, other: "Vector3D", rtol=1e-9, atol=1e-12) -> bool:
        """Compare with tolerance."""
        return bool(
            np.isclose(self.x, other.x, rtol=rtol, atol=atol)
            and np.isclose(self.y, other.y, rtol=rtol, atol=atol)
            and np.isclose(self.z, other.z, rtol=rtol, atol=atol)
        )

    def assert_close(self, other: "Vector3D", rtol=1e-9, atol=1e-12, msg="") -> None:
        """Assert close with custom error message."""
        if not self.isclose(other, rtol=rtol, atol=atol):
            error_msg = f"{self} != {other} (rtol={rtol}, atol={atol})"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def __rmatmul__(self, other):
        """Enable matrix multiplication: matrix @ vector."""
        if isinstance(other, np.ndarray):
            result = other @ np.array(self)
            return Vector3D.from_array(result)
        return NotImplemented

    def __add__(self, other) -> "Vector3D":
        """Add two vectors or add scalar to vector."""
        if isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (int, float)):
            return Vector3D(self.x + other, self.y + other, self.z + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtract vector or scalar from vector."""
        if isinstance(other, Vector3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return Vector3D(self.x - other, self.y - other, self.z - other)
        else:
            return NotImplemented

    def __mul__(self, scalar: float) -> "Vector3D":
        """Multiply vector by a scalar."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3D":
        """Multiply vector by a scalar (reverse order)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3D":
        """Divide vector by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain Vector3D type."""
        if ufunc == np.multiply and method == "__call__":
            # Handle scalar * vector multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], (int, float, np.number)) and isinstance(inputs[1], Vector3D):
                    return inputs[1] * float(inputs[0])  # vector * scalar
                elif isinstance(inputs[0], Vector3D) and isinstance(inputs[1], (int, float, np.number)):
                    return inputs[0] * float(inputs[1])  # vector * scalar
        elif ufunc == np.matmul and method == "__call__":
            # Handle matrix @ vector multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], np.ndarray) and isinstance(inputs[1], Vector3D):
                    # Call the __rmatmul__ method directly
                    return inputs[1].__rmatmul__(inputs[0])

        # For other operations, defer to numpy
        return NotImplemented

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @classmethod
    def deserialize(cls, data: dict) -> "Vector3D":
        """Deserialize from dictionary representation."""
        return cls(data["x"], data["y"], data["z"])


@dataclass(frozen=True)
class Position3D:
    """A 3D position that can be converted to homogeneous coordinates [x,y,z,1]."""

    x: float
    y: float
    z: float

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Enable numpy array operations."""
        arr = np.array([self.x, self.y, self.z, 1.0])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        return arr

    def to_array(self) -> np.ndarray:
        """Convert to homogeneous 4D array [x,y,z,1]."""
        return np.array([self.x, self.y, self.z, 1.0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Position3D":
        """Create from 4D homogeneous array [x,y,z,1] or 3D array [x,y,z]."""
        if arr.shape == (4,):
            if arr[3] != 1.0 and arr[3] != 0.0:
                # De-homogenize if needed
                return cls(arr[0] / arr[3], arr[1] / arr[3], arr[2] / arr[3])
            return cls(arr[0], arr[1], arr[2])
        elif arr.shape == (3,):
            return cls(arr[0], arr[1], arr[2])
        else:
            raise ValueError(f"Expected array shape (3,) or (4,), got {arr.shape}")

    def to_point3d(self) -> Point3D:
        """Convert to Point3D."""
        return Point3D(self.x, self.y, self.z)

    def isclose(self, other: "Position3D", rtol=1e-9, atol=1e-12) -> bool:
        """Compare with tolerance."""
        return bool(
            np.isclose(self.x, other.x, rtol=rtol, atol=atol)
            and np.isclose(self.y, other.y, rtol=rtol, atol=atol)
            and np.isclose(self.z, other.z, rtol=rtol, atol=atol)
        )

    def assert_close(self, other: "Position3D", rtol=1e-9, atol=1e-12, msg="") -> None:
        """Assert close with custom error message."""
        if not self.isclose(other, rtol=rtol, atol=atol):
            error_msg = f"{self} != {other} (rtol={rtol}, atol={atol})"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def __matmul__(self, other):
        """Enable matrix multiplication: position @ matrix."""
        if isinstance(other, np.ndarray):
            result = np.array(self) @ other
            return Position3D.from_array(result)
        return NotImplemented

    def __rmatmul__(self, other):
        """Enable matrix multiplication: matrix @ position (most common case)."""
        if isinstance(other, np.ndarray):
            result = other @ np.array(self)
            return Position3D.from_array(result)
        return NotImplemented

    @overload
    def __sub__(self, other: Union["Position3D", Point3D]) -> Vector3D: ...
    @overload
    def __sub__(self, other: Union[Vector3D, "Direction3D"]) -> "Position3D": ...
    @overload
    def __sub__(self, other: int | float) -> "Position3D": ...
    def __sub__(self, other):
        """Subtract position, point, vector, direction, or scalar from position."""
        if isinstance(other, (Position3D, Point3D)):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (Vector3D, Direction3D)):
            return Position3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return Position3D(self.x - other, self.y - other, self.z - other)
        else:
            return NotImplemented

    def __add__(self, other) -> "Position3D":
        """Add a vector, direction, point, or scalar to a position to get a new position."""
        if isinstance(other, (Vector3D, Direction3D, Point3D)):
            return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (int, float)):
            return Position3D(self.x + other, self.y + other, self.z + other)
        else:
            return NotImplemented

    def __mul__(self, scalar: float) -> "Position3D":
        """Multiply position by a scalar."""
        return Position3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Position3D":
        """Multiply position by a scalar (reverse order)."""
        return self.__mul__(scalar)

    def __radd__(self, other) -> "Position3D":
        """Add position to a scalar (reverse order)."""
        return self.__add__(other)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain Position3D type."""
        if ufunc == np.multiply and method == "__call__":
            # Handle scalar * position multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], (int, float, np.number)) and isinstance(inputs[1], Position3D):
                    return inputs[1] * float(inputs[0])  # position * scalar
                elif isinstance(inputs[0], Position3D) and isinstance(inputs[1], (int, float, np.number)):
                    return inputs[0] * float(inputs[1])  # position * scalar
        elif ufunc == np.matmul and method == "__call__":
            # Handle matrix @ position multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], np.ndarray) and isinstance(inputs[1], Position3D):
                    # Call the __rmatmul__ method directly
                    return inputs[1].__rmatmul__(inputs[0])

        # For other operations, defer to numpy
        return NotImplemented

    @classmethod
    def from_point3d(cls, point: Point3D) -> "Position3D":
        """Create Position3D from Point3D."""
        return cls(point.x, point.y, point.z)

    def distance_to(self, other: "Position3D") -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @classmethod
    def deserialize(cls, data: dict) -> "Position3D":
        """Deserialize from dictionary representation."""
        return cls(data["x"], data["y"], data["z"])


@dataclass(frozen=True)
class Direction3D:
    """A 3D direction vector that converts to homogeneous [x,y,z,0]."""

    x: float
    y: float
    z: float

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Enable numpy array operations."""
        arr = np.array([self.x, self.y, self.z, 0.0])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        return arr

    def to_array(self) -> np.ndarray:
        """Convert to homogeneous 4D array [x,y,z,0]."""
        return np.array([self.x, self.y, self.z, 0.0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Direction3D":
        """Create from 4D homogeneous array [x,y,z,0] or 3D array [x,y,z]."""
        if arr.shape == (4,) or arr.shape == (3,):
            return cls(arr[0], arr[1], arr[2])
        else:
            raise ValueError(f"Expected array shape (3,) or (4,), got {arr.shape}")

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Direction3D":
        """Return normalized direction vector."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return Direction3D(self.x / mag, self.y / mag, self.z / mag)

    @overload
    def dot(self, other: "Direction3D") -> float: ...
    @overload
    def dot(self, other: Vector3D) -> float: ...
    def dot(self, other) -> float:
        """Calculate dot product with another direction."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Direction3D") -> "Direction3D":
        """Calculate cross product with another direction."""
        return Direction3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def to_vector3d(self) -> Vector3D:
        """Convert to Vector3D."""
        return Vector3D(self.x, self.y, self.z)

    def isclose(self, other: "Direction3D", rtol=1e-9, atol=1e-12) -> bool:
        """Compare with tolerance."""
        return bool(
            np.isclose(self.x, other.x, rtol=rtol, atol=atol)
            and np.isclose(self.y, other.y, rtol=rtol, atol=atol)
            and np.isclose(self.z, other.z, rtol=rtol, atol=atol)
        )

    def assert_close(self, other: "Direction3D", rtol=1e-9, atol=1e-12, msg="") -> None:
        """Assert close with custom error message."""
        if not self.isclose(other, rtol=rtol, atol=atol):
            error_msg = f"{self} != {other} (rtol={rtol}, atol={atol})"
            if msg:
                error_msg = f"{msg}: {error_msg}"
            raise AssertionError(error_msg)

    def __rmatmul__(self, other):
        """Enable matrix multiplication: matrix @ direction."""
        if isinstance(other, np.ndarray):
            result = other @ np.array(self)
            return Direction3D.from_array(result)
        return NotImplemented

    def __mul__(self, scalar: float) -> "Direction3D":
        """Multiply direction by a scalar."""
        return Direction3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Direction3D":
        """Multiply direction by a scalar (reverse order)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Direction3D":
        """Divide direction by a scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide direction by zero")
        return Direction3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions to maintain Direction3D type."""
        if ufunc == np.multiply and method == "__call__":
            # Handle scalar * direction multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], (int, float, np.number)) and isinstance(inputs[1], Direction3D):
                    return inputs[1] * float(inputs[0])  # direction * scalar
                elif isinstance(inputs[0], Direction3D) and isinstance(inputs[1], (int, float, np.number)):
                    return inputs[0] * float(inputs[1])  # direction * scalar
        elif ufunc == np.matmul and method == "__call__":
            # Handle matrix @ direction multiplication
            if len(inputs) == 2:
                if isinstance(inputs[0], np.ndarray) and isinstance(inputs[1], Direction3D):
                    # Call the __rmatmul__ method directly
                    return inputs[1].__rmatmul__(inputs[0])

        # For other operations, defer to numpy
        return NotImplemented

    def __add__(self, other) -> "Direction3D":
        """Add two directions or add scalar to direction."""
        if isinstance(other, Direction3D):
            return Direction3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (int, float)):
            return Direction3D(self.x + other, self.y + other, self.z + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtract direction or scalar from direction."""
        if isinstance(other, Direction3D):
            return Direction3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return Direction3D(self.x - other, self.y - other, self.z - other)
        else:
            return NotImplemented

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @classmethod
    def deserialize(cls, data: dict) -> "Direction3D":
        """Deserialize from dictionary representation."""
        return cls(data["x"], data["y"], data["z"])


@dataclass(frozen=True)
class Ray:
    """A ray defined by origin point and direction vector."""

    origin: Point3D
    direction: Direction3D

    def point_at(self, t: float) -> Point3D:
        """Get point along ray at parameter t."""
        return Point3D(
            self.origin.x + t * self.direction.x,
            self.origin.y + t * self.direction.y,
            self.origin.z + t * self.direction.z,
        )

    @classmethod
    def from_two_points(cls, p1: Point3D, p2: Point3D) -> "Ray":
        """Create ray from two points."""
        direction_vec = Direction3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        return cls(origin=p1, direction=direction_vec.normalize())


@dataclass(frozen=True)
class IntersectionResult:
    """Result of ray-surface intersection calculation."""

    intersects: bool
    point: Optional[Point3D] = None
    distance: Optional[float] = None
    surface_normal: Optional[Direction3D] = None

    @classmethod
    def no_intersection(cls) -> "IntersectionResult":
        """Create result indicating no intersection."""
        return cls(intersects=False)

    @classmethod
    def intersection_at(
        cls, point: Point3D, distance: float, normal: Optional[Direction3D] = None
    ) -> "IntersectionResult":
        """Create result indicating intersection at given point."""
        return cls(intersects=True, point=point, distance=distance, surface_normal=normal)


class RotationMatrix(np.ndarray):
    """A 3x3 rotation matrix that validates its mathematical properties."""

    def __new__(cls, input_array, validate_handedness: bool = True):
        """Create a new rotation matrix from input array or list with validation.

        Args:
            input_array: 3x3 array or nested list representing the rotation matrix
            validate_handedness: If True, enforce right-handed coordinate system (det=+1)
                                If False, allow left-handed systems (det=-1) for legacy compatibility
        """
        # Convert input to numpy array (handles both arrays and lists)
        obj = np.asarray(input_array, dtype=np.float64).view(cls)
        if obj.shape != (3, 3):
            raise ValueError(f"RotationMatrix must be 3x3, got shape {obj.shape}")

        # Validate it's a proper rotation matrix
        cls._validate(obj, validate_handedness=validate_handedness)
        return obj

    @staticmethod
    def _validate(matrix: np.ndarray, validate_handedness: bool = True) -> None:
        """Validate that matrix is a proper rotation matrix.

        Args:
            matrix: The matrix to validate
            validate_handedness: If True, enforce right-handed coordinate system
        """
        # Check if matrix is orthonormal (R^T * R = I)
        should_be_identity = matrix.T @ matrix
        if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
            raise ValueError("Matrix is not orthonormal")

        # Check determinant is +1 (right-handed) or -1 (left-handed)
        det = np.linalg.det(matrix)
        if validate_handedness:
            if not np.isclose(det, 1.0, atol=1e-6):
                raise ValueError(f"Matrix determinant is {det:.6f}, expected +1.0 for right-handed rotation")
        else:
            # Allow both +1 and -1 determinants (right-handed and left-handed)
            if not (np.isclose(det, 1.0, atol=1e-6) or np.isclose(det, -1.0, atol=1e-6)):
                raise ValueError(f"Matrix determinant is {det:.6f}, expected ±1.0 for rotation matrix")

    @classmethod
    def identity(cls) -> "RotationMatrix":
        """Create an identity rotation matrix."""
        return cls(np.eye(3))

    @classmethod
    def deserialize(cls, data) -> "RotationMatrix":
        """Create RotationMatrix from serialized data with automatic handedness detection.

        Tries strict right-handed validation first, falls back to allowing left-handed
        matrices for legacy compatibility.

        Args:
            data: Matrix data (list or array)
        """
        try:
            # Try with strict right-handed validation first
            return cls(data, validate_handedness=True)
        except ValueError:
            # Fall back to allowing left-handed matrices for legacy data
            return cls(data, validate_handedness=False)


class TransformationMatrix(np.ndarray):
    """A 4x4 homogeneous transformation matrix with convenient factory methods."""

    def __new__(cls, input_array):
        """Create a new transformation matrix from input array."""
        obj = np.asarray(input_array).view(cls)
        if obj.shape != (4, 4):
            raise ValueError(f"TransformationMatrix must be 4x4, got shape {obj.shape}")
        return obj

    @classmethod
    def identity(cls) -> "TransformationMatrix":
        """Create an identity transformation matrix."""
        return cls(np.eye(4))

    @classmethod
    def from_translation(cls, translation: Position3D) -> "TransformationMatrix":
        """Create a translation matrix from a Position3D."""
        matrix = np.eye(4)
        matrix[0:3, 3] = [translation.x, translation.y, translation.z]
        return cls(matrix)

    @classmethod
    def from_rotation(cls, rotation_matrix: RotationMatrix) -> "TransformationMatrix":
        """Create a transformation matrix from a 3x3 rotation matrix."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got shape {rotation_matrix.shape}")
        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation_matrix
        return cls(matrix)

    @classmethod
    def from_translation_and_rotation(
        cls, translation: Position3D, rotation_matrix: RotationMatrix
    ) -> "TransformationMatrix":
        """Create a transformation matrix from a 3x3 rotation matrix."""
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got shape {rotation_matrix.shape}")
        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation_matrix
        matrix[0:3, 3] = [translation.x, translation.y, translation.z]
        return cls(matrix)

    def get_rotation(self) -> RotationMatrix:
        """Extract the 3x3 rotation matrix."""
        if self.shape != (4, 4):
            raise ValueError("Input must be a 4x4 matrix.")

        # Extract the 3x3 upper-left submatrix as column vectors
        c1 = self[:3, 0]
        c2 = self[:3, 1]
        c3 = self[:3, 2]

        # Normalize columns to get pure rotation matrix, assemble
        rotation_matrix = np.column_stack((c1 / np.linalg.norm(c1), c2 / np.linalg.norm(c2), c3 / np.linalg.norm(c3)))

        return RotationMatrix(rotation_matrix)

    def get_translation(self) -> Position3D:
        """Extract the translation vector as Position3D."""
        return Position3D(x=self[0, 3], y=self[1, 3], z=self[2, 3])
