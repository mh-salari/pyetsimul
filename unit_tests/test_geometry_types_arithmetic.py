"""Unit tests for geometry types arithmetic operations."""

import numpy as np
import pytest

from pyetsimul.types import Direction3D, Point2D, Point3D, Position3D, Vector3D


class TestPoint2D:
    """Test Point2D arithmetic operations."""

    @staticmethod
    def test_scalar_multiplication() -> None:
        """Test scalar multiplication with Point2D."""
        p = Point2D(2, 3)

        # Test regular scalar multiplication
        result1 = p * 5
        assert result1 == Point2D(10, 15)

        result2 = 5 * p
        assert result2 == Point2D(10, 15)

        # Test float scalar
        result3 = p * 2.5
        assert result3 == Point2D(5.0, 7.5)

        result4 = 2.5 * p
        assert result4 == Point2D(5.0, 7.5)

    @staticmethod
    def test_numpy_scalar_multiplication() -> None:
        """Test numpy scalar multiplication with Point2D."""
        p = Point2D(2, 3)

        # Test numpy scalars
        result1 = np.float64(3.5) * p
        assert isinstance(result1, Point2D)
        assert result1 == Point2D(7.0, 10.5)

        result2 = p * np.int32(4)
        assert isinstance(result2, Point2D)
        assert result2 == Point2D(8, 12)

        result3 = np.float32(1.5) * p
        assert isinstance(result3, Point2D)
        assert result3 == Point2D(3.0, 4.5)

    @staticmethod
    def test_addition_subtraction() -> None:
        """Test addition and subtraction with Point2D."""
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)

        # Addition
        result_add = p1 + p2
        assert result_add == Point2D(4, 6)

        # Subtraction
        result_sub = p2 - p1
        assert result_sub == Point2D(2, 2)


class TestPoint3D:
    """Test Point3D arithmetic operations."""

    @staticmethod
    def test_scalar_multiplication() -> None:
        """Test scalar multiplication with Point3D."""
        p = Point3D(1, 2, 3)

        # Test regular scalar multiplication
        result1 = p * 5
        assert result1 == Point3D(5, 10, 15)

        result2 = 5 * p
        assert result2 == Point3D(5, 10, 15)

        # Test float scalar
        result3 = p * 2.5
        assert result3 == Point3D(2.5, 5.0, 7.5)

        result4 = 2.5 * p
        assert result4 == Point3D(2.5, 5.0, 7.5)

    @staticmethod
    def test_numpy_scalar_multiplication() -> None:
        """Test numpy scalar multiplication with Point3D."""
        p = Point3D(1, 2, 3)

        # Test numpy scalars
        result1 = np.float64(2.0) * p
        assert isinstance(result1, Point3D)
        assert result1 == Point3D(2.0, 4.0, 6.0)

        result2 = p * np.int32(3)
        assert isinstance(result2, Point3D)
        assert result2 == Point3D(3, 6, 9)

    @staticmethod
    def test_point_vector_operations() -> None:
        """Test Point3D operations with Vector3D."""
        p = Point3D(1, 2, 3)
        v = Vector3D(4, 5, 6)

        # Point + Vector = Point
        result_add = p + v
        assert isinstance(result_add, Point3D)
        assert result_add == Point3D(5, 7, 9)

        # Point - Point = Vector
        p2 = Point3D(10, 12, 15)
        result_sub = p2 - p
        assert isinstance(result_sub, Vector3D)
        assert result_sub == Vector3D(9, 10, 12)


class TestVector3D:
    """Test Vector3D arithmetic operations."""

    @staticmethod
    def test_scalar_multiplication() -> None:
        """Test scalar multiplication with Vector3D."""
        v = Vector3D(1, 2, 3)

        # Test regular scalar multiplication
        result1 = v * 5
        assert result1 == Vector3D(5, 10, 15)

        result2 = 5 * v
        assert result2 == Vector3D(5, 10, 15)

    @staticmethod
    def test_numpy_scalar_multiplication() -> None:
        """Test numpy scalar multiplication with Vector3D."""
        v = Vector3D(1, 2, 3)

        # Test numpy scalars
        result1 = np.float64(3.5) * v
        assert isinstance(result1, Vector3D)
        assert result1 == Vector3D(3.5, 7.0, 10.5)

        result2 = v * np.int32(2)
        assert isinstance(result2, Vector3D)
        assert result2 == Vector3D(2, 4, 6)

    @staticmethod
    def test_vector_arithmetic() -> None:
        """Test vector addition and subtraction."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)

        # Addition
        result_add = v1 + v2
        assert result_add == Vector3D(5, 7, 9)

        # Subtraction
        result_sub = v2 - v1
        assert result_sub == Vector3D(3, 3, 3)

    @staticmethod
    def test_dot_product() -> None:
        """Test dot product."""
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)

        dot_result = v1.dot(v2)
        assert dot_result == 32  # 1*4 + 2*5 + 3*6 = 32

    @staticmethod
    def test_cross_product() -> None:
        """Test cross product."""
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)

        cross_result = v1.cross(v2)
        assert cross_result == Vector3D(0, 0, 1)

    @staticmethod
    def test_magnitude_and_normalize() -> None:
        """Test magnitude and normalization."""
        v = Vector3D(3, 4, 0)

        # Magnitude
        mag = v.magnitude()
        assert mag == 5.0

        # Normalize
        normalized = v.normalize()
        assert normalized == Vector3D(0.6, 0.8, 0.0)
        assert abs(normalized.magnitude() - 1.0) < 1e-15


class TestPosition3D:
    """Test Position3D arithmetic operations."""

    @staticmethod
    def test_scalar_multiplication() -> None:
        """Test scalar multiplication with Position3D."""
        pos = Position3D(1, 2, 3)

        # Test regular scalar multiplication
        result1 = pos * 5
        assert result1 == Position3D(5, 10, 15)

        result2 = 5 * pos
        assert result2 == Position3D(5, 10, 15)

    @staticmethod
    def test_numpy_scalar_multiplication() -> None:
        """Test numpy scalar multiplication with Position3D."""
        pos = Position3D(1, 2, 3)

        # Test numpy scalars
        result1 = np.float64(2.0) * pos
        assert isinstance(result1, Position3D)
        assert result1 == Position3D(2.0, 4.0, 6.0)

        result2 = pos * np.int32(3)
        assert isinstance(result2, Position3D)
        assert result2 == Position3D(3, 6, 9)

    @staticmethod
    def test_position_vector_operations() -> None:
        """Test Position3D operations with Vector3D."""
        pos = Position3D(1, 2, 3)
        v = Vector3D(4, 5, 6)

        # Position + Vector = Position
        result_add = pos + v
        assert isinstance(result_add, Position3D)
        assert result_add == Position3D(5, 7, 9)

        # Position - Position = Vector
        pos2 = Position3D(10, 12, 15)
        result_sub = pos2 - pos
        assert isinstance(result_sub, Vector3D)
        assert result_sub == Vector3D(9, 10, 12)

    @staticmethod
    def test_homogeneous_array_conversion() -> None:
        """Test that Position3D converts to 4D homogeneous arrays."""
        pos = Position3D(1, 2, 3)
        arr = np.array(pos)

        assert arr.shape == (4,)
        assert np.array_equal(arr, [1, 2, 3, 1])

    @staticmethod
    def test_distance_calculation() -> None:
        """Test distance calculation between positions."""
        pos1 = Position3D(0, 0, 0)
        pos2 = Position3D(3, 4, 0)

        distance = pos1.distance_to(pos2)
        assert distance == 5.0


class TestDirection3D:
    """Test Direction3D arithmetic operations."""

    @staticmethod
    def test_scalar_multiplication() -> None:
        """Test scalar multiplication with Direction3D."""
        d = Direction3D(1, 2, 3)

        # Test regular scalar multiplication
        result1 = d * 5
        assert result1 == Direction3D(5, 10, 15)

        result2 = 5 * d
        assert result2 == Direction3D(5, 10, 15)

    @staticmethod
    def test_numpy_scalar_multiplication() -> None:
        """Test numpy scalar multiplication with Direction3D."""
        d = Direction3D(1, 2, 3)

        # Test numpy scalars
        result1 = np.float64(2.0) * d
        assert isinstance(result1, Direction3D)
        assert result1 == Direction3D(2.0, 4.0, 6.0)

        result2 = d * np.int32(3)
        assert isinstance(result2, Direction3D)
        assert result2 == Direction3D(3, 6, 9)

    @staticmethod
    def test_direction_arithmetic() -> None:
        """Test direction addition and subtraction."""
        d1 = Direction3D(1, 2, 3)
        d2 = Direction3D(4, 5, 6)

        # Addition
        result_add = d1 + d2
        assert result_add == Direction3D(5, 7, 9)

        # Subtraction
        result_sub = d2 - d1
        assert result_sub == Direction3D(3, 3, 3)

    @staticmethod
    def test_dot_product() -> None:
        """Test dot product."""
        d1 = Direction3D(1, 2, 3)
        d2 = Direction3D(4, 5, 6)

        dot_result = d1.dot(d2)
        assert dot_result == 32  # 1*4 + 2*5 + 3*6 = 32

    @staticmethod
    def test_homogeneous_array_conversion() -> None:
        """Test that Direction3D converts to 4D homogeneous arrays."""
        d = Direction3D(1, 2, 3)
        arr = np.array(d)

        assert arr.shape == (4,)
        assert np.array_equal(arr, [1, 2, 3, 0])


class TestMatrixOperations:
    """Test matrix operations with all types."""

    @staticmethod
    def test_matrix_multiplication() -> None:
        """Test matrix multiplication with structured types."""
        # Rotation matrix (90 degrees around Z axis)
        rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Test Position3D
        pos = Position3D(1, 0, 0)
        rotated_pos = rotation @ pos
        assert isinstance(rotated_pos, Position3D)
        rotated_pos.assert_close(Position3D(0, 1, 0), atol=1e-15)

        # Test Direction3D
        d = Direction3D(1, 0, 0)
        rotated_dir = rotation @ d
        assert isinstance(rotated_dir, Direction3D)
        rotated_dir.assert_close(Direction3D(0, 1, 0), atol=1e-15)


class TestTypeConversions:
    """Test conversions between types."""

    @staticmethod
    def test_point3d_to_position3d() -> None:
        """Test Point3D to Position3D conversion."""
        p = Point3D(1, 2, 3)
        pos = p.to_position3d()
        assert isinstance(pos, Position3D)
        assert pos == Position3D(1, 2, 3)

    @staticmethod
    def test_position3d_from_point3d() -> None:
        """Test Position3D from Point3D creation."""
        p = Point3D(1, 2, 3)
        pos = Position3D.from_point3d(p)
        assert isinstance(pos, Position3D)
        assert pos == Position3D(1, 2, 3)

    @staticmethod
    def test_vector3d_direction3d_conversion() -> None:
        """Test Vector3D and Direction3D conversions."""
        d = Vector3D(1, 2, 3).to_direction3d()
        assert isinstance(d, Direction3D)
        assert d == Direction3D(1, 2, 3)

        v2 = d.to_vector3d()
        assert isinstance(v2, Vector3D)
        assert v2 == Vector3D(1, 2, 3)


class TestClosenessComparison:
    """Test closeness comparison methods."""

    @staticmethod
    def test_isclose_methods() -> None:
        """Test isclose methods for all types."""
        # Point2D
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.0000001, 2.0000001)
        assert p1.isclose(p2, rtol=1e-6)
        assert not p1.isclose(p2, rtol=1e-8)

        # Point3D
        p3d1 = Point3D(1.0, 2.0, 3.0)
        p3d2 = Point3D(1.0000001, 2.0000001, 3.0000001)
        assert p3d1.isclose(p3d2, rtol=1e-6)

        # Vector3D
        v1 = Vector3D(1.0, 2.0, 3.0)
        v2 = Vector3D(1.0000001, 2.0000001, 3.0000001)
        assert v1.isclose(v2, rtol=1e-6)

        # Position3D
        pos1 = Position3D(1.0, 2.0, 3.0)
        pos2 = Position3D(1.0000001, 2.0000001, 3.0000001)
        assert pos1.isclose(pos2, rtol=1e-6)

        # Direction3D
        d1 = Direction3D(1.0, 2.0, 3.0)
        d2 = Direction3D(1.0000001, 2.0000001, 3.0000001)
        assert d1.isclose(d2, rtol=1e-6)

    @staticmethod
    def test_assert_close_methods() -> None:
        """Test assert_close methods for all types."""
        # Should not raise for close values
        p1 = Point2D(1.0, 2.0)
        p2 = Point2D(1.0000001, 2.0000001)
        p1.assert_close(p2, rtol=1e-6)  # Should not raise

        # Should raise for distant values
        p3 = Point2D(2.0, 3.0)
        with pytest.raises(AssertionError):
            p1.assert_close(p3, rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_zero_vector_normalization() -> None:
        """Test that normalizing zero vector raises error."""
        zero_vector = Vector3D(0, 0, 0)
        with pytest.raises(ValueError, match="Cannot normalize zero vector"):
            zero_vector.normalize()

    @staticmethod
    def test_array_conversion_consistency() -> None:
        """Test that __array__ and to_array methods are consistent."""
        # Test Vector3D (3D)
        v = Vector3D(1, 2, 3)
        arr1 = np.array(v)
        arr2 = v.to_array()
        assert np.array_equal(arr1, arr2)

        # Test Position3D (4D homogeneous)
        pos = Position3D(1, 2, 3)
        arr3 = np.array(pos)
        arr4 = pos.to_array()
        assert np.array_equal(arr3, arr4)

    @staticmethod
    def test_mixed_arithmetic_operations() -> None:
        """Test complex mixed arithmetic operations."""
        # Real-world scenario: point + scalar * vector
        p = Point3D(1, 2, 3)
        v = Vector3D(4, 5, 6)

        # This should work with all scalar types
        result1 = p + v * 2
        result2 = p + v * np.float64(2.0)
        result3 = p + 2 * v
        result4 = p + np.float64(2.0) * v

        expected = Point3D(9, 12, 15)
        assert result1 == expected
        assert result2 == expected
        assert result3 == expected
        assert result4 == expected

        # All should return Point3D
        assert isinstance(result1, Point3D)
        assert isinstance(result2, Point3D)
        assert isinstance(result3, Point3D)
        assert isinstance(result4, Point3D)
