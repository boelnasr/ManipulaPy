import numpy as np

class Transformations:
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """Create a rotation matrix for a rotation around the X-axis."""
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """Create a rotation matrix for a rotation around the Y-axis."""
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """Create a rotation matrix for a rotation around the Z-axis."""
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def vector_2_matrix(vector: np.ndarray) -> np.ndarray:
        """
        Calculate the pose from the position and Euler angles (ZYX order).

        Args:
            vector (np.ndarray): A 6-element array where the first 3 elements are x, y, z translation,
                                and the last 3 elements are rotation angles.

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        translation_component = vector[:3]
        rotation_component = vector[3:]

        # Rotation matrices for each Euler angle
        Rz = Transformations.rotation_matrix_z(rotation_component[0])
        Ry = Transformations.rotation_matrix_y(rotation_component[1])
        Rx = Transformations.rotation_matrix_x(rotation_component[2])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Construct the transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = translation_component

        return transform_matrix