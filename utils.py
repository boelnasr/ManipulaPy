import numpy as np

def extract_r_list(Slist):
    """
    Extracts the r_list from the given Slist.
    
    Parameters:
        Slist (list): A list of S vectors representing the joint screws.
        
    Returns:
        np.ndarray: An array of r vectors.
    """
    
    r_list = []
    for S in np.array(Slist).T:
        omega = S[:3]
        v = S[3:]
        if np.linalg.norm(omega) != 0:
            r = -np.cross(omega, v) / np.linalg.norm(omega)**2
            r_list.append(r)
        else:
            r_list.append([0, 0, 0])  # For prismatic joints
    return np.array(r_list)

def extract_omega_list(Slist):
    """
    Extracts the first three elements from each sublist in the given list and returns them as a numpy array.

    Parameters:
        Slist (list): A list of sublists.

    Returns:
        np.array: A numpy array containing the first three elements from each sublist.
    """
    
    return np.array(Slist)[:, :3]

def skew_symmetric(v):
    """
    Returns the skew symmetric matrix of a 3D vector.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def transform_from_twist(S, theta):
    """
    Computes the transformation matrix from a twist and a joint angle.
    """
    omega = S[:3]
    v = S[3:]
    if np.linalg.norm(omega) == 0:  # Prismatic joint
        return np.vstack((np.eye(3), v * theta)).T
    else:  # Revolute joint
        skew_omega = skew_symmetric(omega)
        R = np.eye(3) + np.sin(theta) * skew_omega + (1 - np.cos(theta)) * np.dot(skew_omega, skew_omega)
        p = np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * skew_omega + (theta - np.sin(theta)) * np.dot(skew_omega, skew_omega), v)
        return np.vstack((np.hstack((R, p.reshape(-1, 1))), [0, 0, 0, 1]))

def adjoint_transform(T):
    """
    Computes the adjoint transformation matrix for a given transformation matrix.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    skew_p = skew_symmetric(p)
    return np.vstack((np.hstack((R, np.zeros((3, 3)))), np.hstack((skew_p @ R, R))))

def logm(T):
    """
    Computes the logarithm of a transformation matrix.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    omega, theta = rotation_logm(R)
    if np.linalg.norm(omega) < 1e-6:
        v = p / theta
    else:
        G_inv = 1 / theta * np.eye(3) - 0.5 * skew_symmetric(omega) + (1 / theta - 0.5 / np.tan(theta / 2)) * np.dot(skew_symmetric(omega), skew_symmetric(omega))
        v = np.dot(G_inv, p)
    return np.hstack((omega * theta, v))

def rotation_logm(R):
    """
    Computes the logarithm of a rotation matrix.
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-6:
        return np.zeros(3), theta
    else:
        omega = 1 / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return omega, theta
def logm_to_twist(logm):
    """
    Convert the logarithm of a transformation matrix to a twist vector.

    Parameters:
        logm (numpy.darray): The logarithm of a transformation matrix.

    Returns:
        numpy.array: The corresponding twist vector.
    """
    if logm.shape != (4, 4):
        raise ValueError("logm must be a 4x4 matrix.")

    # Extract the skew-symmetric part for angular velocity
    omega_matrix = logm[0:3, 0:3]
    omega = skew_symmetric_to_vector(omega_matrix)

    # Extract the linear velocity part
    v = logm[0:3, 3]

    return np.hstack((omega, v))

def skew_symmetric_to_vector(skew_symmetric):
    """
    Convert a skew-symmetric matrix to a vector.
    """
    return np.array([skew_symmetric[2, 1], skew_symmetric[0, 2], skew_symmetric[1, 0]])
def se3ToVec(se3_matrix):
    """
    Convert an se(3) matrix to a twist vector.

    Parameters:
        se3_matrix (numpy.ndarray): A 4x4 matrix from the se(3) Lie algebra.

    Returns:
        numpy.ndarray: A 6-dimensional twist vector.
    """
    if se3_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")

    # Extract the angular velocity vector from the skew-symmetric part
    omega = np.array([se3_matrix[2, 1], se3_matrix[0, 2], se3_matrix[1, 0]])

    # Extract the linear velocity vector
    v = se3_matrix[0:3, 3]

    # Combine into a twist vector
    twist = np.hstack((omega, v))

    return twist

def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix and position vector."""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p

def TransInv(T):
    """Inverts a homogeneous transformation matrix."""
    R, p = TransToRp(T)
    Rt = R.T
    return np.vstack((np.hstack((Rt, -Rt @ p.reshape(-1, 1))), [0, 0, 0, 1]))

def MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix."""
    R, p = TransToRp(T)
    omega, theta = rotation_logm(R)
    if np.linalg.norm(omega) < 1e-6:
        return np.vstack((np.hstack((np.zeros((3, 3)), p.reshape(-1, 1))), [0, 0, 0, 0]))
    else:
        omega_mat = skew_symmetric(omega)
        G_inv = 1 / theta * np.eye(3) - 0.5 * omega_mat + (1 / theta - 0.5 / np.tan(theta / 2)) * omega_mat @ omega_mat
        v = G_inv @ p
        return np.vstack((np.hstack((omega_mat, v.reshape(-1, 1))), [0, 0, 0, 0]))

def rotation_logm(R):
    """Computes the logarithm of a rotation matrix."""
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-6:
        return np.zeros(3), 0
    else:
        omega = (1 / (2 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return omega, theta
