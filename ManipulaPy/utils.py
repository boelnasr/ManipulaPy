import numpy as np
from scipy.linalg import expm
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
def NearZero(z):
    """
    Determines if a given number is near zero.

    Parameters:
        z (float): The number to check.

    Returns:
        bool: True if the number is near zero, False otherwise.
    """
    
    return abs(z) < 1e-6
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
def MatrixLog6(T):
    """
    Compute the matrix logarithm of a given transformation matrix T.
    
    Parameters:
        T (ndarray): The transformation matrix of shape (4, 4).
        
    Returns:
        ndarray: The matrix logarithm of T, with shape (4, 4).
    """
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                                                            [[0, 0, 0, 0]]]
    
def MatrixLog3(R):
    
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)
    

def VecToso3(omg):
   
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])
def VecTose3(V):
    
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]

def MatrixExp6(se3mat):
    """
    Computes the matrix exponential of a matrix in se(3).

    Parameters:
    se3mat (np.ndarray): A 4x4 matrix representing a twist in se(3).

    Returns:
    np.ndarray: The corresponding 4x4 transformation matrix in SE(3).
    """
    if se3mat.shape != (4, 4):
        raise ValueError("Input matrix must be of shape (4, 4)")

    # Extract the angular velocity vector (omega) and linear velocity vector (v) from the se3 matrix
    omega = np.array([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0]])
    v = np.array([se3mat[0, 3], se3mat[1, 3], se3mat[2, 3]])

    # Compute the magnitude of omega
    omega_magnitude = np.linalg.norm(omega)

    if omega_magnitude < 1e-6:
        # If omega is very small, use the first-order Taylor expansion of matrix exponential
        return np.eye(4) + se3mat

    # Compute the skew-symmetric matrix of omega
    omega_skew = skew_symmetric(omega)

    # Compute the matrix exponential of the skew-symmetric matrix of omega
    omega_exp = expm(omega_skew * omega_magnitude)

    # Compute the additional term for the linear velocity part
    omega_skew_squared = np.dot(omega_skew, omega_skew)
    v_term = (np.eye(3) * omega_magnitude + (1 - np.cos(omega_magnitude)) * omega_skew + (omega_magnitude - np.sin(omega_magnitude)) * omega_skew_squared) / omega_magnitude**2
    v_term = np.dot(v_term, v)

    # Construct the final transformation matrix
    T = np.eye(4)
    T[:3, :3] = omega_exp
    T[:3, 3] = v_term

    return T