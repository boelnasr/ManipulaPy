import numpy as np

def extract_r_list(Slist):
    
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
    Extracts the angular velocity vectors (omega) for each joint from the screw axis list (Slist).
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
