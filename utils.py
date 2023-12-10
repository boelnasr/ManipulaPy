import numpy as np


def extract_r_list(Slist):
    r_list = []
    Slist_array = np.array(Slist)  # Convert Slist to a NumPy array
    for S in Slist_array.T:  # Transpose Slist to iterate over screw axes
        omega = S[:3]
        v = S[3:]
        if np.linalg.norm(omega) != 0:
            r = -np.cross(omega, v) / np.linalg.norm(omega)**2
            r_list.append(r)
        else:
            r_list.append([0, 0, 0])  # For prismatic joints, if any
    return np.array(r_list)

def extract_omega_list(Slist):
    omega_list = []
    Slist_array = np.array(Slist)  # Convert Slist to a NumPy array
    Slist_array = np.array(Slist)  # Ensure Slist is a NumPy array
    omega_list = Slist_array[:, :3]  # Extract the first 3 elements of each row
    return omega_list
