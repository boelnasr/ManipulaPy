from urdfpy import URDF
import numpy as np
import modern_robotics as mr
import pybullet as p
import pybullet_data
import time
from kinematics import SerialManipulator
from dynamics import ManipulatorDynamics
import utils
class URDFToSerialManipulator:
    """
    A class to convert URDF files to SerialManipulator objects and simulate them using PyBullet.
    """

    def __init__(self, urdf_name: str):
        """
        Initializes the URDFToSerialManipulator object.

        Args:
            urdf_name (str): Path to the URDF file.
        """
        self.urdf_name = urdf_name
        self.robot_data = self.load_urdf(urdf_name)
        self.serial_manipulator = self.initialize_serial_manipulator()

    @staticmethod
    def transform_to_xyz(T: np.ndarray) -> np.ndarray:
        """
        Extracts the XYZ position from a transformation matrix.

        Args:
            T (np.ndarray): A 4x4 transformation matrix.

        Returns:
            np.ndarray: A 3-element array representing XYZ position.
        """
        return np.array(T[0:3, 3])

    @staticmethod
    def get_joint(robot: URDF, link_name: str):
        """
        Retrieves a joint from the robot by link name.
        Args:
            robot (URDF): The robot object.
            link_name (str): The name of the link.
        Returns:
            The joint object if found, otherwise None.
        """
        for link in robot.links:
            if link.name == link_name:
                return link
        return None

    @staticmethod
    def w_p_to_slist(w: np.ndarray, p: np.ndarray, robot_dof: int) -> np.ndarray:
        """
        Converts angular and linear positions to a screw axis list.

        Args:
            w (np.ndarray): Angular positions.
            p (np.ndarray): Linear positions.
            robot_dof (int): Degree of freedom of the robot.

        Returns:
            np.ndarray: Screw axis list.
        """
        Slist = []
        for i in range(robot_dof):
            w_ = w[i]
            p_ = p[i]
            v_ = np.cross(w_, -p_)
            Slist.append([w_[0], w_[1], w_[2], v_[0], v_[1], v_[2]])
        return np.transpose(Slist)

    def load_urdf(self, urdf_name: str) -> dict:
        """
        Loads a URDF file and extracts necessary data.

        Args:
            urdf_name (str): Path to the URDF file.

        Returns:
            dict: A dictionary containing robot data.
        """
        robot = URDF.load(urdf_name)
        joint_num = len(robot.actuated_joints)

        p_ = []  # Positions of each joint
        w_ = []  # Rotation axes of each joint
        M_list = np.eye(4)  # Home position matrix
        Glist = []  # Inertia matrices
        for joint in robot.actuated_joints:
            child_link = self.get_joint(robot, joint.child)
            p_.append(self.transform_to_xyz(robot.link_fk()[child_link]))
            G = np.eye(6)
            if child_link.inertial:
                G[0:3, 0:3] = child_link.inertial.inertia
                G[3:6, 3:6] = child_link.inertial.mass * np.eye(3)
            Glist.append(G)
            child_M = robot.link_fk()[child_link]
            child_w = np.array(child_M[0:3, 0:3] @ np.array(joint.axis).T)
            w_.append(child_w)
            if child_link.inertial and child_link.inertial.origin is not None:
                child_M = child_M @ child_link.inertial.origin
            M_list = np.dot(M_list, child_M)
        Slist = self.w_p_to_slist(w_, p_, joint_num)
        Blist = mr.Adjoint(mr.TransInv(M_list)) @ Slist
        return {"M": M_list, "Slist": Slist, "Blist": Blist, "Glist": Glist, "actuated_joints_num": joint_num}

    def initialize_serial_manipulator(self) -> SerialManipulator:
        """
        Initializes a SerialManipulator object using the extracted URDF data.
        Returns:
            SerialManipulator: The initialized serial manipulator object.
        """
        data = self.robot_data
        return SerialManipulator(
            M_list=data["M"],  # Use 'M' instead of 'Mlist'
            omega_list=data["Slist"][:, :3],
            B_list=data["Blist"],
            S_list=data["Slist"],
            r_list=utils.extract_r_list(data["Slist"]),
            G_list=data["Glist"])
    def initialize_manipulator_dynamics(self):
        """
        Initializes the ManipulatorDynamics object using the extracted URDF data.
        """
        data = self.robot_data
        Glist = self.extract_inertia_matrices()

        # Initialize the ManipulatorDynamics object
        self.manipulator_dynamics = ManipulatorDynamics(
            M_list=data["M"],
            omega_list=data["Slist"][:, :3],
            r_list=data["Slist"][:, 3:],
            b_list=data["Blist"],
            S_list=data["Slist"],
            B_list=data["Blist"],
            Glist=Glist
        )

    def extract_inertia_matrices(self):
        """
        Extracts the spatial inertia matrices from the URDF data.
        """
        Glist = []
        for link in self.robot.links:
            if link.inertial:
                inertia = link.inertial.inertia
                mass = link.inertial.mass
                G = np.zeros((6, 6))
                G[:3, :3] = inertia
                G[3:, 3:] = mass * np.eye(3)
                Glist.append(G)
            else:
                Glist.append(np.zeros((6, 6)))  # Add zero matrix for links without inertia
        return Glist

    def simulate_robot(self):
        """
        Simulates the robot using PyBullet.
        """
        M = self.robot_data["M"]
        Slist = self.robot_data["Slist"]
        Blist = self.robot_data["Blist"]
        Mlist = self.robot_data["M"]
        Glist = self.robot_data["Glist"]
        actuated_joints_num = self.robot_data["actuated_joints_num"]
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robotID = p.loadURDF(self.urdf_name, [0, 0, 0], [0, 0, 0, 1], useFixedBase=1)
        numJoints = p.getNumJoints(robotID)
        p.resetBasePositionAndOrientation(robotID, [0, 0, 0], [0, 0, 0, 1])
        for i in range(numJoints):
            p.setJointMotorControl2(robotID, i, p.POSITION_CONTROL, targetVelocity=0, force=0)
        for i in range(numJoints):
            p.resetJointState(robotID, i, np.pi/3.0)
        # Simulation loop
        timeStep = 1/240.0
        p.setTimeStep(timeStep)
        while p.isConnected():
            # Perform simulation steps here...
            # You can use the code from your script to perform the necessary calculations
            # and control the robot in the PyBullet simulation environment.
            # Step simulation
            p.stepSimulation()
            time.sleep(timeStep)
        # Disconnect PyBullet
        p.disconnect()

    def simulate_robot_with_desired_angles(self, desired_angles):
        """
        Simulates the robot using PyBullet with desired joint angles.

        Args:
            desired_angles (np.ndarray): Desired joint angles.
        """
        # Connect to PyBullet and set up the environment
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the robot from the URDF file
        robotID = p.loadURDF(self.urdf_name, [0, 0, 0], [0, 0, 0, 1], useFixedBase=1)

        # Set the desired joint angles using POSITION_CONTROL
        numJoints = p.getNumJoints(robotID)
        
        for i in range(numJoints):
            if i < len(desired_angles):
                # Apply position control to each joint
                p.setJointMotorControl2(robotID, i, p.POSITION_CONTROL, targetPosition=desired_angles[i], force=1000)
            else:
                # If desired_angles list is shorter than numJoints, set remaining joints to a default position
                p.setJointMotorControl2(robotID, i, p.POSITION_CONTROL, targetPosition=0, force=1000)
        # Simulation loop
        timeStep = 1/100.0
        p.setTimeStep(timeStep)
        while p.isConnected():
            p.stepSimulation()
            time.sleep(timeStep)  # Simulation time step
        time.sleep(10*np.exp(100))
        # Disconnect from PyBullet
        p.disconnect()