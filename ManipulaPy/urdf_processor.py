#!/usr/bin/env python3


from urchin.urdf import URDF 
import numpy as np
import pybullet as p
import pybullet_data
import time
from .kinematics import SerialManipulator
from .dynamics import ManipulatorDynamics
from . import utils


class URDFToSerialManipulator:
    """
    A class to convert URDF files to SerialManipulator objects and simulate them using PyBullet.
    """

    def __init__(self, urdf_name: str):
        """
        Initializes the object with the given urdf_name.

        Parameters:
            urdf_name (str): The name of the URDF file.

        Returns:
            None
        """
        self.urdf_name = urdf_name
        self.robot = URDF.load(urdf_name)
        self.robot_data = self.load_urdf(urdf_name)
        self.serial_manipulator = self.initialize_serial_manipulator()
        self.dynamics = self.initialize_manipulator_dynamics()

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
            Given a robot URDF and a link name, returns the joint associated with that link.

        Parameters:
            robot (URDF): The robot URDF object.
            link_name (str): The name of the link to find the joint for.

        Returns:
            Joint or None: The joint associated with the link, or None if no joint is found.
        """
        for link in robot.links:
            if link.name == link_name:
                return link
        return None

    @staticmethod
    def w_p_to_slist(w: np.ndarray, p: np.ndarray, robot_dof: int) -> np.ndarray:
        """
        Convert the input angular velocity and linear position vectors to the screw list representation.

        :param w: A numpy array representing the angular velocity vectors.
        :param p: A numpy array representing the linear position vectors.
        :param robot_dof: An integer representing the number of degrees of freedom of the robot.
        :return: A numpy array representing the screw list.
        """
        Slist = []
        for i in range(robot_dof):
            w_ = w[i]
            p_ = p[i]
            v_ = np.cross(-1*w_, p_)
            Slist.append([w_[0], w_[1], w_[2], v_[0], v_[1], v_[2]])
        return np.transpose(Slist)

    def load_urdf(self, urdf_name: str) -> dict:
        """
        Load the URDF file and extract the necessary information to create the robot model.

        Parameters:
            urdf_name (str): The name of the URDF file to load.

        Returns:
            dict: A dictionary containing the following:
                - "M" (np.ndarray): The home position matrix.
                - "Slist" (list): The screw axes of each joint.
                - "Blist" (list): The inertia matrices.
                - "Glist" (list): The mass matrices.
                - "actuated_joints_num" (int): The number of actuated joints in the robot.
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
        Tsb_inv = np.linalg.inv(M_list)
        Ad_Tsb_inv = utils.adjoint_transform(Tsb_inv)
        # Replace mr.Adjoint and mr.TransInv with custom functions
        Blist = np.dot(Ad_Tsb_inv, Slist)
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
        # Initialize the ManipulatorDynamics object
        self.manipulator_dynamics = ManipulatorDynamics(
            M_list=data["M"],
            omega_list=data["Slist"][:, :3],
            r_list=utils.extract_r_list(data["Slist"]),
            b_list=None,  # Assuming b_list is not provided in URDF data
            S_list=data["Slist"],
            B_list=data["Blist"],
            Glist=data["Glist"]
        )
    def initialize_manipulator_dynamics(self):
        """
        Initializes the ManipulatorDynamics object using the extracted URDF data.
        """
        data = self.robot_data
        # Initialize the ManipulatorDynamics object
        self.manipulator_dynamics = ManipulatorDynamics(
            M_list=data["M"],
            omega_list=data["Slist"][:, :3],
            r_list=utils.extract_r_list(data["Slist"]),
            b_list=None,  # Assuming b_list is not provided in URDF data
            S_list=data["Slist"],
            B_list=data["Blist"],
            Glist=data["Glist"]
        )
        return self.manipulator_dynamics
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

    def visualize_robot(self):
        """
        Visualizes the URDF model using matplotlib.
        """
        
        self.robot.show()
    

    def visualize_trajectory(self, cfg_trajectory=None, loop_time=3.0, use_collision=False):
        # Filter out fixed joints
        actuated_joints = [joint for joint in self.robot.joints if joint.joint_type != 'fixed']
        
        if cfg_trajectory is not None:
            if isinstance(cfg_trajectory, np.ndarray):
                expected_columns = len(actuated_joints)
                if cfg_trajectory.shape[1] != expected_columns:
                    raise ValueError(f"Expected cfg_trajectory to have {expected_columns} columns, got {cfg_trajectory.shape[1]}.")
                cfg_trajectory = {joint.name: cfg_trajectory[:, i] for i, joint in enumerate(actuated_joints) if i < cfg_trajectory.shape[1]}
            elif isinstance(cfg_trajectory, dict):
                if len(cfg_trajectory) != len(actuated_joints):
                    raise ValueError(f"Expected cfg_trajectory keys to match the number of robot joints ({len(actuated_joints)}), got {len(cfg_trajectory)}.")
            else:
                raise TypeError("cfg_trajectory must be either a numpy array or a dictionary mapping joint names to configurations.")
        else:
            cfg_trajectory = {joint.name: [0, np.pi / 2] for joint in actuated_joints}

        self.robot.animate(cfg_trajectory=cfg_trajectory, loop_time=loop_time, use_collision=use_collision)

    def print_joint_info(self):
        """
        Prints the number of joints and their names.
        """
        joint_names = [joint.name for joint in self.robot.joints]
        print(f"Number of joints: {len(joint_names)}")
        for i, joint_name in enumerate(joint_names):
            print(f"Joint {i}: {joint_name}")



