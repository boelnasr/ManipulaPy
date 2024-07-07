# ManipulaPy

ManipulaPy is a comprehensive Python package for robotic manipulator analysis and simulation. It offers a range of functionalities, from kinematic calculations to dynamic analysis and path planning, making it a versatile tool for both educational and research purposes in the field of robotics.

## Features

- **Kinematic Analysis**: Compute forward and inverse kinematics for serial manipulators.
- **Dynamic Analysis**: Perform calculations related to the dynamics of manipulators, including mass matrix computation, gravity forces, and velocity quadratic forces.
- **Path Planning**: Implement various path planning algorithms for robotic manipulators.
- **Singularity Analysis**: Analyze and identify singular configurations of robotic manipulators.
- **URDF Processing**: Parse and process URDF (Unified Robot Description Format) files for simulation and analysis.
- **Controllers**: Implement various control strategies such as PD, PID, robust, adaptive, and feedforward controllers, along with Kalman filter-based control.
- **Simulation**: Simulate robotic manipulator motion using PyBullet.
- **Visualization**: Tools for visualizing joint and end-effector trajectories, and analyzing steady-state response.

## Installation

To install ManipulaPy, run the following command:

```bash
pip install ManipulaPy
```
## Getting Started
To get started with ManipulaPy, you'll need to have a URDF file for your robotic manipulator. The following example shows how to initialize the library with a URDF file and perform basic kinematic and dynamic calculations.
```python
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.path_planning import TrajectoryPlanning as tp
from ManipulaPy.control import ManipulatorController
import numpy as np
from math import pi

# Path to your URDF file
urdf_file_path = "path_to_urdf/robot.urdf"

# Initialize the URDF processor and extract the serial manipulator
urdf_processor = URDFToSerialManipulator(urdf_file_path)
robot = urdf_processor.serial_manipulator
dynamics = ManipulatorDynamics(
    urdf_processor.M_list, urdf_processor.omega_list, urdf_processor.r_list,
    urdf_processor.b_list, urdf_processor.S_list, urdf_processor.B_list, urdf_processor.Glist
)
controller = ManipulatorController(dynamics)

# Example joint angles
thetalist = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, -2*pi/3])
T = robot.forward_kinematics(thetalist)
print("Forward Kinematics:", T)

```
Usage
Kinematics
Perform forward and inverse kinematics for your robot.

```python

# Forward Kinematics
T = robot.forward_kinematics(thetalist)
print("Forward Kinematics:", T)

# Inverse Kinematics
thetalist_sol, success, iterations = robot.iterative_inverse_kinematics(T, thetalist)
print("Inverse Kinematics Solution:", thetalist_sol)
print("Success:", success)
print("Iterations:", iterations)
Dynamics
Calculate mass matrices, velocity quadratic forces, and gravity forces.

```python
Copy code
# Mass Matrix
M = dynamics.mass_matrix(thetalist)
print("Mass Matrix:", M)

# Velocity Quadratic Forces
c = dynamics.velocity_quadratic_forces(thetalist, np.zeros(len(thetalist)))
print("Velocity Quadratic Forces:", c)

# Gravity Forces
g_forces = dynamics.gravity_forces(thetalist)
print("Gravity Forces:", g_forces)
```
## Trajectory Planning
Plan joint space and Cartesian trajectories.
```python

# Joint Space Trajectory
traj = tp.JointTrajectory([0]*6, thetalist, Tf=5, N=100, method=5)
print("Joint Space Trajectory:", traj)

# Cartesian Trajectory
Xstart = np.eye(4)
Xend = np.array([[0, -1, 0, 1.0], [1, 0, 0, 0.0], [0, 0, 1, 0.5], [0, 0, 0, 1]])
cartesian_traj = tp.CartesianTrajectory(Xstart, Xend, Tf=5, N=100, method=5)
print("Cartesian Trajectory:", cartesian_traj)
```
## Controllers
Implement various control strategies for your robot.

```python

# PD Control
Kp = np.eye(len(thetalist))
Kd = np.eye(len(thetalist))
tau = controller.pd_control(thetalist, np.zeros(len(thetalist)), thetalist, np.zeros(len(thetalist)), Kp, Kd)
print("PD Control Torques:", tau)

# PID Control
Ki = np.eye(len(thetalist))
tau = controller.pid_control(thetalist, np.zeros(len(thetalist)), thetalist, np.zeros(len(thetalist)), 0.01, Kp, Ki, Kd)
print("PID Control Torques:", tau)
Singularity Analysis
Analyze singularities, plot manipulability ellipsoids, and estimate workspace.
```
```python

from ManipulaPy.singularity import Singularity

# Initialize the Singularity class with the serial manipulator
singularity_analysis = Singularity(robot)

# Perform singularity analysis
is_singular = singularity_analysis.singularity_analysis(thetalist)
print(f"Is the manipulator at a singularity? {'Yes' if is_singular else 'No'}")

# Plot the manipulability ellipsoid
singularity_analysis.manipulability_ellipsoid(thetalist)

# Define joint limits for the manipulator (example limits)
joint_limits = [(-pi, pi) for _ in range(len(thetalist))]

# Estimate the workspace using Monte Carlo sampling
singularity_analysis.plot_workspace_monte_carlo(joint_limits)

# Calculate the condition number of the Jacobian
cond_number = singularity_analysis.condition_number(thetalist)
print(f"Condition number of the Jacobian: {cond_number}")

# Detect if the manipulator is near a singularity
near_singular = singularity_analysis.near_singularity_detection(thetalist)
print(f"Is the manipulator near a singularity? {'Yes' if near_singular else 'No'}")
```
## Simulation
Simulate robotic manipulator motion using PyBullet.

```python

from ManipulaPy.sim import Simulation

# Define joint limits (example limits)
joint_limits = [(-pi, pi) for _ in range(len(thetalist))]

# Initialize the simulation with the URDF file and joint limits
simulation = Simulation(urdf_file_path, joint_limits)

# Define a simple joint trajectory for the simulation
joint_trajectory = np.linspace([0, 0, 0, 0, 0, 0], [pi/2, pi/4, pi/6, -pi/3, -pi/2, -pi/3], 100)

# Run the simulation
simulation.run(joint_trajectory)
```
## Visualization
Visualize joint and end-effector trajectories and analyze the steady-state response.

```python

# Plot Joint Trajectory
tp.plot_trajectory(traj, Tf=5)

# Plot Cartesian Trajectory
tp.plot_cartesian_trajectory(cartesian_traj, Tf=5)

# Plot Steady-State Response
time = np.linspace(0, 5, 100)
response = np.exp(-time) * np.sin(5 * time) + 1  # Example response
controller.plot_steady_state_response(time, response, set_point=1)
```
## Examples
Check out the examples directory for comprehensive examples demonstrating how to use ManipulaPy for various tasks, including kinematics, dynamics, trajectory planning, control, and simulation.

## Contributing
We welcome contributions to ManipulaPy! If you'd like to contribute, please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and includes tests for new functionality.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

Feel free to reach out if you have any questions or need further assistance!

