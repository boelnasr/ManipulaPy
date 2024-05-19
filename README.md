# ManipulaPy

ManipulaPy is a comprehensive Python package for robotic manipulator analysis and simulation. It offers a range of functionalities, from kinematic calculations to dynamic analysis and path planning, making it a versatile tool for both educational and research purposes in the field of robotics.

## Features

- **Kinematic Analysis**: Compute forward and inverse kinematics for serial manipulators.
- **Dynamic Analysis**: Perform calculations related to the dynamics of manipulators, including mass matrix computation, gravity forces, and velocity quadratic forces.
- **Path Planning**: Implement various path planning algorithms for robotic manipulators.
- **Singularity Analysis**: Analyze and identify singular configurations of robotic manipulators.
- **URDF Processing**: Parse and process URDF (Unified Robot Description Format) files for simulation and analysis.
- **Controllers**: Implement various control strategies such as PD, PID, robust, adaptive, and feedforward controllers, along with Kalman filter-based control.
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
dynamics = ManipulatorDynamics(urdf_processor.M_list, urdf_processor.omega_list, urdf_processor.r_list, urdf_processor.b_list, urdf_processor.S_list, urdf_processor.B_list, urdf_processor.Glist)
controller = ManipulatorController(dynamics)

# Example joint angles
thetalist = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, -2*pi/3])
T = robot.forward_kinematics(thetalist)
print("Forward Kinematics:", T)
```

## Usage

### Kinematics

Perform forward and inverse kinematics for your robot.

```python
# Forward Kinematics
T = robot.forward_kinematics(thetalist)
print("Forward Kinematics:", T)

# Inverse Kinematics
thetalist_sol = robot.inverse_kinematics(T)
print("Inverse Kinematics:", thetalist_sol)
```

### Dynamics

Calculate mass matrices, velocity quadratic forces, and gravity forces.

```python
# Mass Matrix
M = dynamics.mass_matrix(thetalist)
print("Mass Matrix:", M)

# Velocity Quadratic Forces
c = dynamics.velocity_quadratic_forces(thetalist, dthetalist)
print("Velocity Quadratic Forces:", c)

# Gravity Forces
g_forces = dynamics.gravity_forces(thetalist)
print("Gravity Forces:", g_forces)
```

### Trajectory Planning

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

### Controllers

Implement various control strategies for your robot.

```python
# PD Control
tau = controller.pd_control(thetalistd, dthetalistd, thetalist, dthetalist, Kp, Kd)
print("PD Control Torques:", tau)

# PID Control
tau = controller.pid_control(thetalistd, dthetalistd, thetalist, dthetalist, dt, Kp, Ki, Kd)
print("PID Control Torques:", tau)

# Robust Control
tau = controller.robust_control(thetalist, dthetalist, ddthetalist, g, Ftip, disturbance_estimate, adaptation_gain)
print("Robust Control Torques:", tau)

# Adaptive Control
tau = controller.adaptive_control(thetalist, dthetalist, ddthetalist, g, Ftip, measurement_error, adaptation_gain)
print("Adaptive Control Torques:", tau)
```

### Visualization

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

Check out the `examples` directory for comprehensive examples demonstrating how to use ManipulaPy for various tasks, including kinematics, dynamics, trajectory planning, and control.

## Contributing

We welcome contributions to ManipulaPy! If you'd like to contribute, please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the existing style and includes tests for new functionality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to reach out if you have any questions or need further assistance!

