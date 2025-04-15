# ManipulaPy


[![PyPI](https://img.shields.io/pypi/v/ManipulaPy)](https://pypi.org/project/ManipulaPy/)
![Tested](https://img.shields.io/badge/tested-yes-brightgreen.svg)
![CI](https://github.com/boelnasr/ManipulaPy/actions/workflows/test.yml/badge.svg?branch=main)

ManipulaPy is a modular, GPU-accelerated Python package for robotic manipulator simulation, planning, control, and perception. It supports kinematics, dynamics, path planning, control strategies, PyBullet simulation, and even 3D vision with stereo processing and YOLO-based obstacle detection.

---

## 🚀 Features

- **Kinematic Analysis**: Forward and inverse kinematics for serial robots
- **Dynamic Modeling**: Mass matrix, Coriolis/centrifugal, gravity force computation
- **Path Planning**: Joint and Cartesian trajectory generation with time scaling
- **Singularity Analysis**: Detect singularities, plot manipulability ellipsoids, and estimate workspace
- **URDF Processing**: Convert URDF files into manipulatable Python models
- **Control Strategies**: PD, PID, computed torque, robust, adaptive, Kalman filter, and feedforward control
- **Simulation**: Real-time PyBullet simulation of joint-space trajectories
- **Vision & Perception**: Stereo camera modeling, depth/disparity map computation, DBSCAN clustering, YOLO-based object detection
- **Visualization Tools**: Trajectory plotting, steady-state response analysis, 3D cluster rendering

---

## 📦 Installation

```bash
pip install ManipulaPy
```

Or from source:
```bash
git clone https://github.com/boelnasr/ManipulaPy
cd ManipulaPy
pip install .
```

---

## 🛠️ Getting Started

```python
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.kinematics import SerialManipulator
from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.control import ManipulatorController
import numpy as np
from math import pi

urdf_file_path = "path_to_urdf/robot.urdf"
urdf_processor = URDFToSerialManipulator(urdf_file_path)
robot = urdf_processor.serial_manipulator
dynamics = ManipulatorDynamics(
    urdf_processor.M_list, urdf_processor.omega_list,
    urdf_processor.r_list, urdf_processor.b_list,
    urdf_processor.S_list, urdf_processor.B_list,
    urdf_processor.Glist
)
controller = ManipulatorController(dynamics)

# Forward Kinematics
thetalist = np.array([pi, pi/6, pi/4, -pi/3, -pi/2, -2*pi/3])
T = robot.forward_kinematics(thetalist)
print("FK Result:", T)
```

---

## 🤖 Core Functionalities

### Kinematics
```python
# Inverse Kinematics
solution, success, _ = robot.iterative_inverse_kinematics(T, thetalist)
```

### Dynamics
```python
M = dynamics.mass_matrix(thetalist)
c = dynamics.velocity_quadratic_forces(thetalist, np.zeros_like(thetalist))
g = dynamics.gravity_forces(thetalist)
```

### Trajectory Planning
```python
from ManipulaPy.path_planning import TrajectoryPlanning as tp
traj = tp.JointTrajectory([0]*6, thetalist, Tf=5, N=100, method=5)
```

### Control
```python
Kp = np.eye(6)
Kd = np.eye(6)
tau_pd = controller.pd_control(thetalist, np.zeros(6), thetalist, np.zeros(6), Kp, Kd)
```

### Singularity Analysis
```python
from ManipulaPy.singularity import Singularity
s = Singularity(robot)
print("Singular?", s.singularity_analysis(thetalist))
s.manipulability_ellipsoid(thetalist)
s.plot_workspace_monte_carlo([(-pi, pi)] * 6)
```

### Simulation
```python
from ManipulaPy.sim import Simulation
sim = Simulation(urdf_file_path, joint_limits=[(-pi, pi)]*6)
trajectory = np.linspace([0]*6, [pi/2, pi/4, pi/6, -pi/3, -pi/2, -pi/3], 100)
sim.run(trajectory)
```

### Perception
```python
from ManipulaPy.vision import Vision
from ManipulaPy.perception import Perception
vision = Vision([...])
perception = Perception(vision)
points, labels = perception.detect_and_cluster_obstacles()
```

### Visualization
```python
from ManipulaPy.control import ManipulatorController
time = np.linspace(0, 5, 100)
response = np.exp(-time) * np.sin(5 * time) + 1
controller.plot_steady_state_response(time, response, set_point=1)
```

---

## 📁 Examples
Browse the `examples/` folder for full scripts demonstrating:
- Inverse kinematics
- Stereo perception & 3D clustering
- Simulation-based validation

---

## 🤝 Contributing
We welcome contributions! Please fork the repo and submit a pull request. All contributions should include tests and follow PEP8 style.

---

## 📄 License
MIT License — see `LICENSE.md` for details.

---

## 📬 Contact
Created and maintained by **Mohamed Aboelnasr**  
📧 [aboelnasr1997@gmail.com](mailto:aboelnasr1997@gmail.com)

Feel free to reach out with questions or ideas!

---

## ✅ Tested
ManipulaPy includes a suite of unit tests covering kinematics, dynamics, control, and perception modules.
We run them with Python’s built-in `unittest` or `pytest`. See the [`tests/`](./tests) folder for details.

---

> 📌 Latest Version: `v1.1.0`
>
> Now includes full stereo vision, YOLO-based perception, and PyBullet simulation support.
