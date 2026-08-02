.. _doc-index:

ManipulaPy Documentation
========================

.. raw:: html

   <p class="mp-lede">A GPU-accelerated Python toolbox for robot kinematics, dynamics,
   trajectory planning, perception and control — with a unified compute backend that runs
   the same math on NumPy, CuPy, PyTorch or JAX, and returns real gradients through it.</p>

   <div class="mp-badges">
      <a href="https://pypi.org/project/manipulapy/">
         <img src="https://img.shields.io/pypi/v/manipulapy?style=flat-square&logo=pypi&logoColor=white&label=PyPI" alt="PyPI version">
      </a>
      <a href="https://www.python.org/downloads/">
         <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Supported Python versions">
      </a>
      <a href="https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa">
         <img src="https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa/status.svg" alt="JOSS paper status">
      </a>
      <a href="https://github.com/boelnasr/ManipulaPy/blob/main/LICENSE">
         <img src="https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square" alt="AGPL-3.0 license">
      </a>
      <a href="https://github.com/boelnasr/ManipulaPy/actions">
         <img src="https://img.shields.io/github/actions/workflow/status/boelnasr/ManipulaPy/test.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI status">
      </a>
      <a href="https://pypi.org/project/manipulapy/">
         <img src="https://img.shields.io/pypi/dm/manipulapy?style=flat-square&label=downloads" alt="Monthly downloads">
      </a>
   </div>

.. contents:: On this page
   :local:
   :depth: 1
   :backlinks: none

Quick start
-----------

Install the package into a fresh virtual environment:

.. code-block:: bash

   python -m pip install manipulapy          # core: kinematics, dynamics, control
   python -m pip install "manipulapy[cuda]"  # add CUDA-accelerated planning

Load a robot, solve forward kinematics, then invert it — the whole loop in one file:

.. code-block:: python

   import numpy as np
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

   # Built-in 6-DOF xArm model — no external URDF needed
   urdf_processor = URDFToSerialManipulator(xarm_urdf_file)
   robot = urdf_processor.serial_manipulator
   dynamics = urdf_processor.dynamics

   # Forward kinematics at an arbitrary configuration
   theta = np.array([0.5, -0.3, 0.8, 0.0, 0.5, 0.0])
   T_target = robot.forward_kinematics(theta, frame="space")

   # ...and back again
   solution, success, iterations = robot.iterative_inverse_kinematics(
       T_desired=T_target,
       thetalist0=np.zeros(6),
       max_iterations=1000,
   )

   print(f"converged={success} in {iterations} iterations")
   print(f"residual: {np.abs(solution - theta).max():.2e} rad")

The same call runs on a different array library, and differentiates:

.. code-block:: python

   import jax
   from ManipulaPy.backend import use_backend

   # Same robot, same method — JAX arrays instead of NumPy, restored on exit
   with use_backend("jax"):
       T = robot.forward_kinematics(theta)

       # Gradients come from the framework, not finite differences
       dT = jax.jacrev(robot.forward_kinematics)(theta)

   print(f"d(pose)/d(theta) shape: {dT.shape}")

.. note::
   Backends are opt-in (``pip install "manipulapy[jax-cpu]"`` or
   ``[pytorch]``); NumPy remains the default. Gradients are guaranteed for the
   **core math only** — ``utils``, ``kinematics``, ``dynamics``, and
   ``singularity``. See the :doc:`Compute Backends guide <user_guide/Backends>`
   for the full contract.

Capabilities
------------

.. raw:: html

   <div class="mp-grid mp-grid--wide">
      <div class="mp-card">
         <span class="mp-kicker">Core robotics</span>
         <h3 class="mp-card__title"><a href="user_guide/Kinematics.html">Kinematics &amp; dynamics</a></h3>
         <p class="mp-card__body">Forward and inverse kinematics, space and body Jacobians,
            mass matrices, Coriolis and gravity terms, forward and inverse dynamics.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">screw theory</span>
            <span class="mp-tag">cached mass matrix</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Motion</span>
         <h3 class="mp-card__title"><a href="user_guide/Trajectory_Planning.html">Planning &amp; control</a></h3>
         <p class="mp-card__body">Joint and Cartesian trajectory generation, potential-field
            obstacle avoidance, collision checking, PID, computed-torque and adaptive control.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">CUDA kernels</span>
            <span class="mp-tag">quintic scaling</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">New in 1.4</span>
         <h3 class="mp-card__title"><a href="user_guide/Backends.html">Compute backends</a></h3>
         <p class="mp-card__body">One dispatch layer over NumPy, CuPy, PyTorch and JAX.
            Autodiff through kinematics, dynamics, singularity and utils; NumPy stays the
            default and nothing is imported until a backend is requested.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">jax.grad</span>
            <span class="mp-tag">torch.autograd</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Sensing</span>
         <h3 class="mp-card__title"><a href="user_guide/Perception.html">Perception &amp; simulation</a></h3>
         <p class="mp-card__body">Monocular and stereo cameras, YOLO object detection,
            DBSCAN point-cloud clustering, and PyBullet physics with trajectory playback.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">stereo depth</span>
            <span class="mp-tag">PyBullet</span>
         </div>
      </div>
   </div>

Where to start
--------------

.. raw:: html

   <table class="mp-matrix">
      <thead>
         <tr><th>If you want to</th><th>Start here</th><th>Then read</th></tr>
      </thead>
      <tbody>
         <tr>
            <td>Install and run something</td>
            <td><a href="getting_started/index.html">Getting Started</a></td>
            <td><a href="user_guide/Kinematics.html">Kinematics</a></td>
         </tr>
         <tr>
            <td>Load your own robot</td>
            <td><a href="user_guide/URDF_Processor.html">URDF Processor</a></td>
            <td><a href="user_guide/Singularity_Analysis.html">Singularity Analysis</a></td>
         </tr>
         <tr>
            <td>Plan and execute motion</td>
            <td><a href="user_guide/Trajectory_Planning.html">Trajectory Planning</a></td>
            <td><a href="user_guide/Simulation.html">Simulation</a></td>
         </tr>
         <tr>
            <td>Design a controller</td>
            <td><a href="user_guide/Dynamics.html">Dynamics</a></td>
            <td><a href="user_guide/Control.html">Control</a></td>
         </tr>
         <tr>
            <td>Differentiate the math</td>
            <td><a href="user_guide/Backends.html">Compute Backends</a></td>
            <td><a href="api/backend.html">API: backend</a></td>
         </tr>
         <tr>
            <td>Run on the GPU</td>
            <td><a href="user_guide/CUDA_Kernels.html">CUDA Kernels</a></td>
            <td><a href="user_guide/Path_Planning.html">Path Planning</a></td>
         </tr>
         <tr>
            <td>Perceive the workspace</td>
            <td><a href="user_guide/vision.html">Vision</a></td>
            <td><a href="user_guide/Perception.html">Perception</a></td>
         </tr>
      </tbody>
   </table>

Documentation map
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Installation Guide
   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   user_guide/index

Worked examples
---------------

Trajectory planning
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning

   # Setup trajectory planner with GPU acceleration
   joint_limits = np.array([[-np.pi, np.pi]] * 6)
   planner = TrajectoryPlanning(robot, xarm_urdf_file, dynamics, joint_limits)

   # Plan smooth trajectory from start to end
   start_angles = np.zeros(6)
   end_angles = np.array([0.5, -0.3, 0.8, 0.0, 0.5, 0.0])

   trajectory = planner.joint_trajectory(
       thetastart=start_angles,
       thetaend=end_angles,
       Tf=5.0,          # 5 second duration
       N=100,           # 100 waypoints
       method=5         # Quintic time scaling for smoothness
   )

   print(f"Generated trajectory with {trajectory['positions'].shape[0]} points")
   print(f"Start velocity: {trajectory['velocities'][0]}")
   print(f"End velocity: {trajectory['velocities'][-1]}")

   # Visualize the trajectory
   planner.plot_trajectory(trajectory, 5.0, title="Smooth Joint Trajectory")

Computed-torque control
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.control import ManipulatorController

   controller = ManipulatorController(dynamics)

   # Current robot state
   current_pos = np.zeros(6)
   current_vel = np.zeros(6)

   # Desired target state
   desired_pos = np.array([0.2, -0.1, 0.3, 0.0, 0.2, 0.0])
   desired_vel = np.zeros(6)

   # Auto-tune PID gains using Ziegler-Nichols
   ultimate_gain = 50.0  # Found experimentally
   ultimate_period = 0.5
   Kp, Ki, Kd = controller.tune_controller(ultimate_gain, ultimate_period, kind="PID")

   print(f"Auto-tuned gains - Kp: {Kp[0]:.2f}, Ki: {Ki[0]:.2f}, Kd: {Kd[0]:.2f}")

   # Compute optimal control torques
   control_torques = controller.computed_torque_control(
       thetalistd=desired_pos,
       dthetalistd=desired_vel,
       ddthetalistd=np.zeros(6),
       thetalist=current_pos,
       dthetalist=current_vel,
       g=np.array([0, 0, -9.81]),
       dt=0.01,
       Kp=Kp, Ki=Ki, Kd=Kd
   )

   print(f"Control torques: {control_torques}")

PyBullet simulation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.sim import Simulation

   # Create realistic physics simulation
   sim = Simulation(
       urdf_file_path=xarm_urdf_file,
       joint_limits=joint_limits,
       torque_limits=np.array([[-50, 50]] * 6),
       time_step=0.01,
       real_time_factor=1.0
   )

   # Initialize robot and planning systems
   sim.initialize_robot()
   sim.initialize_planner_and_controller()

   # Execute the planned trajectory in simulation
   waypoints = trajectory["positions"][::10]  # Subsample for demonstration

   final_position = sim.run_trajectory(waypoints)
   print(f"Final end-effector position: {final_position}")

Vision and perception
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ManipulaPy.vision import Vision
   from ManipulaPy.perception import Perception

   # Setup camera system
   camera_config = {
       "name": "workspace_camera",
       "translation": [0.0, 0.0, 1.0],  # 1m above workspace
       "rotation": [0, 45, 0],           # Look down at 45°
       "fov": 60,
       "intrinsic_matrix": np.array([
           [500, 0, 320],
           [0, 500, 240],
           [0, 0, 1]
       ], dtype=np.float32),
       "distortion_coeffs": np.zeros(5, dtype=np.float32)
   }

   # Create integrated vision system
   vision = Vision(camera_configs=[camera_config])
   perception = Perception(vision_instance=vision)

   # Detect and analyze obstacles
   obstacle_points, cluster_labels = perception.detect_and_cluster_obstacles(
       camera_index=0,
       depth_threshold=3.0,  # Objects within 3m
       eps=0.1,              # DBSCAN clustering parameter
       min_samples=3         # Minimum points per cluster
   )

   num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
   print(f"Detected {len(obstacle_points)} obstacle points")
   print(f"Found {num_clusters} distinct object clusters")

Installation options
--------------------

.. code-block:: bash

   # Lightweight default install — kinematics, dynamics, control, native URDF parser
   pip install manipulapy

   # Add PyBullet-based simulation
   pip install "manipulapy[simulation]"

   # Add trimesh-based mesh loading
   pip install "manipulapy[urdf]"

   # Add OpenCV / ultralytics / torch for vision and perception
   pip install "manipulapy[vision]"

   # Add scikit-learn for ML clustering
   pip install "manipulapy[ml]"

   # Add CUDA acceleration (CUDA 12.x)
   pip install "manipulapy[cuda]"

   # Add a differentiable compute backend (new in 1.4)
   pip install "manipulapy[pytorch]"    # PyTorch
   pip install "manipulapy[jax-cpu]"    # JAX, CPU
   pip install "manipulapy[jax-cuda]"   # JAX, CUDA 12 (Linux only)

   # Everything (sim + urdf + vision + ml + cuda + pytorch + jax-cpu)
   pip install "manipulapy[all]"

   # Development installation
   git clone https://github.com/boelnasr/ManipulaPy.git
   cd ManipulaPy
   pip install -e ".[dev]"

Measured performance
--------------------

.. raw:: html

   <table class="mp-matrix">
      <thead>
         <tr><th>Workload</th><th>Speedup</th><th>Compared against</th></tr>
      </thead>
      <tbody>
         <tr>
            <td>Trajectory planning</td>
            <td class="mp-num">7&times;</td>
            <td>CUDA vs. CPU, 1000-point trajectories</td>
         </tr>
         <tr>
            <td>Inverse kinematics</td>
            <td class="mp-num">10&times;</td>
            <td>Hybrid neural + iterative approach vs. traditional methods</td>
         </tr>
         <tr>
            <td>Repeated dynamics</td>
            <td class="mp-num">3&times;</td>
            <td>Cached vs. recomputed mass matrices</td>
         </tr>
         <tr>
            <td>Object detection</td>
            <td class="mp-num">30 FPS</td>
            <td>YOLO detection with 3D localization</td>
         </tr>
      </tbody>
   </table>

Release notes
-------------

.. raw:: html

   <div class="mp-rail mp-rail--current">
      <p class="mp-rail__title">v1.4.0 — current</p>
      <ul>
         <li><strong>New:</strong> Unified compute backend system — the same kinematics, dynamics, planning and control code runs on <strong>NumPy, CuPy, PyTorch or JAX</strong> behind one dispatch API (<code>ManipulaPy.backend</code>: <code>set_backend</code>, <code>use_backend</code>, <code>get_backend</code>)</li>
         <li><strong>New:</strong> Differentiable contract for <code>utils</code>, <code>kinematics</code>, <code>dynamics</code> and <code>singularity</code> — <code>jax.grad</code>/<code>jacrev</code>, <code>jit</code>, and <code>torch.autograd</code> are safe on the core math. Every other module runs on all four backends through host-boundary conversion, with no gradient guarantee</li>
         <li><strong>New:</strong> Optional extras <code>[pytorch]</code>, <code>[jax-cpu]</code>, <code>[jax-cuda]</code> — NumPy remains the default and nothing extra is imported until a backend is requested</li>
         <li><strong>Fixed:</strong> SE(3)/SO(3) logarithm conditioning — <code>MatrixLog6</code> no longer discards small rotations and the translation term is derived from <code>MatrixLog3</code>, so values and gradients stay correct at θ ≈ 0 and θ ≈ π (this also corrects NumPy results)</li>
         <li><strong>Note:</strong> JAX eager dispatch is roughly 40× slower than NumPy on single small calls, so time-budgeted solvers such as <code>TracIKSolver</code> need a wider <code>timeout</code> — see the <a href="user_guide/Backends.html">Compute Backends guide</a></li>
      </ul>
   </div>

   <div class="mp-rail">
      <p class="mp-rail__title">v1.3.2</p>
      <ul>
         <li><strong>New:</strong> Modular optional extras — <code>[simulation]</code>, <code>[urdf]</code>, <code>[vision]</code>, <code>[ml]</code>, <code>[cuda]</code>, <code>[all]</code> — default install is now lightweight</li>
         <li><strong>New:</strong> Native NumPy 2.0-compatible URDF parser (<code>ManipulaPy.urdf.URDF</code>) with <code>PackageResolver</code> for <code>package://</code> and <code>file://</code> URIs</li>
         <li><strong>New:</strong> PEP 561 <code>py.typed</code> marker — mypy/pyright now see ManipulaPy as a typed package</li>
         <li><strong>New:</strong> Python 3.12 added to the supported matrix</li>
         <li><strong>Fixed:</strong> CUDA trajectory kernels — corrected quintic acceleration, removed shared-memory and forward-dynamics races, added <code>method=1</code> (linear) support, guarded N≤1 against div-zero</li>
         <li><strong>Fixed:</strong> Repulsive-potential-gradient sign in <code>fused_potential_gradient_kernel</code> — previous versions produced an attracting field</li>
         <li><strong>Fixed:</strong> Simulation methods now raise clear <code>ImportError</code> with <code>pip install ManipulaPy[simulation]</code> hint when PyBullet is missing</li>
         <li><strong>Fixed:</strong> <code>Vision.detect_obstacles</code> default <code>depth_threshold</code> raised from 0.0 → 5.0 m (the old default filtered every detection)</li>
      </ul>
   </div>

Citing ManipulaPy
-----------------

If you use ManipulaPy in your research, please cite:

.. code-block:: bibtex

   @software{manipulapy2026,
     title={ManipulaPy: A Modern Python Library for Robot Manipulation},
     author={Mohamed Aboelnasr},
     year={2026},
     url={https://github.com/boelnasr/ManipulaPy},
     version={1.4.0}
   }

License
-------

ManipulaPy is released under the **AGPL-3.0 License**: the source is freely
available, derivative works must also be open source, modified network services
must offer their source to users, and commercial use is permitted under those
same terms. For commercial licensing options or AGPL compliance questions,
please contact the maintainers.

Contributing to these docs
--------------------------

These pages are generated from the reStructuredText sources in ``docs/``.
Corrections and additions are welcome — open a
`pull request <https://github.com/boelnasr/ManipulaPy/pull/new>`_.

Indices and tables
------------------

* :ref:`genindex` — complete index of all functions, classes, and methods
* :ref:`modindex` — module index for quick navigation
* :ref:`search` — search the documentation
