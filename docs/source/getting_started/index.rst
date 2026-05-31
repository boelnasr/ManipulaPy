.. _getting-started:

Getting Started with ManipulaPy
===============================

Welcome to ManipulaPy! This guide will get you up and running with modern robotics programming in Python.

.. raw:: html

   <div class="getting-started-hero">
      <div class="hero-content">
         <h2>🚀 Ready to Build Robots?</h2>
         <p>ManipulaPy makes robot programming accessible with GPU acceleration, 
            computer vision, and a clean Python API. Let's get started!</p>
      </div>
   </div>

.. contents:: **What you'll learn**
   :local:
   :depth: 2
   :backlinks: none

Installation
------------

🔧 **Quick Install**

The fastest way to get started:

.. code-block:: bash

   pip install manipulapy

.. note::
   As of v1.3.2, the default install is intentionally lightweight — it
   pulls only the core numerical/plotting stack (NumPy, SciPy, matplotlib,
   numba, pillow). Heavier or platform-specific dependencies such as
   ``pybullet``, ``opencv-python``, ``torch``, ``scikit-learn``, and ``trimesh``
   now live in optional extras (see below) so that the base
   install succeeds on minimal images and platforms without prebuilt wheels
   (e.g., Apple Silicon).

🚀 **Recommended Install (with GPU acceleration)**

For the best performance:

.. code-block:: bash

   pip install manipulapy[cuda]

📦 **Full Installation (all features)**

To unlock all capabilities:

.. code-block:: bash

   pip install manipulapy[all]

🛠️ **Development Installation**

If you want to contribute or modify the library:

.. code-block:: bash

   git clone https://github.com/boelnasr/ManipulaPy.git
   cd ManipulaPy
   pip install -e .[dev]

.. note::
   **System Requirements:**

   - Python 3.9 or higher
   - Core (auto-installed): NumPy, SciPy, matplotlib, numba, pillow
   - Optional extras (install on demand):

     - ``[simulation]`` — PyBullet physics simulation
     - ``[urdf]`` — trimesh-based URDF mesh loading
     - ``[vision]`` — OpenCV, Ultralytics/YOLO, PyTorch
     - ``[ml]`` — PyTorch + scikit-learn for learning-based components
     - ``[cuda]`` — CuPy (CUDA 11.x) for GPU acceleration
     - ``[all]`` — everything above

Verify Your Installation
~~~~~~~~~~~~~~~~~~~~~~~~

Let's make sure everything is working:

.. code-block:: python

   import ManipulaPy
   print("🎉 ManipulaPy installed successfully!")
   print(f"Version: {ManipulaPy.__version__}")

Your First Robot
----------------

🤖 **Load a Robot Model**

ManipulaPy comes with built-in robot models. Let's start with the xArm:

.. code-block:: python

   import numpy as np
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

   # Load the built-in xArm robot
   print("📁 Loading xArm robot model...")
   urdf_processor = URDFToSerialManipulator(xarm_urdf_file)
   robot = urdf_processor.serial_manipulator
   dynamics = urdf_processor.dynamics

   print(f"✅ Robot loaded with {len(robot.S_list[0])} degrees of freedom")

🎯 **Forward Kinematics**

Calculate where the robot's end-effector is:

.. code-block:: python

   # Robot at home position (all joints at 0°)
   home_angles = np.zeros(6)
   end_effector_pose = robot.forward_kinematics(home_angles, frame="space")

   print("🏠 Home position:")
   print(f"   Position: {end_effector_pose[:3, 3]}")
   print(f"   Orientation:\n{end_effector_pose[:3, :3]}")

🔄 **Inverse Kinematics**

Find joint angles to reach a target position:

.. code-block:: python

   # Define a target pose
   target_position = np.array([0.5, 0.3, 0.8, 0.0, 0.5, 0.0])
   T_target = robot.forward_kinematics(target_position)

   # Solve inverse kinematics
   print("🎯 Solving inverse kinematics...")
   solution, success, iterations = robot.iterative_inverse_kinematics(
       T_desired=T_target,
       thetalist0=np.zeros(6),
       max_iterations=1000
   )

   if success:
       print(f"✅ Solution found in {iterations} iterations!")
       print(f"🔧 Joint angles: {np.degrees(solution)}°")
   else:
       print("❌ No solution found")

Your First Trajectory
---------------------

⚡ **GPU-Accelerated Planning**

Plan smooth robot motions with CUDA acceleration:

.. code-block:: python

   from ManipulaPy.path_planning import TrajectoryPlanning

   # Set up trajectory planner
   joint_limits = np.array([[-np.pi, np.pi]] * 6)
   planner = TrajectoryPlanning(robot, xarm_urdf_file, dynamics, joint_limits)

   # Plan a smooth trajectory
   start_angles = np.zeros(6)
   end_angles = np.array([0.5, -0.3, 0.8, 0.0, 0.5, 0.0])

   print("📈 Planning trajectory...")
   trajectory = planner.joint_trajectory(
       thetastart=start_angles,
       thetaend=end_angles,
       Tf=5.0,          # 5 seconds
       N=100,           # 100 waypoints
       method=5         # Quintic (smooth) interpolation
   )

   print(f"✅ Generated {trajectory['positions'].shape[0]} waypoints")
   print(f"🚀 Start velocity: {trajectory['velocities'][0]}")
   print(f"🏁 End velocity: {trajectory['velocities'][-1]}")

📊 **Visualize the Trajectory**

See your robot's planned motion:

.. code-block:: python

   # Plot the trajectory
   planner.plot_trajectory(trajectory, 5.0, title="My First Robot Trajectory")

Your First Simulation
---------------------

🎬 **PyBullet Physics Simulation**

.. note::
   Simulation runs on CPU via PyBullet — no GPU or CUDA is required.
   You do, however, need the ``[simulation]`` extra:

   .. code-block:: bash

      pip install "ManipulaPy[simulation]"

Bring your robot to life with realistic physics:

.. code-block:: python

   from ManipulaPy.sim import Simulation

   # Create physics simulation
   print("🎬 Starting simulation...")
   sim = Simulation(
       urdf_file_path=xarm_urdf_file,
       joint_limits=joint_limits,
       torque_limits=np.array([[-50, 50]] * 6),
       time_step=0.01
   )

   # Initialize robot and controllers
   sim.initialize_robot()
   sim.initialize_planner_and_controller()

   # Execute the trajectory in simulation
   waypoints = trajectory["positions"][::10]  # Use every 10th point
   
   print("🏃 Running simulation...")
   final_position = sim.run_trajectory(waypoints)
   print(f"🏁 Final end-effector position: {final_position}")

Your First Control System
-------------------------

🎛️ **Intelligent Robot Control**

.. note::
   The control module is pure NumPy/SciPy — it ships with the default
   install and does **not** require CUDA, a GPU, or any extras.

Control your robot with advanced algorithms:

.. code-block:: python

   from ManipulaPy.control import ManipulatorController

   # Create smart controller
   controller = ManipulatorController(dynamics)

   # Current and desired robot states
   current_pos = np.zeros(6)
   current_vel = np.zeros(6)
   desired_pos = np.array([0.2, -0.1, 0.3, 0.0, 0.2, 0.0])
   desired_vel = np.zeros(6)

   # Auto-tune controller gains
   ultimate_gain = 50.0    # Experiment to find this
   ultimate_period = 0.5   # Measure from oscillations
   Kp, Ki, Kd = controller.tune_controller(ultimate_gain, ultimate_period, kind="PID")

   print(f"🎛️ Auto-tuned gains:")
   print(f"   Kp: {Kp[0]:.2f}, Ki: {Ki[0]:.2f}, Kd: {Kd[0]:.2f}")

   # Compute control torques
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

   print(f"⚡ Control torques: {control_torques}")

Your First Vision System
------------------------

👁️ **Computer Vision & Perception**

.. note::
   The vision and perception modules pull in OpenCV, Ultralytics/YOLO,
   and PyTorch, which are *not* part of the default install. Install the
   ``[vision]`` extra first:

   .. code-block:: bash

      pip install "ManipulaPy[vision]"

Add eyes to your robot:

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

   # Create vision system
   print("👁️ Setting up vision system...")
   vision = Vision(camera_configs=[camera_config])
   perception = Perception(vision_instance=vision)

   # Detect objects in the workspace
   obstacle_points, cluster_labels = perception.detect_and_cluster_obstacles(
       camera_index=0,
       depth_threshold=3.0,  # Objects within 3m
       eps=0.1,              # Clustering parameter
       min_samples=3         # Minimum points per cluster
   )

   num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
   print(f"🔍 Detected {len(obstacle_points)} obstacle points")
   print(f"📦 Found {num_clusters} distinct objects")

What's Next?
------------

🎉 **Congratulations!** You've just built your first robot system with ManipulaPy.

.. raw:: html

   <div class="next-steps">
      <div class="step-category">
         <h4>📚 Learn the Fundamentals</h4>
         <ul>
            <li><a href="../user_guide/Kinematics.html">🔧 Kinematics Deep Dive</a></li>
            <li><a href="../user_guide/Dynamics.html">⚖️ Robot Dynamics</a></li>
            <li><a href="../user_guide/Trajectory_Planning.html">🛤️ Motion Planning</a></li>
         </ul>
      </div>
      <div class="step-category">
         <h4>🎓 Explore Tutorials</h4>
         <ul>
            <li><a href="../tutorials/index.html">🤖 Build a Pick & Place Robot</a></li>
            <li><a href="../tutorials/index.html">👁️ Vision-Guided Manipulation</a></li>
            <li><a href="../tutorials/index.html">🏭 Multi-Robot Coordination</a></li>
         </ul>
      </div>
      <div class="step-category">
         <h4>🛠️ API Reference</h4>
         <ul>
            <li><a href="../api/kinematics.html">📖 Kinematics API</a></li>
            <li><a href="../api/dynamics.html">📖 Dynamics API</a></li>
            <li><a href="../api/control.html">📖 Control API</a></li>
            <li><a href="../api/path_planning.html">📖 Planning API</a></li>
         </ul>
      </div>
   </div>

Common Issues & Solutions
-------------------------

⚠️ **Installation Problems**

.. code-block:: bash

   # If you get permission errors
   pip install --user manipulapy

   # If you need CUDA support
   pip install manipulapy[cuda]
   # Verify CUDA is available
   python -c "import cupy; print('CUDA available!')"

⚠️ **Import Errors**

.. code-block:: python

   # If ManipulaPy modules aren't found
   import sys
   sys.path.append('/path/to/ManipulaPy')
   import ManipulaPy

⚠️ **Simulation Issues**

.. code-block:: python

   # If PyBullet simulation fails
   pip install pybullet
   
   # Test PyBullet installation
   import pybullet as p
   p.connect(p.DIRECT)
   print("PyBullet working!")

⚠️ **Performance Issues**

.. code-block:: python

   # Check if CUDA acceleration is working
   try:
       import cupy
       print("🚀 CUDA acceleration available")
   except ImportError:
       print("⚠️ Install CuPy for GPU acceleration")

💡 **Pro Tips**

.. raw:: html

   <div class="pro-tips">
      <div class="tip">
         <span class="tip-icon">🎯</span>
         <strong>Start Simple</strong><br>
         Begin with forward kinematics before inverse kinematics
      </div>
      <div class="tip">
         <span class="tip-icon">📊</span>
         <strong>Visualize Everything</strong><br>
         Use the plotting functions to understand robot behavior
      </div>
      <div class="tip">
         <span class="tip-icon">⚡</span>
         <strong>Use GPU Acceleration</strong><br>
         Install CUDA for 7x faster computations
      </div>
      <div class="tip">
         <span class="tip-icon">🔧</span>
         <strong>Check Joint Limits</strong><br>
         Always define realistic joint limits for safety
      </div>
   </div>

📞 **Need Help?**

- 📖 Check the :doc:`../api/index` for detailed function documentation
- 🐛 Report bugs on `GitHub Issues <https://github.com/boelnasr/ManipulaPy/issues>`_
- 💬 Join our community discussions
- 📧 Contact the maintainers for support

.. tip::
   Curious what changed in this release? See the
   `v1.3.2 changelog <https://github.com/boelnasr/ManipulaPy/blob/main/CHANGELOG.md>`_
   for highlights — the lightweight default install, the new ``[simulation]``,
   ``[urdf]``, ``[vision]``, and ``[ml]`` extras, plus control/sim/URDF bug fixes.

.. Styles for .getting-started-hero, .next-steps, .step-category, .pro-tips,
   and .tip live in docs/source/_static/custom.css so they adapt to Furo's
   light/dark theme via CSS variables.
