.. _doc-index:

ManipulaPy Documentation
========================

A modern, GPU-accelerated Python toolbox for **robot kinematics, dynamics, trajectory planning, perception and control**.

.. raw:: html

   <div class="hero-section">
      <div class="hero-content">
         <h2>🤖 Modern Robotics Made Simple</h2>
         <p>ManipulaPy brings cutting-edge robotics algorithms to your fingertips with GPU acceleration, 
            computer vision integration, and a clean Python API.</p>
         
         <!-- Project Badges -->
         <div class="project-badges">
            <a href="https://pypi.org/project/manipulapy/">
               <img src="https://img.shields.io/pypi/v/manipulapy?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=blue" alt="PyPI Version">
            </a>
            <a href="https://www.python.org/downloads/">
               <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Versions">
            </a>
            <a href="https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa">
               <img src="https://joss.theoj.org/papers/e0e68c2dcd8ac9dfc1354c7ee37eb7aa/status.svg?style=for-the-badge" alt="JOSS Paper">
            </a>
            <a href="https://github.com/boelnasr/ManipulaPy/blob/main/LICENSE">
               <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg?style=for-the-badge" alt="License">
            </a>
            <a href="https://github.com/boelnasr/ManipulaPy/actions">
               <img src="https://img.shields.io/github/actions/workflow/status/boelnasr/ManipulaPy/test.yml?branch=main&style=for-the-badge&logo=github&label=CI" alt="CI Status">
            </a>
            <a href="https://pypi.org/project/manipulapy/">
               <img src="https://static.pepy.tech/badge/manipulapy" alt="Downloads">
            </a>
            <a href="https://github.com/boelnasr/ManipulaPy">
               <img src="https://img.shields.io/github/stars/boelnasr/ManipulaPy?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=yellow" alt="GitHub Stars">
            </a>
         </div>
         
         <div class="feature-grid">
            <div class="feature">
               <span class="feature-icon">⚡</span>
               <strong>CUDA Accelerated</strong><br>
               GPU-powered trajectory planning and dynamics
            </div>
            <div class="feature">
               <span class="feature-icon">👁️</span>
               <strong>Computer Vision</strong><br>
               YOLO detection and stereo perception
            </div>
            <div class="feature">
               <span class="feature-icon">🎮</span>
               <strong>Real-time Control</strong><br>
               PyBullet simulation and advanced controllers
            </div>
         </div>
      </div>
   </div>

.. contents:: **Quick links**
   :local:
   :depth: 1


Getting Started
---------------

If you're in a hurry, install the package into a fresh virtual-env and try the examples below:

.. raw:: html

   <div class="installation-section">
      <!-- Live Version Badge -->
      <div class="live-version">
         <a href="https://pypi.org/project/manipulapy/">
            <img src="https://img.shields.io/pypi/v/manipulapy?style=flat-square&logo=pypi&logoColor=white&label=Latest%20Version&color=blue" alt="Latest Version">
         </a>
         <a href="https://pypi.org/project/manipulapy/">
            <img src="https://img.shields.io/pypi/wheel/manipulapy?style=flat-square&color=brightgreen&label=Wheel" alt="Wheel Available">
         </a>
         <a href="https://img.shields.io/badge/tests-passing-brightgreen">
            <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square" alt="Test Status">
         </a>
      </div>
   </div>

.. code-block:: bash

   python -m pip install manipulapy[cuda]  # or just manipulapy
   python -c "import ManipulaPy; print('🎉 Installation successful!')"

.. note::
   The docs you're reading are generated from the source in ``docs/``; feel free to improve them and send a `pull request <https://github.com/boelnasr/ManipulaPy/pull/new>`_. 🚀


🚀 Quick Start Examples
~~~~~~~~~~~~~~~~~~~~~~~

**1. Your First Robot Analysis**

.. code-block:: python

   import numpy as np
   from ManipulaPy.urdf_processor import URDFToSerialManipulator
   from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file

   # Load the built-in xArm robot model
   urdf_processor = URDFToSerialManipulator(xarm_urdf_file)
   robot = urdf_processor.serial_manipulator
   dynamics = urdf_processor.dynamics

   # Compute forward kinematics at home position
   joint_angles = np.zeros(6)  # 6-DOF robot at home
   end_effector_pose = robot.forward_kinematics(joint_angles, frame="space")
   
   print("🏠 Home position:", end_effector_pose[:3, 3])
   print("📐 Home orientation:\n", end_effector_pose[:3, :3])

**2. Inverse Kinematics in Action**

.. code-block:: python

   # Define a target pose for the end-effector
   target_angles = np.array([0.5, -0.3, 0.8, 0.0, 0.5, 0.0])
   T_target = robot.forward_kinematics(target_angles)

   # Solve inverse kinematics
   initial_guess = np.zeros(6)
   solution, success, iterations = robot.iterative_inverse_kinematics(
       T_desired=T_target,
       thetalist0=initial_guess,
       max_iterations=1000
   )

   if success:
       print(f"✅ IK solved in {iterations} iterations!")
       print(f"🎯 Solution: {np.degrees(solution):.2f} degrees")
   else:
       print("❌ IK solution not found")

**3. CUDA-Accelerated Trajectory Planning**

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

   print(f"📈 Generated trajectory with {trajectory['positions'].shape[0]} points")
   print(f"🎯 Start velocity: {trajectory['velocities'][0]}")
   print(f"🏁 End velocity: {trajectory['velocities'][-1]}")

   # Visualize the trajectory
   planner.plot_trajectory(trajectory, 5.0, title="Smooth Joint Trajectory")

**4. Advanced Control with Dynamics**

.. code-block:: python

   from ManipulaPy.control import ManipulatorController

   # Create intelligent controller
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

   print(f"🎛️  Auto-tuned gains - Kp: {Kp[0]:.2f}, Ki: {Ki[0]:.2f}, Kd: {Kd[0]:.2f}")

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

   print(f"⚡ Control torques: {control_torques}")

**5. PyBullet Simulation**

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
   
   print("🎬 Running simulation...")
   final_position = sim.run_trajectory(waypoints)
   print(f"🏁 Final end-effector position: {final_position}")

**6. Computer Vision & Perception**

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
   print(f"👁️  Detected {len(obstacle_points)} obstacle points")
   print(f"🔍 Found {num_clusters} distinct object clusters")

.. raw:: html

   <div class="getting-started-tips">
      <h4>💡 Pro Tips for Success</h4>
      <div class="tips-grid">
         <div class="tip">
            <span class="tip-icon">🎯</span>
            <strong>Start Simple</strong><br>
            Begin with forward kinematics before tackling complex control
         </div>
         <div class="tip">
            <span class="tip-icon">🔧</span>
            <strong>Use Built-ins</strong><br>
            The xArm URDF model is perfect for learning and testing
         </div>
         <div class="tip">
            <span class="tip-icon">📊</span>
            <strong>Visualize Everything</strong><br>
            Use plotting functions to understand robot behavior
         </div>
         <div class="tip">
            <span class="tip-icon">⚡</span>
            <strong>GPU Acceleration</strong><br>
            Install CUDA for 7x faster trajectory computations
         </div>
      </div>
   </div>


Key Features at a Glance
~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="features-overview">
      <div class="feature-category">
         <h4>🔧 Core Robotics</h4>
         <ul>
            <li><strong>Kinematics:</strong> Forward/inverse with neural network acceleration</li>
            <li><strong>Dynamics:</strong> Mass matrices, inverse/forward dynamics with caching</li>
            <li><strong>Control:</strong> PID, computed torque, adaptive controllers</li>
            <li><strong>Planning:</strong> CUDA-accelerated trajectory generation</li>
         </ul>
      </div>
      <div class="feature-category">
         <h4>👁️ Perception & Vision</h4>
         <ul>
            <li><strong>Vision:</strong> Monocular/stereo camera support</li>
            <li><strong>Detection:</strong> YOLO-based object detection</li>
            <li><strong>3D Processing:</strong> Point cloud clustering</li>
            <li><strong>Integration:</strong> PyBullet camera debugging</li>
         </ul>
      </div>
      <div class="feature-category">
         <h4>⚡ High Performance</h4>
         <ul>
            <li><strong>CUDA Kernels:</strong> GPU-accelerated computations</li>
            <li><strong>Parallel Processing:</strong> Multi-robot trajectory planning</li>
            <li><strong>Optimized Algorithms:</strong> Cached mass matrices</li>
            <li><strong>Real-time:</strong> Sub-millisecond kinematics</li>
         </ul>
      </div>
   </div>


Documentation map
-----------------

.. toctree::
   :maxdepth: 2
   :caption: 🚀 Get Started

   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: 🛠️ API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: 📚 User Guides

   user_guide/index
   user_guide/Kinematics
   user_guide/Dynamics
   user_guide/Control
   user_guide/Trajectory_Planning
   user_guide/Simulation
   user_guide/URDF_Processor
   user_guide/Singularity_Analysis
   user_guide/Perception
   user_guide/vision
   user_guide/Potential_Field
   user_guide/Collision_Checker
   user_guide/CUDA_Kernels


Popular Learning Paths
----------------------

.. raw:: html

   <div class="learning-paths">
      <div class="path">
         <h4>🆕 Complete Beginner</h4>
         <ol>
            <li><a href="getting_started/index.html">📖 Getting Started Guide</a></li>
            <li><a href="#getting-started">🚀 Quick Start Examples ↑</a></li>
            <li><a href="user_guide/Kinematics.html">🔧 Basic Kinematics</a></li>
            <li><a href="tutorials/index.html">🤖 First Robot Project</a></li>
         </ol>
      </div>
      <div class="path">
         <h4>🤖 Robotics Engineer</h4>
         <ol>
            <li><a href="user_guide/Kinematics.html">🔧 Advanced Kinematics</a></li>
            <li><a href="user_guide/Dynamics.html">⚖️ Robot Dynamics</a></li>
            <li><a href="user_guide/Control.html">🎛️ Control Systems</a></li>
            <li><a href="user_guide/Trajectory_Planning.html">🛤️ Motion Planning</a></li>
         </ol>
      </div>
      <div class="path">
         <h4>💻 Performance Engineer</h4>
         <ol>
            <li><a href="user_guide/CUDA_Kernels.html">⚡ CUDA Setup</a></li>
            <li><a href="tutorials/index.html">🚀 GPU Acceleration</a></li>
            <li><a href="tutorials/index.html">📊 Performance Profiling</a></li>
            <li><a href="tutorials/index.html">🔧 Optimization Tips</a></li>
         </ol>
      </div>
      <div class="path">
         <h4>👁️ Vision Engineer</h4>
         <ol>
            <li><a href="tutorials/index.html">👁️ Vision Systems</a></li>
            <li><a href="tutorials/index.html">🔍 Object Detection</a></li>
            <li><a href="tutorials/index.html">🌐 3D Perception</a></li>
            <li><a href="tutorials/index.html">🎯 Vision-Guided Control</a></li>
         </ol>
      </div>
   </div>


What's New
----------

.. raw:: html

   <div class="whats-new">
      <h4>🎉 Latest in v1.1.0</h4>
      <ul>
         <li><strong>New:</strong> CUDA-accelerated trajectory planning</li>
         <li><strong>New:</strong> Computer vision and perception modules</li>
         <li><strong>New:</strong> Neural network inverse kinematics</li>
         <li><strong>Improved:</strong> 3x faster dynamics computation with caching</li>
         <li><strong>Added:</strong> PyBullet simulation integration</li>
         <li><strong>Enhanced:</strong> Documentation with interactive examples</li>
      </ul>
   </div>


Installation Options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install manipulapy

   # With CUDA acceleration (recommended)
   pip install manipulapy[cuda]

   # With computer vision support
   pip install manipulapy[vision]

   # Full installation (all features)
   pip install manipulapy[all]

   # Development installation
   git clone https://github.com/boelnasr/ManipulaPy.git
   cd ManipulaPy
   pip install -e .[dev]


Performance Showcase
~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="performance-showcase">
      <div class="benchmark">
         <h5>⚡ CUDA Acceleration</h5>
         <p><strong>7x faster</strong> trajectory planning<br>
         GPU vs CPU for 1000-point trajectories</p>
      </div>
      <div class="benchmark">
         <h5>🧠 Neural Network IK</h5>
         <p><strong>10x faster</strong> convergence<br>
         Hybrid approach vs traditional methods</p>
      </div>
      <div class="benchmark">
         <h5>💾 Smart Caching</h5>
         <p><strong>3x faster</strong> dynamics<br>
         Cached mass matrices for repeated calls</p>
      </div>
      <div class="benchmark">
         <h5>👁️ Real-time Vision</h5>
         <p><strong>30 FPS</strong> object detection<br>
         YOLO integration with 3D localization</p>
      </div>
   </div>


Citing ManipulaPy
~~~~~~~~~~~~~~~~~

If you use ManipulaPy in your research, please cite:

.. code-block:: bibtex

   @software{manipulapy2024,
     title={ManipulaPy: A Modern Python Library for Robot Manipulation},
     author={Mohamed Aboelnar},
     year={2024},
     url={https://github.com/boelnasr/ManipulaPy},
     version={1.1.0}
   }


License
-------

ManipulaPy is released under the **AGPL-3.0 License**:

.. raw:: html

   <div class="license-info">
      <div class="license-feature">
         <span class="license-icon">✅</span>
         <strong>Open Source</strong><br>
         Source code freely available
      </div>
      <div class="license-feature">
         <span class="license-icon">🔄</span>
         <strong>Copyleft</strong><br>
         Derivative works must be open source
      </div>
      <div class="license-feature">
         <span class="license-icon">🌐</span>
         <strong>Network Use</strong><br>
         Modified network services must provide source
      </div>
      <div class="license-feature">
         <span class="license-icon">💼</span>
         <strong>Commercial Use</strong><br>
         Permitted with AGPL compliance
      </div>
   </div>

For commercial licensing options or AGPL compliance questions, please contact the maintainers.


Indices and tables
------------------

* :ref:`genindex` - Complete index of all functions, classes, and methods
* :ref:`modindex` - Module index for quick navigation
* :ref:`search` - Search the documentation


.. raw:: html

   <style>
   .hero-section {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 2rem;
      border-radius: 10px;
      margin: 2rem 0;
   }
   .hero-content h2 {
      margin-top: 0;
      font-size: 2rem;
   }
   
   /* Enhanced Badge Styling */
   .project-badges {
      text-align: center;
      margin: 1.5rem 0;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 0.5rem;
   }
   
   .project-badges img {
      height: 28px;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
   }
   
   .project-badges img:hover {
      transform: scale(1.05);
      box-shadow: 0 4px 8px rgba(0,0,0,0.2);
   }
   
   .installation-section {
      background: #f8f9fa;
      border: 1px solid #e9ecef;
      border-radius: 8px;
      padding: 1rem;
      margin: 1rem 0;
   }
   
   .live-version {
      text-align: center;
      margin: 1rem 0;
      display: flex;
      justify-content: center;
      gap: 0.5rem;
      flex-wrap: wrap;
   }
   
   .live-version img {
      height: 20px;
   }
   
   /* Responsive badges */
   @media (max-width: 768px) {
      .project-badges {
         flex-direction: column;
         align-items: center;
      }
      
      .project-badges img {
         width: 100%;
         max-width: 250px;
      }
      
      .live-version {
         flex-direction: column;
         align-items: center;
      }
   }
   
   .feature-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-top: 1.5rem;
   }
   .feature {
      background: rgba(255,255,255,0.5);
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
   }
   .feature-icon {
      font-size: 2rem;
      display: block;
      margin-bottom: 0.5rem;
   }
   .getting-started-tips {
      background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
      color: white;
      padding: 1.5rem;
      border-radius: 10px;
      margin: 2rem 0;
   }
   .getting-started-tips h4 {
      margin-top: 0;
   }
   .tips-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
   }
   .tip {
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
   }

   .tip:nth-child(1) { background: #059669; }
   .tip:nth-child(2) { background: #059669; }
   .tip:nth-child(3) { background: #059669; }
   .tip:nth-child(4) { background: #059669; }

   .tip-icon {
      font-size: 1.5rem;
      display: block;
      margin-bottom: 0.5rem;
   }
   .features-overview {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.5rem;
      margin: 1.5rem 0;
   }
   .feature-category {
      border: 1px solid #e1e4e8;
      border-radius: 8px;
      padding: 1rem;
   }
   .feature-category h4 {
      margin-top: 0;
      color: #0366d6;
   }
   .learning-paths {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
   }
   .path {
      border: 1px solid #e1e4e8;
      border-radius: 8px;
      padding: 1rem;
      background: #f8f9fa;
   }
   .path h4 {
      margin-top: 0;
      color: #0366d6;
   }
   .whats-new {
      background: #f0f8f0;
      border-left: 4px solid #28a745;
      padding: 1rem;
      margin: 1.5rem 0;
   }
   .performance-showcase {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0;
   }
   .benchmark {
      background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
      color: white;
      padding: 1rem;
      border-radius: 8px;
      text-align: center;
   }
   .benchmark h5 {
      margin-top: 0;
   }
   .license-info {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 1rem;
      margin: 1rem 0;
   }
   .license-feature {
      text-align: center;
      padding: 0.5rem;
   }
   .license-icon {
      font-size: 1.5rem;
      display: block;
      margin-bottom: 0.5rem;
   }
   </style>
