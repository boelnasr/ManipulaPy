.. _user_guide_index:
.. _user_guide/index:

User Guide
==========

.. raw:: html

   <p class="mp-lede">Task-oriented guides to every subsystem in ManipulaPy — from screw-theory
   kinematics and rigid-body dynamics through GPU-accelerated planning, perception, and the
   differentiable compute backends introduced in v1.4.</p>

.. contents:: On this page
   :local:
   :depth: 1
   :backlinks: none

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Core Robotics Fundamentals

   Kinematics
   Dynamics
   Control
   URDF_Processor

.. toctree::
   :maxdepth: 2
   :caption: Motion Planning & Simulation

   Trajectory_Planning
   Path_Planning
   Simulation
   Singularity_Analysis
   Collision_Checker
   Potential_Field

.. toctree::
   :maxdepth: 2
   :caption: Perception & Intelligence

   vision
   Perception

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   Backends
   CUDA_Kernels

Reading tracks
--------------

The guides are independent, but they build on each other in a few natural
orders. Pick the track that matches what you are trying to do.

.. raw:: html

   <table class="mp-matrix">
      <thead>
         <tr><th>Track</th><th>Suggested order</th></tr>
      </thead>
      <tbody>
         <tr>
            <td>New to manipulation</td>
            <td>
               <a href="Kinematics.html">Kinematics</a> &rarr;
               <a href="URDF_Processor.html">URDF Processor</a> &rarr;
               <a href="Simulation.html">Simulation</a> &rarr;
               <a href="Trajectory_Planning.html">Trajectory Planning</a>
            </td>
         </tr>
         <tr>
            <td>Control engineering</td>
            <td>
               <a href="Dynamics.html">Dynamics</a> &rarr;
               <a href="Control.html">Control</a> &rarr;
               <a href="Collision_Checker.html">Collision Checker</a> &rarr;
               <a href="Singularity_Analysis.html">Singularity Analysis</a>
            </td>
         </tr>
         <tr>
            <td>Performance and GPU</td>
            <td>
               <a href="CUDA_Kernels.html">CUDA Kernels</a> &rarr;
               <a href="Trajectory_Planning.html">Trajectory Planning</a> &rarr;
               <a href="Potential_Field.html">Potential Field</a> &rarr;
               <a href="Backends.html">Compute Backends</a>
            </td>
         </tr>
         <tr>
            <td>Differentiable robotics</td>
            <td>
               <a href="Backends.html">Compute Backends</a> &rarr;
               <a href="Kinematics.html">Kinematics</a> &rarr;
               <a href="Dynamics.html">Dynamics</a> &rarr;
               <a href="Singularity_Analysis.html">Singularity Analysis</a>
            </td>
         </tr>
         <tr>
            <td>Perception-driven systems</td>
            <td>
               <a href="vision.html">Vision</a> &rarr;
               <a href="Perception.html">Perception</a> &rarr;
               <a href="Path_Planning.html">Path Planning</a> &rarr;
               <a href="Control.html">Control</a>
            </td>
         </tr>
      </tbody>
   </table>

Core robotics
-------------

.. raw:: html

   <div class="mp-grid mp-grid--wide">
      <div class="mp-card">
         <span class="mp-kicker">Introductory</span>
         <h3 class="mp-card__title"><a href="Kinematics.html">Kinematics</a></h3>
         <p class="mp-card__body">Robot geometry, forward and inverse kinematics,
            space and body Jacobians, and workspace analysis.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">screw axes</span>
            <span class="mp-tag">workspace plots</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Intermediate</span>
         <h3 class="mp-card__title"><a href="Dynamics.html">Dynamics</a></h3>
         <p class="mp-card__body">Mass matrices, inverse and forward dynamics,
            Coriolis and centrifugal terms, and gravity compensation.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">cached inertia</span>
            <span class="mp-tag">GPU accelerated</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Intermediate</span>
         <h3 class="mp-card__title"><a href="Control.html">Control</a></h3>
         <p class="mp-card__body">PID, computed-torque and adaptive controllers,
            Ziegler-Nichols auto-tuning, and response analysis.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">auto-tuning</span>
            <span class="mp-tag">step response</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Introductory</span>
         <h3 class="mp-card__title"><a href="URDF_Processor.html">URDF Processor</a></h3>
         <p class="mp-card__body">Load URDF files, extract link and joint parameters,
            and build <code>SerialManipulator</code> and dynamics objects.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">native parser</span>
            <span class="mp-tag">built-in models</span>
         </div>
      </div>
   </div>

Motion planning and simulation
------------------------------

.. raw:: html

   <div class="mp-grid mp-grid--wide">
      <div class="mp-card">
         <span class="mp-kicker">Intermediate</span>
         <h3 class="mp-card__title"><a href="Trajectory_Planning.html">Trajectory Planning</a></h3>
         <p class="mp-card__body">CUDA-accelerated joint and Cartesian trajectory
            generation with time scaling and collision avoidance.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">CUDA kernels</span>
            <span class="mp-tag">quintic scaling</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Intermediate</span>
         <h3 class="mp-card__title"><a href="Path_Planning.html">Path Planning</a></h3>
         <p class="mp-card__body">Batch path generation, obstacle-aware routing, and the
            planning interfaces that feed the trajectory layer.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">batched</span>
            <span class="mp-tag">obstacle aware</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Introductory</span>
         <h3 class="mp-card__title"><a href="Simulation.html">Simulation</a></h3>
         <p class="mp-card__body">PyBullet physics simulation, trajectory execution,
            and interactive robot control. Runs on CPU; no GPU required.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">PyBullet</span>
            <span class="mp-tag">real-time</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Intermediate</span>
         <h3 class="mp-card__title"><a href="Collision_Checker.html">Collision Checker</a></h3>
         <p class="mp-card__body">Self-collision detection, environment obstacles,
            and safety monitoring during execution.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">convex hull</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Advanced</span>
         <h3 class="mp-card__title"><a href="Potential_Field.html">Potential Field</a></h3>
         <p class="mp-card__body">Artificial potential fields for reactive path
            planning: attractive goals and repulsive obstacle gradients.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">fused kernels</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Advanced</span>
         <h3 class="mp-card__title"><a href="Singularity_Analysis.html">Singularity Analysis</a></h3>
         <p class="mp-card__body">Manipulability ellipsoids, condition numbers,
            workspace mapping, and singularity avoidance.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">Monte Carlo</span>
            <span class="mp-tag">differentiable</span>
         </div>
      </div>
   </div>

Perception
----------

.. raw:: html

   <div class="mp-grid mp-grid--wide">
      <div class="mp-card">
         <span class="mp-kicker">Intermediate</span>
         <h3 class="mp-card__title"><a href="vision.html">Vision</a></h3>
         <p class="mp-card__body">Camera configuration, stereo rectification, depth
            processing, multi-camera rigs, and PyBullet virtual cameras.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">stereo</span>
            <span class="mp-tag">calibration</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Advanced</span>
         <h3 class="mp-card__title"><a href="Perception.html">Perception</a></h3>
         <p class="mp-card__body">YOLO object detection, DBSCAN 3D clustering,
            scene understanding, and integration with planning and control.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">YOLO</span>
            <span class="mp-tag">DBSCAN</span>
         </div>
      </div>
   </div>

Advanced topics
---------------

.. raw:: html

   <div class="mp-grid mp-grid--wide">
      <div class="mp-card">
         <span class="mp-kicker">New in 1.4</span>
         <h3 class="mp-card__title"><a href="Backends.html">Compute Backends</a></h3>
         <p class="mp-card__body">Run the same kinematics and dynamics on NumPy, CuPy,
            PyTorch or JAX — with autodiff gradients on the core math, and an explicit
            contract for everything outside it.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">differentiable</span>
            <span class="mp-tag">one API</span>
         </div>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Expert</span>
         <h3 class="mp-card__title"><a href="CUDA_Kernels.html">CUDA Kernels</a></h3>
         <p class="mp-card__body">Custom Numba CUDA kernels, memory management,
            launch configuration, and performance profiling.</p>
         <div class="mp-card__tags">
            <span class="mp-tag">Numba</span>
            <span class="mp-tag">profiling</span>
         </div>
      </div>
   </div>

Quick reference
---------------

.. raw:: html

   <div class="mp-grid">
      <div class="mp-card">
         <span class="mp-kicker">Setup</span>
         <ul>
            <li><a href="../getting_started/index.html">Installation and first run</a></li>
            <li><a href="Kinematics.html#basic-forward-kinematics">First robot analysis</a></li>
            <li><a href="URDF_Processor.html#loading-built-in-models">Built-in robot models</a></li>
            <li><a href="Simulation.html#basic-simulation">Your first simulation</a></li>
         </ul>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Performance</span>
         <ul>
            <li><a href="CUDA_Kernels.html#installation">CUDA setup</a></li>
            <li><a href="Trajectory_Planning.html#cuda-acceleration">GPU trajectories</a></li>
            <li><a href="Dynamics.html#caching">Mass-matrix caching</a></li>
            <li><a href="Backends.html">Choosing a backend</a></li>
         </ul>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Common tasks</span>
         <ul>
            <li><a href="Kinematics.html#inverse-kinematics">Solve inverse kinematics</a></li>
            <li><a href="Control.html#pid-control">Implement PID control</a></li>
            <li><a href="Trajectory_Planning.html#joint-trajectories">Plan trajectories</a></li>
            <li><a href="Collision_Checker.html#basic-collision-detection">Check collisions</a></li>
         </ul>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Perception</span>
         <ul>
            <li><a href="vision.html#camera-setup">Set up cameras</a></li>
            <li><a href="Perception.html#obstacle-detection">Detect obstacles</a></li>
            <li><a href="Perception.html#data-flow-architecture">Data-flow pipeline</a></li>
            <li><a href="vision.html#stereo-vision">Stereo processing</a></li>
         </ul>
      </div>

      <div class="mp-card">
         <span class="mp-kicker">Deeper</span>
         <ul>
            <li><a href="Control.html#computed-torque-control">Computed-torque control</a></li>
            <li><a href="Singularity_Analysis.html#manipulability">Manipulability analysis</a></li>
            <li><a href="Potential_Field.html#artificial-potential-fields">Potential-field planning</a></li>
            <li><a href="CUDA_Kernels.html#custom-kernels">Custom CUDA kernels</a></li>
         </ul>
      </div>
   </div>

Getting help
------------

Each guide ships complete, runnable examples. If something still does not work:

1. Check the :doc:`../api/index` for exact signatures and defaults.

2. Search the `GitHub issue tracker <https://github.com/boelnasr/ManipulaPy/issues>`_
   for the error you are seeing.

3. Open a new issue with a minimal reproducing script, your Python version,
   and the active backend.
