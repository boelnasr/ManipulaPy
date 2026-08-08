.. _doc-index:

ManipulaPy Documentation
========================

.. raw:: html

   <div class="mp-home">
      <section class="mp-hero" aria-labelledby="mp-hero-title">
         <div class="mp-hero__copy">
            <p class="mp-overline">Python robotics, from model to motion</p>
            <h1 id="mp-hero-title">Move from equations to motion.</h1>
            <p class="mp-hero__lede">Build, differentiate, and accelerate robot kinematics, dynamics, planning, perception, and control in Python.</p>
            <a class="mp-primary-action" href="getting_started/index.html">Start building</a>
            <div class="mp-install" aria-label="Installation command">
               <code>python -m pip install manipulapy</code>
            </div>
         </div>
         <figure class="mp-hero__media">
            <img src="_static/images/robotics-lab-hero.webp" width="1600" height="1100" alt="Six-axis industrial robot arm in a research lab">
         </figure>
      </section>

      <section class="mp-paths" aria-labelledby="mp-paths-title" data-reveal>
         <h2 id="mp-paths-title">Choose your path</h2>
         <div class="mp-paths__grid">
            <a class="mp-path mp-path--featured" href="tutorials/notebook_course.html">
               <span>Learn robotics</span>
               <h3>Work through the mathematics</h3>
               <p>Follow executable notebooks from rigid transforms through differentiable dynamics.</p>
            </a>
            <a class="mp-path" href="user_guide/URDF_Processor.html">
               <span>Load a robot</span>
               <h3>Start from a URDF</h3>
               <p>Build the serial model, dynamics, limits, and frames from your robot description.</p>
            </a>
            <a class="mp-path" href="user_guide/Trajectory_Planning.html">
               <span>Plan motion</span>
               <h3>Turn goals into trajectories</h3>
               <p>Generate joint and Cartesian paths, then connect planning to control and simulation.</p>
            </a>
            <a class="mp-path" href="user_guide/Backends.html">
               <span>Accelerate and differentiate</span>
               <h3>Choose the right array backend</h3>
               <p>Run the same core mathematics on NumPy, CuPy, PyTorch, or JAX.</p>
            </a>
         </div>
      </section>

      <section class="mp-motion-gallery" aria-labelledby="mp-motion-title" data-reveal>
         <h2 id="mp-motion-title">See the math move</h2>
         <div class="mp-motion-gallery__grid">
            <figure class="mp-motion-gallery__primary">
               <img src="_static/gifs/workspace.gif" width="550" height="450" loading="lazy" alt="Robot arm tracing its reachable workspace">
               <figcaption>Explore reachable workspaces</figcaption>
            </figure>
            <figure>
               <img src="_static/gifs/joint_trajectory.gif" width="700" height="320" loading="lazy" alt="Joint position, velocity, and acceleration changing along a trajectory">
               <figcaption>Inspect a smooth joint trajectory</figcaption>
            </figure>
            <figure>
               <img src="_static/gifs/ur5_pick_motion.gif" width="480" height="360" loading="lazy" alt="UR5 robot executing a pick motion">
               <figcaption>Execute planned motion in simulation</figcaption>
            </figure>
         </div>
      </section>

      <section class="mp-backends" aria-labelledby="mp-backends-title" data-reveal>
         <p class="mp-overline">One API, four array libraries</p>
         <h2 id="mp-backends-title">Compute where your work belongs.</h2>
         <div class="mp-backends__grid">
            <div><h3>NumPy</h3><p>The lightweight default for local work, teaching, and dependable CPU execution.</p></div>
            <div><h3>CuPy</h3><p>Move compatible array work to NVIDIA GPUs without changing the public mathematics API.</p></div>
            <div><h3>PyTorch</h3><p>Connect robot models to training loops and preserve gradients through supported core math.</p></div>
            <div><h3>JAX</h3><p>Differentiate and compile supported kinematics, dynamics, singularity, and utility operations.</p></div>
         </div>
         <div class="mp-backends__links">
            <a href="user_guide/Backends.html">Read the backend guide</a>
            <a href="api/backend.html">Open the backend API</a>
         </div>
      </section>

      <section class="mp-api-links" aria-labelledby="mp-api-title" data-reveal>
         <h2 id="mp-api-title">Go straight to the reference</h2>
         <div class="mp-api-links__grid">
            <div><h3>Model</h3><a href="api/kinematics.html">Kinematics</a><a href="api/dynamics.html">Dynamics</a><a href="api/urdf_processor.html">URDF processor</a></div>
            <div><h3>Move</h3><a href="api/path_planning.html">Path planning</a><a href="api/control.html">Control</a><a href="api/potential_field.html">Potential fields</a></div>
            <div><h3>Sense</h3><a href="api/vision.html">Vision</a><a href="api/perception.html">Perception</a><a href="api/simulation.html">Simulation</a></div>
            <div><h3>Compute</h3><a href="api/backend.html">Backends</a><a href="api/cuda_kernels.html">CUDA kernels</a><a href="api/utils.html">Utilities</a></div>
         </div>
      </section>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   Installation Guide
   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: User Guides
   :hidden:

   user_guide/index
