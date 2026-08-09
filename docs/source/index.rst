.. _doc-index:

ManipulaPy Documentation
========================

.. raw:: html

   <div class="mp-home">

.. rst-class:: mp-hero

Move from equations to motion.
------------------------------

.. raw:: html

   <div class="mp-hero__copy">
      <p class="mp-overline">Python robotics, from model to motion</p>
      <p class="mp-hero__lede">Build, differentiate, and accelerate robot kinematics, dynamics, planning, perception, and control in Python.</p>
      <a class="mp-primary-action" href="getting_started/index.html">Start building</a>
      <div class="mp-install" aria-label="Installation command">
         <code>python -m pip install manipulapy</code>
      </div>
   </div>
   <figure class="mp-hero__media">
      <img src="_static/images/robotics-lab-hero.webp" width="1600" height="1100" alt="Six-axis industrial robot arm in a research lab">
   </figure>

.. only:: latex

   Build, differentiate, and accelerate robot kinematics, dynamics, planning,
   perception, and control in Python.

   **Install:** ``python -m pip install manipulapy``

   Begin with :doc:`Getting Started <getting_started/index>`.

.. rst-class:: mp-paths

Choose your path
----------------

.. raw:: html

   <div class="mp-paths__grid" data-reveal>
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

.. only:: latex

   Choose a learning route:

   * :doc:`Learn robotics through executable notebooks <tutorials/notebook_course>`
   * :doc:`Load a robot from a URDF <user_guide/URDF_Processor>`
   * :doc:`Plan joint and Cartesian motion <user_guide/Trajectory_Planning>`
   * :doc:`Use accelerated array backends <user_guide/Backends>`

.. rst-class:: mp-motion-gallery

See the math move
-----------------

.. raw:: html

   <div class="mp-motion-gallery__grid" data-reveal>
      <figure class="mp-motion-gallery__primary">
         <picture class="mp-motion-gallery__media">
            <source media="(prefers-reduced-motion: reduce)" srcset="_static/images/workspace-still.png">
            <img src="_static/gifs/workspace.gif" width="550" height="450" loading="lazy" alt="Robot arm tracing its reachable workspace">
         </picture>
         <figcaption>Explore reachable workspaces</figcaption>
      </figure>
      <figure>
         <picture class="mp-motion-gallery__media">
            <source media="(prefers-reduced-motion: reduce)" srcset="_static/images/joint-trajectory-still.png">
            <img src="_static/gifs/joint_trajectory.gif" width="700" height="320" loading="lazy" alt="Joint position, velocity, and acceleration changing along a trajectory">
         </picture>
         <figcaption>Inspect a smooth joint trajectory</figcaption>
      </figure>
      <figure>
         <picture class="mp-motion-gallery__media">
            <source media="(prefers-reduced-motion: reduce)" srcset="_static/images/ur5-pick-motion-still.png">
            <img src="_static/gifs/ur5_pick_motion.gif" width="480" height="360" loading="lazy" alt="UR5 robot executing a pick motion">
         </picture>
         <figcaption>Execute planned motion in simulation</figcaption>
      </figure>
   </div>

.. only:: epub

   .. figure:: _static/images/workspace-still.png
      :alt: Robot arm tracing its reachable workspace

      Explore reachable workspaces

   .. figure:: _static/images/joint-trajectory-still.png
      :alt: Joint position, velocity, and acceleration changing along a trajectory

      Inspect a smooth joint trajectory

   .. figure:: _static/images/ur5-pick-motion-still.png
      :alt: UR5 robot executing a pick motion

      Execute planned motion in simulation

.. only:: latex

   .. figure:: _static/images/workspace-still.png
      :alt: Robot arm tracing its reachable workspace

      Explore reachable workspaces

   .. figure:: _static/images/joint-trajectory-still.png
      :alt: Joint position, velocity, and acceleration changing along a trajectory

      Inspect a smooth joint trajectory

   .. figure:: _static/images/ur5-pick-motion-still.png
      :alt: UR5 robot executing a pick motion

      Execute planned motion in simulation

   The motion studies show a robot tracing its reachable workspace, joint
   position, velocity, and acceleration changing along a smooth trajectory,
   and a UR5 robot executing a planned pick motion in simulation.

.. rst-class:: mp-backends

Compute where your work belongs.
--------------------------------

.. raw:: html

   <p class="mp-overline">One API, four array libraries</p>
   <div class="mp-backends__grid" data-reveal>
      <div><h3>NumPy</h3><p>The lightweight default for local work, teaching, and dependable CPU execution.</p></div>
      <div><h3>CuPy</h3><p>Move compatible array work to NVIDIA GPUs without changing the public mathematics API.</p></div>
      <div><h3>PyTorch</h3><p>Connect robot models to training loops and preserve gradients through supported core math.</p></div>
      <div><h3>JAX</h3><p>Differentiate and compile supported kinematics, dynamics, singularity, and utility operations.</p></div>
   </div>
   <div class="mp-backends__links">
      <a href="user_guide/Backends.html">Read the backend guide</a>
      <a href="api/backend.html">Open the backend API</a>
   </div>

.. only:: latex

   **NumPy** is the lightweight CPU default. **CuPy** moves compatible array
   work to NVIDIA GPUs. **PyTorch** connects robot models to training loops and
   supported gradients. **JAX** differentiates and compiles supported robotics
   operations.

   Continue with the :doc:`Compute Backends guide <user_guide/Backends>` or
   open the :doc:`backend API <api/backend>`.

.. rst-class:: mp-api-links

Go straight to the reference
----------------------------

.. raw:: html

   <div class="mp-api-links__grid" data-reveal>
      <div><h3>Model</h3><a href="api/kinematics.html">Kinematics</a><a href="api/dynamics.html">Dynamics</a><a href="api/urdf_processor.html">URDF processor</a></div>
      <div><h3>Move</h3><a href="api/path_planning.html">Path planning</a><a href="api/control.html">Control</a><a href="api/potential_field.html">Potential fields</a></div>
      <div><h3>Sense</h3><a href="api/vision.html">Vision</a><a href="api/perception.html">Perception</a><a href="api/simulation.html">Simulation</a></div>
      <div><h3>Compute</h3><a href="api/backend.html">Backends</a><a href="api/cuda_kernels.html">CUDA kernels</a><a href="api/utils.html">Utilities</a></div>
   </div>

.. only:: latex

   **Model:** :doc:`Kinematics <api/kinematics>`,
   :doc:`Dynamics <api/dynamics>`, and
   :doc:`URDF processor <api/urdf_processor>`.

   **Move:** :doc:`Path planning <api/path_planning>`,
   :doc:`Control <api/control>`, and
   :doc:`Potential fields <api/potential_field>`.

   **Sense:** :doc:`Vision <api/vision>`,
   :doc:`Perception <api/perception>`, and
   :doc:`Simulation <api/simulation>`.

   **Compute:** :doc:`Backends <api/backend>`,
   :doc:`CUDA kernels <api/cuda_kernels>`, and
   :doc:`Utilities <api/utils>`.

.. raw:: html

   </div>
   <footer class="mp-home__legal">

ManipulaPy is released under the **AGPL-3.0 License**: the source is freely
available, derivative works must also be open source, modified network services
must offer their source to users, and commercial use is permitted under those
same terms. For commercial licensing options or AGPL compliance questions,
please contact the maintainers.

.. raw:: html

   </footer>

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
