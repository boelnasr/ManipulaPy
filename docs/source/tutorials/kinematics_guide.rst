Kinematics with the Franka Panda
================================

This tutorial uses the bundled Franka Panda model to move between joint space,
tool pose, and tool velocity. The calculations below are literal excerpts from
the tested tutorial module, so the documentation and gallery share one source
of truth.

What you will build
-------------------

You will load the Panda, calculate a tool pose from a seven-joint arm
configuration, map joint rates to a tool twist, solve for a target pose, and
validate that solution. The Panda has seven revolute arm joints; its model also
exposes one gripper degree of freedom, which is not part of the arm-vector
calculations in this tutorial.

Before you begin
----------------

Use Python 3.9+ with ManipulaPy installed. The examples load the bundled URDF,
so no robot download or hand-written kinematic parameters are needed. Joint
angles are in radians and distances are in metres.

Load the bundled Panda
----------------------

``load_panda`` returns the serial manipulator, the seven arm-joint names and
limits, and the model's full exposed degree-of-freedom count.

.. literalinclude:: ../../examples/kinematics_tutorial.py
   :language: python
   :dedent: 0
   :start-after: # [load-panda-start]
   :end-before: # [load-panda-end]

Expected output::

   seven arm-joint names; 7 arm degrees of freedom; 8 exposed model degrees of freedom

Forward kinematics: joints to pose
----------------------------------

Forward kinematics maps the fixed seven-element ``HOME`` configuration to a
homogeneous pose matrix for the Panda tool. Its upper-left 3 by 3 block is the
rotation and its final column contains the position in metres.

.. literalinclude:: ../../examples/kinematics_tutorial.py
   :language: python
   :dedent: 0
   :start-after: # [forward-kinematics-start]
   :end-before: # [forward-kinematics-end]

Expected output::

   pose.shape == (4, 4)

Jacobian: joints to tool velocity
---------------------------------

The space Jacobian maps the seven joint rates to a six-component tool twist.
The first three twist components are angular velocity in rad/s; the final
three are linear velocity in m/s. Singular values provide a compact signal of
how close that configuration is to losing a direction of motion.

.. literalinclude:: ../../examples/kinematics_tutorial.py
   :language: python
   :dedent: 0
   :start-after: # [velocity-kinematics-start]
   :end-before: # [velocity-kinematics-end]

Expected output::

   jacobian.shape == (6, 7)
   twist.shape == (6,)

Inverse kinematics: pose to joints
----------------------------------

Inverse kinematics starts from ``HOME`` and asks the public iterative solver
to reach the pose produced from ``TARGET``. The returned configuration respects
the Panda arm's limits and the solver reports whether it converged.

.. literalinclude:: ../../examples/kinematics_tutorial.py
   :language: python
   :dedent: 0
   :start-after: # [inverse-kinematics-start]
   :end-before: # [inverse-kinematics-end]

Expected output::

   success is True; iterations <= 20

Validate the result
-------------------

Validate IK by running forward kinematics on the solution and comparing its
pose with the target. Do not require joint-vector equality: several joint
configurations can produce the same tool pose, and a numerical solver may
choose any valid one. Translation and rotation pose residuals directly test
the result the robot must achieve.

.. literalinclude:: ../../examples/kinematics_tutorial.py
   :language: python
   :dedent: 0
   :start-after: # [validation-start]
   :end-before: # [validation-end]

Expected output::

   translation residual < 1e-5 m
   rotation residual < 1e-5 rad

Troubleshooting
---------------

If loading fails, verify that you are using the installed ManipulaPy package
and that its bundled data are available. If an IK attempt does not converge for
your own target, start from a nearby feasible configuration and check whether
the target is near a singular configuration. The :doc:`URDF processor guide
<../user_guide/URDF_Processor>` explains how the robot description becomes a
serial manipulator.

Go deeper
---------

Continue with the :doc:`kinematics notebook <notebook_course>`, the
:doc:`kinematics user guide <../user_guide/Kinematics>`, and the
:doc:`kinematics API reference <../api/kinematics>`. For singularity metrics
and interpretation, see the :doc:`singularity analysis guide
<../user_guide/Singularity_Analysis>`.
