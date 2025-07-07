Kinematics Tutorial
===================

This tutorial covers the fundamentals of robot kinematics using ManipulaPy.

Introduction
------------

Robot kinematics deals with the geometry of robot motion without considering forces.

Basic Forward Kinematics
------------------------

.. code-block:: python

    import numpy as np
    from ManipulaPy.kinematics import SerialManipulator
    
    # Create manipulator
    robot = SerialManipulator(
        M_list=M,
        omega_list=omega,
        S_list=S_list,
        B_list=B_list
    )
    
    # Compute forward kinematics
    joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    T = robot.forward_kinematics(joint_angles)

Inverse Kinematics
------------------

.. code-block:: python

    # Define desired end-effector pose
    T_desired = np.eye(4)
    T_desired[:3, 3] = [0.5, 0.3, 0.4]  # Position
    
    # Solve inverse kinematics
    solution, success, iterations = robot.iterative_inverse_kinematics(
        T_desired=T_desired,
        thetalist0=np.zeros(6),
        eomg=1e-6,
        ev=1e-6
    )

See Also
--------

- :doc:`../user_guide/Kinematics` - Complete kinematics guide
- :doc:`../api/kinematics` - Kinematics API reference
