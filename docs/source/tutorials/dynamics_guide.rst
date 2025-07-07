Dynamics Tutorial  
=================

This tutorial covers robot dynamics and physics simulation using ManipulaPy.

Introduction
------------

Robot dynamics deals with the relationship between forces, torques, and motion.

Mass Matrix Computation
-----------------------

.. code-block:: python

    from ManipulaPy.dynamics import ManipulatorDynamics
    
    # Create dynamics object
    dynamics = ManipulatorDynamics(
        M_list=M,
        omega_list=omega,
        S_list=S_list,
        B_list=B_list,
        Glist=Glist
    )
    
    # Compute mass matrix
    joint_angles = np.zeros(6)
    M = dynamics.mass_matrix(joint_angles)

Inverse Dynamics
---------------

.. code-block:: python

    # Compute required torques
    joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    joint_velocities = np.zeros(6)
    joint_accelerations = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0])
    gravity = [0, 0, -9.81]
    external_forces = np.zeros(6)
    
    torques = dynamics.inverse_dynamics(
        joint_angles,
        joint_velocities, 
        joint_accelerations,
        gravity,
        external_forces
    )

Forward Dynamics
---------------

.. code-block:: python

    # Simulate robot motion
    applied_torques = np.array([10, 5, 3, 2, 1, 0.5])
    
    accelerations = dynamics.forward_dynamics(
        joint_angles,
        joint_velocities,
        applied_torques,
        gravity,
        external_forces
    )

See Also
--------

- :doc:`../user_guide/Dynamics` - Complete dynamics guide
- :doc:`../api/dynamics` - Dynamics API reference
