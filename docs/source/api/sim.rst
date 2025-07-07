Simulation
==========

.. currentmodule:: ManipulaPy.sim

The simulation module provides comprehensive PyBullet-based simulation capabilities for robotic manipulators.

.. automodule:: ManipulaPy.sim
    :members:
    :undoc-members:
    :show-inheritance:

Classes
-------

.. autoclass:: Simulation
    :members:
    :undoc-members:
    :show-inheritance:

Examples
--------

Basic simulation setup:

.. code-block:: python

    from ManipulaPy.sim import Simulation
    
    # Initialize simulation
    sim = Simulation(
        urdf_file_path="robot.urdf",
        joint_limits=[(-3.14, 3.14)] * 6,
        torque_limits=[(-10, 10)] * 6
    )
    
    # Setup and run
    sim.setup_simulation()
    sim.initialize_robot()
    sim.manual_control()

See Also
--------

- :doc:`../user_guide/Simulation` - Detailed simulation guide
- :doc:`control` - Control system documentation
- :doc:`path_planning` - Trajectory planning documentation
