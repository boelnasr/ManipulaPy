# ManipulaPy/ManipulaPy_data/xarm/__init__.py

import os

# Define the path to the XArm6 URDF file
urdf_file = os.path.join(os.path.dirname(__file__), 'xarm6_robot.urdf')

# Make the path available as a module attribute
__all__ = ['urdf_file']
