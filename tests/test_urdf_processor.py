#!/usr/bin/env python3

import unittest
import os
from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as xarm_urdf_file


class TestURDFProcessor(unittest.TestCase):
    def setUp(self):
        self.urdf_path = xarm_urdf_file

    def test_urdf_load(self):
        """Test loading a URDF file."""
        # Check that the file exists
        self.assertTrue(os.path.isfile(self.urdf_path))

        # Try to load the URDF
        try:
            processor = URDFToSerialManipulator(self.urdf_path)
            self.assertTrue(hasattr(processor, "robot"))
            self.assertTrue(hasattr(processor, "robot_data"))
        except Exception as e:
            self.fail(f"Failed to load URDF: {e}")

    def test_serial_manipulator_creation(self):
        """Test creation of SerialManipulator from URDF."""
        processor = URDFToSerialManipulator(self.urdf_path)

        # Check if the SerialManipulator was created
        self.assertIsNotNone(processor.serial_manipulator)

        # Check basic properties
        self.assertTrue(hasattr(processor.serial_manipulator, "M_list"))
        self.assertTrue(hasattr(processor.serial_manipulator, "S_list"))
        self.assertTrue(hasattr(processor.serial_manipulator, "B_list"))

    def test_dynamics_creation(self):
        """Test creation of ManipulatorDynamics from URDF."""
        processor = URDFToSerialManipulator(self.urdf_path)

        # Check if the ManipulatorDynamics was created
        self.assertIsNotNone(processor.dynamics)

        # Check basic properties
        self.assertTrue(hasattr(processor.dynamics, "Glist"))
        self.assertTrue(hasattr(processor.dynamics, "M_list"))
        self.assertTrue(hasattr(processor.dynamics, "S_list"))
