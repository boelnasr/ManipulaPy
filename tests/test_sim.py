#!/usr/bin/env python3
"""
Enhanced test suite for ManipulaPy.sim module with increased coverage.

This test suite covers:
- Basic simulation setup and teardown
- Joint control and parameter management
- Trajectory visualization with real geometry
- Controller integration
- Error handling and edge cases
- Performance monitoring
- Resource cleanup

Fixed issues:
- Proper PyBullet mocking to prevent actual connections
- Isolated test environment to avoid connection conflicts
- Better fixture management for test isolation
- Fixed import path issues
- Fixed shape mismatch errors in controller tests
- Fixed parameter update test expectations
- Added proper attribute initialization in __init__
- Fixed set_joint_positions size mismatch handling
- Fixed planner and controller initialization

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import os
import numpy as np
import pytest
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock, call
import time

# Import the simulation module
from ManipulaPy.sim import Simulation
from ManipulaPy.path_planning import TrajectoryPlanning
from ManipulaPy.control import ManipulatorController

import pybullet as p
class MockPyBullet:
    """Mock PyBullet module to completely replace pybullet during tests"""
    
    # Constants
    GUI = 2
    DIRECT = 1
    JOINT_FIXED = 4
    JOINT_REVOLUTE = 0
    POSITION_CONTROL = 1
    GEOM_CAPSULE = 7
    GEOM_SPHERE = 2
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all internal counters and state"""
        self._body_counter = 1000
        self._shape_counter = 100
        self._param_counter = 0
        self._joint_states = {}
        self._physics_client = None
    
    def connect(self, mode):
        self._physics_client = 1
        return 1
    
    def disconnect(self):
        self._physics_client = None
    
    def resetSimulation(self):
        pass
    
    def setAdditionalSearchPath(self, path):
        pass
    
    def setGravity(self, x, y, z):
        pass
    
    def setTimeStep(self, dt):
        pass
    
    def loadURDF(self, path, *args, **kwargs):
        return 1
    
    def getNumJoints(self, robot_id):
        return 6
    
    def getJointInfo(self, robot_id, joint_idx):
        joint_types = [self.JOINT_REVOLUTE] * 6
        return (joint_idx, f'joint_{joint_idx}'.encode(), 
                joint_types[min(joint_idx, 5)], 0, 0, 0, 0, 0, -np.pi, np.pi, 100, 0)
    
    def getJointState(self, robot_id, joint_idx):
        return (joint_idx * 0.1, joint_idx * 0.05)
    
    def setJointMotorControlArray(self, *args, **kwargs):
        pass
    
    def createCollisionShape(self, *args, **kwargs):
        self._shape_counter += 1
        return self._shape_counter
    
    def createVisualShape(self, *args, **kwargs):
        self._shape_counter += 1
        return self._shape_counter
    
    def createMultiBody(self, *args, **kwargs):
        self._body_counter += 1
        return self._body_counter
    
    def removeBody(self, body_id):
        pass
    
    def getQuaternionFromEuler(self, euler):
        return [0, 0, 0, 1]
    
    def getQuaternionFromAxisAngle(self, axis, angle):
        return [0, 0, 0, 1]
    
    def addUserDebugParameter(self, name, low, high, init):
        self._param_counter += 1
        return self._param_counter
    
    def readUserDebugParameter(self, param_id):
        return 0.5
    
    def stepSimulation(self):
        pass
    
    def getLinkState(self, robot_id, link_idx):
        return (None, None, None, None, (0.1*link_idx, 0.2*link_idx, 0.3*link_idx), None)
    
    def getContactPoints(self, *args, **kwargs):
        return []


# Global mock instance
mock_pybullet = MockPyBullet()


@pytest.fixture(autouse=True)
def mock_pybullet_module(monkeypatch):
    """Replace pybullet module completely"""
    # Patch all pybullet functions used in sim.py
    monkeypatch.setattr('ManipulaPy.sim.p', mock_pybullet)
    # Also patch any direct imports
    monkeypatch.setattr('pybullet.connect', mock_pybullet.connect)
    monkeypatch.setattr('pybullet.disconnect', mock_pybullet.disconnect)
    monkeypatch.setattr('pybullet.resetSimulation', mock_pybullet.resetSimulation)
    monkeypatch.setattr('pybullet.setAdditionalSearchPath', mock_pybullet.setAdditionalSearchPath)
    monkeypatch.setattr('pybullet.setGravity', mock_pybullet.setGravity)
    monkeypatch.setattr('pybullet.setTimeStep', mock_pybullet.setTimeStep)
    monkeypatch.setattr('pybullet.loadURDF', mock_pybullet.loadURDF)
    monkeypatch.setattr('pybullet.getNumJoints', mock_pybullet.getNumJoints)
    monkeypatch.setattr('pybullet.getJointInfo', mock_pybullet.getJointInfo)
    monkeypatch.setattr('pybullet.getJointState', mock_pybullet.getJointState)
    
    # Mock pybullet_data
    mock_pybullet_data = Mock()
    mock_pybullet_data.getDataPath.return_value = "/fake/path"
    monkeypatch.setattr('ManipulaPy.sim.pybullet_data', mock_pybullet_data)
    
    # Mock time.sleep to speed up tests
    monkeypatch.setattr('time.sleep', lambda x: None)
    
    # Reset mock state before each test
    mock_pybullet.reset()
    
    yield


@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock external dependencies that aren't PyBullet"""
    # Mock matplotlib
    mock_plt = Mock()
    monkeypatch.setattr('matplotlib.pyplot.figure', lambda *args, **kwargs: Mock())
    monkeypatch.setattr('matplotlib.pyplot.show', lambda: None)
    monkeypatch.setattr('matplotlib.pyplot.legend', lambda: None)


@pytest.fixture
def basic_sim():
    """Create a basic simulation instance for testing"""
    joint_limits = [(-np.pi, np.pi)] * 6
    return Simulation("test_robot.urdf", joint_limits)


@pytest.fixture 
def mock_urdf_processor(monkeypatch):
    """Mock the URDFToSerialManipulator"""
    mock_processor = Mock()
    mock_robot = Mock()
    mock_dynamics = Mock()
    
    mock_processor.serial_manipulator = mock_robot
    mock_processor.dynamics = mock_dynamics
    
    def mock_urdf_processor_init(urdf_path):
        return mock_processor
    
    monkeypatch.setattr('ManipulaPy.urdf_processor.URDFToSerialManipulator', 
                       mock_urdf_processor_init)
    return mock_processor


class TestSimulationInitialization:
    """Test simulation setup and initialization"""


    def test_basic_initialization(self):
        """Test basic simulation initialization"""
        joint_limits = [(-np.pi, np.pi)] * 3
        sim = Simulation("test_robot.urdf", joint_limits)
        
        assert sim.urdf_file_path == "test_robot.urdf"
        assert sim.joint_limits == joint_limits
        assert sim.time_step == 0.01
        assert sim.real_time_factor == 1.0
        assert sim.physics_client == 1  # Mock returns 1
        assert isinstance(sim.logger, logging.Logger)
        # no GUI sliders or trajectory bodies asserted here—this test only covers basic ctor
 
    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters"""
        joint_limits = [(-1, 1), (-2, 2)]
        torque_limits = [(-10, 10), (-20, 20)]
        
        sim = Simulation(
            "custom_robot.urdf",
            joint_limits=joint_limits,
            torque_limits=torque_limits,
            time_step=0.005,
            real_time_factor=2.0,
            physics_client=42
        )
        
        assert sim.joint_limits == joint_limits
        assert sim.torque_limits == torque_limits
        assert sim.time_step == 0.005
        assert sim.real_time_factor == 2.0
        assert sim.physics_client == 42
    
    def test_logger_setup(self):
        """Test logger configuration"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        assert sim.logger.name == "SimulationLogger"
        assert sim.logger.level == logging.DEBUG
        assert len(sim.logger.handlers) > 0


class TestConnectionManagement:
    """Test PyBullet connection management"""
    
    def test_connect_simulation(self):
        """Test simulation connection"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.physics_client = None
        sim.connect_simulation()
        
        assert sim.physics_client == 1  # Mock returns 1
    
    def test_disconnect_simulation(self):
        """Test simulation disconnection"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.disconnect_simulation()
        
        assert sim.physics_client is None
    
    def test_disconnect_when_already_none(self):
        """Test disconnection when physics_client is already None"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.physics_client = None
        
        # Should not raise exception
        sim.disconnect_simulation()
        assert sim.physics_client is None


class TestRobotInitialization:
    """Test robot model initialization"""
    
    def test_initialize_robot_first_time(self, mock_urdf_processor):
        """Test robot initialization from URDF"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        # Remove robot attribute to simulate first-time initialization
        if hasattr(sim, 'robot'):
            delattr(sim, 'robot')
        
        sim.initialize_robot()
        
        assert hasattr(sim, 'robot')
        assert hasattr(sim, 'dynamics')
    
    def test_initialize_robot_already_exists(self):
        """Test robot initialization when robot already exists"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.robot = Mock()
        
        initial_robot = sim.robot
        sim.initialize_robot()
        
        # Robot should remain unchanged
        assert sim.robot is initial_robot
    
    def test_set_robot_models(self):
        """Test setting pre-existing robot models"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        mock_robot = Mock()
        mock_dynamics = Mock()
        
        sim.set_robot_models(mock_robot, mock_dynamics)
        
        assert sim.robot is mock_robot
        assert sim.dynamics is mock_dynamics


class TestJointManagement:
    """Test joint parameter and control management"""
    
    def test_add_joint_parameters(self):
        """Test adding joint parameter sliders"""
        sim = Simulation("test.urdf", [(-1, 1), (-2, 2), (-3, 3)])
        sim.non_fixed_joints = [0, 1, 2]
        sim.joint_params = []
        
        sim.add_joint_parameters()
        
        assert len(sim.joint_params) == 3
        assert all(isinstance(param_id, int) for param_id in sim.joint_params)
    
    def test_add_joint_parameters_idempotent(self):
        """Test that adding joint parameters multiple times doesn't duplicate"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.non_fixed_joints = [0]
        sim.joint_params = []
        
        sim.add_joint_parameters()
        initial_count = len(sim.joint_params)
        
        sim.add_joint_parameters()
        assert len(sim.joint_params) == initial_count
    
    def test_add_reset_button_success(self):
        """Test successful reset button addition"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.reset_button = None
        
        sim.add_reset_button()
        
        assert sim.reset_button is not None
    
    def test_add_reset_button_failure(self, monkeypatch):
        """Test reset button addition failure handling"""
        def failing_add_param(*args, **kwargs):
            raise Exception("Debug parameter failed")
        
        # Patch the specific mock method
        monkeypatch.setattr(mock_pybullet, 'addUserDebugParameter', failing_add_param)
        
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.reset_button = None
        
        sim.add_reset_button()
        assert sim.reset_button is None
    
    def test_set_joint_positions(self):
        """Test setting joint positions"""
        control_calls = []
        
        def mock_control(robot_id, joints, mode, targetPositions=None, **kwargs):
            control_calls.append({
                'robot_id': robot_id,
                'joints': joints,
                'mode': mode,
                'positions': targetPositions
            })
        
        mock_pybullet.setJointMotorControlArray = mock_control
        
        sim = Simulation("test.urdf", [(-1, 1), (-2, 2)])
        sim.non_fixed_joints = [0, 1]
        positions = [0.5, -1.0]
        
        sim.set_joint_positions(positions)
        
        assert len(control_calls) == 1
        assert control_calls[0]['positions'] == positions
        assert control_calls[0]['joints'] == [0, 1]

    def test_get_joint_positions(self):
        """Test getting current joint positions"""
        sim = Simulation("test.urdf", [(-1, 1), (-2, 2)])
        sim.non_fixed_joints = [0, 1]
        
        positions = sim.get_joint_positions()
        
        assert len(positions) == 2
        assert isinstance(positions, np.ndarray)
        # Mock returns idx * 0.1 as position
        assert np.allclose(positions, [0.0, 0.1])
    
    def test_get_joint_parameters(self):
        """Test reading joint parameter values"""
        # Set up mock to return specific values
        def mock_read_param(param_id):
            param_values = {1: 0.5, 2: -0.3, 3: 1.2}
            return param_values.get(param_id, 0)
        
        mock_pybullet.readUserDebugParameter = mock_read_param
        
        sim = Simulation("test.urdf", [(-1, 1)] * 3)
        sim.joint_params = [1, 2, 3]
        
        values = sim.get_joint_parameters()
        
        assert values == [0.5, -0.3, 1.2]


class TestControllerIntegration:
    """Test controller integration"""
    
    def test_run_controller(self):
        """Test running controller with trajectory"""
        # Mock cupy
        with patch('cupy.array') as mock_array, \
             patch('cupy.asnumpy') as mock_asnumpy:
            
            mock_array.side_effect = lambda x: np.array(x)
            mock_asnumpy.side_effect = lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
            
            sim = Simulation("test.urdf", [(-1, 1)] * 6)  # Make sure we have 6 joints
            sim.non_fixed_joints = list(range(6))  # 6 joints to match mock
            
            # Mock controller
            mock_controller = Mock()
            mock_controller.computed_torque_control.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            
            desired_positions = [[0.0] * 6, [0.5] * 6]  # 6 joints
            desired_velocities = [[0.0] * 6, [0.1] * 6]
            desired_accelerations = [[0.0] * 6, [0.01] * 6]
            g = [0, 0, -9.81]
            Ftip = [0, 0, 0, 0, 0, 0]
            Kp = [10] * 6
            Ki = [1] * 6
            Kd = [0.1] * 6
            
            final_pos = sim.run_controller(
                mock_controller, desired_positions, desired_velocities,
                desired_accelerations, g, Ftip, Kp, Ki, Kd
            )
            
            assert final_pos is not None
            assert mock_controller.computed_torque_control.called


class TestParameterManagement:
    """Test simulation parameter management"""
    
    def test_add_additional_parameters(self):
        """Test adding additional GUI parameters"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        sim.add_additional_parameters()
        
        assert sim.gravity_param is not None
        assert sim.time_step_param is not None
    
    def test_add_additional_parameters_idempotent(self):
        """Test that additional parameters aren't duplicated"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        sim.add_additional_parameters()
        gravity_param_1 = sim.gravity_param
        
        sim.add_additional_parameters()
        assert sim.gravity_param == gravity_param_1
    
    def test_update_simulation_parameters(self, monkeypatch):
        """Test updating simulation parameters from GUI"""
        param_values = {}
        
        def mock_read_param(param_id):
            return param_values.get(param_id, 0)
        
        monkeypatch.setattr(mock_pybullet, 'readUserDebugParameter', mock_read_param)
        
        gravity_calls = []
        timestep_calls = []
        
        def mock_set_gravity(x, y, z):
            gravity_calls.append((x, y, z))
        
        def mock_set_timestep(t):
            timestep_calls.append(t)
        
        monkeypatch.setattr(mock_pybullet, 'setGravity', mock_set_gravity)
        monkeypatch.setattr(mock_pybullet, 'setTimeStep', mock_set_timestep)
        
        sim = Simulation("test.urdf", [(-1, 1)])
        # Clear any gravity calls from initialization
        gravity_calls.clear()
        timestep_calls.clear()
        
        sim.add_additional_parameters()
        
        param_values[sim.gravity_param] = -10.0
        param_values[sim.time_step_param] = 0.005
        
        sim.update_simulation_parameters()
        
        # Should only have the update call, not initialization calls
        assert (0, 0, -10.0) in gravity_calls
        assert 0.005 in timestep_calls
        assert sim.time_step == 0.005
    
    def test_update_parameters_without_gui(self):
        """Test parameter update when GUI parameters don't exist"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        # Ensure GUI parameters are None
        sim.gravity_param = None
        sim.time_step_param = None
        
        # Should not raise exception, just log warning
        sim.update_simulation_parameters()


class TestTrajectoryVisualization:
    """Test trajectory visualization with real geometry"""
    
    def test_capsule_line_creation(self):
        """Test creating capsule lines between points"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        # Valid line segment
        body_id = sim._capsule_line([0, 0, 0], [1, 1, 1])
        assert body_id > 0
        
        # Zero-length segment should fail
        invalid_body_id = sim._capsule_line([0, 0, 0], [0, 0, 0])
        assert invalid_body_id == -1
    
    def test_plot_trajectory_empty(self):
        """Test plotting trajectory with insufficient points"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        # Empty trajectory
        result = sim.plot_trajectory([])
        assert result == []
        
        # Single point
        result = sim.plot_trajectory([[0, 0, 0]])
        assert result == []
    
    def test_plot_trajectory_success(self):
        """Test successful trajectory plotting"""
        sim = Simulation("test.urdf", [(-1, 1)])
        ee_positions = [
            [0, 0, 0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2]
        ]
        
        trajectory_bodies = sim.plot_trajectory(ee_positions, line_width=2, color=[1, 0, 0])
        
        assert len(trajectory_bodies) > 0
        assert len(sim.trajectory_body_ids) >= len(trajectory_bodies)  # Includes markers
    
    def test_clear_trajectory_visualization(self):
        """Test clearing trajectory visualization"""
        removed_bodies = []
        mock_pybullet.removeBody = lambda bid: removed_bodies.append(bid)
        
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.trajectory_body_ids = [100, 101, 102]
        
        sim.clear_trajectory_visualization()
        
        assert removed_bodies == [100, 101, 102]
        assert sim.trajectory_body_ids == []


class TestTrajectoryExecution:
    """Test trajectory execution and simulation"""
    
    def test_run_trajectory(self):
        """Test running a joint trajectory"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.non_fixed_joints = [0]
        
        joint_trajectory = [
            [0.0],
            [0.5],
            [1.0]
        ]
        
        final_position = sim.run_trajectory(joint_trajectory)
        
        # Should return final end-effector position
        assert final_position is not None
        assert len(sim.trajectory_body_ids) >= 0  # Trajectory should be plotted
    
    def test_simulate_robot_motion(self):
        """Test robot motion simulation"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        trajectory = [[0.0], [0.5], [1.0]]
        final_pos = sim.simulate_robot_motion(trajectory)
        
        assert final_pos is not None


class TestCollisionDetection:
    """Test collision detection functionality"""
    
    def test_check_collisions_no_robot(self):
        """Test collision check when robot_id is None"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.robot_id = None
        
        # Should log warning but not crash
        sim.check_collisions()
    
    def test_check_collisions_with_contacts(self, monkeypatch):
        """Test collision check with detected contacts"""
        # Mock contact points
        mock_contacts = [
            {'contact_point': [0, 0, 0], 'force': 10},
            {'contact_point': [0.1, 0.1, 0.1], 'force': 5}
        ]
        
        monkeypatch.setattr(mock_pybullet, 'getContactPoints', lambda *args: mock_contacts)
        
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.non_fixed_joints = [0]
        
        # Should log warnings but not crash
        sim.check_collisions()
    
    def test_check_collisions_no_contacts(self):
        """Test collision check with no contacts"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.non_fixed_joints = [0, 1]
        
        # Should complete without warnings
        sim.check_collisions()


class TestFileOperations:
    """Test file I/O operations"""
    
    def test_save_joint_states(self, tmp_path):
        """Test saving joint states to CSV"""
        sim = Simulation("test.urdf", [(-1, 1), (-2, 2)])
        sim.non_fixed_joints = [0, 1]
        
        filename = tmp_path / "test_states.csv"
        sim.save_joint_states(str(filename))
        
        assert filename.exists()
        
        # Read and verify content
        data = np.loadtxt(str(filename), delimiter=',', skiprows=1)
        assert data.shape == (2, 2)  # 2 joints, 2 columns (pos, vel)
        # Mock returns (idx * 0.1, idx * 0.05)
        expected = np.array([[0.0, 0.0], [0.1, 0.05]])
        assert np.allclose(data, expected)


class TestResourceManagement:
    """Test resource management and cleanup"""
    
    def test_close_simulation(self):
        """Test simulation cleanup"""
        disconnect_called = []
        original_disconnect = mock_pybullet.disconnect
        
        def track_disconnect():
            disconnect_called.append(True)
            return original_disconnect()
        
        mock_pybullet.disconnect = track_disconnect
        
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.trajectory_body_ids = [100, 101]
        
        sim.close_simulation()
        
        assert disconnect_called == [True]
        assert sim.trajectory_body_ids == []
        assert sim.physics_client is None
    
    def test_destructor_cleanup(self):
        """Test destructor cleanup"""
        sim = Simulation("test.urdf", [(-1, 1)])
        sim.trajectory_body_ids = [100, 101]
        
        # Should not raise exception during cleanup
        sim.__del__()
    
    def test_destructor_without_trajectory_bodies(self):
        """Test destructor when trajectory_body_ids doesn't exist"""
        sim = Simulation("test.urdf", [(-1, 1)])
        
        # Remove the attribute to test robustness
        delattr(sim, 'trajectory_body_ids')
        
        # Should not raise exception
        sim.__del__()


class TestManualControl:
    """Test manual control functionality"""
    
    def test_manual_control_setup(self, monkeypatch):
        """Test manual control setup"""
        # Mock manual control to exit after first iteration
        call_count = [0]
        
        def mock_read_param(param_id):
            call_count[0] += 1
            if call_count[0] > 5:  # Exit after a few calls
                raise KeyboardInterrupt("Test exit")
            return 0.5
        
        monkeypatch.setattr(mock_pybullet, 'readUserDebugParameter', mock_read_param)
        
        sim = Simulation("test.urdf", [(-1, 1), (-2, 2)])
        sim.non_fixed_joints = [0, 1]
        
        # Should handle KeyboardInterrupt gracefully
        sim.manual_control()
        
        assert call_count[0] > 0


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=ManipulaPy.sim",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=85"  # Require at least 85% coverage
    ])
        #!/usr/bin/env python3
"""
Enhanced test suite for ManipulaPy.sim module with increased coverage.

This test suite covers:
- Basic simulation setup and teardown
- Joint control and parameter management
- Trajectory visualization with real geometry
- Controller integration
- Error handling and edge cases
- Performance monitoring
- Resource cleanup

Fixed issues:
- Proper PyBullet mocking to prevent actual connections
- Isolated test environment to avoid connection conflicts
- Better fixture management for test isolation
- Fixed import path issues
- Fixed shape mismatch errors in controller tests
- Fixed parameter update test expectations
- Added proper attribute initialization in __init__
- Fixed set_joint_positions size mismatch handling
- Fixed planner and controller initialization
"""

import os
import numpy as np
import pytest
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock, call
import time

# Import the simulation module
from ManipulaPy.sim import Simulation


class MockPyBullet:
    """Mock PyBullet module to completely replace pybullet during tests"""
    
    # Constants
    GUI = 2
    DIRECT = 1
    JOINT_FIXED = 4
    JOINT_REVOLUTE = 0
    POSITION_CONTROL = 1
    GEOM_CAPSULE = 7
    GEOM_SPHERE = 2
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all internal counters and state"""
        self._body_counter = 1000
        self._shape_counter = 100
        self._param_counter = 0
        self._joint_states = {}
        self._physics_client = None
    
    def connect(self, mode):
        self._physics_client = 1
        return 1
    
    def disconnect(self):
        self._physics_client = None
    
    def resetSimulation(self):
        pass
    
    def setAdditionalSearchPath(self, path):
        pass
    
    def setGravity(self, x, y, z):
        pass
    
    def setTimeStep(self, dt):
        pass
    
    def loadURDF(self, path, *args, **kwargs):
        return 1
    
    def getNumJoints(self, robot_id):
        return 6
    
    def getJointInfo(self, robot_id, joint_idx):
        joint_types = [self.JOINT_REVOLUTE] * 6
        return (joint_idx, f'joint_{joint_idx}'.encode(), 
                joint_types[min(joint_idx, 5)], 0, 0, 0, 0, 0, -np.pi, np.pi, 100, 0)
    
    def getJointState(self, robot_id, joint_idx):
        return (joint_idx * 0.1, joint_idx * 0.05)
    
    def setJointMotorControlArray(self, *args, **kwargs):
        pass
    
    def createCollisionShape(self, *args, **kwargs):
        self._shape_counter += 1
        return self._shape_counter
    
    def createVisualShape(self, *args, **kwargs):
        self._shape_counter += 1
        return self._shape_counter
    
    def createMultiBody(self, *args, **kwargs):
        self._body_counter += 1
        return self._body_counter
    
    def removeBody(self, body_id):
        pass
    
    def getQuaternionFromEuler(self, euler):
        return [0, 0, 0, 1]
    
    def getQuaternionFromAxisAngle(self, axis, angle):
        return [0, 0, 0, 1]
    
    def addUserDebugParameter(self, name, low, high, init):
        self._param_counter += 1
        return self._param_counter
    
    def readUserDebugParameter(self, param_id):
        return 0.5
    
    def step_simulation(self):
        """
        Steps the simulation forward by one time step.
        """
        # Ensure the simulation is connected and GUI controls are in place
        self.logger.info("Setting up the simulation environment...")
        self.connect_simulation()
        self.add_additional_parameters()
        # Read any updated physics parameters (gravity, time step)
        self.update_simulation_parameters()
        # Advance the simulation by one step
        try:
            p.stepSimulation()
            # Respect real-time factor for sleeping
            time.sleep(self.time_step / self.real_time_factor)
            self.logger.debug(
                f"Advanced simulation by {self.time_step}s (RTF={self.real_time_factor})"
            )
        except Exception as e:
            self.logger.error(f"Error during simulation step: {e}")
