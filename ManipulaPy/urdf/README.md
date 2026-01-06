# ManipulaPy Native URDF Parser

A modern, NumPy 2.0+ compatible URDF parser built specifically for ManipulaPy robotics workflows.

## Features

- **Zero external URDF dependencies** - Only NumPy required for core functionality
- **NumPy 2.0+ compatible** - Works with latest NumPy versions
- **Direct ManipulaPy integration** - Convert to `SerialManipulator` and `ManipulatorDynamics`
- **Optional visualization** - Lazy-loaded trimesh/pybullet backends
- **Xacro support** - Macro expansion for parameterized URDFs
- **Multi-robot scenes** - Manage multiple robots with world-frame kinematics
- **URDF modification** - Programmatic calibration and payload simulation

## Quick Start

```python
from ManipulaPy.urdf import URDF

# Load a URDF file
robot = URDF.load("path/to/robot.urdf")

# Basic properties
print(f"Robot: {robot.name}")
print(f"Links: {len(robot.links)}")
print(f"Joints: {len(robot.joints)}")
print(f"DOFs: {robot.num_dofs}")

# Forward kinematics
import numpy as np
config = np.zeros(robot.num_dofs)
fk = robot.link_fk(config, use_names=True)
print(f"End effector: {fk[robot.end_effector_link.name]}")

# Convert to ManipulaPy manipulator
manipulator = robot.to_serial_manipulator()
dynamics = robot.to_manipulator_dynamics()
```

## Loading URDFs

### Basic Loading

```python
from ManipulaPy.urdf import URDF

# Load from file
robot = URDF.load("robot.urdf")

# Load with mesh loading enabled
robot = URDF.load("robot.urdf", load_meshes=True)

# Load with custom mesh directory
robot = URDF.load("robot.urdf", load_meshes=True, mesh_dir="/path/to/meshes")
```

### Backend Selection

The parser supports multiple backends for compatibility:

```python
# Native builtin parser (default, NumPy 2.0 compatible)
robot = URDF.load("robot.urdf", backend="builtin")

# Legacy urchin backend (if installed)
robot = URDF.load("robot.urdf", backend="urchin")

# PyBullet backend
robot = URDF.load("robot.urdf", backend="pybullet")
```

### Xacro Support

Xacro files are automatically expanded:

```python
# Direct xacro loading
robot = URDF.load("robot.urdf.xacro")

# With arguments
robot = URDF.load("robot.urdf.xacro")  # Arguments parsed from file
```

## Forward Kinematics

### Single Configuration

```python
import numpy as np
from ManipulaPy.urdf import URDF

robot = URDF.load("robot.urdf")

# Joint configuration (radians for revolute, meters for prismatic)
config = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# Get all link transforms (Link -> 4x4 matrix)
fk = robot.link_fk(config)

# Get transforms with link names as keys
fk = robot.link_fk(config, use_names=True)
print(fk["end_effector_link"])

# Get specific links only
fk = robot.link_fk(config, links=["link1", "link2"])
```

### Batch Forward Kinematics

Optimized vectorized FK for multiple configurations:

```python
# Multiple configurations (N, num_dofs)
configs = np.random.uniform(-np.pi, np.pi, (1000, robot.num_dofs))

# Batch FK - returns {link_name: (N, 4, 4) array}
batch_fk = robot.link_fk_batch(configs)

# Access individual results
end_effector_transforms = batch_fk["end_effector_link"]  # (1000, 4, 4)
```

## ManipulaPy Integration

### SerialManipulator Conversion

```python
from ManipulaPy.urdf import URDF

robot = URDF.load("ur5.urdf")

# Convert to SerialManipulator
manipulator = robot.to_serial_manipulator()

# Use standard ManipulaPy methods
T_ee = manipulator.forward_kinematics(config)
J = manipulator.jacobian(config)
```

### Dynamics Conversion

```python
# Convert to ManipulatorDynamics
dynamics = robot.to_manipulator_dynamics()

# Compute dynamics
M = dynamics.mass_matrix(config)
C = dynamics.coriolis_matrix(config, velocity)
g = dynamics.gravity_vector(config)
```

### Extract Screw Axes

```python
# Get screw axis parameters directly
screws = robot.extract_screw_axes()

M = screws["M"]           # Home configuration matrix
S_list = screws["S_list"] # Space-frame screw axes
B_list = screws["B_list"] # Body-frame screw axes
G_list = screws["G_list"] # Spatial inertia matrices
```

## Joint Types

The parser supports all standard URDF joint types:

| Type | DOF | Description |
|------|-----|-------------|
| `fixed` | 0 | No motion |
| `revolute` | 1 | Rotation with limits |
| `continuous` | 1 | Unlimited rotation |
| `prismatic` | 1 | Linear translation |
| `planar` | 3 | 2D translation + rotation |
| `floating` | 6 | Full 6-DOF motion |

### Mimic Joints

Mimic joints automatically follow their master joint:

```python
robot = URDF.load("gripper.urdf")

# Only actuated (non-mimic) joints count in num_dofs
print(robot.num_dofs)  # e.g., 1 for symmetric gripper

# Mimic joints handled automatically in FK
config = np.array([0.02])  # Only master joint value needed
fk = robot.link_fk(config)
```

## Transmissions

Access transmission data for actuator information:

```python
robot = URDF.load("robot_with_transmissions.urdf")

# List all transmissions
for trans in robot.transmissions:
    print(f"Transmission: {trans.name}")
    print(f"  Type: {trans.type}")
    for joint in trans.joints:
        print(f"  Joint: {joint.name}")
    for actuator in trans.actuators:
        print(f"  Actuator: {actuator.name}, Reduction: {actuator.mechanical_reduction}")
```

## Multi-Robot Scenes

Manage multiple robots in a shared workspace:

```python
from ManipulaPy.urdf import URDF, Scene

# Load robots
robot1 = URDF.load("ur5.urdf")
robot2 = URDF.load("ur5.urdf")

# Create scene
scene = Scene("workcell")

# Add robots with base transforms
scene.add_robot("left_arm", robot1, base_xyz=[0, 0.5, 0])
scene.add_robot("right_arm", robot2, base_xyz=[0, -0.5, 0], base_rpy=[0, 0, np.pi])

# World-frame FK for all robots
configs = {
    "left_arm": np.zeros(6),
    "right_arm": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
}
world_fk = scene.world_link_fk(configs)

# Access transforms
left_ee = world_fk["left_arm"]["ee_link"]
right_ee = world_fk["right_arm"]["ee_link"]

# Get all collision geometry in world frame
collision_geoms = scene.get_all_collision_geometry(configs)
```

## URDF Modification

Programmatically modify URDFs for calibration and simulation:

```python
from ManipulaPy.urdf import URDF, URDFModifier

robot = URDF.load("robot.urdf")

# Create modifier (deep copies the URDF)
modifier = URDFModifier(robot)

# Joint modifications
modifier.set_joint_origin("joint1", xyz=[0, 0, 0.089159])
modifier.set_joint_limits("joint2", lower=-2.0, upper=2.0)
modifier.offset_joint_zero("joint3", offset=0.002)  # Calibration offset

# Link modifications
modifier.set_link_mass("link4", 2.5)
modifier.set_link_com("link4", [0, 0, 0.1])

# Payload simulation
modifier.add_payload("ee_link", mass=2.0, com=[0, 0, 0.05])

# Mass scaling (for uncertainty analysis)
modifier.scale_masses(1.1)  # +10% mass

# Get modified URDF
calibrated_robot = modifier.urdf

# Export to file
modifier.save("robot_calibrated.urdf")
```

### Batch Calibration

Apply calibration from file:

```python
from ManipulaPy.urdf import URDFModifier, load_calibration

# Load calibration data (YAML or JSON)
calibration = load_calibration("robot_calibration.yaml")
# Format: {"joint1": {"offset": 0.01}, "joint2": {"offset": -0.005, "lower": -1.5}}

# Apply to robot
modifier = URDFModifier(robot)
modifier.apply_calibration(calibration)
```

## Validation

Validate URDF structure:

```python
from ManipulaPy.urdf import URDF, validate_urdf

robot = URDF.load("robot.urdf")
result = validate_urdf(robot)

if result.valid:
    print("URDF is valid")
else:
    for issue in result.issues:
        print(f"{issue.severity.name}: {issue.message}")
```

Note: Invalid URDFs (cycles, multiple roots) are detected during loading and raise `ValueError`.

## Package Resolution

Resolve `package://` URIs from ROS packages:

```python
from ManipulaPy.urdf import PackageResolver

# Create resolver with custom package paths
resolver = PackageResolver(
    package_map={"my_robot": "/path/to/my_robot"},
    search_paths=["/opt/ros/noetic/share"]
)

# Resolve URI
abs_path = resolver.resolve("package://my_robot/urdf/robot.urdf")

# Factory method for URDF-relative resolution
resolver = PackageResolver.for_urdf("/path/to/robot.urdf")
```

You can also configure package resolution via environment variables:

```bash
# Search path(s) for package://<pkg>/... resolution
export MANIPULAPY_PACKAGE_PATH="/opt/ros/noetic/share:/path/to/your_ws/src"

# Explicit map of package name -> path (JSON file or JSON string)
export MANIPULAPY_PACKAGE_MAP="/path/to/package_map.json"
```

## Visualization

```python
from ManipulaPy.urdf import URDF

robot = URDF.load("robot.urdf", load_meshes=True)

# Show robot (opens viewer window)
robot.show()

# Show at specific configuration
robot.show(config=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
```

## API Reference

### URDF Class

| Property | Description |
|----------|-------------|
| `name` | Robot name |
| `links` | List of Link objects |
| `joints` | List of Joint objects |
| `num_dofs` | Number of actuated DOFs |
| `root_link` | Root link of kinematic tree |
| `end_effector_link` | Primary end effector link |
| `end_links` | All leaf links |
| `actuated_joints` | Non-fixed, non-mimic joints |
| `joint_limits` | List of (lower, upper) tuples |
| `transmissions` | List of Transmission objects |

| Method | Description |
|--------|-------------|
| `load(filename, ...)` | Load URDF from file |
| `link_fk(cfg, ...)` | Forward kinematics |
| `link_fk_batch(cfgs, ...)` | Batch forward kinematics |
| `extract_screw_axes()` | Get screw parameters |
| `to_serial_manipulator()` | Convert to SerialManipulator |
| `to_manipulator_dynamics()` | Convert to ManipulatorDynamics |
| `show(cfg)` | Visualize robot |

### Scene Class

| Method | Description |
|--------|-------------|
| `add_robot(name, urdf, ...)` | Add robot to scene |
| `remove_robot(name)` | Remove robot from scene |
| `world_link_fk(configs)` | World-frame FK for all robots |
| `world_end_effector_fk(configs)` | World-frame EE FK |
| `get_all_collision_geometry(configs)` | Get collision geometry |

### URDFModifier Class

| Method | Description |
|--------|-------------|
| `set_joint_origin(name, xyz, rpy)` | Modify joint origin |
| `set_joint_limits(name, lower, upper, ...)` | Modify joint limits |
| `offset_joint_zero(name, offset)` | Add zero offset |
| `set_link_mass(name, mass)` | Modify link mass |
| `set_link_com(name, com)` | Modify link COM |
| `add_payload(link, mass, ...)` | Add payload mass |
| `scale_masses(scale)` | Scale all masses |
| `apply_calibration(data)` | Apply calibration dict |
| `to_urdf_string()` | Export to XML string |
| `save(filename)` | Save to file |

## Migration from urchin

If migrating from the old urchin-based approach:

1. **Import change**: `from urchin import URDF` -> `from ManipulaPy.urdf import URDF`
2. **API is largely compatible** - `link_fk()`, `joint_limits`, etc. work similarly
3. **Backend fallback**: Use `backend="urchin"` if you need exact urchin behavior
4. **NumPy 2.0**: The native parser works with NumPy 2.0+; urchin does not

## Dependencies

**Required:**
- `numpy` (1.19.2+, including 2.0+)

**Optional:**
- `trimesh` - Mesh loading and visualization
- `pybullet` - Alternative backend and visualization
- `scipy` - Joint calibration offsets
- `pyyaml` - YAML calibration files
