# ManipulaPy URDF Parser Troubleshooting Guide

This guide helps diagnose and resolve common issues when using the native URDF parser.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [URDF Loading Errors](#urdf-loading-errors)
3. [Forward Kinematics Issues](#forward-kinematics-issues)
4. [Dynamics/Inertia Issues](#dynamicsinertia-issues)
5. [Conversion Issues](#conversion-issues)
6. [Performance Issues](#performance-issues)
7. [NumPy Compatibility](#numpy-compatibility)
8. [Backend Selection](#backend-selection)

---

## Installation Issues

### NumPy Version Conflicts

**Symptom:** Import errors mentioning NumPy version incompatibility.

**Solution:**
```python
# Check NumPy version
import numpy as np
print(np.__version__)

# The native parser supports NumPy 1.19.2+ and NumPy 2.0+
# If using NumPy 2.0+, ensure scipy is also updated:
# pip install 'scipy>=1.14'
```

### Missing Optional Dependencies

**Symptom:** `ImportError` when using visualization or mesh loading.

**Solution:**
```bash
# For visualization
pip install trimesh

# For mesh loading (DAE files)
pip install trimesh[easy]

# For xacro support
pip install xacro

# For ROS package resolution
pip install rospkg
```

---

## URDF Loading Errors

### File Not Found

**Symptom:** `FileNotFoundError` when loading URDF.

**Solution:**
```python
from pathlib import Path

# Use absolute path
urdf_path = Path("/absolute/path/to/robot.urdf")
robot = URDF.load(urdf_path)

# Or resolve relative to script
urdf_path = Path(__file__).parent / "robot.urdf"
robot = URDF.load(urdf_path)
```

### Invalid XML

**Symptom:** `xml.etree.ElementTree.ParseError`

**Solution:**
```python
# Validate XML first
import xml.etree.ElementTree as ET

try:
    tree = ET.parse("robot.urdf")
except ET.ParseError as e:
    print(f"XML error at line {e.position[0]}: {e}")
```

Common XML issues:
- Missing closing tags
- Invalid characters (use `&amp;` for `&`, `&lt;` for `<`)
- Encoding issues (use UTF-8)

### Cyclic Kinematic Chain

**Symptom:** `ValueError: URDF contains cyclic kinematic chain`

**Cause:** A link is both an ancestor and descendant of another link.

**Solution:**
```python
# The parser detects cycles during loading
# Check your URDF for joints that create loops

# Debug: List parent-child relationships
for joint in robot.joints:
    print(f"{joint.parent} -> {joint.child}")
```

### Multiple Root Links

**Symptom:** `ValueError: URDF has multiple root links`

**Cause:** More than one link has no parent joint.

**Solution:**
```python
# Find all links without parents
all_links = {link.name for link in robot.links}
child_links = {joint.child for joint in robot.joints}
root_links = all_links - child_links
print(f"Root links: {root_links}")  # Should be exactly 1
```

### Missing Mesh Files

**Symptom:** Warning about missing mesh files during loading.

**Solution:**
```python
# Option 1: Provide mesh directory
robot = URDF.load("robot.urdf", mesh_dir="/path/to/meshes")

# Option 2: Disable mesh loading
robot = URDF.load("robot.urdf", load_meshes=False)

# Option 3: Use package resolver
from ManipulaPy.urdf import PackageResolver

resolver = PackageResolver(
    package_map={"my_robot": "/path/to/my_robot_package"}
)
robot = URDF.load("robot.urdf", package_resolver=resolver)
```

### package:// URI Resolution

**Symptom:** `Could not resolve package://...`

**Solution:**
```python
from ManipulaPy.urdf import URDF, PackageResolver

# Manual package mapping
resolver = PackageResolver(
    package_map={
        "ur_description": "/opt/ros/noetic/share/ur_description",
        "my_robot": "/home/user/catkin_ws/src/my_robot",
    }
)
robot = URDF.load("robot.urdf", package_resolver=resolver)

# Or set ROS package path
import os
os.environ["ROS_PACKAGE_PATH"] = "/opt/ros/noetic/share:/home/user/catkin_ws/src"
```

---

## Forward Kinematics Issues

### Wrong Number of DOFs

**Symptom:** Configuration array has wrong size.

**Solution:**
```python
import numpy as np

robot = URDF.load("robot.urdf")
print(f"Expected DOFs: {robot.num_dofs}")
print(f"Actuated joints: {[j.name for j in robot.actuated_joints]}")

# Create correct configuration
cfg = np.zeros(robot.num_dofs)
```

### Unexpected FK Results

**Symptom:** FK returns unexpected transforms.

**Debug steps:**
```python
import numpy as np

robot = URDF.load("robot.urdf")

# 1. Check joint order
for i, joint in enumerate(robot.actuated_joints):
    print(f"DOF {i}: {joint.name} ({joint.joint_type.name})")

# 2. Check joint axes
for joint in robot.actuated_joints:
    print(f"{joint.name}: axis = {joint.axis}")

# 3. Verify at zero config
cfg = np.zeros(robot.num_dofs)
fk = robot.link_fk(cfg, use_names=True)

for link_name, T in fk.items():
    pos = T[:3, 3]
    print(f"{link_name}: position = {pos}")

# 4. Compare with PyBullet reference
import pybullet as p
p.connect(p.DIRECT)
pb_robot = p.loadURDF("robot.urdf")
# ... compare results
```

### Mimic Joints Not Working

**Symptom:** Mimic joints don't follow master joint.

**Solution:**
```python
# Check mimic joint configuration
for joint in robot.joints:
    if joint.mimic is not None:
        print(f"{joint.name} mimics {joint.mimic.joint}")
        print(f"  multiplier: {joint.mimic.multiplier}")
        print(f"  offset: {joint.mimic.offset}")

# Mimic joints are handled automatically in FK
# Only provide values for non-mimic actuated joints
print(f"Actuated (non-mimic) joints: {robot.num_dofs}")
```

---

## Dynamics/Inertia Issues

### Mass Matrix Not Positive Definite

**Symptom:** Eigenvalues of mass matrix include negatives.

**Cause:** Usually due to incorrect inertial parameters or near-singular configurations.

**Solution:**
```python
import numpy as np

robot = URDF.load("robot.urdf")
dynamics = robot.to_manipulator_dynamics()

# Check at safe configuration (away from singularities)
cfg = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])[:robot.num_dofs]
M = dynamics.mass_matrix(cfg)

eigenvalues = np.linalg.eigvalsh(M)
print(f"Mass matrix eigenvalues: {eigenvalues}")

if np.any(eigenvalues <= 0):
    print("Warning: Mass matrix not positive definite")
    print("Check inertial parameters in URDF")
```

### Missing Inertial Properties

**Symptom:** Warning about missing inertial properties.

**Solution:**
```python
# Check which links are missing inertial
for link in robot.links:
    if link.inertial is None:
        print(f"Warning: {link.name} has no inertial properties")

# The parser uses defaults (mass=1.0, I=eye(3)) for missing inertials
# For accurate dynamics, add <inertial> to all moving links
```

### Incorrect Inertia Values

**Common issues:**
```xml
<!-- Wrong: Inertia in wrong units -->
<inertia ixx="0.1" ... />  <!-- Should be kg*m^2 -->

<!-- Wrong: Non-physical inertia (negative or violating triangle inequality) -->
<inertia ixx="1" iyy="1" izz="10" ... />  <!-- Izz should be <= Ixx + Iyy -->

<!-- Correct example for a 1kg, 10cm cube -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.00167" ixy="0" ixz="0"
           iyy="0.00167" iyz="0" izz="0.00167"/>
</inertial>
```

---

## Conversion Issues

### SerialManipulator Conversion Fails

**Symptom:** `to_serial_manipulator()` raises error.

**Solution:**
```python
# Check robot is a serial chain
if len(robot.end_links) > 1:
    print("Warning: Branched robot detected")
    print(f"End links: {[l.name for l in robot.end_links]}")

    # Specify which branch to use
    manipulator = robot.to_serial_manipulator(tip_link="specific_ee_link")
```

### FK Mismatch After Conversion

**Symptom:** SerialManipulator FK differs from URDF FK.

**Debug:**
```python
import numpy as np

robot = URDF.load("robot.urdf")
manipulator = robot.to_serial_manipulator()

cfg = np.random.uniform(-1, 1, robot.num_dofs)

# URDF FK
urdf_fk = robot.link_fk(cfg, use_names=True)
urdf_ee = urdf_fk[robot.end_effector_link.name]

# SerialManipulator FK
sm_ee = manipulator.forward_kinematics(cfg)

# Compare
diff = np.linalg.norm(urdf_ee - sm_ee)
print(f"FK difference: {diff}")

if diff > 1e-10:
    print("Mismatch detected - check joint ordering and screw axes")
```

---

## Performance Issues

### Slow FK Computation

**Solution:** Use batch FK for multiple configurations:
```python
import numpy as np

robot = URDF.load("robot.urdf")

# Slow: individual calls
configs = np.random.uniform(-np.pi, np.pi, (1000, robot.num_dofs))
for cfg in configs:
    fk = robot.link_fk(cfg)  # Slow!

# Fast: batch FK
batch_fk = robot.link_fk_batch(configs)  # ~50x faster
```

### Slow URDF Loading

**Solution:**
```python
# 1. Disable mesh loading if not needed
robot = URDF.load("robot.urdf", load_meshes=False)

# 2. Cache loaded robot
_robot_cache = {}

def get_robot(urdf_path):
    if urdf_path not in _robot_cache:
        _robot_cache[urdf_path] = URDF.load(urdf_path)
    return _robot_cache[urdf_path]
```

---

## NumPy Compatibility

### NumPy 2.0 Issues

**Symptom:** Errors about NumPy API changes.

**Solution:**
```python
# The native URDF parser supports NumPy 2.0
# Make sure scipy is also updated:
# pip install 'scipy>=1.14'

# Check versions
import numpy as np
import scipy
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
```

### Array Type Warnings

**Symptom:** Warnings about array creation or type casting.

**Solution:**
```python
import numpy as np

# Always use explicit dtype
cfg = np.array([0.1, 0.2, 0.3], dtype=np.float64)

# For integer arrays
indices = np.array([0, 1, 2], dtype=np.int64)
```

---

## Backend Selection

### When to Use Different Backends

```python
from ManipulaPy.urdf import URDF

# Default: Native builtin parser (recommended)
# - NumPy 2.0 compatible
# - No external dependencies
# - Fast batch FK
robot = URDF.load("robot.urdf", backend="builtin")

# Legacy: urchin backend
# - For compatibility with existing code
# - Does NOT support NumPy 2.0
robot = URDF.load("robot.urdf", backend="urchin")

# PyBullet backend
# - Uses PyBullet's URDF parser
# - Good for simulation integration
robot = URDF.load("robot.urdf", backend="pybullet")
```

### Backend Not Available

**Symptom:** `ImportError` for backend.

**Solution:**
```python
# Check available backends
def check_backends():
    backends = {"builtin": True}

    try:
        import urchin
        backends["urchin"] = True
    except ImportError:
        backends["urchin"] = False

    try:
        import pybullet
        backends["pybullet"] = True
    except ImportError:
        backends["pybullet"] = False

    return backends

print(check_backends())
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check the tests:** Look at `tests/test_urdf_*.py` for usage examples
2. **Enable debug output:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. **Report issues:** https://github.com/anthropics/ManipulaPy/issues

### Minimal Reproducible Example

When reporting issues, include:
```python
# 1. Python and package versions
import sys
import numpy as np
from ManipulaPy import __version__

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"ManipulaPy: {__version__}")

# 2. Minimal code to reproduce
from ManipulaPy.urdf import URDF
robot = URDF.load("problem.urdf")
# ... minimal steps to trigger issue

# 3. URDF file (or minimal version that triggers issue)
```
