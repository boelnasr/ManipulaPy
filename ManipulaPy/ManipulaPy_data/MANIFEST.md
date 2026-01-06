# ManipulaPy Robot Data Manifest

This directory contains URDF models and mesh assets for 25+ robot manipulators from 7 manufacturers.

**Total Size:** ~143 MB
**Last Updated:** 2026-01-05

---

## 📁 Directory Structure

### Production URDFs (Ready to Use)

These folders contain compiled URDF files ready for immediate use with ManipulaPy.

```
ManipulaPy_data/
├── __init__.py          # Robot database API
├── MANIFEST.md          # This file
│
├── abb/                 # ABB industrial robots (12 KB)
│   └── irb2400.urdf
├── fanuc/               # Fanuc industrial robots (20 KB)
│   ├── lrmate200ib.urdf
│   └── m16ib.urdf
├── fanuc_crx/           # Fanuc CRX collaborative robots (42 MB)
│   ├── crx5ia.urdf, crx10ia.urdf, crx10ia_l.urdf
│   ├── crx20ia_l.urdf, crx30ia.urdf
│   └── meshes/
├── franka_panda/        # Franka Emika Panda (11 MB)
│   ├── panda.urdf
│   └── meshes/
├── kinova/              # Kinova Gen3 and Jaco (4.3 MB)
│   ├── gen3/, jaco/
│   └── meshes/
├── kuka_iiwa/           # KUKA LBR iiwa robots (20 MB)
│   ├── iiwa7/, iiwa14/
│   └── meshes/
├── robotiq/             # Robotiq grippers (36 KB)
│   ├── robotiq_2f_85.urdf
│   └── robotiq_2f_140.urdf
├── universal_robots/    # Universal Robots UR series (59 MB)
│   ├── ur3/, ur5/, ur10/, ur3e/, ur5e/, ur10e/, ur16e/
│   └── meshes/
├── xarm/                # UFactory xArm series (5.2 MB)
│   ├── xarm6_robot.urdf
│   ├── xarm6_robot_white.urdf
│   ├── xarm6_with_gripper.urdf
│   ├── xarm_description/meshes/
│   └── xarm_gripper/meshes/
│
└── _source/             # ROS source packages (for developers)
    └── (see below)
```

### Source Packages (Xacro/ROS Developers)

The `_source/` directory contains ROS package structures with xacro source files for developers who need to customize robot models.

```
_source/
├── fanuc_crx_description/      # CRX series xacro sources (560 KB)
│   ├── urdf/*.xacro            # Xacro macro files
│   ├── robot/*.urdf.xacro      # Top-level robot definitions
│   ├── launch/, rviz/          # ROS launch and visualization configs
│   ├── meshes/                 # Meshes (smaller, symbolic copies)
│   └── package.xml             # ROS package metadata
├── fanuc_lrmate_description/   # LRMate series xacro sources (212 KB)
├── fanuc_m10_description/      # M10 series xacro sources (128 KB)
├── fanuc_m20_description/      # M20 series xacro sources (212 KB)
├── fanuc_r1000ia_description/  # R1000iA series xacro sources (128 KB)
└── fanuc_r2000_description/    # R2000 series xacro sources (128 KB)
```

**Note:** The `_source/` folder contains ROS packages with xacro files. If you just need to use a robot, use the production URDFs above. If you need to modify or customize robot definitions, these packages provide the source files.

---

## 🤖 Available Robots

### Universal Robots (7 models)
- **ur3** - 6-DOF, 3 kg payload, 500 mm reach
- **ur5** - 6-DOF, 5 kg payload, 850 mm reach
- **ur10** - 6-DOF, 10 kg payload, 1300 mm reach
- **ur3e** - 6-DOF e-Series, 3 kg payload, 500 mm reach
- **ur5e** - 6-DOF e-Series, 5 kg payload, 850 mm reach
- **ur10e** - 6-DOF e-Series, 12.5 kg payload, 1300 mm reach
- **ur16e** - 6-DOF e-Series, 16 kg payload, 900 mm reach

### Franka Emika (1 model)
- **panda** (alias: franka_panda) - 7-DOF, 3 kg payload, 855 mm reach

### KUKA (2 models)
- **iiwa7** - 7-DOF, 7 kg payload, 800 mm reach
- **iiwa14** (alias: kuka_iiwa) - 7-DOF, 14 kg payload, 820 mm reach

### Kinova (3 models)
- **gen3** (alias: kinova_gen3) - 7-DOF, 4 kg payload, 902 mm reach
- **jaco_6dof** - 6-DOF, 1.6 kg payload, 900 mm reach
- **jaco_7dof** - 7-DOF, 1.6 kg payload, 900 mm reach

### Fanuc (7 models)
- **fanuc_lrmate** - LR Mate 200iB, 6-DOF, 5 kg payload, 704 mm reach
- **fanuc_m16ib** - M-16iB, 6-DOF, 16 kg payload, 1885 mm reach
- **crx5ia** - CRX-5iA collaborative, 6-DOF, 5 kg payload, 994 mm reach
- **crx10ia** - CRX-10iA collaborative, 6-DOF, 10 kg payload, 1249 mm reach
- **crx10ia_l** - CRX-10iA/L long reach, 6-DOF, 10 kg payload, 1418 mm reach
- **crx20ia_l** - CRX-20iA/L, 6-DOF, 20 kg payload, 1418 mm reach
- **crx30ia** - CRX-30iA heavy payload, 6-DOF, 30 kg payload, 1252 mm reach

### ABB (1 model)
- **abb_irb2400** - IRB 2400, 6-DOF, 7-20 kg payload, 1550 mm reach

### UFactory (2 models)
- **xarm6** - xArm6, 6-DOF, 5 kg payload, 700 mm reach
- **xarm6_gripper** - xArm6 with gripper, 6-DOF, 5 kg payload, 700 mm reach

### Robotiq (2 models)
- **robotiq_2f_85** - 2F-85 adaptive gripper, 85 mm stroke
- **robotiq_2f_140** - 2F-140 adaptive gripper, 140 mm stroke

---

## 📖 Usage

### Basic Usage

```python
from ManipulaPy.ManipulaPy_data import get_robot_urdf, list_robots

# List all available robots
print(list_robots())

# Get path to a robot's URDF
urdf_path = get_robot_urdf('ur5')
urdf_path = get_robot_urdf('panda')
urdf_path = get_robot_urdf('iiwa14')

# Load robot directly with ManipulaPy
from ManipulaPy.urdf_processor import load_robot
robot = load_robot(get_robot_urdf('ur5'))
```

### Load with Native URDF Parser

```python
from ManipulaPy.urdf import URDF
from ManipulaPy.ManipulaPy_data import get_robot_urdf

# Load URDF
robot = URDF.load(get_robot_urdf('panda'))

# Access robot properties
print(f"DOF: {robot.num_dofs}")
print(f"Joints: {robot.actuated_joint_names}")

# Forward kinematics
fk = robot.link_fk([0.0] * robot.num_dofs)

# Convert to SerialManipulator
manipulator = robot.to_serial_manipulator()
```

### Query Robot Information

```python
from ManipulaPy.ManipulaPy_data import (
    get_robot_info,
    list_manufacturers,
    get_robots_by_dof,
    print_robot_catalog
)

# Get robot details
info = get_robot_info('ur5')
print(f"Name: {info['name']}")
print(f"DOF: {info['dof']}, Payload: {info['payload']}")

# Filter by manufacturer
ur_robots = list_robots('Universal Robots')
fanuc_robots = list_robots('Fanuc')

# Filter by DOF
six_dof = get_robots_by_dof(6)
seven_dof = get_robots_by_dof(7)

# Print full catalog
print_robot_catalog()
```

---

## 🔧 Mesh Path Resolution

### Universal Robots URDFs

Universal Robots URDFs use `package://` URIs for mesh references:

```xml
<mesh filename="package://ur_description/meshes/ur5/visual/base.dae"/>
```

**Resolution:** The native ManipulaPy URDF parser's `PackageResolver` automatically handles these paths:

```python
from ManipulaPy.urdf import URDF

# PackageResolver automatically handles package:// URIs
robot = URDF.load(get_robot_urdf('ur5'))  # Works automatically
```

The resolver searches:
1. Explicit package map
2. Environment variables (ROS_PACKAGE_PATH, AMENT_PREFIX_PATH, MANIPULAPY_PACKAGE_PATH)
3. Parent directories of the URDF file
4. ROS package discovery (rospack/ament_index if available)

### Custom Package Mapping

If needed, you can configure custom package paths:

```python
from ManipulaPy.urdf import URDF, PackageResolver

# Create custom resolver
resolver = PackageResolver()
resolver.add_package('ur_description', '/path/to/ur_description')

# Load with custom resolver
robot = URDF.load(urdf_path, resolver=resolver)
```

---

## 📝 Adding New Robots

To add a new robot to the database:

1. **Add robot files:**
   ```
   ManipulaPy_data/
   └── manufacturer_name/
       ├── model_name/
       │   └── model_name.urdf
       └── meshes/
           ├── visual/
           └── collision/
   ```

2. **Update database in `__init__.py`:**
   ```python
   ROBOT_DATABASE = {
       'robot_key': {
           'name': 'Robot Full Name',
           'manufacturer': 'Manufacturer Name',
           'dof': 6,
           'payload': '10 kg',
           'reach': '1000 mm',
           'urdf': 'manufacturer_name/model_name/model_name.urdf',
           'description': 'Brief description',
       },
       # ... existing robots
   }
   ```

3. **Test availability:**
   ```python
   from ManipulaPy.ManipulaPy_data import check_robot_available
   assert check_robot_available('robot_key')
   ```

---

## 🧪 Validation

Validate all robot assets:

```python
from ManipulaPy.ManipulaPy_data import ROBOT_DATABASE, check_robot_available

# Check all robots
for robot in ROBOT_DATABASE:
    status = "✓" if check_robot_available(robot) else "✗"
    print(f"{status} {robot}")
```

---

## 📊 Statistics

| Category | Count | Size |
|----------|-------|------|
| **Manufacturers** | 7 | - |
| **Robot Models** | 25 | - |
| **URDF Files** | 69 | - |
| **Total Size** | ~143 MB | - |

### Breakdown by Manufacturer

| Manufacturer | Models | Size |
|--------------|--------|------|
| Universal Robots | 7 | 59 MB |
| Fanuc | 7 | 42 MB |
| KUKA | 2 | 20 MB |
| Franka Emika | 1 | 11 MB |
| UFactory | 2 | 5.2 MB |
| Kinova | 3 | 4.3 MB |
| Robotiq | 2 | 36 KB |
| ABB | 1 | 12 KB |

---

## 🔍 Troubleshooting

### Robot Not Found

```python
ValueError: Unknown robot: 'myrobot'
```

**Solution:** Check available robots:
```python
from ManipulaPy.ManipulaPy_data import list_robots
print(list_robots())
```

### URDF File Not Found

```python
FileNotFoundError: URDF file not found: /path/to/robot.urdf
```

**Solution:** Verify the file exists:
```python
from ManipulaPy.ManipulaPy_data import get_robot_info
info = get_robot_info('ur5')
print(f"Available: {info['available']}")
print(f"Path: {info['urdf_path']}")
```

### Mesh Files Not Found

If you get mesh loading errors, ensure the URDF file's parent directory is in your search path, or use the native parser which handles this automatically:

```python
from ManipulaPy.urdf import URDF
robot = URDF.load(urdf_path)  # Automatically sets up mesh paths
```

---

## 📚 Related Documentation

- **URDF Parser:** `/ManipulaPy/urdf/README.md`
- **URDF Troubleshooting:** `/ManipulaPy/urdf/TROUBLESHOOTING.md`
- **Examples:** `/Examples/intermediate_examples/`

---

## 🎯 Best Practices

1. **Use the database API** instead of hardcoding paths:
   ```python
   # ✓ Good
   urdf_path = get_robot_urdf('ur5')

   # ✗ Avoid
   urdf_path = 'ManipulaPy_data/universal_robots/ur5/ur5.urdf'
   ```

2. **Check availability** before loading:
   ```python
   if check_robot_available('ur5'):
       robot = URDF.load(get_robot_urdf('ur5'))
   ```

3. **Use the native parser** for best compatibility:
   ```python
   from ManipulaPy.urdf import URDF
   robot = URDF.load(get_robot_urdf('panda'))  # NumPy 2.0+ compatible
   ```

4. **For production:** Use compiled URDFs from production folders
5. **For development:** Use xacro sources from `_source/*_description` packages

---

**Maintained by:** ManipulaPy Team
**License:** See individual robot packages for specific licenses
