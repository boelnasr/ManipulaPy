# Changelog Maintenance Guide

## Overview

The `CHANGELOG.md` file tracks all notable changes to the ManipulaPy project. It helps users and developers understand what has changed between versions.

## Format

We follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format with these principles:

1. **Human-readable** - Written for humans, not machines
2. **One entry per version** - Each version gets its own section
3. **Grouped by type** - Changes are categorized
4. **Reverse chronological** - Newest versions at the top
5. **Semantic Versioning** - Versions follow [SemVer](https://semver.org/)

## Categories

### Added
New features or functionality.

**Examples:**
```markdown
### Added
- Support for 7-DOF manipulators
- URDF file loading capability
- New trajectory planning algorithms
- `get_end_effector_pose()` method in kinematics module
```

### Changed
Changes to existing functionality (backward compatible).

**Examples:**
```markdown
### Changed
- Improved accuracy of forward kinematics calculations
- Updated default IK solver parameters for better convergence
- Refactored control module for better performance
- Changed default plot colors for better visibility
```

### Deprecated
Features that will be removed in future versions (still work now).

**Examples:**
```markdown
### Deprecated
- `old_inverse_kinematics()` will be removed in v2.0.0, use `iterative_inverse_kinematics()` instead
- `compute_jacobian()` is deprecated in favor of `jacobian()`
```

### Removed
Features that have been removed.

**Examples:**
```markdown
### Removed
- Removed deprecated `legacy_control()` method
- Dropped support for Python 3.7
- Removed unused `utils.old_transform()` function
```

### Fixed
Bug fixes (backward compatible).

**Examples:**
```markdown
### Fixed
- Fixed memory leak in trajectory optimization
- Corrected singularity handling in IK solver
- Fixed incorrect joint limits enforcement
- Resolved crash when loading certain URDF files
```

### Security
Security vulnerability fixes.

**Examples:**
```markdown
### Security
- Fixed path traversal vulnerability in URDF loader
- Updated dependencies to patch CVE-2024-XXXXX
- Sanitized user input in trajectory planner
```

## Daily Workflow

### When Making Changes

1. **After every significant change**, add an entry to `CHANGELOG.md`

2. **Add to `[Unreleased]` section** under the appropriate category:

```markdown
## [Unreleased]

### Fixed
- Fixed IK convergence for large rotations
- Corrected SE(3) error calculation in kinematics module

### Added
- Added `clear_yolo_cache()` for memory management
```

3. **Commit the changelog** with your changes:
```bash
git add CHANGELOG.md
git commit -m "feat: Add clear_yolo_cache function"
```

### Writing Good Entries

**Bad entries:**
```markdown
- Fixed bug
- Updated code
- Improved performance
- Changed stuff
```

**Good entries:**
```markdown
- Fixed memory leak in YOLO model instantiation (50-100x speedup)
- Corrected SE(3) error calculation in IK algorithm for large rotations
- Eliminated GPU-CPU transfers in control module (2-3x speedup)
- Added type hints to kinematics, dynamics, and control modules
```

**Rules:**
- ✅ Describe **what** changed and **why** it matters
- ✅ Be specific and actionable
- ✅ Include performance impact if relevant
- ✅ Reference issue numbers: `(#123)`
- ❌ Don't be vague or generic
- ❌ Don't use jargon without explanation

## Release Workflow

### When Creating a Release

#### Step 1: Update CHANGELOG.md

```markdown
## [Unreleased]

### Added
- Feature 1
- Feature 2

### Fixed
- Bug 1
- Bug 2
```

**Becomes:**

```markdown
## [Unreleased]

### Added

### Changed

### Fixed

## [1.2.0] - 2025-11-15

### Added
- Feature 1
- Feature 2

### Fixed
- Bug 1
- Bug 2
```

#### Step 2: Determine Version Number

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR (X.0.0)**: Breaking changes (incompatible API changes)
  ```
  1.1.3 → 2.0.0
  ```

- **MINOR (0.X.0)**: New features (backward compatible)
  ```
  1.1.3 → 1.2.0
  ```

- **PATCH (0.0.X)**: Bug fixes (backward compatible)
  ```
  1.1.3 → 1.1.4
  ```

**Decision tree:**
```
Breaking API changes? → MAJOR
  No ↓
New features added? → MINOR
  No ↓
Bug fixes only? → PATCH
```

**Examples:**

| Changes | Version Bump | Example |
|---------|--------------|---------|
| Fixed IK bug | PATCH | 1.1.3 → 1.1.4 |
| Added type hints | MINOR | 1.1.3 → 1.2.0 |
| Fixed IK + Added cache API | MINOR | 1.1.3 → 1.2.0 |
| Removed Python 3.7 support | MAJOR | 1.1.3 → 2.0.0 |

#### Step 3: Update Version in Code

Update version in these files:

1. **`pyproject.toml`**:
```toml
[project]
name = "manipulapy"
version = "1.2.0"  # ← Update this
```

2. **`ManipulaPy/__init__.py`**:
```python
__version__ = "1.2.0"  # ← Update this
```

#### Step 4: Update Comparison Links

At the bottom of `CHANGELOG.md`:

```markdown
[Unreleased]: https://github.com/DR-ROBOTICS-RESEARCH-GROUP/ManipulaPy/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/DR-ROBOTICS-RESEARCH-GROUP/ManipulaPy/compare/v1.1.3...v1.2.0
[1.1.3]: https://github.com/DR-ROBOTICS-RESEARCH-GROUP/ManipulaPy/releases/tag/v1.1.3
```

#### Step 5: Commit and Tag

```bash
# Commit changelog and version updates
git add CHANGELOG.md pyproject.toml ManipulaPy/__init__.py
git commit -m "chore: Release v1.2.0"

# Create annotated tag
git tag -a v1.2.0 -m "Release v1.2.0

Highlights:
- Fixed critical IK algorithm bug
- 50-100x vision performance improvement
- Added comprehensive type hints
"

# Push to remote
git push origin main
git push origin v1.2.0
```

#### Step 6: Create GitHub Release

1. Go to: https://github.com/DR-ROBOTICS-RESEARCH-GROUP/ManipulaPy/releases/new
2. Select the tag: `v1.2.0`
3. Title: `v1.2.0 - Performance & Correctness Improvements`
4. Copy the changelog section into the description
5. Click "Publish release"

## Example Changelog Entry

Here's a complete example of a good changelog entry:

```markdown
## [1.2.0] - 2025-11-15

### Added
- Type hints to core modules (kinematics, dynamics, control, vision, transformations)
- `clear_yolo_cache()` function for explicit YOLO model memory management
- Comprehensive documentation files for all major fixes
  - `IK_ALGORITHM_FIX.md` - Mathematical analysis of IK corrections
  - `CONTROL_BOTTLENECK_FIX.md` - Performance optimization details
  - `VISION_BOTTLENECK_FIX.md` - YOLO caching implementation

### Fixed
- **CRITICAL**: Inverse kinematics algorithm SE(3) error calculation
  - Previously used incorrect `log(T_desired) - log(T_current)` formulation
  - Now uses correct `log(T_desired @ inv(T_current))` as per Modern Robotics textbook
  - Fixes convergence failures, especially for large rotations
  - Impact: IK now mathematically correct for all cases
- **PERFORMANCE**: Control module GPU-CPU transfer bottleneck
  - Eliminated 6+ PCIe transfers per control cycle
  - Changed from CuPy (GPU) to NumPy (CPU) for consistency with dynamics module
  - Impact: 2-3x faster control loops, real-time capable
- **PERFORMANCE**: Vision module YOLO model reinstantiation
  - Implemented global model caching in `detect_objects()`
  - Impact: 50-100x faster subsequent detections, enables real-time vision (15-30 FPS)
- IK accuracy benchmark now uses same error metric and tolerances as IK algorithm
  - Fixed orientation tolerance mismatch (was 10x too lenient)
  - Eliminates false positives in success rate reporting

### Changed
- Control module return types from `cp.ndarray` to `NDArray[np.float64]` (backward compatible)
- IK algorithm variable names from Greek symbols to English for better compatibility
  - `θ` → `theta`, `Δθ` → `delta_theta`, `normΔ` → `norm_delta`
- Vision module `detect_objects()` now caches models globally (transparent to users)

### Breaking Changes
**None** - All changes are backward compatible.

### Upgrade Notes
See "Upgrade Guide" section in CHANGELOG.md for migration details.
```

## Tips and Best Practices

### DO ✅

- **Update the changelog with every PR/commit**
- **Be specific** - "Fixed IK bug" → "Fixed IK convergence for rotations > 90°"
- **Include impact** - "50-100x speedup", "enables real-time vision"
- **Reference issues** - "Fixed memory leak (#123)"
- **Group related changes** - Multiple IK fixes under one bullet
- **Use present tense** - "Add", "Fix", "Change" (not "Added", "Fixed")
- **Link to detailed docs** - "See `IK_ALGORITHM_FIX.md` for details"

### DON'T ❌

- **Don't skip the changelog** - Even small fixes deserve an entry
- **Don't be vague** - "Improved code" tells users nothing
- **Don't include internal changes** - Refactoring without user impact
- **Don't duplicate git history** - Summarize, don't copy commit messages
- **Don't forget to update on release** - Move `[Unreleased]` to versioned section

## Tools

### Automatic Changelog Generation

For projects with many contributors, consider:

1. **GitHub Actions** - Auto-generate from PRs
2. **Conventional Commits** - Structured commit messages
3. **Release-please** - Automated releases

### Manual Template

Quick template for adding entries:

```markdown
### Fixed
- Fixed [WHAT] in [WHERE] ([IMPACT/WHY])
  - [Additional details]
  - [Performance numbers]

### Added
- [FEATURE NAME] for [USE CASE]
  - [How to use]
  - [Benefits]
```

## Common Scenarios

### Bug Fix Release (Patch)

```markdown
## [1.1.4] - 2025-11-20

### Fixed
- Fixed memory leak in trajectory planner when using 1000+ waypoints
- Corrected joint limit checking in 7-DOF configurations
```

**Version:** 1.1.3 → 1.1.4

### Feature Release (Minor)

```markdown
## [1.2.0] - 2025-11-15

### Added
- URDF file loading support
- Real-time trajectory visualization
- Collision detection module

### Fixed
- Fixed IK convergence issues
```

**Version:** 1.1.3 → 1.2.0

### Breaking Change (Major)

```markdown
## [2.0.0] - 2026-01-01

### Changed
- **BREAKING**: Renamed `get_jacobian()` to `jacobian()` for consistency
- **BREAKING**: `Robot` class now requires explicit DOF specification

### Removed
- **BREAKING**: Removed deprecated `legacy_control()` method
- **BREAKING**: Dropped Python 3.7 support

### Migration Guide
See MIGRATION.md for step-by-step upgrade instructions.
```

**Version:** 1.2.0 → 2.0.0

## Summary

1. **Update `CHANGELOG.md`** after every significant change
2. **Use clear, specific descriptions** of what changed and why
3. **Follow the categories** (Added, Changed, Fixed, etc.)
4. **Determine version numbers** using Semantic Versioning
5. **Create git tags** when releasing
6. **Keep it human-readable** - write for your users, not machines

---

**Remember:** A good changelog is a love letter to your users. It shows you care about their experience and helps them understand what's changing in the software they depend on.
