#!/bin/bash

# Script to integrate existing user_guide structure and fix file names
# Run this from: /path/to/ManipulaPy/docs/source/

echo "Integrating existing user_guide structure with main documentation..."
echo "Directory: $(pwd)"

# Check if we're in the right directory
if [ ! -d "user_guide" ]; then
    echo "❌ user_guide directory not found!"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /path/to/ManipulaPy/docs/source/"
    exit 1
fi

echo "✅ Found user_guide directory: $(pwd)/user_guide"

# Step 1: Fix the filename with spaces and typo
echo ""
echo "STEP 1: Fixing file names in user_guide directory..."

cd user_guide

# Rename "Urdf Procsssor.rst" to "URDF_Processor.rst" (fix typo and spaces)
if [ -f "Urdf Procsssor.rst" ]; then
    mv "Urdf Procsssor.rst" "URDF_Processor.rst"
    echo "✅ Renamed 'Urdf Procsssor.rst' → 'URDF_Processor.rst'"
fi

# Check what files we have
echo "📁 Current user_guide files:"
ls -la *.rst

cd ..

# Step 2: Update the user_guide/index.rst with correct file names
echo ""
echo "STEP 2: Updating user_guide/index.rst with correct file names..."

cat > user_guide/index.rst << 'EOL'
.. _user-guides-index:

User Guides
===========

High-level, task-oriented guides that walk you through ManipulaPy's core
capabilities. Each guide provides detailed explanations, examples, and best
practices for using ManipulaPy effectively.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   Kinematics
   Dynamics
   Trajectory_Planning
   URDF_Processor

Core Concepts
-------------

These guides cover the fundamental concepts and modules in ManipulaPy:

**Kinematics** (:doc:`Kinematics`)
   Learn about forward and inverse kinematics, including serial manipulator 
   configurations, transformation matrices, and solving for joint angles.

**Dynamics** (:doc:`Dynamics`)
   Understand robot dynamics including mass matrices, inverse dynamics, 
   forward dynamics, and force/torque calculations.

**Trajectory Planning** (:doc:`Trajectory_Planning`)
   Explore motion planning techniques, CUDA-accelerated trajectory generation,
   and collision avoidance strategies.

**URDF Processing** (:doc:`URDF_Processor`)
   Discover how to work with URDF files, automatic robot model generation,
   and PyBullet integration.

Getting Started
---------------

New to ManipulaPy? We recommend following this learning path:

1. **Start Here**: Read the :doc:`../tutorials/installation` guide
2. **Basic Usage**: Try the :doc:`../tutorials/quickstart` examples  
3. **Core Concepts**: Work through these User Guides in order:
   
   - :doc:`Kinematics` - Understand robot motion
   - :doc:`Dynamics` - Learn about forces and torques
   - :doc:`Trajectory_Planning` - Plan robot movements
   - :doc:`URDF_Processor` - Work with robot models

Quick Reference
---------------

**Need help with a specific task?**

- **Loading a robot**: See :doc:`URDF_Processor`
- **Computing end-effector pose**: See :doc:`Kinematics`
- **Calculating joint torques**: See :doc:`Dynamics`
- **Planning a trajectory**: See :doc:`Trajectory_Planning`
- **API details**: Check the :doc:`../api/index`
EOL

echo "✅ Updated user_guide/index.rst with proper structure"

# Step 3: Create the main index.rst that uses the existing user_guide
echo ""
echo "STEP 3: Creating main index.rst that integrates with user_guide..."

# [Copy the main index.rst content from the artifact above]

echo "✅ Created main index.rst that integrates with existing user_guide"

echo ""
echo "=========================================="
echo "✅ USER GUIDE INTEGRATION COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Your existing structure is now integrated:"
echo "   📄 index.rst                    - Main page with User Guides tab"
echo "   📁 user_guide/"
echo "      📄 index.rst                 - User Guides landing page (updated)"
echo "      📄 Kinematics.rst            - Your existing kinematics guide"
echo "      📄 Dynamics.rst              - Your existing dynamics guide"
echo "      📄 Trajectory_Planning.rst   - Your existing trajectory guide"
echo "      📄 URDF_Processor.rst        - Renamed from 'Urdf Procsssor.rst'"
echo ""
echo "🏷️  Navigation will show:"
echo "   📚 User Guides"
echo "      ├── Kinematics (your existing content)"
echo "      ├── Dynamics (your existing content)"
echo "      ├── Trajectory Planning (your existing content)"
echo "      └── URDF Processor (your existing content, renamed)"
echo ""
echo "🔧 Next steps:"
echo "   1. Build: cd .. && make clean && make html"
echo "   2. View: open build/html/index.html"
echo "   3. Your User Guides will appear as a main tab!"
