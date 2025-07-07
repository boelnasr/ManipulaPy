.. _api-index:

======================
🔧 API Reference Guide
======================

Welcome to the low-level API reference for **ManipulaPy**.

This section contains auto-generated documentation for all key modules, classes, and functions in the library, based on the actual Python source code using **Sphinx autodoc**.

.. note::

   - This section is **not a tutorial** – it’s for precise function/class definitions.  
   - For interactive examples and walkthroughs, see :ref:`tutorial-index`.  
   - For explanations, workflows, and configuration, see the :ref:`user_guide_index`.

---------------------------------------------------
📦 Module Overview
---------------------------------------------------

.. rubric:: 🔩 Core Manipulator Stack

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   kinematics
   dynamics
   path_planning
   control
   urdf_processor

.. rubric:: 🧠 Planning, Simulation & Optimization

.. toctree::
   :maxdepth: 2
   :caption: Planning & Simulation

   simulation
   potential_field
   cuda_kernels

.. rubric:: 👁️ Perception & Vision

.. toctree::
   :maxdepth: 2
   :caption: Perception Modules

   perception
   vision

.. rubric:: 🧪 Advanced Analysis

.. toctree::
   :maxdepth: 2
   :caption: Analysis Tools

   singularity

---------------------------------------------------
📖 How to Use This Reference
---------------------------------------------------

- **Module headers** show the import path: e.g., ``ManipulaPy.control.ManipulatorController``  
- **Function signatures** are exact and include optional/default parameters  
- **Class attributes** and properties are listed, with docstrings if provided  
- **NumPy/CuPy compatibility** is noted where applicable  
- **Examples** may be included for select methods/classes

---------------------------------------------------
🔍 Quick Navigation
---------------------------------------------------

- :ref:`genindex` – Alphabetical list of all documented symbols  
- :ref:`modindex` – Index of Python modules  
- :ref:`search` – Full-text search across the documentation
