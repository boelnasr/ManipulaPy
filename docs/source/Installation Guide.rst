.. _installation-guide:

==================
Installation Guide
==================

ManipulaPy 1.4.0 ships with a **lightweight default install** and a set
of **optional extras** that pull in heavy or platform-specific
dependencies on demand. Most workflows only need one or two extras.

System Requirements
===================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Requirement
   * - Python
     - 3.9 -- 3.12 (3.12 added in v1.3.2)
   * - OS
     - Linux, macOS, Windows
   * - GPU (optional)
     - NVIDIA CUDA 11.x (driver >= 470) or 12.x (driver >= 535)
   * - PyBullet (optional)
     - Required only for simulation; CPU-only, no GPU needed

Default install
===============

.. code-block:: bash

   pip install ManipulaPy

This pulls in only **NumPy 2.x, SciPy, Matplotlib, Numba, and Pillow**.
Kinematics, dynamics, control, native URDF parsing, and CPU trajectory
generation work out of the box.

.. versionchanged:: 1.3.2
   The default install is now lightweight. Previous versions installed
   PyBullet, OpenCV, scikit-learn, and trimesh by default;
   these are now opt-in via extras.

Optional extras
===============

.. list-table::
   :header-rows: 1
   :widths: 22 53 25

   * - Extra
     - What it adds
     - Install
   * - ``[simulation]``
     - PyBullet physics simulation
     - ``pip install "ManipulaPy[simulation]"``
   * - ``[urdf]``
     - ``trimesh`` mesh loading for the native URDF backend
     - ``pip install "ManipulaPy[urdf]"``
   * - ``[vision]``
     - OpenCV, Ultralytics (YOLO), and PyTorch
     - ``pip install "ManipulaPy[vision]"``
   * - ``[vision-headless]``
     - ``opencv-python-headless`` + Ultralytics for CI/servers
     - ``pip install "ManipulaPy[vision-headless]"``
   * - ``[ml]``
     - PyTorch + scikit-learn (DBSCAN clustering for perception)
     - ``pip install "ManipulaPy[ml]"``
   * - ``[cuda]``
     - CuPy 12.x for GPU-accelerated kernels (default, driver >= 525)
     - ``pip install "ManipulaPy[cuda]"``
   * - ``[gpu-cuda11]``
     - CuPy 11.x for legacy CUDA 11.x toolchains (driver 470 - 524)
     - ``pip install "ManipulaPy[gpu-cuda11]"``
   * - ``[gpu-cuda12]``
     - Explicit CUDA 12.x alias for ``[cuda]``
     - ``pip install "ManipulaPy[gpu-cuda12]"``
   * - ``[gpu-rocm]``
     - CuPy ROCm 4.3 build for AMD GPUs
     - ``pip install "ManipulaPy[gpu-rocm]"``
   * - ``[gpu-pycuda]``
     - PyCUDA backend (alternative to CuPy)
     - ``pip install "ManipulaPy[gpu-pycuda]"``
   * - ``[pytorch]``
     - PyTorch compute backend (autodiff on the core math)
     - ``pip install "ManipulaPy[pytorch]"``
   * - ``[jax-cpu]``
     - JAX compute backend, CPU build (autodiff on the core math)
     - ``pip install "ManipulaPy[jax-cpu]"``
   * - ``[jax-cuda]``
     - JAX compute backend, CUDA 12 build (Linux only)
     - ``pip install "ManipulaPy[jax-cuda]"``
   * - ``[jax-tpu]``
     - JAX compute backend for a Google Cloud TPU VM (Linux only; release evidence pending Task 17)
     - ``pip install "ManipulaPy[jax-tpu]"``
   * - ``[all]``
     - All non-TPU runtime extras, including CPU JAX; excludes ``jax-tpu``
     - ``pip install "ManipulaPy[all]"``
   * - ``[minimal]``
     - Backwards-compatible pre-1.3.2 set (core + PyBullet)
     - ``pip install "ManipulaPy[minimal]"``
   * - ``[dev]``
     - Test, lint, and type-check tooling
     - ``pip install "ManipulaPy[dev]"``
   * - ``[docs]``
     - Sphinx + theme for building the documentation
     - ``pip install "ManipulaPy[docs]"``
   * - ``[ci]``
     - CI-only test deps (headless OpenCV, pytest plugins)
     - ``pip install "ManipulaPy[ci]"``

Combine extras with a comma:

.. code-block:: bash

   pip install "ManipulaPy[simulation,vision,cuda]"

GPU extras detail
=================

The default ``[cuda]`` extra installs ``cupy-cuda12x`` — the version
that ships with Ubuntu 22.04's NVIDIA apt repos and is what v1.3.2
was validated against on driver 580. For older CUDA 11.x toolchains
(driver 470 – 524), use the ``[gpu-cuda11]`` extra:

.. code-block:: bash

   # CUDA 12.x (driver >= 525) -- the default `[cuda]` extra
   pip install "ManipulaPy[cuda]"

   # Legacy CUDA 11.x (driver 470 - 524)
   pip install "ManipulaPy[gpu-cuda11]"

   # AMD ROCm 4.3
   pip install "ManipulaPy[gpu-rocm]"

.. note::

   On macOS the CuPy wheels are skipped automatically
   (``sys_platform != 'darwin'``); install ManipulaPy without the GPU
   extras on Apple Silicon.

If GPU acceleration is unavailable, ManipulaPy automatically falls
back to NumPy/Numba CPU paths -- no code changes needed.

Compute backend extras
======================

.. versionadded:: 1.4.0

The core math dispatches through a single backend protocol, so these
extras select an *array library* rather than adding features. NumPy is
the default and the base install stays NumPy-only -- nothing is
imported until a backend is explicitly requested.

.. code-block:: bash

   pip install ManipulaPy                # NumPy only (default)
   pip install "ManipulaPy[pytorch]"     # + PyTorch >= 2.7.1
   pip install "ManipulaPy[jax-cpu]"     # + JAX >= 0.6 (CPU)
   pip install "ManipulaPy[jax-cuda]"    # + JAX >= 0.6 (CUDA 12, Linux only)
   pip install "ManipulaPy[jax-tpu]"     # + JAX >= 0.6 (Google Cloud TPU VM, Linux only)
   pip install "ManipulaPy[cuda]"        # + CuPy 13.x (CUDA 12)

.. warning::

   ``[jax-tpu]`` targets a Google Cloud one-chip ``v5litepod-1`` (TPU v5e),
   not local hardware. The planned supported domain is real ``float32``,
   ``float64``, and ``int64``. X64 is required and can increase TPU resource
   use and compilation cost; a linalg compilation has exceeded 60 seconds.
   Complex TPU inputs must fail fast under Task 17. Packaging is not proof of
   support: real release evidence remains pending the Task-17 contract gate in
   `tests/test_tpu_contract.py <../../tests/test_tpu_contract.py>`_ and
   `.github/workflows/tpu-release.yml <../../.github/workflows/tpu-release.yml>`_.

The PyTorch and JAX backends make ``utils``, ``kinematics``, ``dynamics``
and ``singularity`` differentiable; the remaining modules run under every
backend through host-boundary conversion and carry no gradient
guarantee. See :doc:`user_guide/Backends` for the full contract.

Apple Silicon (M1 / M2 / M3)
============================

The default ``pip install ManipulaPy`` installs cleanly on macOS ARM --
all CPU features (kinematics, IK, dynamics, trajectory planning, native
URDF parsing, perception clustering) work natively.

PyPI does not ship a ``pybullet`` wheel for Apple Silicon, so a plain
``pip install "ManipulaPy[simulation]"`` triggers a from-source Clang
build that fails on some macOS ARM toolchains. If you need the
``[simulation]`` extra, install PyBullet from conda-forge first (it
provides prebuilt ARM64 wheels):

.. code-block:: bash

   conda create -n manipulapy python=3.11
   conda activate manipulapy
   conda install -c conda-forge pybullet
   pip install ManipulaPy            # pybullet already satisfied -- no compile

Alternatively, run ManipulaPy inside a Linux container on Apple Silicon;
installation succeeds there, including the simulation extra.

.. note::

   CUDA / GPU acceleration is not available on macOS by design -- the
   ``[cuda]`` extra is skipped automatically (``sys_platform != 'darwin'``)
   and ManipulaPy falls back to the NumPy / Numba CPU paths.

Development install
===================

.. code-block:: bash

   git clone https://github.com/boelnasr/ManipulaPy.git
   cd ManipulaPy
   pip install -e ".[dev]"

For documentation work, add ``[docs]``; for the full feature set while
developing, combine extras as needed (for example
``pip install -e ".[dev,all]"``).

Upgrading
=========

.. code-block:: bash

   pip install --upgrade ManipulaPy
   pip install --upgrade "ManipulaPy[cuda]"   # if you already use the cuda extra

When upgrading from 1.3.1 or earlier, run
``ManipulaPy.check_dependencies()`` after install to see which extras
you may now need to request explicitly -- several previously bundled
dependencies (PyBullet, OpenCV, scikit-learn, trimesh) are now
opt-in.

Verifying the install
=====================

.. code-block:: python

   import ManipulaPy
   ManipulaPy.check_dependencies()

This prints a per-feature availability table and the exact
``pip install`` command for any missing extra.

See also
========

- :doc:`getting_started/index` -- quick-start tutorial.
- :doc:`user_guide/Backends` -- choosing and using a compute backend.
- :doc:`api/index` -- full API reference.
