Compute Backends User Guide
============================

ManipulaPy's core math runs on **NumPy, CuPy, PyTorch, or JAX** behind a single
dispatch API. You select the array library; the kinematics, dynamics, planning
and control code is unchanged. Under PyTorch and JAX the core math is
autodiff-safe, so gradients come from the framework rather than from finite
differences.

.. note::
   New in v1.4.0. The default backend is NumPy and the public return contract
   under the default backend is frozen, so existing code — including the ROS
   wrapper — is unaffected by this feature.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Selecting a Backend
-------------------

Two entry points, both from ``ManipulaPy.backend``:

.. code-block:: python

   from ManipulaPy.backend import set_backend, use_backend

   # Process-wide, until changed
   set_backend("torch")

   # Scoped — restores the previous backend on exit
   with use_backend("jax"):
       T = robot.forward_kinematics(thetalist)

``use_backend`` is the safer choice in library code and tests: it restores the
previous backend even if the block raises.

Backends register lazily behind an import probe, so a base install with none of
the optional libraries present behaves exactly as before and never imports
torch or jax.

Available Backends
------------------

.. list-table::
   :header-rows: 1
   :widths: 14 16 14 14 42

   * - Name
     - Install
     - Concrete
     - Gradients
     - Notes
   * - ``numpy``
     - built in
     - yes
     - no
     - Default. The frozen public return contract is defined here.
   * - ``cupy``
     - ``[cuda]``
     - yes
     - no
     - GPU arrays; same eager semantics as NumPy.
   * - ``torch``
     - ``[pytorch]``
     - no
     - yes
     - ``torch.autograd`` and ``torch.jit.trace`` safe on core math.
   * - ``jax``
     - ``[jax-cpu]`` / ``[jax-cuda]`` / ``[jax-tpu]``
     - no
     - yes
     - ``jax.grad`` / ``jax.jacrev`` / ``jit`` safe on core math.

**Concrete** means the backend's arrays hold materialised values. It is false
for tracing backends, where an array may be a symbolic tracer rather than
data. ManipulaPy uses this to decide whether value-dependent optimisations are
safe — most visibly, the mass-matrix cache is bypassed under tracing backends,
because caching on a tracer would either detach the gradient or key the cache
on a value that does not exist yet.

The Differentiable Contract
---------------------------

This is the most important section of this guide: **gradients are guaranteed
for core math only.**

.. warning::
   The differentiable contract covers ``utils``, ``kinematics``, ``dynamics``,
   and ``singularity``. Every other module *runs* under all four backends, but
   through **host-boundary conversion** — arrays are converted to NumPy, the
   existing implementation runs, and the result is converted back. Those paths
   are portable, **not differentiable**. A gradient taken through them is not
   meaningful.

Host-bound components, and why:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Reason
   * - ``trac_ik`` SLSQP solve
     - SciPy optimiser, host-only
   * - ``sim`` (PyBullet)
     - external physics engine
   * - ``potential_field`` ConvexHull
     - SciPy/Qhull, host-only
   * - ``cuda_kernels`` / planning kernels
     - Numba CUDA kernels take device pointers, not backend arrays

What *is* covered:

.. code-block:: python

   import jax
   from ManipulaPy.backend import use_backend

   with use_backend("jax"):
       # forward kinematics is differentiable end to end
       dT = jax.jacrev(robot.forward_kinematics)(thetalist)

       # so is inverse dynamics
       dtau = jax.jacrev(
           lambda q: dynamics.inverse_dynamics(q, dq, ddq, g, Ftip)
       )(thetalist)

The same works under PyTorch with ``torch.autograd.functional.jacobian`` or an
ordinary ``.backward()`` on a scalar loss.

Gradients at singular configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SE(3)/SO(3) logarithm is the delicate part of any differentiable robotics
stack, because the rotation angle is a branch point at :math:`\theta = 0` and
:math:`\theta = \pi`. ManipulaPy's implementations are conditioned so that
gradients stay finite and correct at both endpoints, using Taylor branches
whose cutoffs are sized by where the **backward** pass stops cancelling — not
by where the forward value looks acceptable.

This distinction matters more than it sounds. A ``where``-style mask selects
the correct *value*, but both branches are still evaluated during
differentiation, so a singular inactive branch contributes ``0 * inf = NaN`` to
the gradient while the value looks perfectly fine. Several defects of exactly
this shape were found and fixed while building the contract; they affected
NumPy results too, not only the new backends.

Performance Notes
-----------------

.. warning::
   **JAX eager dispatch is roughly 40x slower than NumPy on small problems**
   (single forward kinematics call: ~5.3 ms vs ~0.13 ms). Per-call dispatch
   overhead dominates at robot-sized array shapes. This is a property of eager
   JAX, not of ManipulaPy.

The practical consequence is that **any time-budgeted algorithm behaves
differently under JAX**. Most notably, :class:`TracIKSolver` takes a
``timeout`` (default 0.2 s) and does not converge within that budget under the
JAX backend. With a wider budget it reaches the same answer as NumPy. The
default is deliberately unchanged, because raising it for one backend would
slow failure detection for everyone else:

.. code-block:: python

   with use_backend("jax"):
       solver.solve(T_desired, theta_init, timeout=5.0)   # widen explicitly

JAX pays off when you ``jit`` a whole computation or batch with ``vmap``, where
the dispatch overhead is compiled away. It does not pay off for one-shot calls.

For raw throughput on large batches, CuPy remains the fastest option, and the
Numba CUDA kernels are still used for trajectory generation regardless of the
active backend.

JAX and float64
---------------

JAX defaults to float32 and its promotion lattice deliberately avoids widening
to float64. ManipulaPy needs float64 to match NumPy numerics, so the JAX
backend calls ``jax.config.update("jax_enable_x64", True)`` when it is
imported.

.. important::
   This is **process-global JAX state**. It is set only when the ``jax``
   backend is actually requested — importing ManipulaPy alone does not touch
   it — but once set it is not reverted, including on exit from a
   ``use_backend("jax")`` block. If your application shares the process with
   other JAX code that depends on float32 defaults, be aware of this.

Every dtype-combining operation in the JAX backend computes its result dtype
with ``numpy.result_type`` and casts explicitly, so mixed-dtype expressions
promote the way NumPy would rather than the way JAX would. The casts lower to
``convert_element_type``, so they remain traceable and differentiable.

Installation
------------

.. code-block:: bash

   pip install ManipulaPy                # NumPy only
   pip install ManipulaPy[pytorch]       # + PyTorch
   pip install ManipulaPy[jax-cpu]       # + JAX (CPU)
   pip install ManipulaPy[jax-cuda]      # + JAX (CUDA 12, Linux only)
   pip install "ManipulaPy[jax-tpu]"     # + JAX (Google Cloud TPU VM, Linux only)
   pip install ManipulaPy[cuda]          # + CuPy (CUDA 12)

.. warning::
   ``[jax-tpu]`` is for a Google Cloud one-chip ``v5litepod-1`` (TPU v5e), not
   a local accelerator. Its planned supported domain is real ``float32``,
   ``float64``, and ``int64``. X64 is required and may increase resource use
   and compilation cost; a linalg compilation has taken more than 60 seconds.
   Complex TPU inputs must fail fast under the release gate. This package extra does
   **not** prove TPU support: release evidence is pending the TPU release contract
   gate, `tests/test_tpu_contract.py <../../../tests/test_tpu_contract.py>`_,
   and `.github/workflows/tpu-release.yml <../../../.github/workflows/tpu-release.yml>`_.

Gotchas
-------

**Return types change with the backend.** Under a non-default backend, public
methods return that backend's native arrays by design. The frozen return
contract applies to the default NumPy backend. Convert explicitly at your own
boundaries:

.. code-block:: python

   from ManipulaPy.backend import get_backend
   arr = get_backend().to_numpy(result)

**Complex dtypes are supported but unused.** The backends agree on complex
semantics for conformance, but no ManipulaPy call site produces complex arrays.

**Backend selection is not thread-local.** ``set_backend`` affects the whole
process. Use ``use_backend`` scopes rather than switching backends from
multiple threads.

**Integer inputs are promoted to float64** before core math runs, so passing an
integer joint vector is safe on every backend.

See Also
--------

- :doc:`Kinematics` — the FK/IK surface the contract covers
- :doc:`Dynamics` — mass matrix, inverse and forward dynamics
- :doc:`CUDA_Kernels` — Numba CUDA kernels, which sit outside backend dispatch
