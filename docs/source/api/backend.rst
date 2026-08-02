.. _api-backend:

=========================
Backend API Reference
=========================

This page documents ``ManipulaPy.backend``, the compute-backend dispatch
package. The core numerical modules call through a single active backend
instead of importing NumPy or CuPy directly, so the same kinematics, dynamics
and analysis code runs on **NumPy, CuPy, PyTorch, or JAX**.

.. versionadded:: 1.4.0

.. tip::
   For the conceptual tour — how to select a backend, what is differentiable,
   and the performance trade-offs — see :doc:`../user_guide/Backends`.

.. warning::
   The differentiable contract covers ``utils``, ``kinematics``, ``dynamics``,
   and ``singularity`` only. Other modules run under every backend through
   host-boundary conversion and carry **no gradient guarantee**. See
   :doc:`../user_guide/Backends` for the full list.

Quick Navigation
================

.. contents::
   :local:
   :depth: 2
   :backlinks: none

Selection API
=============

.. currentmodule:: ManipulaPy.backend

NumPy is registered and activated eagerly at import time and is the process
default. CuPy, PyTorch and JAX are registered lazily on first request, behind
an ``importlib.util.find_spec`` probe, so ManipulaPy stays importable on
machines where those libraries are absent.

.. autofunction:: set_backend

.. autofunction:: get_backend

.. autofunction:: use_backend

.. autofunction:: register

.. autofunction:: get_registered

``set_backend``, ``use_backend`` and ``get_backend`` are also re-exported at the
package root, so ``from ManipulaPy import use_backend`` is equivalent to
importing from ``ManipulaPy.backend``.

**Usage**

.. code-block:: python

   from ManipulaPy.backend import get_backend, set_backend, use_backend

   set_backend("torch")            # process-wide, until changed

   with use_backend("jax"):        # scoped; restored even if the block raises
       T = robot.forward_kinematics(thetalist)

   arr = get_backend().to_numpy(T)  # convert at your own boundary

.. note::
   Selection is process-wide and explicit. There is no environment sniffing,
   no per-call backend argument, and no thread-local stack — the registry and
   the active-backend reference are guarded by one re-entrant lock, but all
   threads share the same active backend.

Registered Backends
===================

.. list-table::
   :header-rows: 1
   :widths: 14 26 15 15 30

   * - Name
     - Class
     - ``is_concrete``
     - ``gpu_capable``
     - Registration
   * - ``numpy``
     - ``NumpyBackend``
     - ``True``
     - ``False``
     - eager (default)
   * - ``cupy``
     - ``CupyBackend``
     - ``True``
     - ``True``
     - lazy; ``ImportError`` if CuPy is missing
   * - ``torch``
     - ``TorchBackend``
     - ``False``
     - ``False``
     - lazy; ``ImportError`` if PyTorch is missing
   * - ``jax``
     - ``JaxBackend``
     - ``False``
     - ``False``
     - lazy; ``ImportError`` if JAX is missing

``is_concrete`` is ``False`` for the tracing backends, whose arrays may be
autograd/``jit`` tracers rather than materialised values; value-keyed caches
such as the mass-matrix cache are bypassed when it is ``False``.
``gpu_capable`` is the single dispatch-boundary predicate for the Numba CUDA
kernel path — only CuPy advertises it, so the GPU trajectory kernels launch
from the CuPy backend rather than from a direct hardware probe.

ArrayBackend Protocol
=====================

.. autoclass:: ArrayBackend
   :members:
   :undoc-members:
   :show-inheritance:

The method set was fixed by a call-site audit of the numerical modules and is
intentionally minimal. Signatures mirror NumPy semantics, so an existing
``np.<name>(...)`` call maps to ``backend.<name>(...)`` without changing
arguments, and no method mutates its input.

Errors
======

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Exception
     - Raised when
   * - ``ValueError``
     - ``register`` is given a name that is already registered, or a
       selection function is given an unknown name (the message lists the
       registered names).
   * - ``ImportError``
     - ``"cupy"``, ``"torch"`` or ``"jax"`` is requested but the library is
       not installed. The message names the extra to install.

See Also
========

* :doc:`../user_guide/Backends` -- backend user guide, differentiable contract, and gotchas
* :doc:`kinematics` -- SerialManipulator, covered by the differentiable contract
* :doc:`dynamics` -- ManipulatorDynamics, covered by the differentiable contract
* :doc:`singularity` -- singularity analysis, covered by the differentiable contract
* :doc:`cuda_kernels` -- Numba CUDA kernels, which sit outside backend dispatch
