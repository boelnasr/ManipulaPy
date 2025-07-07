# Installation Guide

This page shows the **fastest way** to get ManipulaPy on your machine and explains the
optional *extras* published on PyPI — they mirror the names you see in
`pyproject.toml` (`gpu‑cuda11` & co.).

.. note::
All commands assume you are inside a fresh virtual environment (`python -m venv`
or Conda).  ManipulaPy supports **Python 3.8 – 3.12** on Linux, macOS (CPU‑only),
and Windows/WSL2.

## System requirements

\===================  =============================================================
Component            Minimum / recommended
\===================  =============================================================
CPU                  x86‑64 or Apple Silicon (runs in Rosetta)
RAM                  4 GB min / 8 GB recommended
GPU (optional)       *NVIDIA* CUDA 11.8 / 12.4 or *AMD* ROCm 5.7
\===================  =============================================================

## Quick install (CPU‑only)

If you do **not** need GPU acceleration right away:

.. code-block:: bash

pip install manipulapy

This brings in *all* core features — kinematics, dynamics, perception, trajectory
planning, PyBullet sim, OpenCV, Torch, scikit‑learn, etc.  Nothing else to add.

## GPU extras

ManipulaPy ships several "extras" so you can choose the CUDA flavour that matches your
driver:

\==========================  When to use                                      Command
\==========================  ------------------------------------------------  ----------------------------------------
CUDA 11.x (470+ drivers)    Most LTS distro packages                         `pip install manipulapy[gpu-cuda11]`
CUDA 12.x (535+ drivers)    Latest NVIDIA toolkits / Ubuntu 24.04            `pip install manipulapy[gpu-cuda12]`
AMD ROCm 5.6+               Recent Radeon/MI cards                           `pip install manipulapy[gpu-rocm]`
Legacy PyCUDA               Prefer `pycuda` API over CuPy                 `pip install manipulapy[gpu-pycuda]`
Meta‑extra (defaults CUDA11) Installs CuPy 11 build                          `pip install manipulapy[gpu]`
\==========================  ------------------------------------------------  ----------------------------------------

After the install you can verify CUDA access:

.. code-block:: python

import cupy as cp
print("CUDA available:", cp.cuda.is\_available())
print("Device count:", cp.cuda.runtime.getDeviceCount())

## Development & docs

* **Editable dev install** (tests, Black, MyPy, coverage):

  .. code-block:: bash

  git clone [https://github.com/boelnasr/ManipulaPy.git](https://github.com/boelnasr/ManipulaPy.git)
  cd ManipulaPy
  pip install -e .\[dev]

* **Build the documentation locally**:

  .. code-block:: bash

  pip install manipulapy\[docs]
  sphinx-build -b html docs docs/\_build/html

## Conda & Docker

ManipulaPy is mirrored on **conda‑forge** and as ready‑made Docker images:

.. code-block:: bash

# Conda CPU build

conda install -c conda-forge manipulapy

# Docker (CPU)

docker pull manipulapy/manipulapy\:cpu-latest

# Docker (GPU, CUDA 12)

docker pull manipulapy/manipulapy\:cuda12-latest
docker run --gpus all -it manipulapy/manipulapy\:cuda12-latest

## Troubleshooting

* `ImportError: No module named 'cupy'` – install the matching extra (see table).

* `CUDA out of memory` – lower batch sizes or hide the GPU:

  .. code-block:: bash

  CUDA\_VISIBLE\_DEVICES="" python your\_script.py  # forces CPU fallback

* **Windows build errors** – ensure *Microsoft C++ Build Tools* are installed.

## Update / Upgrade

.. code-block:: bash

pip install --upgrade manipulapy   # upgrades the core
pip install --upgrade "manipulapy\[gpu-cuda12]"  # upgrade with extras

## Next steps

* \:doc:`quickstart` – run your first pick‑and‑place
* \:doc:`user_guide/kinematics` – learn the API in depth
* \:doc:`examples/basic_manipulation` – full end‑to‑end notebook
