"""Shared robot loading + conventions for the notebook course.

Running robot: Franka Emika Panda. The URDF has 7 revolute joints plus a fixed
``panda_joint8`` flange, so ManipulaPy's screw list has 8 columns while FK /
Jacobian accept either a 7- or 8-vector. The course standardises on the 7
actuated joints; see notebook 02 for the full explanation.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
from ManipulaPy import ManipulaPy_data
from ManipulaPy.urdf_processor import URDFToSerialManipulator

N_JOINTS = 7
PANDA_URDF = os.path.join(
    os.path.dirname(ManipulaPy_data.__file__), "franka_panda", "panda.urdf"
)

# A non-singular, visually clear default configuration (radians), 7 actuated joints.
HOME = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])


def load_panda():
    """Return ``(SerialManipulator, ManipulatorDynamics)`` for the Panda."""
    proc = URDFToSerialManipulator(PANDA_URDF)
    return proc.serial_manipulator, proc.dynamics


def joint_limits():
    """Conservative Panda joint limits (rad) for the 7 actuated joints."""
    return [
        (-2.90, 2.90),
        (-1.76, 1.76),
        (-2.90, 2.90),
        (-3.07, -0.07),
        (-2.90, 2.90),
        (-0.02, 3.75),
        (-2.90, 2.90),
    ]


_PB_URDF = None


def panda_pybullet_urdf():
    """Path to a PyBullet-loadable copy of the bundled Panda URDF.

    The shipped ``panda.urdf`` references its meshes through ROS
    ``package://franka_description/...`` URIs. ManipulaPy's own URDF parser
    resolves those, but PyBullet's C++ loader does not, so ``Simulation``
    cannot open the file as-is. This writes a copy with plain relative mesh
    paths into a temp directory (meshes symlinked, copied as a fallback) and
    returns the copy's path.

    The copy also gets a default ``<inertial>`` block per link — the same
    mass-1/identity values PyBullet would assume anyway — purely to silence
    its per-link "No inertial data" warnings. The notebooks only pose the
    robot kinematically, so these values are never exercised.
    """
    global _PB_URDF
    if _PB_URDF is None or not os.path.exists(_PB_URDF):
        import re
        import shutil
        import tempfile

        src_dir = os.path.dirname(PANDA_URDF)
        tmp_dir = tempfile.mkdtemp(prefix="manipulapy_panda_pb_")
        with open(PANDA_URDF) as f:
            text = f.read().replace("package://franka_description/meshes/", "meshes/")
        inertial = (
            '<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="1"/>'
            '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
        )
        text = re.sub(r'(<link name="[^"]*">)', r"\1" + inertial, text)
        # Self-closing links (the flange) need expanding to take the block.
        text = re.sub(
            r'<link name="([^"]*)"/>', r'<link name="\1">' + inertial + "</link>", text
        )
        path = os.path.join(tmp_dir, "panda.urdf")
        with open(path, "w") as f:
            f.write(text)
        try:
            os.symlink(os.path.join(src_dir, "meshes"), os.path.join(tmp_dir, "meshes"))
        except OSError:
            shutil.copytree(os.path.join(src_dir, "meshes"), os.path.join(tmp_dir, "meshes"))
        _PB_URDF = path
    return _PB_URDF


@contextlib.contextmanager
def quiet_pybullet():
    """Silence C-level stderr inside the block (PyBullet URDF-importer chatter).

    The bundled Panda URDF carries no ``<inertial>`` tags, so PyBullet prints a
    warning per link straight to file descriptor 2, bypassing Python's stderr.
    """
    saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull)
        os.close(saved_fd)


def sim_snapshot(name, target=(0.25, 0.0, 0.45), distance=1.5, yaw=55, pitch=-20,
                 width=900, height=620, outdir=None):
    """Photograph the current PyBullet world to ``_figures/<name>.png``.

    Uses the software renderer, so it works in DIRECT (headless) mode; the PNG
    is committed alongside the TikZ figures and the returned
    ``IPython.display.Image`` embeds it in the notebook.
    """
    import matplotlib.image as mpimg
    import pybullet as p
    from IPython.display import Image

    view = p.computeViewMatrixFromYawPitchRoll(list(target), distance, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(60, width / height, 0.05, 5.0)
    img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
    outdir = outdir or os.path.join(os.path.dirname(__file__), "..", "_figures")
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{name}.png")
    mpimg.imsave(png, rgb)
    return Image(filename=png)
