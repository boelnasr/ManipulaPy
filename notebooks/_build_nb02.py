"""Build and execute notebooks/02_forward_kinematics.ipynb.

Run from the notebooks/ directory so `_shared` is importable at execute time:
    cd notebooks && python3 _build_nb02.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "02_forward_kinematics.ipynb")


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


cells = [
    md(
        "# 02 · Forward Kinematics — Product of Exponentials\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/boelnasr/ManipulaPy/blob/main/"
        "notebooks/02_forward_kinematics.ipynb)\n"
        "\n"
        "> **ManipulaPy teaching course — notebook 2 of 11.** Running robot: Franka Panda.\n"
        "\n"
        "**Forward kinematics (FK)** answers: *given the joint angles, where is the "
        "end-effector?* In notebook 01 we learned that following one screw axis "
        "$\\mathcal{S}$ for an angle $\\theta$ produces a rigid-body transform "
        "$e^{[\\mathcal{S}]\\theta}$. A robot arm is just a **chain of screws**, so its FK "
        "is a *product of exponentials* — one factor per joint. That is the whole idea of "
        "this notebook, applied to the 7-DOF Panda."
    ),
    md(
        "### Running on Colab or another cloud platform?\n"
        "\n"
        "The next cell bootstraps the environment on Google Colab (clone the repo + install "
        "ManipulaPy). It is a **no-op when you run locally** from a clone of the repo."
    ),
    code(
        "# Cloud bootstrap (no-op when running locally from the repo).\n"
        "import sys\n"
        'if "google.colab" in sys.modules:\n'
        "    !git clone -q https://github.com/boelnasr/ManipulaPy.git\n"
        "    %cd ManipulaPy/notebooks\n"
        "    !pip install -q -e ..\n"
        '    print("Colab setup complete.")'
    ),
    code(
        "import os, sys\n"
        'sys.path.insert(0, os.path.join(os.getcwd(), "_shared"))\n'
        "from tikz import render_tikz_file, setup_pgf, embed_pgf_fig\n"
        "import numpy as np\n"
        "from helpers import load_panda, HOME, N_JOINTS, joint_limits\n"
        "from ManipulaPy.utils import MatrixExp6, VecTose3, adjoint_transform, TransInv\n"
        "np.set_printoptions(precision=4, suppress=True)\n"
        "\n"
        "sm, dyn = load_panda()\n"
        'print("Panda loaded |", N_JOINTS, "actuated joints")'
    ),

    # --- 1. the FK problem ---
    md(
        "## 1. The forward-kinematics problem\n"
        "\n"
        "A serial manipulator is an open chain of links connected by joints. Choosing the "
        "joint values $\\theta=(\\theta_1,\\dots,\\theta_n)$ fixes the pose of every link, "
        "including the end-effector. FK is the map\n"
        "\n"
        "$$\\theta \\;\\longmapsto\\; T_{sb}(\\theta)\\in SE(3),$$\n"
        "\n"
        "the homogeneous transform of the end-effector frame $\\{b\\}$ relative to the "
        "fixed space frame $\\{s\\}$. It is always well-defined and unique (unlike inverse "
        "kinematics in notebook 04)."
    ),

    # --- 2. PoE formula ---
    md(
        "## 2. The Product of Exponentials (PoE) formula\n"
        "\n"
        "Fix the robot at its **zero configuration** $\\theta=0$ and record the "
        "end-effector pose there: call it the **home configuration** $M$. Each joint $i$ "
        "has a screw axis $\\mathcal{S}_i$ expressed in the space frame (built from the "
        "joint's axis direction and a point on it, exactly as in notebook 01). Then\n"
        "\n"
        "$$T_{sb}(\\theta)=e^{[\\mathcal{S}_1]\\theta_1}\\,e^{[\\mathcal{S}_2]\\theta_2}"
        "\\cdots e^{[\\mathcal{S}_n]\\theta_n}\\,M.$$\n"
        "\n"
        "Read it right-to-left: start at the home pose $M$, then apply each joint's screw "
        "motion in turn. The figure shows the structure for the Panda."
    ),
    code('render_tikz_file("_figures/src/poe_chain.tex", name="poe_chain")'),
    md(
        "ManipulaPy stores the screw axes as the columns of `sm.S_list` (space frame) and "
        "`sm.B_list` (body frame), and the home configuration as `sm.M_list`. The home "
        "configuration is exactly the FK at the zero configuration:"
    ),
    code(
        "M = np.array(sm.M_list)\n"
        'print("home configuration M =\\n", M)\n'
        "assert np.allclose(sm.forward_kinematics(np.zeros(N_JOINTS)), M)\n"
        'print("\\nFK(0) == M :", True)'
    ),

    # --- 3. the 7-vs-8 convention ---
    md(
        "## 3. The Panda's 7-vs-8 joint convention\n"
        "\n"
        "A small but important detail. The Panda URDF has **7 revolute joints** plus a "
        "**fixed `panda_joint8` flange** (and the hand). ManipulaPy includes that fixed "
        "frame in the screw list, so `S_list` has **8 columns** even though only **7 joints "
        "actuate**. `forward_kinematics` accepts either a 7-vector (the actuated joints) or "
        "an 8-vector; the fixed flange contributes no motion. **This course always uses the "
        "7 actuated joints** (`N_JOINTS = 7`)."
    ),
    code(
        'print("S_list shape:", sm.S_list.shape, " (6 x 8: 7 revolute + 1 fixed flange)")\n'
        'print("B_list shape:", sm.B_list.shape)\n'
        "# Passing 7 or 8 joint values gives the same pose at the zero configuration:\n"
        "T7 = sm.forward_kinematics(np.zeros(7))\n"
        "T8 = sm.forward_kinematics(np.zeros(8))\n"
        'print("FK(zeros7) == FK(zeros8):", np.allclose(T7, T8))'
    ),

    # --- 4. PoE by hand ---
    md(
        "## 4. Building the PoE product by hand\n"
        "\n"
        "To make the formula concrete, let us evaluate it ourselves from `S_list` and `M` "
        "and check it against `forward_kinematics`. We multiply the per-joint exponentials "
        "left-to-right and post-multiply by $M$ — using the first 7 screw columns."
    ),
    code(
        "theta = HOME.copy()\n"
        "S = np.array(sm.S_list)[:, :N_JOINTS]      # 6 x 7 actuated screw axes\n"
        "\n"
        "T = np.eye(4)\n"
        "for i in range(N_JOINTS):\n"
        "    T = T @ MatrixExp6(VecTose3(S[:, i] * theta[i]))\n"
        "T = T @ M                                  # post-multiply by the home pose\n"
        "\n"
        "T_fk = sm.forward_kinematics(theta, frame='space')\n"
        'print("hand-built PoE =\\n", T)\n'
        'print("\\nmatches forward_kinematics(space):", np.allclose(T, T_fk))\n'
        "assert np.allclose(T, T_fk)"
    ),

    # --- 5. space vs body ---
    md(
        "## 5. Space frame vs body frame\n"
        "\n"
        "The PoE formula has a twin. The **space form** above uses space-frame screws "
        "$\\mathcal{S}_i$ and post-multiplies $M$. The **body form** uses body-frame screws "
        "$\\mathcal{B}_i$ and *pre*-multiplies $M$:\n"
        "\n"
        "$$T_{sb}(\\theta)=M\\,e^{[\\mathcal{B}_1]\\theta_1}\\cdots e^{[\\mathcal{B}_n]\\theta_n}.$$\n"
        "\n"
        "The two describe the **same** physical pose. The screw sets are related by the "
        "adjoint of the home configuration (from notebook 01), "
        "$\\mathcal{B}_i=[\\mathrm{Ad}_{M^{-1}}]\\,\\mathcal{S}_i$. ManipulaPy selects the "
        "form via the `frame` argument."
    ),
    code(
        "T_space = sm.forward_kinematics(HOME, frame='space')\n"
        "T_body  = sm.forward_kinematics(HOME, frame='body')\n"
        'print("space form ee position:", T_space[:3, 3])\n'
        'print("body  form ee position:", T_body[:3, 3])\n'
        'print("same pose:", np.allclose(T_space, T_body))\n'
        "\n"
        "# Verify the adjoint relation B_i = Ad_{M^-1} S_i for the actuated joints.\n"
        "Ad_Minv = adjoint_transform(TransInv(M))\n"
        "B = np.array(sm.B_list)[:, :N_JOINTS]\n"
        "assert np.allclose(Ad_Minv @ S, B, atol=1e-6)\n"
        'print("B_i == Ad_{M^-1} S_i :", True)'
    ),

    # --- 6. using FK ---
    md(
        "## 6. Using forward kinematics\n"
        "\n"
        "With FK in hand we can trace where the end-effector goes as the robot moves. Below "
        "we sweep joint 1 across its range and plot the end-effector's $x$–$y$ path (the "
        "arm swings about the base axis), rendered with the TikZ/PGF backend."
    ),
    code(
        "plt = setup_pgf()\n"
        "q = HOME.copy()\n"
        "angles = np.linspace(-np.pi, np.pi, 120)\n"
        "xy = []\n"
        "for a in angles:\n"
        "    q[0] = a\n"
        "    xy.append(sm.forward_kinematics(q)[:2, 3])\n"
        "xy = np.array(xy)\n"
        "\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot(xy[:, 0], xy[:, 1], color='tab:blue')\n"
        "ax.scatter([0], [0], color='black', marker='+', s=80, label='base')\n"
        "ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')\n"
        "ax.set_title('End-effector path as joint 1 sweeps'); ax.legend()\n"
        'embed_pgf_fig(fig, name="fk_joint1_sweep")'
    ),

    # --- 7. visualizing the robot ---
    md(
        "## 7. Visualizing the robot\n"
        "\n"
        "The Product-of-Exponentials math is easier to trust when you can *see* the arm. "
        "ManipulaPy's **URDF module** exposes the full kinematic structure through "
        "`robot.link_fk`, which returns the world frame of every link at a given "
        "configuration — **no mesh files and no GUI**, so it renders headless (and on "
        "Colab). We use it to draw the Panda as a 3D stick figure at several "
        "configurations.\n"
        "\n"
        "*(The mesh-based PyBullet simulator is covered in notebook 09; it needs the Franka "
        "mesh package, which isn't bundled with ManipulaPy, so we visualize from the "
        "kinematic structure here.)*"
    ),
    code(
        "from ManipulaPy.urdf_processor import URDFToSerialManipulator\n"
        "from helpers import PANDA_URDF\n"
        "\n"
        "# The URDF module gives every link's world frame via link_fk (mesh-free, headless).\n"
        "_robot = URDFToSerialManipulator(PANDA_URDF).robot\n"
        "_arm = [j.name for j in _robot.actuated_joints][:N_JOINTS]\n"
        '_chain = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",\n'
        '          "panda_link5", "panda_link6", "panda_link7", "panda_link8", "panda_hand"]\n'
        "\n"
        "def link_points(q):\n"
        "    '''World positions of the main-chain link frames at config q (7-vector).'''\n"
        "    lfk = _robot.link_fk(cfg={n: float(q[i]) for i, n in enumerate(_arm)})\n"
        '    name2T = {getattr(L, "name", str(L)): T for L, T in lfk.items()}\n'
        "    return np.array([name2T[n][:3, 3] for n in _chain if n in name2T])\n"
        "\n"
        "# Cross-check: the URDF module's fingertip frame agrees with SerialManipulator FK.\n"
        "_lfk = _robot.link_fk(cfg={n: float(HOME[i]) for i, n in enumerate(_arm)})\n"
        '_name2T = {getattr(L, "name", str(L)): T for L, T in _lfk.items()}\n'
        'assert np.allclose(_name2T["panda_leftfinger"][:3, 3],\n'
        "                   sm.forward_kinematics(HOME)[:3, 3], atol=1e-6)\n"
        'print("URDF link_fk agrees with forward_kinematics at the end-effector :", True)'
    ),
    code(
        "plt = setup_pgf()\n"
        "\n"
        "def _equal3d(ax, P, pad=0.1):\n"
        "    lo, hi = P.min(0) - pad, P.max(0) + pad\n"
        "    c = (lo + hi) / 2; r = (hi - lo).max() / 2\n"
        "    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)\n"
        "\n"
        "poses = [\n"
        '    (np.zeros(N_JOINTS), "zero configuration"),\n'
        '    (HOME, "HOME"),\n'
        '    (np.array([np.pi/2, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785]), "joint 1 = +90 deg"),\n'
        '    (np.array([0.0, 0.4, 0.0, -1.0, 0.0, 1.7, 0.785]), "joints 2 and 4 moved"),\n'
        "]\n"
        "\n"
        "fig = plt.figure(figsize=(7.4, 6.4))\n"
        "for k, (q, title) in enumerate(poses, 1):\n"
        '    ax = fig.add_subplot(2, 2, k, projection="3d")\n'
        "    P = link_points(q)\n"
        '    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color="tab:blue", lw=2.5, ms=3)\n'
        '    ax.scatter(*P[0], color="black", marker="s", s=30)            # base\n'
        "    ee = sm.forward_kinematics(q)[:3, 3]\n"
        '    ax.scatter(*ee, color="tab:red", marker="*", s=110)           # end-effector (FK)\n'
        "    _equal3d(ax, np.vstack([P, ee]))\n"
        "    ax.set_title(title, fontsize=9)\n"
        '    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")\n'
        '    ax.view_init(elev=18, azim=-60)\n'
        'fig.suptitle("Panda kinematic skeleton (URDF module link_fk)")\n'
        'embed_pgf_fig(fig, name="panda_skeleton_montage")'
    ),

    # --- smoke test ---
    md("## Smoke test\n\nAsserts the key invariants of this notebook in one cell."),
    code(
        "M = np.array(sm.M_list)\n"
        "# Home configuration is FK at zero.\n"
        "assert np.allclose(sm.forward_kinematics(np.zeros(N_JOINTS)), M)\n"
        "# Hand-built PoE equals forward_kinematics.\n"
        "S = np.array(sm.S_list)[:, :N_JOINTS]\n"
        "T = np.eye(4)\n"
        "for i in range(N_JOINTS):\n"
        "    T = T @ MatrixExp6(VecTose3(S[:, i] * HOME[i]))\n"
        "T = T @ M\n"
        "assert np.allclose(T, sm.forward_kinematics(HOME, frame='space'))\n"
        "# Space and body forms agree.\n"
        "assert np.allclose(sm.forward_kinematics(HOME, frame='space'),\n"
        "                   sm.forward_kinematics(HOME, frame='body'))\n"
        "# Every FK output is a valid SE(3) element.\n"
        "Tt = sm.forward_kinematics(HOME)\n"
        "assert np.allclose(Tt[3], [0, 0, 0, 1]) and np.isclose(np.linalg.det(Tt[:3, :3]), 1.0)\n"
        'print("nb02 forward kinematics: smoke OK")'
    ),

    # --- exercises ---
    md(
        "## Try it\n"
        "\n"
        "1. Compute the end-effector pose at $\\theta=0$ and read off its height "
        "($z$-coordinate). What physical configuration is this?\n"
        "2. Move only joint 4 by $-\\pi/2$ from `HOME` and report how far (in metres) the "
        "end-effector moves.\n"
        "3. Re-derive the body-form PoE by hand (pre-multiply $M$, use `sm.B_list`) and "
        "confirm it equals `forward_kinematics(..., frame='body')`.\n"
        "\n"
        "*Next up — notebook 03: differentiating FK to get the **Jacobian**, velocity "
        "mapping, and the Panda's 7-DOF redundancy.*"
    ),
    md(
        "## References\n"
        "\n"
        "1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and "
        "Control*, Cambridge University Press, 2017. — Chapter 4, *Forward Kinematics* "
        "(Product of Exponentials, space and body forms).\n"
        "2. R. M. Murray, Z. Li, and S. S. Sastry, *A Mathematical Introduction to "
        "Robotic Manipulation*, CRC Press, 1994. — Origin of the product-of-exponentials "
        "formula.\n"
        "3. B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics: Modelling, "
        "Planning and Control*, Springer, 2009.\n"
        "4. ManipulaPy documentation — https://manipulapy.readthedocs.io/ · "
        "source — https://github.com/boelnasr/ManipulaPy\n"
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": HERE}})
with open(NB_PATH, "w") as f:
    nbf.write(nb, f)
print("wrote and executed", NB_PATH)
