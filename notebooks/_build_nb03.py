"""Build and execute notebooks/03_velocity_kinematics_jacobians.ipynb.

Run from the notebooks/ directory so `_shared` is importable at execute time:
    cd notebooks && python3 _build_nb03.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "03_velocity_kinematics_jacobians.ipynb")


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


cells = [
    md(
        "# 03 · Velocity Kinematics and the Jacobian\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/boelnasr/ManipulaPy/blob/notebooks/tutorials/"
        "notebooks/03_velocity_kinematics_jacobians.ipynb)\n"
        "\n"
        "> **ManipulaPy teaching course — notebook 3 of 11.** Running robot: Franka Panda.\n"
        "\n"
        "Forward kinematics (notebook 02) maps joint *angles* to an end-effector *pose*. "
        "Differentiating that map relates joint *velocities* to end-effector *velocity* — "
        "and the matrix that does it is the **Jacobian** $J(\\theta)$. The Jacobian is the "
        "workhorse of robotics: it drives velocity control, inverse kinematics (notebook "
        "04), force/torque relationships, singularity analysis (notebook 06), and — for the "
        "7-DOF Panda — **redundancy**."
    ),
    md(
        "### Running on Colab or another cloud platform?\n"
        "\n"
        "The next cell bootstraps the environment on Google Colab. It is a **no-op when you "
        "run locally** from a clone of the repo."
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
        "from helpers import load_panda, HOME, N_JOINTS\n"
        "from ManipulaPy.utils import adjoint_transform\n"
        "np.set_printoptions(precision=4, suppress=True)\n"
        "\n"
        "sm, dyn = load_panda()\n"
        'print("Panda loaded |", N_JOINTS, "actuated joints")'
    ),

    # --- 1. velocity kinematics ---
    md(
        "## 1. Velocity kinematics\n"
        "\n"
        "Differentiating the forward kinematics with respect to time relates the joint-rate "
        "vector $\\dot\\theta$ to the end-effector **twist** $\\mathcal{V}=[\\omega;\\,v]$ "
        "(the angular + linear velocity from notebook 01):\n"
        "\n"
        "$$\\mathcal{V}=J(\\theta)\\,\\dot\\theta.$$\n"
        "\n"
        "$J(\\theta)\\in\\mathbb{R}^{6\\times n}$ is the **Jacobian**. Each column is the "
        "instantaneous screw axis of one joint, expressed in the chosen frame — so the "
        "Jacobian is literally \"the joints' screw axes at the current configuration.\""
    ),

    # --- 2. space and body Jacobians ---
    md(
        "## 2. The space and body Jacobians\n"
        "\n"
        "As with forward kinematics, there are two frames. The **space Jacobian** $J_s$ "
        "expresses the twist in the fixed frame; the **body Jacobian** $J_b$ expresses it "
        "in the end-effector frame. They describe the same motion and are related by the "
        "adjoint of the current pose, $J_s=[\\mathrm{Ad}_{T_{sb}}]\\,J_b$.\n"
        "\n"
        "Geometrically, $J$ is the linear map that sends the unit ball of joint rates to "
        "the **manipulability ellipsoid** of achievable end-effector velocities — its "
        "principal axes are the singular values of $J$ (we return to this in notebook 06)."
    ),
    code('render_tikz_file("_figures/src/jacobian_map.tex", name="jacobian_map")'),
    code(
        "Js = sm.jacobian(HOME, frame='space')\n"
        "Jb = sm.jacobian(HOME, frame='body')\n"
        'print("space Jacobian shape:", Js.shape, " (6 x 7)")\n'
        "\n"
        "# The two Jacobians are related by the adjoint of the current pose.\n"
        "T = sm.forward_kinematics(HOME, frame='space')\n"
        "assert np.allclose(Js, adjoint_transform(T) @ Jb, atol=1e-5)\n"
        'print("J_s == Ad_T J_b :", True)'
    ),

    # --- 3. end-effector velocity ---
    md(
        "## 3. From joint rates to end-effector velocity\n"
        "\n"
        "The forward velocity map $\\mathcal{V}=J\\dot\\theta$ is implemented directly by "
        "`end_effector_velocity`. We check it against the matrix product:"
    ),
    code(
        "dtheta = np.array([0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.1])\n"
        "V = sm.end_effector_velocity(HOME, dtheta, frame='space')\n"
        'print("end-effector twist V = [omega; v]:", V)\n'
        "assert np.allclose(V, Js @ dtheta, atol=1e-9)\n"
        'print("\\nV == J @ dtheta :", True)'
    ),

    # --- 4. resolved-rate ---
    md(
        "## 4. The inverse problem: resolved-rate motion\n"
        "\n"
        "More often we want the reverse: *what joint rates produce a desired end-effector "
        "velocity?* Inverting $\\mathcal{V}=J\\dot\\theta$ gives "
        "$\\dot\\theta=J^{\\dagger}\\mathcal{V}$ with the Moore–Penrose pseudoinverse "
        "$J^{\\dagger}$ — the basis of **resolved-rate** control. ManipulaPy exposes it as "
        "`joint_velocity`."
    ),
    code(
        "V_desired = np.array([0, 0, 0, 0.05, 0.0, 0.0])   # move +x at 5 cm/s, no rotation\n"
        "dtheta_sol = sm.joint_velocity(HOME, V_desired, frame='space')\n"
        'print("joint rates:", dtheta_sol)\n'
        "\n"
        "# The recovered joint rates reproduce the requested twist.\n"
        "assert np.allclose(Js @ dtheta_sol, V_desired, atol=1e-3)\n"
        'print("\\nJ @ dtheta_sol == V_desired :", True)'
    ),
    md(
        "### Resolved rates, live in the simulator\n"
        "\n"
        "Integrating $\\dot\\theta=J^{\\dagger}\\mathcal{V}$ over time turns the rate "
        "command into a motion. Below we hold the same twist (+$x$ at 5 cm/s) for 4 s, "
        "re-solving `joint_velocity` at every step as the configuration changes, then "
        "replay the result in ManipulaPy's **simulation module** (headless PyBullet). "
        "`plot_trajectory` draws the end-effector's path as real geometry — the straight "
        "orange line the controller was asked to produce."
    ),
    code(
        "dt, T_total = 0.05, 4.0\n"
        "qk = HOME.copy()\n"
        "q_traj, ee_path = [qk.copy()], [sm.forward_kinematics(qk)[:3, 3]]\n"
        "for _ in range(int(T_total / dt)):\n"
        "    dq = sm.joint_velocity(qk, V_desired, frame='space')  # re-solve each step\n"
        "    qk = qk + dq * dt\n"
        "    q_traj.append(qk.copy())\n"
        "    ee_path.append(sm.forward_kinematics(qk)[:3, 3])\n"
        "ee_path = np.array(ee_path)\n"
        "\n"
        "disp = ee_path[-1] - ee_path[0]\n"
        'print("end-effector displacement:", np.round(disp, 4), "m  (expect ~[0.2, 0, 0])")\n'
        "assert np.allclose(disp, [0.2, 0, 0], atol=5e-3)          # 5 cm/s * 4 s, straight\n"
        "off_axis = np.abs(ee_path[:, 1:] - ee_path[0, 1:]).max()\n"
        'print("max off-axis deviation  :", round(off_axis * 1000, 2), "mm")'
    ),
    code(
        "import os\n"
        'os.environ.setdefault("MANIPULAPY_PYBULLET_CONNECT", "DIRECT")  # headless; remove to watch in a GUI\n'
        "import logging\n"
        "from helpers import panda_pybullet_urdf, joint_limits, sim_snapshot, quiet_pybullet\n"
        "from ManipulaPy.sim import Simulation\n"
        "import pybullet as p\n"
        "\n"
        "with quiet_pybullet():\n"
        "    sim = Simulation(panda_pybullet_urdf(), joint_limits())\n"
        "# Simulation's logger defaults to DEBUG; keep the notebook output clean.\n"
        'logging.getLogger("SimulationLogger").setLevel(logging.WARNING)\n'
        "\n"
        "# Pose the arm at the final configuration; fingers (also non-fixed) padded with 0.\n"
        "for j, qj in zip(sim.non_fixed_joints, list(q_traj[-1]) + [0.0, 0.0]):\n"
        "    p.resetJointState(sim.robot_id, j, qj)\n"
        "sim.plot_trajectory(list(ee_path), line_width=5, color=[0.9, 0.4, 0.1])\n"
        'img = sim_snapshot("sim_resolved_rate", target=(0.5, 0, 0.5), distance=1.1, yaw=35, pitch=-15)\n'
        "sim.disconnect_simulation()\n"
        "img"
    ),

    # --- 5. redundancy and null space ---
    md(
        "## 5. Redundancy and the null space\n"
        "\n"
        "The Panda has **7 joints** but the task space (a twist) is only **6-dimensional**, "
        "so $J$ is $6\\times7$ — wider than it is tall. At a non-singular configuration it "
        "has rank 6, leaving a **1-dimensional null space**: a direction in joint-rate "
        "space that produces **zero** end-effector motion. Moving along it reshapes the arm "
        "(elbow swivel) while the end-effector stays put. This *self-motion* is what "
        "redundancy buys — room to avoid joint limits, obstacles, and singularities while "
        "still holding the task pose."
    ),
    code(
        "rank = np.linalg.matrix_rank(Js)\n"
        'print("rank(J):", rank, " -> null-space dimension:", N_JOINTS - rank)\n'
        "\n"
        "# A null-space joint velocity produces (numerically) zero end-effector twist.\n"
        "_, _, Vt = np.linalg.svd(Js)\n"
        "null_dir = Vt[rank]                       # the 1-D null-space basis vector\n"
        'print("end-effector twist from null-space motion:", np.round(Js @ null_dir, 9))\n'
        "assert np.allclose(Js @ null_dir, 0, atol=1e-9)"
    ),
    md(
        "To *see* the self-motion, we follow the null space as a curve: take small steps "
        "along the (re-computed) null direction. The arm posture changes substantially "
        "while the end-effector barely moves."
    ),
    code(
        "# Follow the null-space curve in small steps, keeping the direction consistent.\n"
        "q = HOME.copy()\n"
        "ee0 = sm.forward_kinematics(HOME)[:3, 3]\n"
        "prev = None\n"
        "for _ in range(20):\n"
        "    _, _, Vt = np.linalg.svd(sm.jacobian(q, 'space'))\n"
        "    d = Vt[6]\n"
        "    if prev is not None and np.dot(d, prev) < 0:\n"
        "        d = -d\n"
        "    prev = d\n"
        "    q = q + 0.04 * d\n"
        "ee1 = sm.forward_kinematics(q)[:3, 3]\n"
        'print("largest joint change:", np.round(np.degrees(np.abs(q - HOME)).max(), 1), "deg")\n'
        'print("end-effector drift  :", round(np.linalg.norm(ee1 - ee0) * 1000, 2), "mm")\n'
        "assert np.linalg.norm(ee1 - ee0) < 5e-3        # < 5 mm: essentially the same pose"
    ),
    code(
        "# Draw the two postures (mesh-free, via the URDF module's link_fk) -- same fingertip.\n"
        "from ManipulaPy.urdf_processor import URDFToSerialManipulator\n"
        "from helpers import PANDA_URDF\n"
        "_robot = URDFToSerialManipulator(PANDA_URDF).robot\n"
        "_arm = [j.name for j in _robot.actuated_joints][:N_JOINTS]\n"
        '_chain = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",\n'
        '          "panda_link5", "panda_link6", "panda_link7", "panda_link8", "panda_hand"]\n'
        "def link_points(qq):\n"
        "    lfk = _robot.link_fk(cfg={n: float(qq[i]) for i, n in enumerate(_arm)})\n"
        '    name2T = {getattr(L, "name", str(L)): T for L, T in lfk.items()}\n'
        "    return np.array([name2T[n][:3, 3] for n in _chain if n in name2T])\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig = plt.figure(figsize=(5.6, 4.6))\n"
        'ax = fig.add_subplot(projection="3d")\n'
        "for qq, c, lab in [(HOME, 'tab:gray', 'HOME'), (q, 'tab:blue', 'null-space motion')]:\n"
        "    P = link_points(qq)\n"
        '    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color=c, lw=2.5, ms=3, label=lab)\n'
        "ee = sm.forward_kinematics(HOME)[:3, 3]\n"
        'ax.scatter(*ee, color="tab:red", marker="*", s=140, label="end-effector (fixed)")\n'
        "P2 = np.vstack([link_points(HOME), link_points(q)])\n"
        "lo, hi = P2.min(0) - 0.1, P2.max(0) + 0.1\n"
        "c0 = (lo + hi) / 2; r = (hi - lo).max() / 2\n"
        "ax.set_xlim(c0[0]-r, c0[0]+r); ax.set_ylim(c0[1]-r, c0[1]+r); ax.set_zlim(c0[2]-r, c0[2]+r)\n"
        'ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.view_init(elev=18, azim=-60)\n'
        'ax.set_title("Redundancy: same end-effector, different posture"); ax.legend(fontsize=8)\n'
        'embed_pgf_fig(fig, name="null_space_motion")'
    ),

    # --- 6. manipulability preview ---
    md(
        "## 6. A first look at manipulability\n"
        "\n"
        "How *easily* can the arm move its end-effector in each direction? The "
        "**manipulability ellipsoid** answers this: the linear-velocity block "
        "$J_v$ of the body Jacobian maps the unit joint-rate ball to an ellipsoid whose "
        "axes are the singular values of $J_v$. A scalar summary is "
        "$w=\\sqrt{\\det(J_vJ_v^{\\top})}$ — larger means more dexterous, zero means a "
        "singularity. Below we draw the velocity ellipse in the $x$–$z$ plane; notebook 06 "
        "treats singularities and manipulability in full."
    ),
    code(
        "Jv = sm.jacobian(HOME, frame='body')[3:6]      # linear-velocity rows [v]\n"
        "w = np.sqrt(np.linalg.det(Jv @ Jv.T))\n"
        'print("manipulability measure w =", round(float(w), 5))\n'
        "\n"
        "# x-z velocity ellipse: 2x2 block for (v_x, v_z).\n"
        "Jxz = Jv[[0, 2], :]\n"
        "A = Jxz @ Jxz.T\n"
        "evals, evecs = np.linalg.eigh(A)\n"
        "t = np.linspace(0, 2 * np.pi, 200)\n"
        "circle = np.vstack([np.cos(t), np.sin(t)])\n"
        "ellipse = evecs @ np.diag(np.sqrt(evals)) @ circle\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(4.2, 4.2))\n"
        "ax.plot(ellipse[0], ellipse[1], color='teal')\n"
        "for i in range(2):\n"
        "    vec = evecs[:, i] * np.sqrt(evals[i])\n"
        "    ax.annotate('', xy=vec, xytext=(0, 0),\n"
        "                arrowprops=dict(arrowstyle='->', color='tab:orange'))\n"
        "ax.set_aspect('equal'); ax.set_xlabel('$v_x$'); ax.set_ylabel('$v_z$')\n"
        "ax.set_title('End-effector velocity ellipse (x-z) at HOME')\n"
        'embed_pgf_fig(fig, name="velocity_ellipse")'
    ),

    # --- smoke test ---
    md("## Smoke test\n\nAsserts the key invariants of this notebook in one cell."),
    code(
        "Js = sm.jacobian(HOME, frame='space')\n"
        "Jb = sm.jacobian(HOME, frame='body')\n"
        "T = sm.forward_kinematics(HOME, frame='space')\n"
        "# Shape, forward map, frame relation.\n"
        "assert Js.shape == (6, N_JOINTS)\n"
        "dq = np.array([0.1, -0.2, 0.15, 0.05, -0.1, 0.2, 0.1])\n"
        "assert np.allclose(sm.end_effector_velocity(HOME, dq, 'space'), Js @ dq, atol=1e-9)\n"
        "assert np.allclose(Js, adjoint_transform(T) @ Jb, atol=1e-5)\n"
        "# Resolved-rate realizes a desired twist.\n"
        "Vd = np.array([0, 0, 0, 0.05, 0, 0])\n"
        "assert np.allclose(Js @ sm.joint_velocity(HOME, Vd, 'space'), Vd, atol=1e-3)\n"
        "# Redundancy: rank 6, one null direction with zero end-effector twist.\n"
        "assert np.linalg.matrix_rank(Js) == 6\n"
        "_, _, Vt = np.linalg.svd(Js)\n"
        "assert np.allclose(Js @ Vt[6], 0, atol=1e-9)\n"
        "# Manipulability is positive at this non-singular pose.\n"
        "Jv = Jb[3:6]\n"
        "assert np.sqrt(np.linalg.det(Jv @ Jv.T)) > 0\n"
        'print("nb03 velocity kinematics: smoke OK")'
    ),

    # --- exercises ---
    md(
        "## Try it\n"
        "\n"
        "1. Command a pure end-effector rotation about $z$ (twist "
        "$[0,0,1,0,0,0]$) at `HOME` with `joint_velocity` and confirm the resulting joint "
        "rates reproduce it through the Jacobian.\n"
        "2. Compute the manipulability $w=\\sqrt{\\det(J_vJ_v^\\top)}$ at `HOME` and at a "
        "near-singular configuration (e.g. straighten joint 4 toward $0$). How does $w$ "
        "change?\n"
        "3. Take two steps along the null space with opposite sign and verify the "
        "end-effector returns to (approximately) the same pose while the elbow swivels both "
        "ways.\n"
        "\n"
        "*Next up — notebook 04: **inverse kinematics**, turning a desired pose back into "
        "joint angles, including how redundancy is resolved.*"
    ),
    md(
        "## References\n"
        "\n"
        "1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and "
        "Control*, Cambridge University Press, 2017. — Chapter 5, *Velocity Kinematics and "
        "Statics* (space/body Jacobians, manipulability, singularities).\n"
        "2. R. M. Murray, Z. Li, and S. S. Sastry, *A Mathematical Introduction to "
        "Robotic Manipulation*, CRC Press, 1994.\n"
        "3. B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics: Modelling, "
        "Planning and Control*, Springer, 2009. — Redundancy resolution and the "
        "pseudoinverse.\n"
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
