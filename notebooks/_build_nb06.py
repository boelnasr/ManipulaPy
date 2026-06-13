"""Build and execute notebooks/06_singularities.ipynb.

Run from the notebooks/ directory so `_shared` is importable at execute time:
    cd notebooks && python3 _build_nb06.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "06_singularities.ipynb")


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


cells = [
    md(
        "# 06 · Singularities and Manipulability\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/boelnasr/ManipulaPy/blob/notebooks/tutorials/"
        "notebooks/06_singularities.ipynb)\n"
        "\n"
        "> **ManipulaPy teaching course — notebook 6 of 11.** Running robot: Franka Panda.\n"
        "\n"
        "The Jacobian (notebook 03) maps joint rates to end-effector twists. At most "
        "configurations it has full rank and the arm can move its end-effector in every "
        "direction. At a **singularity** it loses rank: some direction of end-effector "
        "motion becomes instantaneously impossible, no matter how the joints move. "
        "Singularities are where resolved-rate control (notebook 03) and inverse "
        "kinematics (notebook 04) get into trouble — joint speeds blow up and the damping "
        "we added earns its keep. This notebook measures how close a configuration is to a "
        "singularity, and shows what is lost when it reaches one, using "
        "`ManipulaPy.singularity.Singularity`."
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
        "from tikz import setup_pgf, embed_pgf_fig\n"
        "import numpy as np\n"
        "from helpers import load_panda, HOME, N_JOINTS\n"
        "from ManipulaPy.singularity import Singularity\n"
        "np.set_printoptions(precision=4, suppress=True)\n"
        "\n"
        "sm, dyn = load_panda()\n"
        "sing = Singularity(sm)\n"
        'print("Panda loaded | singularity tools ready")'
    ),

    # --- 1. singular values ---
    md(
        "## 1. The Jacobian's singular values\n"
        "\n"
        "Everything about singularities lives in the **singular value decomposition** of "
        "the Jacobian, $J=U\\Sigma V^{\\top}$. The singular values "
        "$\\sigma_1\\ge\\dots\\ge\\sigma_6$ are the lengths of the manipulability "
        "ellipsoid's axes (notebook 03): the largest is the easiest direction of "
        "end-effector motion, the **smallest** $\\sigma_6$ is the hardest. A singularity "
        "is exactly $\\sigma_6=0$ — one axis of the ellipsoid has collapsed to a line, and "
        "that direction of motion is lost.\n"
        "\n"
        "Three scalar summaries follow from the spectrum:\n"
        "\n"
        "- **manipulability** $w=\\sqrt{\\det(JJ^{\\top})}=\\sigma_1\\sigma_2\\cdots"
        "\\sigma_6$ — the ellipsoid's volume; $0$ at a singularity;\n"
        "- the **smallest singular value** $\\sigma_6$ — distance to rank loss in the "
        "worst direction;\n"
        "- the **condition number** $\\kappa=\\sigma_1/\\sigma_6$ — how lopsided the "
        "ellipsoid is; $\\infty$ at a singularity."
    ),
    code(
        "J = sm.jacobian(HOME, frame='space')\n"
        "sigmas = np.linalg.svd(J, compute_uv=False)\n"
        "w = np.sqrt(np.linalg.det(J @ J.T))\n"
        'print("singular values at HOME:", np.round(sigmas, 4))\n'
        'print("manipulability  w = %.4f" % w)\n'
        'print("smallest sigma_6 = %.4f" % sigmas[-1])\n'
        'print("condition number = %.1f" % (sigmas[0] / sigmas[-1]))\n'
        "assert np.isclose(w, np.prod(sigmas))      # w is the product of the singular values"
    ),
    md(
        "Put `HOME` next to a **near-singular** configuration — most of the way from "
        "`HOME` toward the Panda's all-zero pose, which (we will see) is singular. The "
        "spectrum tells the story at a glance: the large singular values barely move, while "
        "$\\sigma_6$ collapses toward zero. That single shrinking bar *is* the approach to "
        "rank loss."
    ),
    code(
        "q_near = 0.235 * HOME             # the deepest near-singular point on the HOME->zero path\n"
        "sig_home = np.linalg.svd(sm.jacobian(HOME, frame='space'), compute_uv=False)\n"
        "sig_near = np.linalg.svd(sm.jacobian(q_near, frame='space'), compute_uv=False)\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.4, 3.4))\n"
        "x = np.arange(1, 7)\n"
        "ax.bar(x - 0.2, sig_home, 0.4, label='HOME', color='tab:blue')\n"
        "ax.bar(x + 0.2, sig_near, 0.4, label='near-singular', color='tab:red')\n"
        "ax.set_xlabel('singular value index'); ax.set_ylabel('$\\\\sigma_i$')\n"
        "ax.set_xticks(x); ax.set_title('Jacobian singular-value spectrum')\n"
        "ax.legend()\n"
        'embed_pgf_fig(fig, name="singular_value_spectrum")'
    ),
    md(
        "### The manipulability ellipsoid in 3-D\n"
        "\n"
        "Geometry makes \"losing a direction\" concrete. The end-effector linear velocities "
        "reachable with **unit-norm joint speed** form an ellipsoid whose semi-axes are the "
        "*linear* singular values. At `HOME` it is a healthy, rounded ellipsoid — the hand "
        "moves comfortably in every direction. At the near-singular pose it flattens into a "
        "**pancake**: its shortest axis ($\\sigma\\approx0.04$) is the direction the "
        "end-effector can barely move at all."
    ),
    code(
        "def linear_ellipsoid(q):\n"
        "    Jv = sm.jacobian(q, frame='space')[3:6]          # linear-velocity rows\n"
        "    U, S, _ = np.linalg.svd(Jv)\n"
        "    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]\n"
        "    sphere = np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])\n"
        "    pts = U @ np.diag(S) @ sphere.reshape(3, -1)      # image of the unit sphere\n"
        "    return [pts[k].reshape(u.shape) for k in range(3)]\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig = plt.figure(figsize=(7.8, 3.8))\n"
        "for k, (q, title, color) in enumerate(\n"
        "        [(HOME, 'HOME (healthy)', 'tab:blue'),\n"
        "         (q_near, 'near-singular (collapsed)', 'tab:red')], 1):\n"
        "    ax = fig.add_subplot(1, 2, k, projection='3d')\n"
        "    X, Y, Z = linear_ellipsoid(q)\n"
        "    ax.plot_surface(X, Y, Z, color=color, alpha=0.6, linewidth=0)\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.set_xlabel('$v_x$'); ax.set_ylabel('$v_y$'); ax.set_zlabel('$v_z$')\n"
        "    lim = 1.3\n"
        "    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)\n"
        "    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=-60)\n"
        "fig.suptitle('Linear-velocity manipulability ellipsoid')\n"
        'embed_pgf_fig(fig, name="manipulability_ellipsoid_3d")'
    ),

    # --- 2. approaching a singularity ---
    md(
        "## 2. Approaching a singularity\n"
        "\n"
        "Walk the arm continuously from `HOME` toward the all-zero pose and track all three "
        "measures. As the configuration approaches the singularity the manipulability $w$ "
        "and smallest singular value $\\sigma_6$ plunge toward zero while the condition "
        "number $\\kappa$ blows up — three views of the same collapse."
    ),
    code(
        "ts = np.linspace(0.0, 0.765, 40)     # 0 = HOME, 1 = all-zero (singular) pose\n"
        "W, Smin, Cond = [], [], []\n"
        "for t in ts:\n"
        "    q = (1 - t) * HOME\n"
        "    s = np.linalg.svd(sm.jacobian(q, frame='space'), compute_uv=False)\n"
        "    W.append(np.prod(s)); Smin.append(s[-1]); Cond.append(s[0] / s[-1])\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, axL = plt.subplots(figsize=(5.6, 3.6))\n"
        "axL.semilogy(ts, W, color='tab:blue', label='manipulability $w$')\n"
        "axL.semilogy(ts, Smin, color='tab:green', label='smallest $\\\\sigma_6$')\n"
        "axL.set_xlabel('path from HOME (0) toward the singular pose (1)')\n"
        "axL.set_ylabel('$w$ and $\\\\sigma_6$ (log)')\n"
        "axR = axL.twinx()\n"
        "axR.semilogy(ts, Cond, color='tab:red', ls='--', label='condition number $\\\\kappa$')\n"
        "axR.set_ylabel('condition number $\\\\kappa$ (log)')\n"
        "lines = axL.get_lines() + axR.get_lines()\n"
        "axL.legend(lines, [l.get_label() for l in lines], loc='center left', fontsize=8)\n"
        "axL.set_title('Approaching a singularity')\n"
        'embed_pgf_fig(fig, name="singularity_approach")'
    ),
    md(
        "### The singularity landscape\n"
        "\n"
        "That sweep followed one path; sweeping **two** joints maps manipulability over a "
        "whole slice of configuration space. Bright regions are dexterous, the dark "
        "trenches are where $w$ collapses toward a singularity. The structure is not a "
        "single point — singularities form **surfaces** the arm must steer around, which "
        "is exactly what makes them a planning concern (notebook 07)."
    ),
    code(
        "q2s = np.linspace(-1.76, 1.76, 50)\n"
        "q4s = np.linspace(-3.07, -0.07, 50)\n"
        "Wmap = np.zeros((len(q4s), len(q2s)))\n"
        "for i, a in enumerate(q2s):\n"
        "    for j, b in enumerate(q4s):\n"
        "        q = HOME.copy(); q[1] = a; q[3] = b\n"
        "        J = sm.jacobian(q, frame='space')\n"
        "        Wmap[j, i] = np.sqrt(max(np.linalg.det(J @ J.T), 0))\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.4, 3.9))\n"
        "im = ax.contourf(np.degrees(q2s), np.degrees(q4s), Wmap, levels=20, cmap='viridis')\n"
        "fig.colorbar(im, ax=ax, label='manipulability $w$')\n"
        "ax.set_xlabel('joint 2 angle (deg)'); ax.set_ylabel('joint 4 angle (deg)')\n"
        "ax.set_title('Manipulability landscape over (joint 2, joint 4)')\n"
        'embed_pgf_fig(fig, name="manipulability_landscape")'
    ),

    # --- 3. detecting ---
    md(
        "## 3. Detecting a singularity\n"
        "\n"
        "`singularity_analysis` returns a boolean — it flags a configuration as singular "
        "when the smallest singular value drops below $10^{-4}$. The Panda's **all-zero "
        "pose** is a genuine singularity (its Jacobian is exactly rank-deficient), so the "
        "detector fires there and not at `HOME`. `condition_number` gives the underlying "
        "$\\kappa$ as a number."
    ),
    code(
        "q_singular = np.zeros(N_JOINTS)         # the Panda's all-zero pose\n"
        'print("HOME       : singular? %-5s  condition number %.1f"\n'
        "      % (sing.singularity_analysis(HOME), sing.condition_number(HOME)))\n"
        'print("all-zero   : singular? %-5s  condition number %.1e"\n'
        "      % (sing.singularity_analysis(q_singular), sing.condition_number(q_singular)))\n"
        "assert not sing.singularity_analysis(HOME)\n"
        "assert sing.singularity_analysis(q_singular)\n"
        "\n"
        "# near_singularity_detection wraps the condition number against a cutoff you choose\n"
        "# (kappa >= 1 always, so pick a value like 50-100 for 'near singular', not the\n"
        "# permissive default).\n"
        'print("near-singular by kappa>50?  HOME:", sing.near_singularity_detection(HOME, threshold=50),\n'
        '      " near pose:", sing.near_singularity_detection(q_near, threshold=50))'
    ),

    # --- 4. what is lost ---
    md(
        "## 4. What is lost, and why it matters\n"
        "\n"
        "At a singularity the lost direction is the column of $U$ paired with "
        "$\\sigma_6$: the end-effector simply cannot be moved that way by any joint "
        "motion. Just *near* a singularity the direction is still available, but only at a "
        "steep price — producing one unit of end-effector velocity along it needs joint "
        "speeds of order $1/\\sigma_6$. As $\\sigma_6\\to0$ that cost explodes, which is "
        "exactly why the resolved-rate pseudoinverse (notebook 03) and the inverse "
        "kinematics step (notebook 04) blow up near singularities — and why the **damped** "
        "least-squares solver, which caps that $1/\\sigma_6$ factor, is the standard fix."
    ),
    code(
        "for label, q in [('HOME', HOME), ('near-singular', q_near)]:\n"
        "    s6 = np.linalg.svd(sm.jacobian(q, frame='space'), compute_uv=False)[-1]\n"
        "    print('%-13s sigma_6 = %.4f  ->  worst-direction joint-speed cost ~ 1/sigma_6 = %5.1f'\n"
        "          % (label, s6, 1.0 / s6))\n"
        "\n"
        "# Moving the end-effector in its hardest direction costs far more joint speed\n"
        "# near the singularity than at HOME.\n"
        "s6_home = np.linalg.svd(sm.jacobian(HOME, frame='space'), compute_uv=False)[-1]\n"
        "s6_near = np.linalg.svd(sm.jacobian(q_near, frame='space'), compute_uv=False)[-1]\n"
        "assert (1 / s6_near) > 3 * (1 / s6_home)"
    ),

    # --- smoke test ---
    md("## Smoke test\n\nAsserts the key invariants of this notebook in one cell."),
    code(
        "sm2, _ = load_panda()\n"
        "sg2 = Singularity(sm2)\n"
        "J = sm2.jacobian(HOME, frame='space')\n"
        "s = np.linalg.svd(J, compute_uv=False)\n"
        "# Manipulability is the product of the singular values, positive away from singularities.\n"
        "assert np.isclose(np.sqrt(np.linalg.det(J @ J.T)), np.prod(s))\n"
        "assert s[-1] > 1e-4\n"
        "# HOME is well-conditioned; the all-zero pose is singular.\n"
        "assert not sg2.singularity_analysis(HOME)\n"
        "assert sg2.singularity_analysis(np.zeros(N_JOINTS))\n"
        "# Condition number grows toward the singular pose.\n"
        "assert sg2.condition_number(0.235 * HOME) > sg2.condition_number(HOME)\n"
        'print("nb06 singularities: smoke OK")'
    ),

    # --- exercises ---
    md(
        "## Try it\n"
        "\n"
        "1. Find the column of $U$ (from `np.linalg.svd(J, full_matrices=True)`) paired "
        "with $\\sigma_6$ at the near-singular pose — that twist is the direction the "
        "end-effector struggles to move. Confirm `joint_velocity` for that twist returns "
        "very large joint rates.\n"
        "2. Re-solve that same twist with the damped pseudoinverse "
        "$J^{\\top}(JJ^{\\top}+\\lambda^2 I)^{-1}$ for a few $\\lambda$ and watch the joint "
        "rates stay bounded — the notebook 04 fix, seen from the singularity side.\n"
        "3. Sweep a different joint (e.g. joint 6) and see whether you can drive $w$ to a "
        "local minimum — not every joint reaches a singularity within its limits.\n"
        "\n"
        "*Next up — notebook 07: **control** — turning the dynamics of notebook 05 into "
        "torques that track a desired motion.*"
    ),
    md(
        "## References\n"
        "\n"
        "1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and "
        "Control*, Cambridge University Press, 2017. — Chapter 5, *Velocity Kinematics and "
        "Statics* (singularities, manipulability ellipsoid).\n"
        "2. T. Yoshikawa, *Manipulability of Robotic Mechanisms*, Int. J. Robotics "
        "Research, 1985. — Origin of the manipulability measure $w=\\sqrt{\\det(JJ^{\\top})}$.\n"
        "3. B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics: Modelling, "
        "Planning and Control*, Springer, 2009. — Singularities and damped least squares.\n"
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
