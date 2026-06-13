"""Build and execute notebooks/05_dynamics.ipynb.

Run from the notebooks/ directory so `_shared` is importable at execute time:
    cd notebooks && python3 _build_nb05.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "05_dynamics.ipynb")


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


cells = [
    md(
        "# 05 · Dynamics\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/boelnasr/ManipulaPy/blob/notebooks/tutorials/"
        "notebooks/05_dynamics.ipynb)\n"
        "\n"
        "> **ManipulaPy teaching course — notebook 5 of 11.** Running robot: Franka Panda.\n"
        "\n"
        "Kinematics (notebooks 02–04) never mentioned mass or force. **Dynamics** puts them "
        "in. Everything in this notebook is one equation — the **manipulator equation** — "
        "relating joint torques $\\tau$ to motion:\n"
        "\n"
        "$$\\tau = M(\\theta)\\,\\ddot\\theta \\;+\\; c(\\theta,\\dot\\theta) \\;+\\; "
        "g(\\theta) \\;+\\; J^{\\top}(\\theta)\\,\\mathcal{F}_{\\text{tip}}.$$\n"
        "\n"
        "- $M(\\theta)$ — the **mass (inertia) matrix**\n"
        "- $c(\\theta,\\dot\\theta)$ — **Coriolis and centripetal** forces\n"
        "- $g(\\theta)$ — **gravity**\n"
        "- $J^{\\top}\\mathcal{F}_{\\text{tip}}$ — joint torques from an external "
        "end-effector wrench\n"
        "\n"
        "It answers two questions. **Inverse dynamics**: what torques produce a desired "
        "motion (the engine of control, notebook 07)? **Forward dynamics**: what motion do "
        "given torques produce (the engine of simulation)? ManipulaPy's "
        "`ManipulatorDynamics` provides every term."
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
        "np.set_printoptions(precision=4, suppress=True)\n"
        "\n"
        "sm, dyn = load_panda()           # dyn is a ManipulatorDynamics\n"
        "g = np.array([0.0, 0.0, -9.81])  # gravity in the base frame\n"
        "Ftip = np.zeros(6)               # no external end-effector wrench\n"
        'print("Panda dynamics loaded |", N_JOINTS, "joints")'
    ),

    # --- 1. mass matrix ---
    md(
        "## 1. The mass matrix $M(\\theta)$\n"
        "\n"
        "$M(\\theta)$ maps joint accelerations to the inertial part of the torque, "
        "$\\tau_{\\text{inertia}}=M(\\theta)\\ddot\\theta$. Two properties always hold: it "
        "is **symmetric**, and **positive-definite** — the kinetic energy "
        "$\\tfrac12\\dot\\theta^{\\top}M\\dot\\theta$ is positive for any non-zero motion. "
        "It also depends on the configuration: the arm's effective inertia changes as it "
        "folds and unfolds. The diagonal entries are the effective inertias seen by each "
        "joint; the off-diagonals are inertial **coupling** between joints."
    ),
    code(
        "M = dyn.mass_matrix(HOME)\n"
        'print("M shape:", M.shape)\n'
        "assert np.allclose(M, M.T)                          # symmetric\n"
        "eigs = np.linalg.eigvalsh(M)\n"
        "assert np.all(eigs > 0)                             # positive-definite\n"
        'print("eigenvalues:", np.round(eigs, 3), "(all > 0)")\n'
        'print("diagonal (effective joint inertias):", np.round(np.diag(M), 3))'
    ),
    code(
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(4.6, 3.9))\n"
        "im = ax.imshow(M, cmap='viridis')\n"
        "fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='kg$\\\\cdot$m$^2$')\n"
        "ax.set_xticks(range(N_JOINTS)); ax.set_yticks(range(N_JOINTS))\n"
        "ax.set_xticklabels(range(1, N_JOINTS + 1)); ax.set_yticklabels(range(1, N_JOINTS + 1))\n"
        "ax.set_xlabel('joint'); ax.set_ylabel('joint')\n"
        "ax.set_title('Mass matrix $M(\\\\theta)$ at HOME')\n"
        'embed_pgf_fig(fig, name="mass_matrix_heatmap")'
    ),

    # --- 2. gravity ---
    md(
        "## 2. Gravity $g(\\theta)$\n"
        "\n"
        "`gravity_forces` returns the torque each joint must exert **just to hold the arm "
        "still** against gravity (zero velocity, zero acceleration). At `HOME` the load "
        "falls almost entirely on the shoulder and elbow — the joints with the most arm "
        "hanging off them."
    ),
    code(
        "tau_g = dyn.gravity_forces(HOME, g)\n"
        'print("gravity hold torques (N*m):", np.round(tau_g, 3))\n'
        'print("heaviest joint:", int(np.argmax(np.abs(tau_g))) + 1)'
    ),
    md(
        "### How the gravity load changes with configuration\n"
        "\n"
        "Sweep joint 2 (the shoulder) through its range and watch the torque it must hold. "
        "The load peaks where that link is most **horizontal** — the longest gravity lever "
        "— and passes through lighter values as the arm tucks toward vertical, exactly like "
        "holding a weight at arm's length versus overhead."
    ),
    code(
        "q2_range = np.linspace(-1.7, 1.7, 80)\n"
        "tau2 = []\n"
        "for a in q2_range:\n"
        "    q = HOME.copy(); q[1] = a\n"
        "    tau2.append(dyn.gravity_forces(q, g)[1])\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.2, 3.4))\n"
        "ax.plot(np.degrees(q2_range), tau2, color='tab:red')\n"
        "ax.axhline(0, color='gray', lw=0.8, ls=':')\n"
        "ax.set_xlabel('joint 2 angle (deg)'); ax.set_ylabel('gravity torque on joint 2 (N$\\\\cdot$m)')\n"
        "ax.set_title('Gravity load on the shoulder vs configuration')\n"
        "ax.grid(True, ls=':', alpha=0.5)\n"
        'embed_pgf_fig(fig, name="gravity_joint2_sweep")'
    ),
    md(
        "### The gravity load across the workspace\n"
        "\n"
        "That 1-D sweep is one slice. Sweeping the **shoulder (joint 2)** and **elbow "
        "(joint 4)** together maps the shoulder's gravity torque over a whole plane of "
        "configurations. The diverging bands show where the load reverses sign as the arm "
        "swings through the vertical — a torque map a gravity-compensating controller has "
        "to supply everywhere it operates."
    ),
    code(
        "q2s = np.linspace(-1.76, 1.76, 50)\n"
        "q4s = np.linspace(-3.07, -0.07, 50)\n"
        "G = np.zeros((len(q4s), len(q2s)))\n"
        "for i, a in enumerate(q2s):\n"
        "    for j, b in enumerate(q4s):\n"
        "        q = HOME.copy(); q[1] = a; q[3] = b\n"
        "        G[j, i] = dyn.gravity_forces(q, g)[1]\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.4, 3.9))\n"
        "lim = np.abs(G).max()\n"
        "im = ax.contourf(np.degrees(q2s), np.degrees(q4s), G, levels=20,\n"
        "                 cmap='RdBu_r', vmin=-lim, vmax=lim)\n"
        "fig.colorbar(im, ax=ax, label='gravity torque on joint 2 (N$\\\\cdot$m)')\n"
        "ax.set_xlabel('joint 2 angle (deg)'); ax.set_ylabel('joint 4 angle (deg)')\n"
        "ax.set_title('Shoulder gravity load over the (joint 2, joint 4) plane')\n"
        'embed_pgf_fig(fig, name="gravity_workspace_map")'
    ),

    # --- 3. coriolis ---
    md(
        "## 3. Coriolis and centripetal forces $c(\\theta,\\dot\\theta)$\n"
        "\n"
        "When the joints are **moving**, the coupled links exert velocity-dependent "
        "fictitious forces on each other. `velocity_quadratic_forces` returns them. They "
        "are **quadratic** in the joint velocity (double the speed, quadruple the force) "
        "and vanish exactly when the arm is at rest."
    ),
    code(
        "dq = np.array([0.5, -0.3, 0.4, 0.2, -0.1, 0.3, 0.1])   # joint velocities\n"
        "c = dyn.velocity_quadratic_forces(HOME, dq)\n"
        'print("Coriolis/centripetal torques:", np.round(c, 4))\n'
        "\n"
        "# Zero at rest; quadratic in speed.\n"
        "assert np.allclose(dyn.velocity_quadratic_forces(HOME, np.zeros(N_JOINTS)), 0)\n"
        "c2 = dyn.velocity_quadratic_forces(HOME, 2 * dq)\n"
        'print("doubling speed scales the force by ~%.2fx (expect 4)" % (np.linalg.norm(c2) / np.linalg.norm(c)))\n'
        "assert np.allclose(c2, 4 * c, rtol=1e-6)"
    ),

    # --- 4. inverse dynamics ---
    md(
        "## 4. Inverse dynamics\n"
        "\n"
        "Putting the terms together: given a motion $(\\theta,\\dot\\theta,\\ddot\\theta)$, "
        "`inverse_dynamics` returns the torques that produce it. This is the whole "
        "manipulator equation evaluated in one call — and the basis of torque-level control "
        "(notebook 07). A useful sanity check: at rest with no acceleration, the required "
        "torque is exactly the gravity hold torque from Section 2."
    ),
    code(
        "ddq = np.array([0.2, 0.1, -0.3, 0.15, 0.05, -0.2, 0.1])   # joint accelerations\n"
        "tau = dyn.inverse_dynamics(HOME, dq, ddq, g, Ftip)\n"
        'print("required torques (N*m):", np.round(tau, 3))\n'
        "\n"
        "# At rest and still, inverse dynamics reduces to gravity compensation.\n"
        "tau_static = dyn.inverse_dynamics(HOME, np.zeros(N_JOINTS), np.zeros(N_JOINTS), g, Ftip)\n"
        "assert np.allclose(tau_static, dyn.gravity_forces(HOME, g))\n"
        'print("at rest, inverse dynamics == gravity_forces :", True)'
    ),
    md(
        "### Anatomy of the torque along a motion\n"
        "\n"
        "Inverse dynamics is most revealing watched over a *motion*. Drive every joint "
        "along a slow sinusoid and split the joint-4 torque into its three physical "
        "contributions: the **inertial** term $M(\\theta)\\ddot\\theta$, the **Coriolis** "
        "term $c(\\theta,\\dot\\theta)$, and **gravity** $g(\\theta)$. Gravity sets the "
        "slowly-varying baseline, the inertial term tracks the acceleration, and Coriolis "
        "adds the velocity-dependent ripple — and the three sum **exactly** to the "
        "inverse-dynamics torque."
    ),
    code(
        "tsim = np.linspace(0, 2, 100)\n"
        "amp = np.array([0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4]); wsin = 2 * np.pi * 0.5\n"
        "J4 = 3                                              # report joint 4\n"
        "total, inertial, coriolis, gravity = [], [], [], []\n"
        "for t in tsim:\n"
        "    qt = HOME + amp * np.sin(wsin * t)\n"
        "    dqt = amp * wsin * np.cos(wsin * t)\n"
        "    ddqt = -amp * wsin**2 * np.sin(wsin * t)\n"
        "    inertial.append((dyn.mass_matrix(qt) @ ddqt)[J4])\n"
        "    coriolis.append(dyn.velocity_quadratic_forces(qt, dqt)[J4])\n"
        "    gravity.append(dyn.gravity_forces(qt, g)[J4])\n"
        "    total.append(dyn.inverse_dynamics(qt, dqt, ddqt, g, Ftip)[J4])\n"
        "total = np.array(total)\n"
        "assert np.allclose(total, np.array(inertial) + np.array(coriolis) + np.array(gravity), atol=1e-9)\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.8, 3.6))\n"
        "ax.plot(tsim, total, 'k', lw=2.2, label='total (inverse dynamics)')\n"
        "ax.plot(tsim, inertial, label='inertial $M\\\\ddot\\\\theta$')\n"
        "ax.plot(tsim, coriolis, label='Coriolis $c$')\n"
        "ax.plot(tsim, gravity, label='gravity $g$')\n"
        "ax.set_xlabel('time (s)'); ax.set_ylabel('torque on joint 4 (N$\\\\cdot$m)')\n"
        "ax.set_title('Inverse-dynamics torque, decomposed')\n"
        "ax.legend(fontsize=8, ncol=2); ax.grid(True, ls=':', alpha=0.5)\n"
        'embed_pgf_fig(fig, name="torque_decomposition")'
    ),

    # --- 5. forward dynamics ---
    md(
        "## 5. Forward dynamics\n"
        "\n"
        "The reverse question — *what motion results from applied torques?* — solves the "
        "equation for the acceleration, $\\ddot\\theta = M^{-1}\\big(\\tau - c - g - "
        "J^{\\top}\\mathcal{F}_{\\text{tip}}\\big)$. `forward_dynamics` does exactly this. "
        "Because it inverts what inverse dynamics builds, feeding the torques from Section "
        "4 straight back in must return the **same accelerations** — to machine precision."
    ),
    code(
        "ddq_recovered = dyn.forward_dynamics(HOME, dq, tau, g, Ftip)\n"
        'print("recovered accelerations:", np.round(ddq_recovered, 4))\n'
        'print("max error vs original:", np.max(np.abs(ddq_recovered - ddq)))\n'
        "assert np.allclose(ddq_recovered, ddq, atol=1e-9)       # exact round trip"
    ),

    # --- 6. simulating ---
    md(
        "## 6. Simulating the arm under gravity\n"
        "\n"
        "Integrating `forward_dynamics` over time **is** a physics simulation. We step the "
        "state with semi-implicit Euler: $\\dot\\theta \\mathrel{+}= \\ddot\\theta\\,dt$, "
        "then $\\theta \\mathrel{+}= \\dot\\theta\\,dt$. Released from `HOME` with **zero "
        "torque**, the arm collapses under its own weight. Feed back the gravity torque "
        "$\\tau=g(\\theta)$ at every step — **gravity compensation** — and it hangs "
        "motionless, the simplest model-based controller there is."
    ),
    code(
        "def simulate(torque_fn, q0, T=0.6, dt=0.002):\n"
        "    '''Integrate forward_dynamics from rest under torque_fn(q, dq). Returns (t, Q).'''\n"
        "    q = np.array(q0, float); dq = np.zeros(N_JOINTS)\n"
        "    ts, Q = [0.0], [q.copy()]\n"
        "    for i in range(int(T / dt)):\n"
        "        ddq = dyn.forward_dynamics(q, dq, torque_fn(q, dq), g, Ftip)\n"
        "        dq = dq + ddq * dt          # semi-implicit Euler\n"
        "        q = q + dq * dt\n"
        "        ts.append((i + 1) * dt); Q.append(q.copy())\n"
        "    return np.array(ts), np.array(Q)\n"
        "\n"
        "t_fall, Q_fall = simulate(lambda q, dq: np.zeros(N_JOINTS), HOME)     # released, no torque\n"
        "t_hold, Q_hold = simulate(lambda q, dq: dyn.gravity_forces(q, g), HOME)  # gravity compensation\n"
        "\n"
        'print("free fall  : max joint drift %.1f deg" % np.degrees(np.abs(Q_fall[-1] - HOME).max()))\n'
        'print("grav. comp.: max joint drift %.3f deg" % np.degrees(np.abs(Q_hold[-1] - HOME).max()))\n'
        "assert np.degrees(np.abs(Q_fall[-1] - HOME).max()) > 10     # it really falls\n"
        "assert np.degrees(np.abs(Q_hold[-1] - HOME).max()) < 0.1    # it really holds"
    ),
    code(
        "plt = setup_pgf()\n"
        "fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)\n"
        "for j in range(N_JOINTS):\n"
        "    axes[0].plot(t_fall, np.degrees(Q_fall[:, j] - HOME[j]), label='joint %d' % (j + 1))\n"
        "    axes[1].plot(t_hold, np.degrees(Q_hold[:, j] - HOME[j]))\n"
        "axes[0].set_title('Zero torque: free fall'); axes[1].set_title('Gravity compensation')\n"
        "for ax in axes:\n"
        "    ax.set_xlabel('time (s)'); ax.grid(True, ls=':', alpha=0.5)\n"
        "axes[0].set_ylabel('joint angle change (deg)')\n"
        "axes[0].legend(fontsize=7, ncol=2)\n"
        'embed_pgf_fig(fig, name="forward_dynamics_freefall")'
    ),

    # --- smoke test ---
    md("## Smoke test\n\nAsserts the key invariants of this notebook in one cell."),
    code(
        "sm2, dyn2 = load_panda()\n"
        "# Mass matrix is symmetric positive-definite.\n"
        "M = dyn2.mass_matrix(HOME)\n"
        "assert np.allclose(M, M.T) and np.all(np.linalg.eigvalsh(M) > 0)\n"
        "# Coriolis vanishes at rest.\n"
        "assert np.allclose(dyn2.velocity_quadratic_forces(HOME, np.zeros(N_JOINTS)), 0)\n"
        "# At rest, inverse dynamics equals gravity compensation.\n"
        "tau0 = dyn2.inverse_dynamics(HOME, np.zeros(N_JOINTS), np.zeros(N_JOINTS), g, Ftip)\n"
        "assert np.allclose(tau0, dyn2.gravity_forces(HOME, g))\n"
        "# Inverse dynamics and forward dynamics are inverses (round trip).\n"
        "dq = np.array([0.5, -0.3, 0.4, 0.2, -0.1, 0.3, 0.1])\n"
        "ddq = np.array([0.2, 0.1, -0.3, 0.15, 0.05, -0.2, 0.1])\n"
        "tau = dyn2.inverse_dynamics(HOME, dq, ddq, g, Ftip)\n"
        "assert np.allclose(dyn2.forward_dynamics(HOME, dq, tau, g, Ftip), ddq, atol=1e-9)\n"
        'print("nb05 dynamics: smoke OK")'
    ),

    # --- exercises ---
    md(
        "## Try it\n"
        "\n"
        "1. Read off the effective inertia of joint 1 from $M$ at `HOME`, then fold the arm "
        "(set joints 2 and 4 toward straight) and recompute. Does $M_{11}$ rise or fall, "
        "and why?\n"
        "2. Apply a constant non-gravity torque to a single joint in `simulate` and watch "
        "how the whole arm responds — inertial coupling means one torque moves many "
        "joints.\n"
        "3. Add an end-effector load: set `Ftip` to a downward force and recompute the "
        "inverse-dynamics torques. Which joints pick up the extra work?\n"
        "\n"
        "*Next up — notebook 06: **singularities and manipulability** — where the Jacobian "
        "loses rank and the arm loses the ability to move in some direction.*"
    ),
    md(
        "## References\n"
        "\n"
        "1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and "
        "Control*, Cambridge University Press, 2017. — Chapter 8, *Dynamics of Open "
        "Chains* (mass matrix, Coriolis, gravity, inverse/forward dynamics).\n"
        "2. R. M. Murray, Z. Li, and S. S. Sastry, *A Mathematical Introduction to "
        "Robotic Manipulation*, CRC Press, 1994. — Lagrangian formulation of the "
        "manipulator equation.\n"
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
