"""Build and execute notebooks/04_inverse_kinematics.ipynb.

Run from the notebooks/ directory so `_shared` is importable at execute time:
    cd notebooks && python3 _build_nb04.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "04_inverse_kinematics.ipynb")


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


cells = [
    md(
        "# 04 · Inverse Kinematics\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/boelnasr/ManipulaPy/blob/notebooks/tutorials/"
        "notebooks/04_inverse_kinematics.ipynb)\n"
        "\n"
        "> **ManipulaPy teaching course — notebook 4 of 11.** Running robot: Franka Panda.\n"
        "\n"
        "Forward kinematics (notebook 02) sends joint angles to a pose; the Jacobian "
        "(notebook 03) sends joint *rates* to a twist. **Inverse kinematics** runs the "
        "first map backwards: given a desired end-effector pose $T_d$, find joint angles "
        "$\\theta$ with $T(\\theta)=T_d$. Unlike FK there is no closed-form answer for a "
        "general arm — we solve it **iteratively**, and the Jacobian is what makes each "
        "step. This notebook builds the **damped least-squares** solver by hand, checks it "
        "against ManipulaPy's, and uses the Panda's redundancy to land *a* solution among "
        "many."
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
        "from helpers import load_panda, HOME, N_JOINTS, joint_limits\n"
        "from ManipulaPy.utils import MatrixLog6, se3ToVec, TransInv\n"
        "np.set_printoptions(precision=4, suppress=True)\n"
        "np.random.seed(0)                       # reproducible random seeds/targets\n"
        "\n"
        "sm, dyn = load_panda()\n"
        "# The solver clips against sm.joint_limits, which ships with 8 entries (the URDF's\n"
        "# 7 arm joints + the gripper finger). This course works in the 7 actuated joints,\n"
        "# so install the 7-entry limit set -- IK, FK and the Jacobian are then all 7-DOF.\n"
        "sm.joint_limits = joint_limits()\n"
        'print("Panda loaded |", N_JOINTS, "actuated joints | joint_limits:", len(sm.joint_limits))'
    ),

    # --- 1. the inverse problem ---
    md(
        "## 1. The inverse problem and Newton–Raphson\n"
        "\n"
        "We want $\\theta$ such that $T(\\theta)=T_d$. Writing the pose error as a "
        "**body twist** — the screw that carries the current pose $T(\\theta)$ onto the "
        "target — turns IK into **root finding**:\n"
        "\n"
        "$$F(\\theta)=\\log\\!\\big(T(\\theta)^{-1}T_d\\big)^{\\vee}"
        "=\\mathcal{V}_b(\\theta)\\;\\stackrel{!}{=}\\;0.$$\n"
        "\n"
        "**The scalar idea.** Newton–Raphson solves a one-dimensional equation "
        "$f(x)=0$ by repeatedly following the *tangent line* to where it crosses zero:\n"
        "\n"
        "$$x_{k+1}=x_k-\\frac{f(x_k)}{f'(x_k)}.$$\n"
        "\n"
        "Each step replaces the curve with its linear approximation at $x_k$ and jumps to "
        "that line's root. Near a simple root the error roughly **squares** every step — "
        "quadratic convergence.\n"
        "\n"
        "**Lifting it to a robot.** Now $\\theta$ and the residual $F$ are vectors, so the "
        "scalar derivative $f'$ becomes a matrix — and that matrix is exactly the **body "
        "Jacobian** $J_b(\\theta)$ from notebook 03 (to first order a joint step "
        "$\\Delta\\theta$ changes the error twist by $J_b\\,\\Delta\\theta$). The scalar "
        "reciprocal $1/f'$ becomes the Jacobian inverse, giving the **Newton–Raphson step "
        "for inverse kinematics**:\n"
        "\n"
        "$$\\theta_{k+1}=\\theta_k+J_b(\\theta_k)^{-1}\\,\\mathcal{V}_b(\\theta_k),"
        "\\qquad \\mathcal{V}_b=\\log\\!\\big(T(\\theta_k)^{-1}T_d\\big)^{\\vee}.$$\n"
        "\n"
        "That is the whole algorithm: measure the pose error as a twist, solve a linear "
        "system for the joint correction, step, repeat.\n"
        "\n"
        "**Two practical fixes.** For the 7-DOF Panda $J_b$ is $6\\times7$ — not square, so "
        "it has no inverse. We use the Moore–Penrose **pseudoinverse** "
        "$J_b^{\\dagger}=J_b^{\\top}(J_bJ_b^{\\top})^{-1}$, which takes the minimum-norm "
        "joint step (the **Gauss–Newton** variant). And near a singularity "
        "$J_bJ_b^{\\top}$ becomes ill-conditioned and the step explodes, so we **damp** it "
        "(Levenberg–Marquardt):\n"
        "\n"
        "$$\\Delta\\theta=J_b^{\\top}\\big(J_bJ_b^{\\top}+\\lambda^2 I\\big)^{-1}"
        "\\mathcal{V}_b.$$\n"
        "\n"
        "With $\\lambda=0$ this is the undamped Newton–Raphson step above; a positive "
        "$\\lambda$ trades raw speed for stability — a tradeoff we measure in Section 3. "
        "Notice every ingredient is something earlier notebooks already built: "
        "`forward_kinematics`, `MatrixLog6` (notebook 01), and the body Jacobian "
        "(notebook 03)."
    ),
    md(
        "The picture below is the scalar construction on $f(x)=x^2-2$: from a guess "
        "$x_0$, follow the **tangent** down to where it meets the axis — that intercept is "
        "$x_1$ — then repeat from the curve above $x_1$. The iterates race to the root "
        "$x^{*}=\\sqrt{2}$, and the same geometry, with the Jacobian as the slope, is what "
        "the robot solver runs below."
    ),
    code(
        'render_tikz_file("_figures/src/newton_raphson.tex", name="newton_raphson")'
    ),

    # --- 2. newton-raphson by hand ---
    md(
        "## 2. Newton–Raphson by hand\n"
        "\n"
        "We pick a **reachable** target by running FK on a known configuration `q_target` "
        "(so a solution provably exists), then iterate the step above from a different "
        "starting guess. One routine covers both methods: `lam=0` is pure Newton–Raphson "
        "(the bare pseudoinverse step), and a small `lam` is the damped variant. We record "
        "the twist-error norm at each step for the convergence plot in Section 4."
    ),
    code(
        "q_target = np.array([0.6, -0.5, 0.4, -1.8, 0.3, 1.5, -0.2])   # a known config\n"
        "T_d = sm.forward_kinematics(q_target, frame='space')           # => a reachable pose\n"
        "\n"
        "def newton_ik(T_target, q0, lam=0.0, max_iters=60, tol=1e-8):\n"
        "    '''Newton-Raphson IK in the body frame; lam>0 adds Levenberg-Marquardt\n"
        "    damping. Returns (theta, residual_history).'''\n"
        "    q = np.array(q0, float)\n"
        "    history = []\n"
        "    for _ in range(max_iters):\n"
        "        Tb = sm.forward_kinematics(q, frame='space')\n"
        "        Vb = se3ToVec(MatrixLog6(TransInv(Tb) @ T_target))     # body-frame twist error\n"
        "        history.append(np.linalg.norm(Vb))\n"
        "        if history[-1] < tol:\n"
        "            break\n"
        "        Jb = sm.jacobian(q, frame='body')                      # the 'derivative' of F\n"
        "        dq = Jb.T @ np.linalg.solve(Jb @ Jb.T + lam**2 * np.eye(6), Vb)\n"
        "        q = q + dq                                             # the Newton step\n"
        "    return q, history\n"
        "\n"
        "theta_nr, hist_nr = newton_ik(T_d, HOME, lam=0.0)              # pure Newton-Raphson\n"
        "theta_hand, hist = newton_ik(T_d, HOME, lam=0.05)             # damped variant\n"
        "T_hand = sm.forward_kinematics(theta_hand, frame='space')\n"
        'print("Newton-Raphson (lam=0):  %2d iterations, final twist error %.2e"\n'
        "      % (len(hist_nr), hist_nr[-1]))\n"
        'print("damped         (lam=0.05): %2d iterations, final twist error %.2e"\n'
        "      % (len(hist), hist[-1]))\n"
        "assert np.allclose(T_hand[:3, 3], T_d[:3, 3], atol=1e-4)        # position recovered\n"
        "assert np.allclose(T_hand[:3, :3], T_d[:3, :3], atol=1e-4)      # orientation recovered\n"
        "assert np.allclose(sm.forward_kinematics(theta_nr)[:3, 3], T_d[:3, 3], atol=1e-4)"
    ),

    # --- 3. the library solver + the damping knob ---
    md(
        "## 3. The library solver and the damping knob\n"
        "\n"
        "ManipulaPy packages the same idea — plus joint-limit projection, an SVD-robust "
        "solve, adaptive damping and stagnation recovery — in "
        "`iterative_inverse_kinematics(T_desired, thetalist0)`, which returns "
        "`(theta, success, iterations)`. We recover a target from a perturbed seed, then "
        "sweep the `damping` parameter to see its cost."
    ),
    code(
        "seed = HOME + 0.2                                   # start 0.2 rad off every joint\n"
        "theta, ok, iters = sm.iterative_inverse_kinematics(T_d, seed, max_iterations=2000)\n"
        "T_sol = sm.forward_kinematics(theta, frame='space')\n"
        'print("library IK: success=%s  iterations=%d  position error=%.2e m"\n'
        "      % (ok, iters, np.linalg.norm(T_sol[:3, 3] - T_d[:3, 3])))\n"
        "assert ok and np.allclose(T_sol[:3, 3], T_d[:3, 3], atol=1e-3)\n"
        "\n"
        "# The damping lambda trades stability for speed: small lambda -> near-Newton, fast;\n"
        "# large lambda -> heavily regularised, stable near singularities but slow.\n"
        'print("\\ndamping   iterations")\n'
        "for lam in [5e-3, 2e-2, 1e-1, 3e-1]:\n"
        "    _, ok_l, it_l = sm.iterative_inverse_kinematics(\n"
        "        T_d, seed, damping=lam, max_iterations=3000)\n"
        '    print("  %.0e      %3d%s" % (lam, it_l, "" if ok_l else "  (no convergence)"))'
    ),

    # --- 4. convergence plot ---
    md(
        "## 4. Convergence\n"
        "\n"
        "Newton–Raphson converges **quadratically** near the solution — the error roughly "
        "squares each step. Plotting both by-hand runs on a log axis shows the "
        "characteristic cliff, and that the undamped step ($\\lambda=0$) is a step or two "
        "quicker *here*: at this well-conditioned pose the damping is insurance we don't "
        "need. Near a singularity that same insurance is what stops the curve from blowing "
        "up."
    ),
    code(
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.2, 3.4))\n"
        "ax.semilogy(range(len(hist_nr)), hist_nr, '-o', color='tab:green', ms=4,\n"
        "            label='Newton-Raphson ($\\\\lambda=0$)')\n"
        "ax.semilogy(range(len(hist)), hist, '-s', color='tab:purple', ms=4,\n"
        "            label='damped ($\\\\lambda=0.05$)')\n"
        "ax.set_xlabel('iteration'); ax.set_ylabel('twist-error norm $\\\\|\\\\mathcal{V}_b\\\\|$')\n"
        "ax.set_title('Newton-Raphson IK convergence (Panda)')\n"
        "ax.legend(); ax.grid(True, which='both', ls=':', alpha=0.6)\n"
        'embed_pgf_fig(fig, name="ik_convergence")'
    ),

    # --- 5. a wider family of IK algorithms ---
    md(
        "## 5. A wider family of IK algorithms\n"
        "\n"
        "Newton–Raphson and its damped cousin are the **Jacobian-based iterative** branch, "
        "but they are one part of a larger family. The main approaches:\n"
        "\n"
        "- **Jacobian transpose** — $\\Delta\\theta=\\alpha\\,J_b^{\\top}\\mathcal{V}_b$. "
        "Replaces the inverse with a plain transpose: no linear solve, so each step is "
        "cheap and it *never* blows up at a singularity. The cost is slow, "
        "gradient-descent-like convergence (many steps).\n"
        "- **Pseudoinverse (Gauss–Newton)** — $\\Delta\\theta=J_b^{\\dagger}"
        "\\mathcal{V}_b$. The true Newton step from Section 1: fast (quadratic) but "
        "fragile near singularities.\n"
        "- **Damped least squares (Levenberg–Marquardt)** — $\\Delta\\theta=J_b^{\\top}"
        "(J_bJ_b^{\\top}+\\lambda^2I)^{-1}\\mathcal{V}_b$. Interpolates between the two "
        "($\\lambda\\to0$ is Gauss–Newton; large $\\lambda$ behaves like a scaled "
        "transpose step). This is `iterative_inverse_kinematics`.\n"
        "- **Optimization-based** — recast IK as *minimise* $\\|\\text{pose error}\\|^2$ "
        "*subject to joint limits* and hand it to a numerical optimiser (SQP, BFGS, …). "
        "Naturally handles joint-limit and collision constraints and escapes local minima, "
        "at a higher per-step cost. This is what `trac_ik` runs.\n"
        "- **Analytical / closed-form** — for arms with special geometry (a spherical "
        "wrist, Pieper's condition) the equations solve exactly and instantly, returning "
        "*every* solution. No seed, no iteration — but it must be derived per robot, and "
        "the Panda's offset wrist has no simple closed form.\n"
        "- **Heuristic / geometric** — Cyclic Coordinate Descent (CCD) and FABRIK nudge "
        "one or two joints at a time toward the goal; simple and fast, popular in "
        "animation and for highly redundant chains, but weaker on orientation accuracy.\n"
        "- **Learned** — neural models (e.g. normalising-flow IK) approximate the inverse "
        "map from data, trading exactness for speed and full solution-set coverage.\n"
        "\n"
        "ManipulaPy implements the first two families. We have used the Jacobian-inverse "
        "methods — below we run the **transpose** and the **optimisation** (TRAC-IK) "
        "approaches on the same target to feel the difference."
    ),
    code(
        "# Jacobian TRANSPOSE: swap the inverse for a plain transpose. No linear solve, so\n"
        "# each step is cheap and never blows up at a singularity -- but it takes many more\n"
        "# steps. The scalar alpha is the error-minimising step length along the J^T V_b ray.\n"
        "def jt_ik(T_target, q0, max_iters=1000, tol=1e-4):\n"
        "    '''Jacobian-transpose IK in the body frame. Returns (theta, residual_history).'''\n"
        "    q = np.array(q0, float)\n"
        "    history = []\n"
        "    for _ in range(max_iters):\n"
        "        Tb = sm.forward_kinematics(q, frame='space')\n"
        "        Vb = se3ToVec(MatrixLog6(TransInv(Tb) @ T_target))\n"
        "        history.append(np.linalg.norm(Vb))\n"
        "        if history[-1] < tol:\n"
        "            break\n"
        "        Jb = sm.jacobian(q, frame='body')\n"
        "        JJtV = Jb @ Jb.T @ Vb\n"
        "        alpha = float(Vb @ JJtV / (JJtV @ JJtV + 1e-12))    # optimal step length\n"
        "        q = q + alpha * (Jb.T @ Vb)                         # transpose step, no inverse\n"
        "    return q, history\n"
        "\n"
        "theta_jt, hist_jt = jt_ik(T_d, HOME)\n"
        "T_jt = sm.forward_kinematics(theta_jt, frame='space')\n"
        'print("Jacobian transpose: %d iterations (vs %d for Newton-Raphson) for the same pose"\n'
        "      % (len(hist_jt), len(hist_nr)))\n"
        "assert np.allclose(T_jt[:3, 3], T_d[:3, 3], atol=1e-3)      # reaches the target, slowly"
    ),
    md(
        "### Optimization-based: TRAC-IK\n"
        "\n"
        "`trac_ik` is modelled on the TRAC-IK solver (Beeson & Ames, 2015), whose insight "
        "was that the classic Jacobian solvers have a blind spot: when the solution lies "
        "against a **joint limit**, an unconstrained Newton/DLS step keeps trying to push "
        "past the bound and stalls. TRAC-IK runs **two solvers concurrently** and returns "
        "whichever finishes first:\n"
        "\n"
        "- a **damped-least-squares** solver — the fast, local method we built above; and\n"
        "- a **Sequential Quadratic Programming** optimiser that recasts IK as *minimise* "
        "$f(\\theta)=\\|e(\\theta)\\|^2$ subject to the joint limits as hard **bounds**, "
        "with the analytic gradient $\\nabla f=2J^{\\top}e$ (SLSQP). Being bounds-aware it "
        "handles limit-constrained poses the Jacobian step trips on, and being an "
        "optimiser it can escape local minima.\n"
        "\n"
        "ManipulaPy's implementation adds **diverse initial guesses** (a workspace "
        "heuristic, the joint-limit midpoint, and random restarts), runs sequentially by "
        "default (parallel optional), and — crucially — takes a hard **`timeout`**. It is "
        "an *anytime* solver: when the clock runs out it returns the best configuration "
        "found so far. With a `theta0` warm start it suits real-time control loops, where "
        "the previous step's solution seeds the next.\n"
        "\n"
        "It returns `(theta, success, solve_time_seconds)`:"
    ),
    code(
        "theta_tr, ok_tr, secs = sm.trac_ik(T_d, theta0=HOME, timeout=0.2)\n"
        "T_tr = sm.forward_kinematics(theta_tr, frame='space')\n"
        'print("trac_ik: success=%s in %.1f ms, position error = %.2e m"\n'
        "      % (ok_tr, secs * 1000, np.linalg.norm(T_tr[:3, 3] - T_d[:3, 3])))\n"
        "assert ok_tr"
    ),
    md(
        "### Success vs latency: a solver tradeoff\n"
        "\n"
        "The design goal of `trac_ik` is **bounded latency**, not maximum success rate — "
        "and the clearest way to see what that buys is to put the solvers on the same "
        "targets and measure *both* axes. We solve a batch of reachable poses (FK of "
        "random in-limit configs, so a solution always exists) with three solvers and "
        "record success rate and median solve time.\n"
        "\n"
        "*(The exhaustive multi-start solver is seconds-slow, so this runs on a small "
        "batch and takes about a minute.)*"
    ),
    code(
        "import time, warnings\n"
        '# scipy\'s SLSQP prints a benign out-of-bounds notice on some steps; quiet it.\n'
        'warnings.filterwarnings("ignore", message="Values in x were outside bounds")\n'
        "lims = np.array(joint_limits())\n"
        "\n"
        "def solve_time(fn):\n"
        "    t0 = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t0) * 1000\n"
        "\n"
        "np.random.seed(7)\n"
        "bench_targets = [sm.forward_kinematics(np.random.uniform(lims[:, 0], lims[:, 1]),\n"
        "                                       frame='space') for _ in range(12)]\n"
        "solvers = {\n"
        "    'iterative\\n(1 seed)':   lambda T: sm.iterative_inverse_kinematics(T, HOME, max_iterations=500),\n"
        "    'robust\\n(multi-start)': lambda T: sm.robust_inverse_kinematics(T),\n"
        "    'trac_ik\\n(DLS+SQP)':    lambda T: sm.trac_ik(T, theta0=HOME, timeout=0.2, num_restarts=5),\n"
        "}\n"
        "bench = {}\n"
        "for name, fn in solvers.items():\n"
        "    succ, times = 0, []\n"
        "    for T in bench_targets:\n"
        "        out, ms = solve_time(lambda: fn(T))\n"
        "        succ += bool(out[1]); times.append(ms)\n"
        "    bench[name] = (100 * succ / len(bench_targets), float(np.median(times)))\n"
        "    print('%-22s success %3.0f%%  median %8.1f ms'\n"
        "          % (name.replace('\\n', ' '), bench[name][0], bench[name][1]))"
    ),
    code(
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.6, 3.6))\n"
        "colors = {'iterative\\n(1 seed)': 'tab:blue', 'robust\\n(multi-start)': 'tab:red',\n"
        "          'trac_ik\\n(DLS+SQP)': 'tab:green'}\n"
        "for name, (succ, med) in bench.items():\n"
        "    ax.scatter(med, succ, s=90, color=colors[name], label=name.replace('\\n', ' '), zorder=3)\n"
        "ax.axvspan(0, 50, color='green', alpha=0.06)                 # real-time region\n"
        "ax.set_xscale('log'); ax.set_xlim(5, 8000); ax.set_ylim(0, 108)\n"
        "ax.set_xlabel('median solve time (ms, log scale)'); ax.set_ylabel('success rate (%)')\n"
        "ax.set_title('IK solvers: success vs latency (Panda)')\n"
        "ax.legend(loc='center left', fontsize=8); ax.grid(True, ls=':', alpha=0.5)\n"
        'embed_pgf_fig(fig, name="ik_solver_tradeoff")'
    ),
    md(
        "The three solvers trace out a **tradeoff**, not a ranking:\n"
        "\n"
        "- **single-shot DLS** — fast (~10 ms), but a single seed misses a fraction of "
        "poses;\n"
        "- **multi-start (`robust`)** — approaches 100%, but spends **seconds**, unusable "
        "inside a control loop;\n"
        "- **`trac_ik`** — lives in the real-time corner (tens of ms, capped by its "
        "`timeout`), accepting that within that budget it won't crack every pose.\n"
        "\n"
        "Raising `trac_ik`'s `timeout` or `num_restarts` moves it up and to the right — "
        "more success, more latency. Its latency *profile* over many targets is what a "
        "controller has to budget for: most solves land fast, with a tail at the right "
        "edge where it spent the whole timeout on a pose it could not reach and returned "
        "its best attempt."
    ),
    code(
        "np.random.seed(11)\n"
        "trac_times = []\n"
        "for _ in range(60):\n"
        "    T = sm.forward_kinematics(np.random.uniform(lims[:, 0], lims[:, 1]), frame='space')\n"
        "    out, ms = solve_time(lambda: sm.trac_ik(T, theta0=HOME, timeout=0.2))\n"
        "    trac_times.append(ms)\n"
        "\n"
        "plt = setup_pgf()\n"
        "fig, ax = plt.subplots(figsize=(5.6, 3.2))\n"
        "ax.hist(trac_times, bins=20, color='tab:green', alpha=0.8, edgecolor='white')\n"
        "ax.axvline(np.median(trac_times), color='black', ls='--', lw=1,\n"
        "           label='median %.0f ms' % np.median(trac_times))\n"
        "ax.set_xlabel('solve time (ms)'); ax.set_ylabel('count')\n"
        "ax.set_title('trac_ik solve-time distribution (%d targets)' % len(trac_times))\n"
        "ax.legend()\n"
        'embed_pgf_fig(fig, name="trac_ik_timing")'
    ),

    # --- 6. redundancy ---
    md(
        "## 6. One pose, many solutions\n"
        "\n"
        "The Panda is **redundant** (7 joints, 6-D task), so a reachable pose has a whole "
        "1-parameter family of joint solutions — the null-space self-motion from notebook "
        "03. Inverse kinematics returns *one* of them, and which one depends on the seed. "
        "Solving for `q_target`'s pose starting from `HOME` lands a valid configuration "
        "that reaches the same pose with a **different elbow**."
    ),
    code(
        "theta_alt, ok, _ = sm.iterative_inverse_kinematics(T_d, HOME, max_iterations=4000)\n"
        "T_alt = sm.forward_kinematics(theta_alt, frame='space')\n"
        "\n"
        "# Same pose as q_target ...\n"
        "assert ok and np.allclose(T_alt[:3, 3], T_d[:3, 3], atol=1e-3)\n"
        "assert np.allclose(T_alt[:3, :3], T_d[:3, :3], atol=1e-2)\n"
        "# ... reached by a genuinely different joint vector.\n"
        "joint_gap = np.abs(theta_alt - q_target)\n"
        'print("same end-effector pose, different joints:")\n'
        'print("  max per-joint difference:", round(np.degrees(joint_gap.max()), 1), "deg")\n'
        "assert joint_gap.max() > np.radians(5)         # not just a copy of q_target"
    ),

    # --- 7. when the target is out of reach ---
    md(
        "## 7. When the target is out of reach\n"
        "\n"
        "Not every pose is reachable. A good solver should **fail honestly** rather than "
        "return garbage. Pushing the target 1.5 m past the Panda's reach, "
        "`iterative_inverse_kinematics` reports `success=False` and leaves the arm at its "
        "closest approach. For hard-but-reachable poses, "
        "`robust_inverse_kinematics` retries from multiple seeds and reports which "
        "strategy won."
    ),
    code(
        "T_far = T_d.copy(); T_far[0, 3] += 1.5                 # shove it well outside the workspace\n"
        "theta_f, ok_f, _ = sm.iterative_inverse_kinematics(T_far, HOME, max_iterations=1500)\n"
        "T_f = sm.forward_kinematics(theta_f, frame='space')\n"
        'print("unreachable target: success=%s, residual = %.2f m (honest failure)"\n'
        "      % (ok_f, np.linalg.norm(T_f[:3, 3] - T_far[:3, 3])))\n"
        "assert not ok_f                                        # correctly reports failure\n"
        "\n"
        "# Multi-start robust solver on the reachable pose: returns the winning strategy.\n"
        "theta_r, ok_r, iters_r, strategy = sm.robust_inverse_kinematics(T_d, max_attempts=10)\n"
        "T_r = sm.forward_kinematics(theta_r, frame='space')\n"
        'print("robust IK: success=%s via \'%s\' strategy, position error = %.2e m"\n'
        "      % (ok_r, strategy, np.linalg.norm(T_r[:3, 3] - T_d[:3, 3])))\n"
        "assert ok_r"
    ),

    # --- 8. seeing the solution ---
    md(
        "## 8. Seeing the solution\n"
        "\n"
        "Finally, the payoff: load the meshed Panda into ManipulaPy's PyBullet simulation "
        "(headless, as in notebooks 02–03), pose it at the IK solution, and mark the "
        "target. The fingertip sits on the marker — the joint angles we solved for put the "
        "end-effector exactly where we asked."
    ),
    code(
        "import logging\n"
        'os.environ.setdefault("MANIPULAPY_PYBULLET_CONNECT", "DIRECT")  # headless; remove to watch in a GUI\n'
        "from helpers import panda_pybullet_urdf, sim_snapshot, quiet_pybullet\n"
        "from ManipulaPy.sim import Simulation\n"
        "import pybullet as p\n"
        "\n"
        "with quiet_pybullet():\n"
        "    sim = Simulation(panda_pybullet_urdf(), joint_limits())\n"
        "# Simulation's logger defaults to DEBUG; keep the notebook output clean.\n"
        'logging.getLogger("SimulationLogger").setLevel(logging.WARNING)\n'
        "\n"
        "# Pose the arm at the IK solution; the two gripper joints (non-fixed) pad with 0.\n"
        "for j, qj in zip(sim.non_fixed_joints, list(theta) + [0.0, 0.0]):\n"
        "    p.resetJointState(sim.robot_id, j, qj)\n"
        "\n"
        "# Drop a green marker at the target end-effector position.\n"
        "target_xyz = T_d[:3, 3]\n"
        "marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 0.8, 0, 1])\n"
        "p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker, basePosition=list(target_xyz))\n"
        'img = sim_snapshot("sim_ik_solution", target=tuple(target_xyz), distance=1.5, yaw=50, pitch=-20)\n'
        "sim.disconnect_simulation()\n"
        "img"
    ),

    # --- smoke test ---
    md("## Smoke test\n\nAsserts the key invariants of this notebook in one cell."),
    code(
        "sm2, _ = load_panda(); sm2.joint_limits = joint_limits()\n"
        "T_goal = sm2.forward_kinematics(HOME, frame='space')\n"
        "# Library IK recovers a perturbed seed.\n"
        "theta_s, ok_s, it_s = sm2.iterative_inverse_kinematics(T_goal, HOME + 0.2, max_iterations=2000)\n"
        "assert ok_s, f'IK failed after {it_s} iterations'\n"
        "T_check = sm2.forward_kinematics(theta_s, frame='space')\n"
        "assert np.allclose(T_check[:3, 3], T_goal[:3, 3], atol=1e-3)        # position recovered\n"
        "assert np.allclose(T_check[:3, :3], T_goal[:3, :3], atol=1e-2)      # orientation recovered\n"
        "# Hand-rolled Newton-Raphson agrees with the library at the end-effector.\n"
        "theta_h, _ = newton_ik(T_goal, HOME + 0.2, lam=0.05)\n"
        "assert np.allclose(sm2.forward_kinematics(theta_h)[:3, 3], T_goal[:3, 3], atol=1e-3)\n"
        "# Unreachable target is reported as a failure.\n"
        "T_bad = T_goal.copy(); T_bad[0, 3] += 1.5\n"
        "_, ok_bad, _ = sm2.iterative_inverse_kinematics(T_bad, HOME, max_iterations=1000)\n"
        "assert not ok_bad\n"
        'print("nb04 inverse kinematics: smoke OK")'
    ),

    # --- exercises ---
    md(
        "## Try it\n"
        "\n"
        "1. Solve IK for the same pose `T_d` from three different random seeds "
        "(`np.random.uniform` within `joint_limits()`). Do you always land the same joint "
        "vector? Confirm each still reaches `T_d`.\n"
        "2. Keep `newton_ik` at `lam=0` (pure Newton–Raphson) but start near a "
        "stretched-out (near-singular) configuration. Watch the step sizes blow up — then "
        "add a little damping and see it recover.\n"
        "3. Take the redundant solution `theta_alt` and a null-space step from notebook 03; "
        "verify you can slide between IK solutions without moving the end-effector.\n"
        "\n"
        "*Next up — notebook 05: **dynamics** — the mass matrix, Coriolis and gravity "
        "terms, and the inverse/forward-dynamics round trip.*"
    ),
    md(
        "## References\n"
        "\n"
        "1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and "
        "Control*, Cambridge University Press, 2017. — Chapter 6, *Inverse Kinematics* "
        "(Newton–Raphson on the body twist, numerical IK).\n"
        "2. S. R. Buss, *Introduction to Inverse Kinematics with Jacobian Transpose, "
        "Pseudoinverse and Damped Least Squares methods*, 2004. — The damped least-squares "
        "step used here.\n"
        "3. B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics: Modelling, "
        "Planning and Control*, Springer, 2009. — Redundancy resolution.\n"
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
