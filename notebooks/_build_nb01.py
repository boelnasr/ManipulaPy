"""Build and execute notebooks/01_rigid_body_motions.ipynb (full foundations notebook).

Run from the notebooks/ directory so `_shared` is importable at execute time:
    cd notebooks && python3 _build_nb01.py
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(HERE, "01_rigid_body_motions.ipynb")


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


cells = [
    md(
        "# 01 · Rigid-Body Motions and Screw Axis Theory\n"
        "\n"
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/boelnasr/ManipulaPy/blob/main/"
        "notebooks/01_rigid_body_motions.ipynb)\n"
        "\n"
        "> **ManipulaPy teaching course — notebook 1 of 11.** Running robot: Franka Panda.\n"
        "\n"
        "This is where the whole course begins. Every later notebook — forward "
        "kinematics, Jacobians, dynamics, control — is built on the language introduced "
        "here. We build it from the ground up:\n"
        "\n"
        "1. **Rotations** and the group $SO(3)$\n"
        "2. **Angular velocity** and exponential coordinates for rotation\n"
        "3. **Rigid-body motions** and the group $SE(3)$\n"
        "4. **Screw axes** and **twists**\n"
        "5. **Exponential coordinates** for rigid-body motion\n"
        "6. The **adjoint map** for changing the frame of a twist\n"
        "\n"
        "Throughout, the concepts are demonstrated with the matching functions in "
        "`ManipulaPy.utils`."
    ),
    md(
        "### Running on Colab or another cloud platform?\n"
        "\n"
        "The next cell bootstraps the environment when it detects Google Colab: it clones "
        "the repository (so the shared helpers and figure sources are available) and "
        "installs ManipulaPy. It is a **no-op when you run locally** from a clone of the "
        "repo. Figures are pre-rendered and committed, so they display even without a TeX "
        "installation — install TeX Live + poppler only if you want to re-render them from "
        "the `.tex` sources."
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
        "from ManipulaPy.utils import (\n"
        "    MatrixExp3, MatrixLog3, MatrixExp6, MatrixLog6,\n"
        "    VecToso3, VecTose3, se3ToVec, skew_symmetric_to_vector,\n"
        "    adjoint_transform, TransInv, TransToRp,\n"
        "    rotation_logm, transform_from_twist, extract_screw_list,\n"
        "    euler_to_rotation_matrix, rotation_matrix_to_euler_angles,\n"
        ")\n"
        'np.set_printoptions(precision=4, suppress=True)\n'
        'print("imports OK")'
    ),

    # --- Section A: rotations & SO(3) ---
    md(
        "## 1. Rotations and $SO(3)$\n"
        "\n"
        "A **rotation matrix** $R$ describes the orientation of one frame relative to "
        "another. The set of all rotations forms the **special orthogonal group** "
        "$SO(3)=\\{R\\in\\mathbb{R}^{3\\times3}\\mid R^\\top R=I,\\ \\det R=+1\\}$. "
        "The two defining properties — orthonormal columns and determinant $+1$ — mean a "
        "rotation preserves lengths, angles, and handedness.\n"
        "\n"
        "The figure below shows a body frame $\\{b\\}$ obtained by rotating the fixed "
        "frame $\\{s\\}$ by an angle $\\theta$ about the $\\hat{z}_s$ axis (pointing out "
        "of the page)."
    ),
    code('render_tikz_file("_figures/src/rotation_so3.tex", name="rotation_so3")'),
    md(
        "### Exponential coordinates: from an axis–angle to a rotation\n"
        "\n"
        "Any rotation can be written as a rotation by an angle $\\theta$ about a unit "
        "axis $\\hat{\\omega}$. Stacking these as $\\hat{\\omega}\\theta\\in\\mathbb{R}^3$ "
        "gives the **exponential coordinates** of the rotation. The map back to a matrix "
        "is the matrix exponential of the skew-symmetric form $[\\hat{\\omega}]$ "
        "(Rodrigues' formula):\n"
        "\n"
        "$$R=e^{[\\hat{\\omega}]\\theta}\\in SO(3).$$\n"
        "\n"
        "In ManipulaPy: `VecToso3` builds the $3\\times3$ skew matrix $[\\,\\cdot\\,]$, "
        "`MatrixExp3` exponentiates it, and `MatrixLog3` inverts the map."
    ),
    code(
        "# Rotate 90 degrees about z: x_hat should map to y_hat.\n"
        "omega_hat_theta = np.array([0, 0, np.pi / 2])      # axis * angle\n"
        "so3 = VecToso3(omega_hat_theta)                    # 3x3 skew-symmetric matrix\n"
        "R = MatrixExp3(so3)                                # Rodrigues' formula\n"
        'print("R =\\n", R)\n'
        'print("R @ x_hat =", R @ np.array([1, 0, 0]), "(expect [0, 1, 0])")\n'
        "\n"
        "# SO(3) membership: orthonormal and det = +1.\n"
        'print("R^T R = I :", np.allclose(R.T @ R, np.eye(3)))\n'
        'print("det(R)   :", round(float(np.linalg.det(R)), 6))\n'
        "\n"
        "# Recover the exponential coordinates with the matrix log.\n"
        "recovered = skew_symmetric_to_vector(MatrixLog3(R))\n"
        'print("log(R)   =", recovered, "(expect [0, 0, %.4f])" % (np.pi / 2))'
    ),
    md(
        "### `rotation_logm`: axis and angle directly\n"
        "\n"
        "Often you want the **axis** and **angle** separately rather than the stacked "
        "vector $\\hat{\\omega}\\theta$. `rotation_logm(R)` returns exactly "
        "$(\\hat{\\omega},\\ \\theta)$."
    ),
    code(
        "axis, angle = rotation_logm(R)\n"
        'print("axis  =", axis, " angle =", round(float(angle), 4))\n'
        "assert np.allclose(axis * angle, recovered, atol=1e-9)   # consistent with MatrixLog3"
    ),
    md(
        "### The log is **not** a unique inverse\n"
        "\n"
        "Exponentiation is many-to-one, so the matrix log must *choose* a representative:\n"
        "\n"
        "- It returns the **principal angle** $\\theta\\in[0,\\pi]$.\n"
        "- The axis–angle pair is not unique: $(\\hat{\\omega},\\theta)$, "
        "$(-\\hat{\\omega},-\\theta)$, and $(\\hat{\\omega},\\theta+2\\pi k)$ all give the "
        "**same** $R$.\n"
        "- At $\\theta=\\pi$ the axis is determined only up to sign (both $\\pm\\hat{\\omega}$ "
        "are valid), a genuine singular case.\n"
        "- For $R=I$ ($\\theta=0$) the axis is undefined.\n"
        "\n"
        "The same caveats apply to `MatrixLog6` for $SE(3)$: a **pure translation** has zero "
        "rotation, so its screw is a pure-translation twist (the rotational part vanishes and "
        "the “axis” is the translation direction).\n"
        "\n"
        "Numerically, $\\theta=\\pi$ is the hard case: the log is **ill-conditioned** there, "
        "so the recovered angle carries small error and `MatrixLog3` can degrade. "
        "`rotation_logm` stays robust and is the recommended way to extract an axis–angle."
    ),
    code(
        "# Large rotation: 300 deg about z is reported as its principal equivalent.\n"
        "R_big = MatrixExp3(VecToso3([0, 0, np.deg2rad(300)]))\n"
        "axis_b, angle_b = rotation_logm(R_big)\n"
        'print("input 300 deg -> principal angle %.1f deg about axis %s"\n'
        '      % (np.rad2deg(angle_b), np.round(axis_b, 3)))\n'
        "assert 0 <= angle_b <= np.pi + 1e-9               # principal angle in [0, pi]\n"
        "assert np.allclose(MatrixExp3(VecToso3(axis_b * angle_b)), R_big)  # same rotation\n"
        "\n"
        "# theta = pi: the singular case. The axis is defined only up to sign, and the\n"
        "# log is ill-conditioned, so the recovered angle carries ~1e-8 error.\n"
        "R_pi = MatrixExp3(VecToso3([0, 0, np.pi]))\n"
        "ax_pi, ang_pi = rotation_logm(R_pi)\n"
        'print("theta=pi: angle = %.10f, axis = %s (sign-ambiguous)" % (ang_pi, np.round(ax_pi, 3)))\n'
        'print("angle error vs pi:", abs(ang_pi - np.pi))\n'
        "# Reconstruction is only approximate here -- note the relaxed tolerance.\n"
        "assert np.allclose(MatrixExp3(VecToso3(ax_pi * ang_pi)), R_pi, atol=1e-6)\n"
        'print("rotation_logm stays robust at pi; use it rather than MatrixLog3, which\\n"\n'
        '      "degrades to a near-zero matrix at theta = pi.")'
    ),
    md(
        "### Sidebar: Euler angles (and why we avoid them)\n"
        "\n"
        "Euler angles parameterize a rotation with three scalars. ManipulaPy offers "
        "`euler_to_rotation_matrix` (ZYX) and its inverse. **Watch the units — they are "
        "asymmetric:** the forward map takes **degrees**, the inverse returns **radians**. "
        "Euler angles also suffer **gimbal lock** (a lost degree of freedom at "
        "pitch $=\\pm90^\\circ$), which is why screw/exponential coordinates are preferred "
        "throughout this course."
    ),
    code(
        "R_euler = euler_to_rotation_matrix([0, 0, 90])      # roll, pitch, yaw in DEGREES\n"
        'print("euler [0,0,90] deg -> R @ x_hat =", R_euler @ np.array([1, 0, 0]))\n'
        "back = rotation_matrix_to_euler_angles(R_euler)     # returns RADIANS\n"
        'print("inverse -> %s rad  (= %s deg)" % (np.round(back, 4), np.round(np.rad2deg(back), 1)))\n'
        "assert np.allclose(R_euler @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)"
    ),
    md(
        "### Angular velocity: space vs body\n"
        "\n"
        "When a frame rotates, $R=R(t)$ and its rate $\\dot R$ encodes the **angular "
        "velocity**. There are two equally valid expressions, differing only in the frame "
        "they are written in:\n"
        "\n"
        "$$[\\omega_s]=\\dot R\\,R^\\top \\quad(\\text{space frame}),\\qquad "
        "[\\omega_b]=R^\\top\\dot R \\quad(\\text{body frame}),$$\n"
        "\n"
        "and the two are related by the rotation itself: $\\omega_s = R\\,\\omega_b$. "
        "(Both $[\\omega_s]$ and $[\\omega_b]$ are skew-symmetric — recover the vector with "
        "`skew_symmetric_to_vector`.) We verify all three numerically below."
    ),
    code(
        "# A frame spinning at constant rate about a fixed unit axis.\n"
        "w_hat = np.array([1.0, 2.0, 2.0]); w_hat /= np.linalg.norm(w_hat)\n"
        "theta_dot = 0.7\n"
        "def R_of_t(t):\n"
        "    return MatrixExp3(VecToso3(w_hat * theta_dot * t))\n"
        "\n"
        "t0, h = 0.4, 1e-6\n"
        "R0 = R_of_t(t0)\n"
        "Rdot = (R_of_t(t0 + h) - R_of_t(t0 - h)) / (2 * h)     # finite-difference dR/dt\n"
        "\n"
        "omega_s = skew_symmetric_to_vector(Rdot @ R0.T)        # [w_s] = Rdot R^T\n"
        "omega_b = skew_symmetric_to_vector(R0.T @ Rdot)        # [w_b] = R^T Rdot\n"
        'print("omega_s =", omega_s, " (expect w_hat*theta_dot =", np.round(w_hat * theta_dot, 4), ")")\n'
        'print("omega_b =", omega_b)\n'
        "\n"
        "assert np.allclose(omega_s, w_hat * theta_dot, atol=1e-4)   # space velocity = axis*rate\n"
        "assert np.allclose(omega_s, R0 @ omega_b, atol=1e-4)        # omega_s = R omega_b"
    ),

    # --- Section B: rigid-body motions & SE(3) ---
    md(
        "## 2. Rigid-body motions and $SE(3)$\n"
        "\n"
        "A full rigid-body configuration is a rotation **and** a translation, packed into "
        "a $4\\times4$ **homogeneous transformation matrix**\n"
        "\n"
        "$$T=\\begin{bmatrix}R & p\\\\ 0 & 1\\end{bmatrix}\\in SE(3),\\qquad "
        "R\\in SO(3),\\ p\\in\\mathbb{R}^3.$$\n"
        "\n"
        "These compose by matrix multiplication and invert in closed form "
        "($T^{-1}=\\begin{bmatrix}R^\\top & -R^\\top p\\\\ 0 & 1\\end{bmatrix}$). "
        "ManipulaPy gives `TransToRp` (split $T$ into $R,p$) and `TransInv` "
        "(the closed-form inverse)."
    ),
    code(
        "p = np.array([0.3, -0.1, 0.5])\n"
        "T = np.eye(4)\n"
        "T[:3, :3] = R\n"
        "T[:3, 3] = p\n"
        'print("T =\\n", T)\n'
        "\n"
        "R_back, p_back = TransToRp(T)\n"
        'print("split ok:", np.allclose(R_back, R) and np.allclose(p_back, p))\n'
        "\n"
        "# Closed-form inverse undoes the transform.\n"
        'print("T^{-1} T = I :", np.allclose(TransInv(T) @ T, np.eye(4)))'
    ),

    # --- Section C: screw axes & twists ---
    md(
        "## 3. Screw axes and twists\n"
        "\n"
        "By the **Chasles–Mozzi theorem**, every rigid-body motion is a rotation about "
        "some axis combined with a translation along that same axis — a **screw motion**. "
        "The screw axis is described by:\n"
        "\n"
        "- a unit direction $\\hat{s}$ along the axis,\n"
        "- a point $q$ the axis passes through,\n"
        "- a **pitch** $h$ = (linear speed)/(angular speed) along the axis.\n"
        "\n"
        "The instantaneous motion is a **twist** $\\mathcal{V}=[\\omega;\\,v]$, where "
        "$\\omega$ is the angular velocity and $v$ is the twist's **linear component** "
        "(the two parts together describe the rigid-body velocity field). For motion about "
        "the screw at rate $\\dot{\\theta}$:\n"
        "\n"
        "$$\\mathcal{V}=\\begin{bmatrix}\\omega\\\\ v\\end{bmatrix}"
        "=\\begin{bmatrix}\\hat{s}\\,\\dot{\\theta}\\\\ "
        "(-\\hat{s}\\times q + h\\,\\hat{s})\\,\\dot{\\theta}\\end{bmatrix}.$$\n"
        "\n"
        "A twist, like an angular velocity, is **frame-dependent**. The same physical "
        "motion has a **space twist** $\\mathcal{V}_s$ (expressed in the fixed frame) and a "
        "**body twist** $\\mathcal{V}_b$ (expressed in the moving body frame). Only in the "
        "space frame does $v$ coincide with the velocity of the (possibly imaginary) body "
        "point instantaneously at the origin; in general $v$ is just the linear part of the "
        "twist. The two are related by the adjoint, "
        "$\\mathcal{V}_s=[\\mathrm{Ad}_{T_{sb}}]\\,\\mathcal{V}_b$ — which we use in "
        "Section 5 and again for Jacobians in notebook 03."
    ),
    code('render_tikz_file("_figures/src/screw_axis.tex", name="screw_axis")'),
    md(
        "A **revolute joint** is the special case of zero pitch ($h=0$): the linear part "
        "is purely $v=-\\hat{s}\\times q$. We build such a screw below — this is exactly "
        "the form the Panda's joint screw axes take in the next notebook."
    ),
    code(
        "s_hat = np.array([0.0, 0.0, 1.0])      # axis direction (z)\n"
        "q = np.array([0.3, 0.0, 0.0])          # a point on the axis\n"
        "v = -np.cross(s_hat, q)                 # zero-pitch (revolute) linear part\n"
        "S = np.concatenate([s_hat, v])         # the 6-vector screw axis [omega; v]\n"
        'print("screw axis S =", S)\n'
        "\n"
        "# [V] in se(3) is the 4x4 matrix form, via VecTose3; se3ToVec inverts it.\n"
        "se3 = VecTose3(S)\n"
        'print("se3ToVec(VecTose3(S)) == S :", np.allclose(se3ToVec(se3), S))'
    ),

    # --- Section D: exp coords for rigid motion ---
    md(
        "## 4. Exponential coordinates for rigid-body motion\n"
        "\n"
        "Just as $e^{[\\hat{\\omega}]\\theta}$ turns an axis–angle into a rotation, the "
        "$4\\times4$ matrix exponential $e^{[\\mathcal{S}]\\theta}$ turns a screw axis "
        "$\\mathcal{S}$ followed for an angle $\\theta$ into a rigid-body transform "
        "$T\\in SE(3)$. This single operation is the engine behind the **Product of "
        "Exponentials** forward-kinematics formula in notebook 02.\n"
        "\n"
        "`MatrixExp6` takes the $4\\times4$ matrix $[\\mathcal{S}]\\theta$; `MatrixLog6` "
        "recovers it."
    ),
    code(
        "theta = np.pi / 2\n"
        "T_screw = MatrixExp6(VecTose3(S * theta))    # follow the screw S by theta\n"
        'print("T(theta) =\\n", T_screw)\n'
        "\n"
        "# The zero twist gives the identity transform.\n"
        'print("exp(0) == I :", np.allclose(MatrixExp6(VecTose3(np.zeros(6))), np.eye(4)))\n'
        "\n"
        "# Round trip: log of the transform recovers the original screw * theta.\n"
        "recovered = se3ToVec(MatrixLog6(T_screw))\n"
        'print("log(exp(S*theta)) == S*theta :", np.allclose(recovered, S * theta))'
    ),
    md(
        "### `transform_from_twist`: the same thing, in one call\n"
        "\n"
        "ManipulaPy bundles “build $[\\mathcal{S}]\\theta$ then exponentiate” into "
        "`transform_from_twist(S, theta)`. It is exactly "
        "$e^{[\\mathcal{S}]\\theta}$ — verified here for a **revolute** screw (zero pitch) "
        "and a **prismatic** screw (pure translation, $\\omega=0$)."
    ),
    code(
        "# Revolute: zero-pitch screw about z through q.\n"
        "S_rev = np.concatenate([[0, 0, 1], -np.cross([0, 0, 1], [0.3, 0, 0])])\n"
        "assert np.allclose(transform_from_twist(S_rev, np.pi / 2),\n"
        "                   MatrixExp6(VecTose3(S_rev * (np.pi / 2))))\n"
        "\n"
        "# Prismatic: pure translation along y (omega = 0, v = y_hat).\n"
        "S_pris = np.array([0, 0, 0, 0, 1, 0], float)\n"
        "T_pris = transform_from_twist(S_pris, 0.5)\n"
        'print("prismatic translation =", T_pris[:3, 3], "(expect [0, 0.5, 0])")\n'
        "assert np.allclose(T_pris[:3, 3], [0, 0.5, 0])\n"
        "assert np.allclose(T_pris[:3, :3], np.eye(3))      # no rotation\n"
        "assert np.allclose(transform_from_twist(S_pris, 0.5),\n"
        "                   MatrixExp6(VecTose3(S_pris * 0.5)))"
    ),
    md(
        "### Looking ahead: `extract_screw_list`\n"
        "\n"
        "A serial robot is a *chain* of these screws — one per joint. "
        "`extract_screw_list(omega_list, r_list)` assembles the $6\\times n$ matrix of "
        "joint screw axes from each joint's axis direction ($\\omega$) and a point on it "
        "($r$). That matrix is the input to the **Product of Exponentials** forward "
        "kinematics in notebook 02; here we just build a tiny 2-joint example."
    ),
    code(
        "# Two revolute joints: both about z, at x = 0 and x = 0.3.\n"
        "omega_list = np.array([[0, 0], [0, 0], [1, 1]], float)   # 3 x n axis directions\n"
        "r_list = np.array([[0.0, 0.3], [0, 0], [0, 0]], float)   # 3 x n points on axes\n"
        "Slist = extract_screw_list(omega_list, r_list)\n"
        'print("screw list (6 x 2):\\n", Slist)\n'
        "assert Slist.shape == (6, 2)\n"
        "assert np.allclose(Slist[:3, 0], [0, 0, 1])              # first joint axis is z"
    ),

    # --- Section E: adjoint ---
    md(
        "## 5. The adjoint map\n"
        "\n"
        "A twist is expressed in some frame. To re-express the **same** physical twist in "
        "another frame related by $T$, we use the $6\\times6$ **adjoint** "
        "$[\\mathrm{Ad}_T]$:\n"
        "\n"
        "$$\\mathcal{V}_a=[\\mathrm{Ad}_{T_{ab}}]\\,\\mathcal{V}_b,\\qquad "
        "[\\mathrm{Ad}_T]=\\begin{bmatrix}R & 0\\\\ [p]R & R\\end{bmatrix}.$$\n"
        "\n"
        "This is how joint screw axes get moved between the space frame and the body "
        "frame (notebook 03). ManipulaPy provides `adjoint_transform`."
    ),
    code(
        "Ad = adjoint_transform(T)\n"
        'print("Ad_T shape:", Ad.shape)\n'
        "\n"
        "# Re-express the screw twist in the transformed frame, then map it back.\n"
        "V_b = S.copy()\n"
        "V_a = Ad @ V_b\n"
        "V_b_again = adjoint_transform(TransInv(T)) @ V_a\n"
        'print("adjoint round trip recovers the twist :", np.allclose(V_b_again, V_b))'
    ),

    # --- smoke test ---
    md("## Smoke test\n\nAsserts the key invariants of this notebook in one cell."),
    code(
        "# Rotations\n"
        "R = MatrixExp3(VecToso3([0, 0, np.pi / 2]))\n"
        "assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)\n"
        "assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)\n"
        "# SE(3)\n"
        "T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0.3, -0.1, 0.5]\n"
        "assert np.allclose(TransInv(T) @ T, np.eye(4))\n"
        "# Screw / exponential coordinates round trip\n"
        "S = np.array([0, 0, 1, 0, -0.3, 0], float)\n"
        "assert np.allclose(se3ToVec(MatrixLog6(MatrixExp6(VecTose3(S * 1.1)))), S * 1.1, atol=1e-9)\n"
        "# Adjoint\n"
        "assert adjoint_transform(T).shape == (6, 6)\n"
        "# rotation_logm returns a unit axis and the principal angle\n"
        "ax, ang = rotation_logm(R)\n"
        "assert np.isclose(np.linalg.norm(ax), 1.0) and 0 <= ang <= np.pi + 1e-9\n"
        "# transform_from_twist == MatrixExp6(VecTose3(.))\n"
        "assert np.allclose(transform_from_twist(S, 0.9), MatrixExp6(VecTose3(S * 0.9)))\n"
        "# extract_screw_list builds a 6xn matrix\n"
        "assert extract_screw_list(np.array([[0],[0],[1]],float), np.array([[0.3],[0],[0]],float)).shape == (6, 1)\n"
        "# Euler: degrees in, radians out\n"
        "assert np.allclose(rotation_matrix_to_euler_angles(euler_to_rotation_matrix([0, 0, 90]))[2], np.pi / 2)\n"
        'print("nb01 foundations: smoke OK")'
    ),

    # --- exercises ---
    md(
        "## Try it\n"
        "\n"
        "1. Build the rotation that takes $\\hat{x}_s$ to $\\hat{z}_s$ and verify it is in "
        "$SO(3)$. *(Hint: a $90^\\circ$ rotation about $\\hat{y}$.)*\n"
        "2. Construct the screw axis of a **prismatic** joint sliding along $\\hat{y}$ "
        "(pure translation: $\\omega=0$, $v=\\hat{y}$) and confirm `MatrixExp6` produces a "
        "pure-translation transform for a given displacement.\n"
        "3. Take two transforms $T_1,T_2$ and check that "
        "$[\\mathrm{Ad}_{T_1 T_2}] = [\\mathrm{Ad}_{T_1}][\\mathrm{Ad}_{T_2}]$.\n"
        "\n"
        "*Next up — notebook 02: chaining screw exponentials into the **Product of "
        "Exponentials** forward-kinematics formula on the Panda.*"
    ),
    md(
        "## References\n"
        "\n"
        "1. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and "
        "Control*, Cambridge University Press, 2017. — The primary reference for this "
        "course; ManipulaPy follows its screw-theory / Product-of-Exponentials "
        "conventions. (Chapters 3 *Rigid-Body Motions* and 4 *Forward Kinematics*.)\n"
        "2. R. M. Murray, Z. Li, and S. S. Sastry, *A Mathematical Introduction to "
        "Robotic Manipulation*, CRC Press, 1994. — Foundational treatment of twists, "
        "screws, and exponential coordinates.\n"
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
