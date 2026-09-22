#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Potential-Field Curvature (Hessian) Tensor Field Demo - ManipulaPy

Every other tensor field in this directory lives over the UR5's
*configuration* space (a Jacobian-derived tensor at each joint angle,
lifted into the workspace via forward kinematics). This one lives directly
over free Cartesian *workspace* -- it comes from ManipulaPy's motion
planner, not its kinematics.

ManipulaPy.potential_field.PotentialField defines a scalar artificial
potential U(x) = U_attractive(x) + U_repulsive(x) used for gradient-descent
path planning (attractive toward a goal, repulsive away from obstacles,
compactly supported within ``influence_distance``). It exposes the
gradient directly (``compute_gradient``) but not the Hessian, so this demo
finite-differences the gradient to build the curvature tensor field

    H(x) = d/dx [ grad U(x) ]     (symmetrized: H <- (H + H^T) / 2)

at each point of a grid spanning part of the UR5's reach. Unlike the
Jacobian-derived tensors elsewhere in this directory, H is not guaranteed
positive semi-definite -- it is a basin (positive-definite) near the goal,
but can be a saddle near an obstacle's influence boundary, where the
potential falls off steeply on one side and is flat on the other. This
demo renders each ellipsoid's *shape* from the eigenvalues' magnitudes
(|eigenvalue|, since a negative one has no real square root to use as a
semi-axis) and encodes the *sign* structure -- via trace(H), the field's
mean curvature -- as color on a diverging colormap: warm for a basin,
cool for a saddle/ridge.

Usage:
    python potential_field_hessian_tensor_field_demo.py

Expected Output:
    - Console summary of the mean-curvature range
    - "potential_field_hessian_tensor_field.png" with two panels: a 2-D
      mean-curvature heatmap (with the goal/obstacles marked) and the 3-D
      curvature-ellipsoid tensor field
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

# LaTeX-style (Computer Modern) text via matplotlib's bundled cmr10 font, so
# the figure renders the same everywhere without requiring a system LaTeX install.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

try:
    import ManipulaPy
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import ManipulaPy

from ManipulaPy.potential_field import PotentialField

OUTPUT_DIR = Path(__file__).resolve().parent


def hessian_via_finite_differences(pf, x, goal, obstacles, h=1e-3):
    """Symmetrized Hessian of the potential at ``x``, from finite-differencing its gradient."""
    H = np.zeros((3, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        g_plus = np.asarray(pf.compute_gradient(x + step, goal, obstacles), dtype=float)
        g_minus = np.asarray(pf.compute_gradient(x - step, goal, obstacles), dtype=float)
        H[:, k] = (g_plus - g_minus) / (2 * h)
    return 0.5 * (H + H.T)


def build_curvature_field(pf, goal, obstacles, x_range, y_range, z, resolution=30,
                           mask_radius=0.08):
    """Sample the potential's curvature tensor field over an (x, y) grid at fixed z.

    The repulsive term's 1/d^3 falloff means curvature genuinely diverges as
    d -> 0, not just at a few unlucky grid points -- any sample within
    ``mask_radius`` of an obstacle center is inside its singular core, where
    a curvature glyph is not physically meaningful (a planner would already
    treat that region as collision), so those points are masked out (NaN in
    ``mean_curvature``, ``valid=False``) rather than left to dominate the
    color scale.
    """
    x_vals = np.linspace(*x_range, resolution)
    y_vals = np.linspace(*y_range, resolution)

    points = np.zeros((resolution, resolution, 3))
    tensor_field = np.zeros((resolution, resolution, 3, 3))
    mean_curvature = np.zeros((resolution, resolution))
    valid = np.ones((resolution, resolution), dtype=bool)

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            point = np.array([x, y, z])
            points[i, j] = point
            if any(np.linalg.norm(point - obs) < mask_radius for obs in obstacles):
                valid[i, j] = False
                mean_curvature[i, j] = np.nan
                continue
            H = hessian_via_finite_differences(pf, point, goal, obstacles)
            tensor_field[i, j] = H
            mean_curvature[i, j] = np.trace(H) / 3.0

    return {
        "x_vals": x_vals, "y_vals": y_vals, "points": points,
        "tensor_field": tensor_field, "mean_curvature": mean_curvature, "valid": valid,
    }


def _ellipsoid_surface(center, tensor, scale, sphere_res=14):
    """Build the (X, Y, Z) mesh of the 3-D ellipsoid representing ``|tensor|``'s shape."""
    eigvals, eigvecs = np.linalg.eigh(tensor)
    radii = scale * np.sqrt(np.abs(eigvals))  # magnitude only; sign is carried separately

    u = np.linspace(0, 2 * np.pi, sphere_res)
    v = np.linspace(0, np.pi, sphere_res)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    points = eigvecs @ np.stack([x.ravel(), y.ravel(), z.ravel()])
    points += np.asarray(center).reshape(3, 1)
    shape = x.shape
    return points[0].reshape(shape), points[1].reshape(shape), points[2].reshape(shape)


def _grid_spacing(centers):
    d_rows = np.linalg.norm(np.diff(centers, axis=0), axis=-1).mean()
    d_cols = np.linalg.norm(np.diff(centers, axis=1), axis=-1).mean()
    return min(d_rows, d_cols)


def plot_curvature_field(field_data, goal, obstacles, stride=2, sphere_res=14, fill_frac=0.8):
    """Plot the mean-curvature heatmap and the 3-D curvature-ellipsoid tensor field."""
    x_vals, y_vals = field_data["x_vals"], field_data["y_vals"]
    mean_curvature = field_data["mean_curvature"]
    tensor_field = field_data["tensor_field"]
    points = field_data["points"]
    valid = field_data["valid"]

    # Even outside each obstacle's masked-out singular core, curvature still
    # spans a wide range close to the mask boundary; the color range (not the
    # underlying data, and not each ellipsoid's own size normalization below)
    # is robustly clipped to the bulk of the field so a few remaining
    # near-boundary points don't wash out the whole scale.
    vmax = np.nanpercentile(np.abs(mean_curvature), 95) + 1e-9
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    cmap.set_bad(color="0.85")  # masked (near-obstacle) cells

    fig = plt.figure(figsize=(14, 6.5))
    fig.suptitle(r"Potential-Field Curvature (Hessian) Tensor Field", fontsize=16, fontweight="bold")

    # --- 2-D mean-curvature heatmap ------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
    im = ax1.imshow(mean_curvature.T, origin="lower", extent=extent, aspect="equal", cmap=cmap, norm=norm)
    ax1.scatter(*goal[:2], marker="*", s=260, color="gold", edgecolor="black", linewidth=0.8, label="Goal", zorder=5)
    for k, obs in enumerate(obstacles):
        ax1.scatter(*obs[:2], marker="x", s=120, color="black", linewidth=2.2,
                    label="Obstacle" if k == 0 else None, zorder=5)
    ax1.set_xlabel(r"$X$ (m)")
    ax1.set_ylabel(r"$Y$ (m)")
    ax1.set_title(r"Mean curvature $\;\mathrm{tr}(H)/3$ (warm = basin, cool = saddle)", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(im, ax=ax1, label="Mean curvature", extend="both")

    # --- 3-D curvature-ellipsoid tensor field --------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sampled_points = points[::stride, ::stride]
    target_radius = 0.5 * fill_frac * _grid_spacing(sampled_points)

    resolution = points.shape[0]
    for i in range(0, resolution, stride):
        for j in range(0, resolution, stride):
            if not valid[i, j]:
                continue
            tensor = tensor_field[i, j]
            max_abs_eigval = max(np.abs(np.linalg.eigvalsh(tensor)).max(), 1e-9)
            scale = target_radius / np.sqrt(max_abs_eigval)
            X, Y, Z = _ellipsoid_surface(points[i, j], tensor, scale, sphere_res=sphere_res)
            color = cmap(norm(mean_curvature[i, j]))
            ax2.plot_surface(X, Y, Z, color=color, alpha=0.85, linewidth=0, shade=True)

    ax2.scatter(*goal, marker="*", s=260, color="gold", edgecolor="black", linewidth=0.8, zorder=5)
    for obs in obstacles:
        ax2.scatter(*obs, marker="x", s=120, color="black", linewidth=2.2, zorder=5)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax2, shrink=0.6, label="Mean curvature", extend="both")
    ax2.set_xlabel(r"$X$ (m)")
    ax2.set_ylabel(r"$Y$ (m)")
    ax2.set_zlabel(r"$Z$ (m)")
    ax2.set_title(r"Curvature ellipsoids $\left(H(x)\;@\;x\right)$", fontweight="bold")
    # This is a flat single-z slice, so its true z-extent is just each
    # ellipsoid's own radius -- not a fraction of the (much larger) x/y span.
    # A box_aspect that doesn't match the real x:y:z data ratio silently
    # rescales each axis to fill its allotted box, which stretched every
    # near-spherical ellipsoid here into a tall needle before this fix.
    z_center = float(np.mean(points[..., 2]))
    z_half_extent = 1.3 * target_radius
    ax2.set_zlim(z_center - z_half_extent, z_center + z_half_extent)
    ax2.set_box_aspect([
        x_vals[-1] - x_vals[0], y_vals[-1] - y_vals[0], 2 * z_half_extent,
    ])

    plt.tight_layout()
    save_path = OUTPUT_DIR / "potential_field_hessian_tensor_field.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("   ManipulaPy: Potential-Field Curvature Tensor Field Demo")
    print("=" * 70)

    # A goal and two obstacles inside a UR5-scale (~0.85 m reach) workspace slice.
    goal = np.array([0.9, 0.0, 0.3])
    obstacles = [np.array([0.45, 0.18, 0.3]), np.array([0.55, -0.22, 0.3])]

    pf = PotentialField(attractive_gain=1.5, repulsive_gain=40.0, influence_distance=0.35)

    print(f"\nGoal:      {goal}")
    print(f"Obstacles: {obstacles}")
    print("\nBuilding the potential's curvature (Hessian) tensor field...")
    field_data = build_curvature_field(
        pf, goal, obstacles, x_range=(-0.1, 1.1), y_range=(-0.6, 0.6), z=0.3, resolution=30
    )

    mc = field_data["mean_curvature"]
    n_masked = int((~field_data["valid"]).sum())
    print(f"  Masked (near-obstacle core) grid points: {n_masked}/{mc.size}")
    print(f"  Mean curvature range (unmasked, raw):     "
          f"[{np.nanmin(mc):.3f}, {np.nanmax(mc):.3f}]")
    print(f"  Mean curvature range (5th-95th pct): "
          f"[{np.nanpercentile(mc, 5):.3f}, {np.nanpercentile(mc, 95):.3f}]  "
          f"(figure's color scale)")

    plot_curvature_field(field_data, goal, obstacles)

    print("\nDone.")


if __name__ == "__main__":
    main()
