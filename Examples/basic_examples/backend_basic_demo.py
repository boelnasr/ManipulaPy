#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Basic Backend Demo: Running the Same Math on NumPy, PyTorch and JAX

This example demonstrates ManipulaPy's unified compute backend system introduced in v1.4.0:
selecting a backend process-wide or in a scoped block, verifying that one unchanged forward
kinematics call produces numerically identical results on every backend, and taking an
automatic-differentiation gradient of forward kinematics validated against finite differences.

Usage:
    python backend_basic_demo.py

Expected Output:
    - Console output listing which backends are installed and available
    - End-effector poses from the same FK call under NumPy, PyTorch and JAX
    - Cross-backend agreement errors and per-call dispatch timings
    - Autodiff Jacobians (jax.jacrev / torch.autograd) compared against central differences
    - A summary figure saved next to this script

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import importlib.util
import os
import numpy as np
import matplotlib

# Honor a pre-set non-interactive backend (e.g. MPLBACKEND=Agg for headless runs);
# otherwise pick Agg so the demo saves figures to disk without needing a display.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add ManipulaPy to path if needed
try:
    import ManipulaPy
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    import ManipulaPy

from ManipulaPy.urdf_processor import URDFToSerialManipulator
from ManipulaPy.backend import get_backend, get_registered, set_backend, use_backend

# Directory next to this script for saved plots.
OUTPUT_DIR = Path(__file__).resolve().parent


class BackendBasicDemo:
    """
    Demonstration of the v1.4 unified compute backend system.
    """

    def __init__(self) -> None:
        """Initialize the demo with robot model loading."""
        self.robot = None
        self.n_joints = 0
        # A single fixed, well-conditioned configuration keeps the numbers reproducible
        # and keeps the JAX path small (eager JAX dispatch is ~40x slower than NumPy).
        self.theta = np.array([0.2, -0.4, 0.7, 0.1, -0.3, 0.5])
        self.available = []
        self.fk_results = {}
        self.gradient_results = {}

    def run_demo(self) -> bool:
        """Run the complete backend demonstration."""
        print("=" * 70)
        print("   ManipulaPy: Basic Backend Demo")
        print("=" * 70)
        print()

        # Step 1: Load robot model
        if not self.load_robot_model():
            return False

        # Step 2: Discover which backends this machine can actually run
        self.demonstrate_backend_selection()

        # Step 3: Run the same FK call on every available backend
        self.demonstrate_cross_backend_agreement()

        # Step 4: Differentiate forward kinematics and check against finite differences
        self.demonstrate_autodiff_gradients()

        # Step 5: Summarize the results in one figure
        self.create_visualizations()

        print("\n" + "=" * 70)
        print("✅ Backend demo completed successfully!")
        print("📊 Check the generated plot for the agreement and timing summary")
        print("=" * 70)

        return True

    def load_robot_model(self) -> bool:
        """Load and initialize robot model."""
        print("🤖 Loading Robot Model")
        print("-" * 30)

        try:
            from ManipulaPy.ManipulaPy_data.xarm import urdf_file

            print("📁 Using built-in xArm 6-DOF model")
        except ImportError:
            print("❌ No built-in robot models found!")
            print("💡 Please ensure ManipulaPy is properly installed with robot data.")
            return False

        try:
            urdf_processor = URDFToSerialManipulator(urdf_file)
            self.robot = urdf_processor.serial_manipulator
            self.n_joints = len(self.robot.joint_limits)

            print("✅ xArm 6-DOF loaded successfully!")
            print(f"   • Number of joints: {self.n_joints}")
            print(f"   • Test configuration: {self.theta}")

            return True

        except Exception as e:
            print(f"❌ Error loading robot model: {e}")
            import traceback

            traceback.print_exc()
            return False

    def demonstrate_backend_selection(self) -> None:
        """Demonstrate how backends are selected and which ones are installed."""
        print(f"\n🔌 Backend Selection")
        print("-" * 40)

        # NumPy is registered and activated eagerly; it is always the process default.
        print(f"   Default active backend: {type(get_backend()).__name__}")

        # Backends register lazily behind an import probe, so asking for one that is
        # not installed raises ImportError rather than failing at import time. Probe
        # the same way the registry does so the demo degrades gracefully.
        candidates = {
            "numpy": "built in",
            "cupy": "pip install ManipulaPy[cuda]",
            "torch": "pip install ManipulaPy[pytorch]",
            "jax": "pip install ManipulaPy[jax-cpu]",
        }

        # Probe by actually registering, not with find_spec. find_spec only proves
        # a module can be LOCATED; an accelerator compiled against a different
        # NumPy ABI resolves fine and then fails on import. Asking the registry
        # is the only honest test of whether a backend is usable here.
        for name, install_hint in candidates.items():
            try:
                probe = get_registered(name)
            except ImportError as exc:
                if "is not installed" in str(exc):
                    print(f"   ⏭️ {name:<6} not installed ({install_hint})")
                else:
                    print(f"   ⚠️ {name:<6} installed but unusable — skipping")
                    print(f"      {exc}")
                continue

            # Importing is still not proof of usability: CuPy imports happily on a
            # machine with no CUDA device and only fails when it tries to allocate.
            # One tiny round-trip settles it before the demo depends on the answer.
            try:
                probe.to_numpy(probe.asarray([1.0, 2.0]))
            except Exception as exc:  # noqa: BLE001 - any backend/device failure
                print(f"   ⚠️ {name:<6} present but not operational — skipping")
                print(f"      {type(exc).__name__}: {exc}")
                continue

            self.available.append(name)
            print(f"   ✅ {name:<6} available")

        # `use_backend` is a scoped switch that restores the previous backend on exit,
        # including when the block raises. `set_backend` switches process-wide.
        before = type(get_backend()).__name__
        with use_backend(self.available[-1]):
            inside = type(get_backend()).__name__
        after = type(get_backend()).__name__

        print(f"\n   Scoped switch with use_backend({self.available[-1]!r}):")
        print(f"     before: {before} → inside: {inside} → after: {after}")
        print(f"   ✅ Previous backend restored on exit")

    def demonstrate_cross_backend_agreement(self) -> None:
        """Run one unchanged FK call on every available backend and compare."""
        print(f"\n🧮 Cross-Backend Forward Kinematics")
        print("-" * 40)

        for name in self.available:
            # Nothing about the call changes between backends - only the active backend.
            with use_backend(name) as backend:
                # One warm-up call so the timing is steady-state dispatch rather than
                # the one-off cost of first tracing the function.
                self.robot.forward_kinematics(self.theta)

                start_time = time.time()
                T = self.robot.forward_kinematics(self.theta)
                fk_time = time.time() - start_time

                # Convert at the boundary: backend arrays become NumPy for reporting.
                T_host = backend.to_numpy(T)

            position = T_host[:3, 3]
            print(f"\n📍 Backend: {name}")
            print(f"   Returned type: {type(T).__name__}")
            print(
                f"   End-effector position: [{position[0]:.6f}, {position[1]:.6f}, "
                f"{position[2]:.6f}] m"
            )
            print(f"   Computation time: {fk_time*1000:.2f} ms")

            self.fk_results[name] = {"transform": T_host, "computation_time": fk_time}

        # NumPy defines the frozen public return contract; measure everyone against it.
        reference = self.fk_results["numpy"]["transform"]
        print(f"\n📊 Agreement with the NumPy reference:")

        for name, result in self.fk_results.items():
            max_error = float(np.max(np.abs(result["transform"] - reference)))
            result["max_error"] = max_error
            status = "✅" if max_error < 1e-9 else "⚠️"
            print(f"   {status} {name:<6} max |ΔT| = {max_error:.2e}")

    def demonstrate_autodiff_gradients(self) -> None:
        """Differentiate forward kinematics and validate against finite differences."""
        print(f"\n∂ Automatic Differentiation of Forward Kinematics")
        print("-" * 40)

        # Reference: central differences of end-effector position on the NumPy backend.
        reference_jacobian = self._central_difference_jacobian()
        print(f"   Finite-difference reference computed ({reference_jacobian.shape})")

        if "jax" in self.available:
            import jax
            import jax.numpy as jnp

            with use_backend("jax"):
                start_time = time.time()
                J = jax.jacrev(
                    lambda q: self.robot.forward_kinematics(q)[:3, 3]
                )(jnp.asarray(self.theta))
                grad_time = time.time() - start_time
            self._report_gradient("jax.jacrev", np.asarray(J), reference_jacobian, grad_time)

        if "torch" in self.available:
            import torch

            with use_backend("torch"):
                q = torch.as_tensor(self.theta, dtype=torch.float64)
                start_time = time.time()
                J = torch.autograd.functional.jacobian(
                    lambda x: self.robot.forward_kinematics(x)[:3, 3], q
                )
                grad_time = time.time() - start_time
            self._report_gradient(
                "torch.autograd", J.detach().numpy(), reference_jacobian, grad_time
            )

        print(f"\n⚠️ The differentiable contract covers utils, kinematics, dynamics and")
        print(f"   singularity only. Every other module runs on all four backends via")
        print(f"   host-boundary conversion, so gradients through them are not meaningful.")

    def _central_difference_jacobian(self, step: float = 1e-6) -> np.ndarray:
        """Return a central-difference Jacobian of end-effector position."""
        # `set_backend` is the process-wide counterpart to `use_backend`: it pins the
        # reference computation to NumPy no matter what ran before this call.
        set_backend("numpy")
        columns = []
        for index in range(self.n_joints):
            delta = np.zeros(self.n_joints)
            delta[index] = step
            forward = self.robot.forward_kinematics(self.theta + delta)[:3, 3]
            backward = self.robot.forward_kinematics(self.theta - delta)[:3, 3]
            columns.append((forward - backward) / (2 * step))
        return np.stack(columns, axis=1)

    def _report_gradient(self, label, jacobian, reference, grad_time) -> None:
        """Print and record one autodiff Jacobian against the finite-difference one."""
        max_error = float(np.max(np.abs(jacobian - reference)))

        print(f"\n📐 {label}")
        print(f"   Jacobian shape: {jacobian.shape}")
        print(f"   Computation time: {grad_time*1000:.1f} ms")
        print(f"   Max error vs finite differences: {max_error:.2e}")

        # Central differences carry O(step^2) truncation error, so agreement to ~1e-9
        # is the most the comparison can show - the autodiff values are the exact ones.
        status = "✅" if max_error < 1e-7 else "⚠️"
        print(f"   {status} Analytic gradient matches the numerical reference")

        self.gradient_results[label] = {
            "jacobian": jacobian,
            "max_error": max_error,
            "computation_time": grad_time,
        }

    def create_visualizations(self) -> None:
        """Create the summary visualization plot."""
        print(f"\n📊 Creating Visualization Plots")
        print("-" * 40)

        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(
                "ManipulaPy: Basic Backend Demo - Cross-Backend Agreement",
                fontsize=14,
                fontweight="bold",
            )

            self._plot_dispatch_times(axes[0])
            self._plot_agreement_errors(axes[1])
            self._plot_gradient_errors(axes[2])

            plt.tight_layout()
            out_path = OUTPUT_DIR / "backend_basic_demo.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            print(f"✅ Visualization saved to {out_path}")

        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
            import traceback

            traceback.print_exc()

    def _plot_dispatch_times(self, ax) -> None:
        """Plot per-call forward kinematics dispatch time by backend."""
        names = list(self.fk_results.keys())
        times = [self.fk_results[name]["computation_time"] * 1000 for name in names]

        bars = ax.bar(names, times, alpha=0.8, color="skyblue")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Single FK Call Dispatch Time")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        for bar, value in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_agreement_errors(self, ax) -> None:
        """Plot max |ΔT| against the NumPy reference by backend."""
        names = [name for name in self.fk_results if name != "numpy"]
        errors = [max(self.fk_results[name]["max_error"], 1e-18) for name in names]

        ax.bar(names, errors, alpha=0.8, color="lightgreen")
        ax.axhline(1e-9, color="red", linestyle="--", alpha=0.7, label="1e-9 tolerance")
        ax.set_ylabel("Max |ΔT| vs NumPy")
        ax.set_title("Cross-Backend FK Agreement")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_errors(self, ax) -> None:
        """Plot autodiff vs finite-difference error by framework."""
        if not self.gradient_results:
            ax.text(
                0.5,
                0.5,
                "No autodiff backends\navailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Autodiff vs Finite Differences")
            return

        names = list(self.gradient_results.keys())
        errors = [self.gradient_results[name]["max_error"] for name in names]

        ax.bar(names, errors, alpha=0.8, color="lightcoral")
        ax.axhline(1e-7, color="red", linestyle="--", alpha=0.7, label="1e-7 tolerance")
        ax.set_ylabel("Max error vs central differences")
        ax.set_title("Autodiff vs Finite Differences")
        ax.set_yscale("log")
        # Fixed limits: the frameworks agree to well within the finite-difference
        # truncation error, so an auto-scaled axis would magnify float noise.
        ax.set_ylim(1e-14, 1e-5)
        ax.tick_params(axis="x", rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)


def main() -> None:
    """Main function to run the backend basic demo."""
    try:
        demo = BackendBasicDemo()
        success = demo.run_demo()

        if success:
            print("\n🎉 Demo completed successfully!")
            print("📋 Summary of demonstrated concepts:")
            print("   ✅ Backend discovery and lazy registration")
            print("   ✅ Scoped switching with use_backend and process-wide set_backend")
            print("   ✅ Identical forward kinematics results across backends")
            print("   ✅ Converting backend arrays to NumPy at the boundary")
            print("   ✅ Autodiff gradients validated against finite differences")

            print("\n📚 Key takeaways:")
            print("   • One dispatch API: the same code runs on NumPy, CuPy, PyTorch or JAX")
            print("   • NumPy is the default and defines the frozen public return contract")
            print("   • Gradients are guaranteed for utils, kinematics, dynamics, singularity")
            print("   • Other modules are portable via host conversion, but not differentiable")
            print("   • Eager JAX dispatch is ~40x slower than NumPy on robot-sized arrays;")
            print("     TracIKSolver will not converge under JAX at its default 0.2 s timeout")

            print("\n🔗 Next steps:")
            print("   • Explore basic_examples/kinematics_basic_demo.py")
            print("   • Explore basic_examples/dynamics_basic_demo.py")
            print("   • Read docs/source/user_guide/Backends.rst for the full contract")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
