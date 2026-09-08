#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JEPA training data from ManipulaPy dynamics - ManipulaPy

A world model needs trajectories, and ManipulaPy can produce them without a
simulator at all: :meth:`ManipulatorDynamics.forward_dynamics` gives the exact
joint accelerations for a torque, so integrating it yields ground-truth
rollouts of the real articulated-body dynamics - mass matrix, Coriolis terms,
gravity and all.

Two entry points:

- :func:`generate_rollouts` - excites the arm with smooth random torques and
  integrates the resulting motion, the standard way to collect the broad,
  dynamically consistent coverage a world model needs.
- :func:`rollouts_from_trajectories` - wraps trajectories you already have
  (from :class:`OptimizedTrajectoryPlanning`, a PyBullet run, or a real robot's
  logs) in the same container.

:class:`TrajectoryWindowDataset` then cuts either into the
``(context, actions, targets)`` triples the model trains on.

Copyright (c) 2025 Mohamed Aboelnasr

This file is part of ManipulaPy.

ManipulaPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ManipulaPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ManipulaPy. If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from .features import RobotStateFeaturizer

__all__ = [
    "Rollout",
    "generate_rollouts",
    "rollouts_from_trajectories",
    "TrajectoryWindowDataset",
]

logger = logging.getLogger(__name__)

_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)


@dataclass
class Rollout:
    """One recorded trajectory.

    Attributes:
        states: ``(T, state_dim)`` featurized states.
        actions: ``(T, action_dim)`` action applied *at* each state, so
            ``actions[t]`` is what drove ``states[t]`` to ``states[t + 1]``.
            The final row is recorded but never used as a training input.
        thetas: ``(T, num_joints)`` raw joint positions.
        dthetas: ``(T, num_joints)`` raw joint velocities.
        dt: Control period, in seconds.
        info: Free-form provenance.
    """

    states: NDArray[np.float64]
    actions: NDArray[np.float64]
    thetas: NDArray[np.float64]
    dthetas: NDArray[np.float64]
    dt: float
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __post_init__(self) -> None:
        if self.states.shape[0] != self.actions.shape[0]:
            raise ValueError(
                f"states has {self.states.shape[0]} steps but actions has "
                f"{self.actions.shape[0]}; they must be aligned"
            )


def _limits_to_arrays(
    joint_limits: Optional[Sequence[Tuple[Optional[float], Optional[float]]]],
    num_joints: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split joint limits into finite lower/upper arrays.

    Unbounded entries (``None``) become +/-inf, so clipping against the result
    is always safe.
    """
    if joint_limits is None:
        return (
            np.full(num_joints, -np.inf),
            np.full(num_joints, np.inf),
        )
    lower = np.empty(num_joints, dtype=np.float64)
    upper = np.empty(num_joints, dtype=np.float64)
    for i in range(num_joints):
        lo, hi = joint_limits[i]
        lower[i] = -np.inf if lo is None else float(lo)
        upper[i] = np.inf if hi is None else float(hi)
    if np.any(lower > upper):
        raise ValueError("joint_limits contains a lower bound above its upper")
    return lower, upper


def _sample_torque_sequence(
    steps: int,
    num_joints: int,
    scale: NDArray[np.float64],
    smoothness: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Smooth random torques as a first-order autoregressive process.

    White-noise torques excite almost nothing: the arm's inertia integrates
    them away and the resulting states barely move off the initial pose. An
    AR(1) process with ``smoothness`` close to one produces the correlated,
    slowly varying commands that actually swing the arm through its workspace,
    while the ``sqrt(1 - rho^2)`` innovation keeps the marginal variance at
    ``scale`` regardless of how much correlation is asked for.
    """
    rho = float(np.clip(smoothness, 0.0, 0.999))
    innovation = np.sqrt(max(1.0 - rho * rho, 1e-12))
    tau = np.empty((steps, num_joints), dtype=np.float64)
    current = rng.normal(scale=scale)
    for t in range(steps):
        tau[t] = current
        current = rho * current + innovation * rng.normal(scale=scale)
    return tau


def generate_rollouts(
    dynamics: Any,
    featurizer: Optional[RobotStateFeaturizer] = None,
    *,
    num_rollouts: int = 16,
    steps: int = 128,
    dt: float = 0.02,
    substeps: int = 4,
    torque_scale: float = 5.0,
    smoothness: float = 0.9,
    joint_limits: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    velocity_limit: Optional[float] = 10.0,
    gravity: Sequence[float] = _DEFAULT_GRAVITY,
    initial_states: Optional[Sequence[NDArray[np.float64]]] = None,
    seed: Optional[int] = None,
) -> List[Rollout]:
    """Roll the true dynamics forward under smooth random torques.

    Integration is semi-implicit (symplectic) Euler with ``substeps`` inner
    steps per control period: velocity is updated first and the new velocity
    advances the position. For the stiff second-order systems manipulator
    dynamics produce, that is markedly more stable than explicit Euler at the
    same step size, which matters because a diverging rollout is worse than a
    coarse one - it poisons the dataset with states the robot can never reach.

    Cost scales as ``num_rollouts * steps * substeps`` evaluations of
    ``forward_dynamics``, each of which builds a mass matrix and its
    derivatives - tens of milliseconds for a 6-DOF arm. Generate a dataset
    once and keep it, rather than regenerating per training run; or feed
    trajectories you already have through :func:`rollouts_from_trajectories`.

    Args:
        dynamics: A :class:`~ManipulaPy.dynamics.ManipulatorDynamics`, or
            anything exposing ``forward_dynamics(theta, dtheta, tau, g, Ftip)``.
        featurizer: Featurizer for the recorded states. Built from ``dynamics``
            with end-effector features enabled when omitted.
        num_rollouts: Number of trajectories.
        steps: Control steps per trajectory.
        dt: Control period, in seconds.
        substeps: Integration substeps per control step.
        torque_scale: Standard deviation of the excitation torque, in N*m.
            May be a scalar or a per-joint sequence.
        smoothness: AR(1) coefficient of the torque process, in ``[0, 1)``.
            Higher is smoother and swings the arm further.
        joint_limits: Position limits used to clamp the state. Taken from
            ``dynamics.joint_limits`` when omitted.
        velocity_limit: Magnitude cap on joint velocity, or None for no cap.
        gravity: Gravity vector passed to ``forward_dynamics``.
        initial_states: Optional sequence of ``(theta, dtheta)`` pairs, or of
            ``theta`` arrays. Sampled uniformly inside the joint limits when
            omitted.
        seed: Seed for the torque and initial-state RNG.

    Returns:
        A list of :class:`Rollout`. Trajectories that diverge to non-finite
        values are truncated at the last finite step and kept if long enough,
        or dropped otherwise; a warning names how many were affected.

    Raises:
        ValueError: If every rollout diverged, which means the excitation is
            too aggressive or ``dt`` too large for this robot.
    """
    if featurizer is None:
        featurizer = RobotStateFeaturizer(dynamics)
    num_joints = featurizer.num_joints

    if joint_limits is None:
        joint_limits = getattr(dynamics, "joint_limits", None)
    lower, upper = _limits_to_arrays(joint_limits, num_joints)
    # Sampling needs a finite box even when the robot has continuous joints.
    sample_lower = np.where(np.isfinite(lower), lower, -np.pi)
    sample_upper = np.where(np.isfinite(upper), upper, np.pi)

    scale = np.broadcast_to(
        np.asarray(torque_scale, dtype=np.float64), (num_joints,)
    ).astype(np.float64, copy=True)
    g = np.asarray(gravity, dtype=np.float64)
    ftip = np.zeros(6, dtype=np.float64)
    rng = np.random.default_rng(seed)
    h = float(dt) / max(int(substeps), 1)

    rollouts: List[Rollout] = []
    diverged = 0
    for r in range(int(num_rollouts)):
        theta, dtheta = _initial_state(
            r, initial_states, sample_lower, sample_upper, num_joints, rng
        )
        taus = _sample_torque_sequence(int(steps), num_joints, scale, smoothness, rng)

        thetas = np.empty((steps, num_joints), dtype=np.float64)
        dthetas = np.empty((steps, num_joints), dtype=np.float64)
        valid = int(steps)
        for t in range(int(steps)):
            thetas[t] = theta
            dthetas[t] = dtheta
            tau = taus[t]
            for _ in range(max(int(substeps), 1)):
                ddtheta = np.asarray(
                    dynamics.forward_dynamics(theta, dtheta, tau, g, ftip),
                    dtype=np.float64,
                ).reshape(-1)
                if not np.all(np.isfinite(ddtheta)):
                    valid = t
                    break
                dtheta = dtheta + ddtheta * h
                if velocity_limit is not None:
                    dtheta = np.clip(dtheta, -velocity_limit, velocity_limit)
                theta = theta + dtheta * h
                # A joint that hits a stop stays there with zero velocity
                # rather than tunnelling through it; without this the dataset
                # would contain configurations the hardware cannot reach.
                clamped = np.clip(theta, lower, upper)
                dtheta = np.where(clamped != theta, 0.0, dtheta)
                theta = clamped
            if valid != steps:
                break

        if valid < 2:
            diverged += 1
            continue
        if valid < steps:
            diverged += 1

        states = featurizer.featurize_batch(thetas[:valid], dthetas[:valid])
        rollouts.append(
            Rollout(
                states=states,
                actions=taus[:valid].copy(),
                thetas=thetas[:valid].copy(),
                dthetas=dthetas[:valid].copy(),
                dt=float(dt),
                info={"rollout_index": r, "truncated": valid < steps},
            )
        )

    if not rollouts:
        raise ValueError(
            "every rollout diverged; reduce torque_scale, reduce dt, or raise "
            "substeps"
        )
    if diverged:
        logger.warning(
            "%d of %d rollouts hit non-finite dynamics and were truncated or "
            "dropped",
            diverged,
            num_rollouts,
        )
    return rollouts


def _initial_state(
    index: int,
    initial_states: Optional[Sequence[Any]],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    num_joints: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resolve the starting ``(theta, dtheta)`` for rollout ``index``."""
    if initial_states is None:
        theta = rng.uniform(lower, upper)
        return theta.astype(np.float64), np.zeros(num_joints, dtype=np.float64)

    entry = initial_states[index % len(initial_states)]
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        theta = np.asarray(entry[0], dtype=np.float64).reshape(-1)
        dtheta = np.asarray(entry[1], dtype=np.float64).reshape(-1)
    else:
        theta = np.asarray(entry, dtype=np.float64).reshape(-1)
        dtheta = np.zeros(num_joints, dtype=np.float64)
    if theta.shape[0] != num_joints or dtheta.shape[0] != num_joints:
        raise ValueError(f"initial_states[{index}] does not have {num_joints} joints")
    return theta.copy(), dtheta.copy()


def rollouts_from_trajectories(
    trajectories: Sequence[Dict[str, NDArray[np.float64]]],
    featurizer: RobotStateFeaturizer,
    *,
    dt: float,
    position_key: str = "positions",
    velocity_key: str = "velocities",
    action_key: str = "torques",
) -> List[Rollout]:
    """Wrap existing trajectories as :class:`Rollout` objects.

    Accepts the dictionaries produced by
    :meth:`OptimizedTrajectoryPlanning.joint_trajectory` and
    :meth:`~OptimizedTrajectoryPlanning.inverse_dynamics_trajectory`, or any
    equivalent logs from simulation or hardware.

    Args:
        trajectories: Dicts holding at least positions and velocities. When the
            action key is absent, joint accelerations are used as the action if
            present, and zeros otherwise - so a purely kinematic trajectory
            still trains a (weakly action-conditioned) model.
        featurizer: Featurizer applied to each state.
        dt: Control period of the trajectories, in seconds.
        position_key: Key holding ``(T, num_joints)`` joint positions.
        velocity_key: Key holding ``(T, num_joints)`` joint velocities.
        action_key: Key holding ``(T, action_dim)`` actions.

    Returns:
        A list of :class:`Rollout`.
    """
    out: List[Rollout] = []
    for i, traj in enumerate(trajectories):
        if position_key not in traj:
            raise KeyError(
                f"trajectory {i} has no {position_key!r} key; got " f"{sorted(traj)}"
            )
        thetas = np.asarray(traj[position_key], dtype=np.float64)
        if thetas.ndim != 2:
            raise ValueError(
                f"trajectory {i} positions must be (T, num_joints), got "
                f"{thetas.shape}"
            )
        dthetas = (
            np.asarray(traj[velocity_key], dtype=np.float64)
            if velocity_key in traj
            else np.zeros_like(thetas)
        )
        if action_key in traj:
            actions = np.asarray(traj[action_key], dtype=np.float64)
        elif "accelerations" in traj:
            actions = np.asarray(traj["accelerations"], dtype=np.float64)
        else:
            actions = np.zeros_like(thetas)
        if actions.shape[0] != thetas.shape[0]:
            raise ValueError(
                f"trajectory {i}: actions have {actions.shape[0]} steps but "
                f"positions have {thetas.shape[0]}"
            )

        out.append(
            Rollout(
                states=featurizer.featurize_batch(thetas, dthetas),
                actions=actions,
                thetas=thetas,
                dthetas=dthetas,
                dt=float(dt),
                info={"source": "trajectory", "index": i},
            )
        )
    return out


class TrajectoryWindowDataset(Dataset):
    """Cut rollouts into JEPA training windows.

    Sample ``i`` maps to a rollout and an anchor step ``t``, and yields:

    - ``context``: ``states[t - C + 1 : t + 1]``, the window the context
      encoder sees;
    - ``actions``: ``actions[t : t + K]``, the commands applied from ``t``;
    - ``targets``: for each ``k``, the window ending at ``t + 1 + k`` - the
      state the robot actually reached, in the same window shape the context
      encoder consumes.

    Args:
        rollouts: Rollouts to draw from.
        context_len: Steps per window (``C``).
        horizon: Prediction horizon (``K``).
        dtype: Torch dtype of the produced tensors.

    Raises:
        ValueError: If no rollout is long enough for ``context_len + horizon``.
    """

    def __init__(
        self,
        rollouts: Sequence[Rollout],
        context_len: int,
        horizon: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_len < 1 or horizon < 1:
            raise ValueError("context_len and horizon must be >= 1")
        self.rollouts = list(rollouts)
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.dtype = dtype

        self._index: List[Tuple[int, int]] = []
        for r, rollout in enumerate(self.rollouts):
            length = len(rollout)
            first = self.context_len - 1
            last = length - 1 - self.horizon
            for t in range(first, last + 1):
                self._index.append((r, t))

        if not self._index:
            longest = max((len(r) for r in self.rollouts), default=0)
            raise ValueError(
                f"no rollout is long enough: need at least "
                f"{self.context_len + self.horizon} steps, longest has "
                f"{longest}"
            )

    @property
    def state_dim(self) -> int:
        """Width of one featurized state."""
        return int(self.rollouts[0].states.shape[1])

    @property
    def action_dim(self) -> int:
        """Width of one action."""
        return int(self.rollouts[0].actions.shape[1])

    def all_states(self) -> NDArray[np.float64]:
        """Every state in every rollout, stacked - for normalizer fitting."""
        return np.concatenate([r.states for r in self.rollouts], axis=0)

    def all_actions(self) -> NDArray[np.float64]:
        """Every action in every rollout, stacked."""
        return np.concatenate([r.actions for r in self.rollouts], axis=0)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r, t = self._index[idx]
        rollout = self.rollouts[r]
        states, actions = rollout.states, rollout.actions
        c, k = self.context_len, self.horizon

        context = states[t - c + 1 : t + 1]
        action_seq = actions[t : t + k]
        targets = np.stack(
            [states[t + 2 + j - c : t + 2 + j] for j in range(k)], axis=0
        )
        return {
            "context": torch.as_tensor(context, dtype=self.dtype),
            "actions": torch.as_tensor(action_seq, dtype=self.dtype),
            "targets": torch.as_tensor(targets, dtype=self.dtype),
        }
