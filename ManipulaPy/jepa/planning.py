#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Latent-space planning with a trained JEPA - ManipulaPy

A world model earns its keep at planning time. :class:`LatentMPCPlanner` runs
the cross-entropy method over action sequences, scoring each candidate by
rolling the JEPA predictor forward and measuring how close the resulting
embeddings come to a goal embedding.

Every candidate is evaluated in one batched forward pass through a small MLP,
so a population of a few hundred sequences costs a fraction of what the same
search would cost against the true articulated-body dynamics - and unlike a
dynamics rollout, it stays differentiable and never diverges. The trade is
fidelity: the plan is only as good as the model, so
:meth:`LatentMPCPlanner.plan` is meant to be re-run each control step
(receding horizon) with the freshly measured state.

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

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

from .model import ManipulatorJEPA

__all__ = ["LatentMPCPlanner"]

ArrayLike = Union[np.ndarray, torch.Tensor]


class LatentMPCPlanner:
    """Cross-entropy-method planner over a JEPA's latent dynamics.

    Args:
        model: A trained :class:`~ManipulaPy.jepa.model.ManipulatorJEPA`.
        horizon: Planning horizon in control steps. May exceed the model's
            trained horizon, at the usual cost in accuracy.
        num_samples: Candidate action sequences per CEM iteration.
        num_elites: Best candidates kept to refit the sampling distribution.
        num_iterations: CEM refinement iterations.
        action_low: Lower bound per action dimension, scalar or per-dimension.
        action_high: Upper bound per action dimension.
        action_cost: Weight on mean squared action magnitude, which keeps the
            planner from buying a marginally better embedding with a torque
            spike the robot cannot deliver.
        smoothness_cost: Weight on the squared difference between consecutive
            actions, penalizing chattering plans.
        discount: Per-step weight ``discount ** k`` on the goal distance.
            Values below one favour reaching the goal sooner.
        terminal_weight: Extra weight on the final step's goal distance.
        elite_momentum: Blend factor between the previous and refitted
            sampling distribution, in ``[0, 1)``. Higher is more conservative.
        device: Torch device. Defaults to the model's device.
        seed: Seed for the sampling RNG.

    Example:
        >>> planner = LatentMPCPlanner(model, horizon=8)      # doctest: +SKIP
        >>> plan = planner.plan(context_window, goal_window)  # doctest: +SKIP
        >>> tau = plan["actions"][0]                          # doctest: +SKIP
    """

    def __init__(
        self,
        model: ManipulatorJEPA,
        *,
        horizon: int = 8,
        num_samples: int = 256,
        num_elites: int = 32,
        num_iterations: int = 4,
        action_low: Union[float, Sequence[float]] = -10.0,
        action_high: Union[float, Sequence[float]] = 10.0,
        action_cost: float = 1e-3,
        smoothness_cost: float = 0.0,
        discount: float = 1.0,
        terminal_weight: float = 1.0,
        elite_momentum: float = 0.1,
        device: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        if num_elites > num_samples:
            raise ValueError(
                f"num_elites ({num_elites}) cannot exceed num_samples "
                f"({num_samples})"
            )
        if num_elites < 1 or num_samples < 1 or num_iterations < 1:
            raise ValueError(
                "num_samples, num_elites and num_iterations must all be >= 1"
            )
        if not 0.0 <= elite_momentum < 1.0:
            raise ValueError(f"elite_momentum must be in [0, 1), got {elite_momentum}")

        self.model = model.eval()
        self.device = torch.device(
            device if device is not None else model.state_mean.device
        )
        self.model.to(self.device)

        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.num_iterations = int(num_iterations)
        self.action_cost = float(action_cost)
        self.smoothness_cost = float(smoothness_cost)
        self.discount = float(discount)
        self.terminal_weight = float(terminal_weight)
        self.elite_momentum = float(elite_momentum)

        action_dim = model.config.action_dim
        self.action_low = self._bounds(action_low, action_dim)
        self.action_high = self._bounds(action_high, action_dim)
        if torch.any(self.action_low >= self.action_high):
            raise ValueError("action_low must be strictly below action_high")

        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(seed))

        # Warm start: a plan from the previous control step, shifted by one.
        self._last_mean: Optional[torch.Tensor] = None

    def _bounds(
        self, value: Union[float, Sequence[float]], action_dim: int
    ) -> torch.Tensor:
        """Broadcast a scalar or per-dimension bound to ``(action_dim,)``."""
        tensor = torch.as_tensor(
            np.broadcast_to(np.asarray(value, dtype=np.float32), (action_dim,)).copy(),
            device=self.device,
            dtype=torch.float32,
        )
        return tensor

    @torch.no_grad()
    def goal_embedding(self, goal: ArrayLike) -> torch.Tensor:
        """Embed a goal window with the target encoder.

        The target encoder is deliberate: the predictor was trained to land on
        *its* outputs, so scoring against the context encoder would measure
        distance in a slightly different space than the one the model was
        optimized in.

        Args:
            goal: ``(context_len, state_dim)`` window describing the desired
                state, or an already-computed ``(embed_dim,)`` embedding.

        Returns:
            ``(1, embed_dim)`` embedding.
        """
        goal_t = torch.as_tensor(
            np.asarray(goal) if not isinstance(goal, torch.Tensor) else goal,
            device=self.device,
            dtype=torch.float32,
        )
        if goal_t.dim() == 1 and goal_t.shape[0] == self.model.config.embed_dim:
            return goal_t.unsqueeze(0)
        return self.model.encode(goal_t, use_target=True)[:1]

    @torch.no_grad()
    def plan(
        self,
        context_states: ArrayLike,
        goal: ArrayLike,
        *,
        warm_start: bool = True,
    ) -> Dict[str, Any]:
        """Search for an action sequence that drives the latent state to ``goal``.

        Args:
            context_states: ``(context_len, state_dim)`` window of recent
                featurized states - what the robot has just been doing.
            goal: Goal window or goal embedding, see :meth:`goal_embedding`.
            warm_start: Seed the search with the previous call's plan, shifted
                one step. This is what makes receding-horizon replanning cheap:
                consecutive control steps face nearly the same problem.

        Returns:
            Dict with:

            - ``actions``: ``(horizon, action_dim)`` best plan, as float64
              NumPy, ready to hand to a ManipulaPy controller;
            - ``cost``: scalar cost of that plan;
            - ``predicted_embeddings``: ``(horizon, embed_dim)`` latent
              rollout under it;
            - ``goal_distance``: distance from the final predicted embedding
              to the goal.
        """
        action_dim = self.model.config.action_dim
        z_goal = self.goal_embedding(goal)
        z0 = self.model.encode(
            torch.as_tensor(
                (
                    np.asarray(context_states)
                    if not isinstance(context_states, torch.Tensor)
                    else context_states
                ),
                device=self.device,
                dtype=torch.float32,
            ),
            use_target=True,
        )[:1]

        mean = self._initial_mean(warm_start, action_dim)
        std = (self.action_high - self.action_low).unsqueeze(0) / 2.0
        std = std.expand(self.horizon, action_dim).clone()

        best_actions = mean.clone()
        best_embeddings = self.model.predictor.rollout(
            z0, self.model.normalize_actions(mean.unsqueeze(0))
        )[0]
        best_cost = torch.tensor(float("inf"), device=self.device)

        for _ in range(self.num_iterations):
            noise = torch.randn(
                (self.num_samples, self.horizon, action_dim),
                generator=self.generator,
                device=self.device,
                dtype=torch.float32,
            )
            candidates = torch.clamp(
                mean.unsqueeze(0) + noise * std.unsqueeze(0),
                min=self.action_low,
                max=self.action_high,
            )

            costs, embeddings = self._score(z0, z_goal, candidates)
            order = torch.argsort(costs)
            elites = candidates[order[: self.num_elites]]

            if costs[order[0]] < best_cost:
                best_cost = costs[order[0]]
                best_actions = candidates[order[0]].clone()
                best_embeddings = embeddings[order[0]].clone()

            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0).clamp_min(1e-4)
            mean = self.elite_momentum * mean + (1.0 - self.elite_momentum) * new_mean
            std = self.elite_momentum * std + (1.0 - self.elite_momentum) * new_std

        self._last_mean = mean.detach().clone()
        final_distance = torch.linalg.vector_norm(best_embeddings[-1] - z_goal[0])
        return {
            "actions": best_actions.double().cpu().numpy(),
            "cost": float(best_cost),
            "predicted_embeddings": best_embeddings.double().cpu().numpy(),
            "goal_distance": float(final_distance),
        }

    def _initial_mean(self, warm_start: bool, action_dim: int) -> torch.Tensor:
        """Starting CEM mean: the shifted previous plan, or the bound midpoint."""
        midpoint = ((self.action_low + self.action_high) / 2.0).unsqueeze(0)
        if not warm_start or self._last_mean is None:
            return midpoint.expand(self.horizon, action_dim).clone()
        shifted = torch.roll(self._last_mean, shifts=-1, dims=0)
        shifted[-1] = midpoint[0]
        return shifted

    def _score(
        self,
        z0: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple:
        """Cost of each candidate sequence, plus its latent rollout.

        Every candidate shares the same starting embedding, so ``z0`` is
        expanded rather than re-encoded - the encoder runs once per plan, not
        once per sample.
        """
        population = candidates.shape[0]
        z_start = z0.expand(population, -1)
        normalized = self.model.normalize_actions(candidates)
        embeddings = self.model.predictor.rollout(z_start, normalized)

        distance = torch.linalg.vector_norm(embeddings - z_goal.unsqueeze(1), dim=-1)
        weights = torch.pow(
            torch.as_tensor(
                self.discount, device=distance.device, dtype=distance.dtype
            ),
            torch.arange(self.horizon, device=distance.device, dtype=distance.dtype),
        )
        weights = weights / weights.sum()
        cost = (distance * weights.unsqueeze(0)).sum(dim=1)
        cost = cost + self.terminal_weight * distance[:, -1]

        if self.action_cost > 0.0:
            cost = cost + self.action_cost * candidates.pow(2).mean(dim=(1, 2))
        if self.smoothness_cost > 0.0 and self.horizon > 1:
            jerk = (candidates[:, 1:] - candidates[:, :-1]).pow(2).mean(dim=(1, 2))
            cost = cost + self.smoothness_cost * jerk
        return cost, embeddings

    def reset(self) -> None:
        """Forget the warm-start plan.

        Call this when the goal changes or the robot is repositioned, so the
        next search does not start from a plan made for a different problem.
        """
        self._last_mean = None
