#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Manipulator JEPA model - ManipulaPy

:class:`ManipulatorJEPA` assembles the three parts of a joint-embedding
predictive architecture over robot state:

1. a **context encoder** that embeds a window of past states, optionally with
   joints masked out;
2. an **action-conditioned predictor** that advances that embedding through a
   sequence of actions entirely in latent space;
3. a **target encoder**, an exponential-moving-average copy of the context
   encoder held under ``no_grad``, that embeds the windows actually reached.

The loss compares (2) against (3). Nothing is ever decoded back to joint
angles, which is the whole point: the model is free to discard whatever about
the future is not predictable, instead of being penalized for failing to
hallucinate it.

State normalization lives in the model as buffers, so a checkpoint carries the
statistics its weights were trained under and cannot be silently mispaired with
differently scaled data.

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

import copy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .config import JEPAConfig
from .losses import jepa_loss
from .masking import sample_joint_mask
from .modules import LatentPredictor, StateWindowEncoder

__all__ = ["ManipulatorJEPA"]

ArrayLike = Union[np.ndarray, torch.Tensor]


class ManipulatorJEPA(nn.Module):
    """Joint-embedding predictive architecture over manipulator state.

    Args:
        config: The :class:`~ManipulaPy.jepa.config.JEPAConfig` describing the
            architecture. Stored on the instance and written into checkpoints.

    Example:
        >>> cfg = JEPAConfig(num_joints=6, state_dim=27, action_dim=6)
        >>> model = ManipulatorJEPA(cfg)                    # doctest: +SKIP
        >>> out = model(context, actions, targets)          # doctest: +SKIP
        >>> out["loss"].backward()                          # doctest: +SKIP
        >>> model.update_target(0.996)                      # doctest: +SKIP
    """

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        if not isinstance(config, JEPAConfig):
            raise TypeError(f"config must be a JEPAConfig, got {type(config).__name__}")
        self.config = config

        if config.seed is not None:
            torch.manual_seed(int(config.seed))

        self.context_encoder = StateWindowEncoder(
            state_dim=config.state_dim,
            context_len=config.context_len,
            embed_dim=config.embed_dim,
            hidden=config.encoder_hidden,
            dropout=config.dropout,
            normalize=config.normalize_embeddings,
        )
        # The target encoder is a frozen EMA copy. It is registered as a
        # submodule so it moves with .to()/.cuda() and lands in the state dict,
        # but every parameter is detached from autograd.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        self.predictor = LatentPredictor(
            embed_dim=config.embed_dim,
            action_dim=config.action_dim,
            horizon=config.horizon,
            hidden=config.predictor_hidden,
            dropout=config.dropout,
            normalize=config.normalize_embeddings,
        )

        self.register_buffer(
            "state_mean", torch.zeros(config.state_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "state_std", torch.ones(config.state_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_mean", torch.zeros(config.action_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.ones(config.action_dim, dtype=torch.float32)
        )

    # -- normalization ---------------------------------------------------

    def fit_normalizer(
        self,
        states: ArrayLike,
        actions: Optional[ArrayLike] = None,
        eps: float = 1e-6,
    ) -> None:
        """Set the input statistics from a sample of training data.

        Args:
            states: Any array whose last axis is ``state_dim``.
            actions: Any array whose last axis is ``action_dim``. Optional.
            eps: Floor on the standard deviation, so a constant feature
                normalizes to zero instead of dividing by zero.
        """
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        flat = states_t.reshape(-1, states_t.shape[-1])
        if flat.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"states have width {flat.shape[-1]}, expected "
                f"{self.config.state_dim}"
            )
        self.state_mean.copy_(flat.mean(dim=0))
        self.state_std.copy_(flat.std(dim=0).clamp_min(eps))

        if actions is not None:
            actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
            flat_a = actions_t.reshape(-1, actions_t.shape[-1])
            if flat_a.shape[-1] != self.config.action_dim:
                raise ValueError(
                    f"actions have width {flat_a.shape[-1]}, expected "
                    f"{self.config.action_dim}"
                )
            self.action_mean.copy_(flat_a.mean(dim=0))
            self.action_std.copy_(flat_a.std(dim=0).clamp_min(eps))

    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Apply the stored state statistics."""
        mean = self.state_mean.to(states.dtype)
        std = self.state_std.to(states.dtype)
        return (states - mean) / std

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply the stored action statistics."""
        mean = self.action_mean.to(actions.dtype)
        std = self.action_std.to(actions.dtype)
        return (actions - mean) / std

    # -- encoding --------------------------------------------------------

    def encode(
        self,
        states: ArrayLike,
        mask: Optional[torch.Tensor] = None,
        *,
        use_target: bool = False,
    ) -> torch.Tensor:
        """Embed a batch of raw (unnormalized) state windows.

        Args:
            states: ``(B, context_len, state_dim)`` or a single
                ``(context_len, state_dim)`` window, which is treated as a
                batch of one.
            mask: Optional boolean mask, True where hidden.
            use_target: Embed with the EMA target encoder under ``no_grad``
                instead of the context encoder. This is what
                :class:`~ManipulaPy.jepa.planning.LatentMPCPlanner` uses for
                goals, so plans are scored against the same encoder the
                predictor was trained to match.

        Returns:
            ``(B, embed_dim)`` embeddings.
        """
        x = self._as_state_tensor(states)
        x = self.normalize_states(x)
        if use_target:
            with torch.no_grad():
                return self.target_encoder(x, mask)
        return self.context_encoder(x, mask)

    def predict(
        self,
        context_states: ArrayLike,
        actions: ArrayLike,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed a context window and roll it forward over ``actions``.

        Args:
            context_states: ``(B, context_len, state_dim)`` raw states.
            actions: ``(B, K, action_dim)`` raw actions. ``K`` may exceed the
                trained horizon.
            mask: Optional context mask.

        Returns:
            ``(B, K, embed_dim)`` predicted embeddings.
        """
        z0 = self.encode(context_states, mask)
        a = self.normalize_actions(self._as_action_tensor(actions))
        return self.predictor.rollout(z0, a)

    # -- training --------------------------------------------------------

    def forward(
        self,
        context_states: ArrayLike,
        actions: ArrayLike,
        target_states: ArrayLike,
        context_mask: Optional[torch.Tensor] = None,
        *,
        apply_random_mask: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the JEPA objective for one batch.

        Args:
            context_states: ``(B, context_len, state_dim)`` past states.
            actions: ``(B, K, action_dim)`` actions applied after the context.
            target_states: ``(B, K, context_len, state_dim)`` the windows
                actually reached, one per horizon step. Each is a full window
                so the target encoder sees exactly the input shape the context
                encoder does - the two halves stay architecturally identical,
                which is what makes the EMA coupling meaningful.
            context_mask: Explicit mask, True where hidden. Overrides random
                masking when given.
            apply_random_mask: Sample a joint mask from
                ``config.joint_mask_prob``. Defaults to ``self.training``, so
                masking is a training-time augmentation and evaluation sees
                the full context.
            generator: Optional RNG for reproducible masks.

        Returns:
            Dict with ``loss``, its ``prediction`` / ``variance`` /
            ``covariance`` components, the ``predicted`` and ``target``
            embeddings, and ``embedding_std`` - the mean per-dimension
            standard deviation of the target embeddings, the number to watch
            for representation collapse.
        """
        ctx = self._as_state_tensor(context_states)
        act = self._as_action_tensor(actions)
        tgt = self._as_target_tensor(target_states)

        batch, horizon = act.shape[0], act.shape[1]
        if tgt.shape[0] != batch or tgt.shape[1] != horizon:
            raise ValueError(
                f"target_states has shape {tuple(tgt.shape)}, expected "
                f"({batch}, {horizon}, {self.config.context_len}, "
                f"{self.config.state_dim})"
            )

        if context_mask is None:
            if apply_random_mask is None:
                apply_random_mask = self.training
            if apply_random_mask and self.config.joint_mask_prob > 0.0:
                context_mask = sample_joint_mask(
                    batch_size=batch,
                    context_len=self.config.context_len,
                    num_joints=self.config.num_joints,
                    state_dim=self.config.state_dim,
                    prob=self.config.joint_mask_prob,
                    generator=generator,
                    device=ctx.device,
                )

        ctx_norm = self.normalize_states(ctx)
        z0 = self.context_encoder(ctx_norm, context_mask)
        predicted = self.predictor.rollout(z0, self.normalize_actions(act))

        # Targets are always encoded from the *complete* window: the whole
        # point is to regress toward a representation of what really happened.
        with torch.no_grad():
            tgt_norm = self.normalize_states(tgt)
            flat = tgt_norm.reshape(
                batch * horizon, self.config.context_len, self.config.state_dim
            )
            target = self.target_encoder(flat).reshape(
                batch, horizon, self.config.embed_dim
            )

        out = jepa_loss(
            predicted,
            target,
            kind=self.config.pred_loss,
            discount=self.config.discount,
            variance_weight=self.config.variance_weight,
            covariance_weight=self.config.covariance_weight,
        )
        out["predicted"] = predicted
        out["target"] = target
        with torch.no_grad():
            flat_target = target.reshape(-1, target.shape[-1])
            out["embedding_std"] = (
                flat_target.std(dim=0).mean()
                if flat_target.shape[0] > 1
                else target.new_zeros(())
            )
        return out

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """Move the target encoder toward the context encoder.

        ``theta_target <- m * theta_target + (1 - m) * theta_context``, applied
        to parameters and buffers alike. Call this once per optimizer step,
        after ``optimizer.step()``.

        Args:
            momentum: EMA coefficient in ``[0, 1]``. ``1.0`` freezes the target
                encoder; ``0.0`` copies the context encoder outright.
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        for online, target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
        for online, target in zip(
            self.context_encoder.buffers(), self.target_encoder.buffers()
        ):
            target.copy_(online)

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Write weights and configuration to ``path``."""
        torch.save(
            {"config": self.config.to_dict(), "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Optional[Any] = None,
    ) -> "ManipulatorJEPA":
        """Rebuild a model saved by :meth:`save`.

        Args:
            path: Checkpoint path.
            map_location: Passed through to ``torch.load``.

        Returns:
            The restored model, in eval mode.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(JEPAConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    # -- input coercion --------------------------------------------------

    def _tensor(self, data: ArrayLike) -> torch.Tensor:
        """Coerce to a float tensor on this model's device and dtype."""
        reference = self.state_mean
        if isinstance(data, torch.Tensor):
            return data.to(device=reference.device, dtype=reference.dtype)
        return torch.as_tensor(
            np.asarray(data), device=reference.device, dtype=reference.dtype
        )

    def _as_state_tensor(self, states: ArrayLike) -> torch.Tensor:
        x = self._tensor(states)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(
                f"expected (B, context_len, state_dim) states, got " f"{tuple(x.shape)}"
            )
        if x.shape[1] != self.config.context_len or (
            x.shape[2] != self.config.state_dim
        ):
            raise ValueError(
                f"states shape {tuple(x.shape)} does not match config "
                f"(context_len={self.config.context_len}, "
                f"state_dim={self.config.state_dim})"
            )
        return x

    def _as_action_tensor(self, actions: ArrayLike) -> torch.Tensor:
        a = self._tensor(actions)
        if a.dim() == 2:
            a = a.unsqueeze(0)
        if a.dim() != 3:
            raise ValueError(
                f"expected (B, K, action_dim) actions, got {tuple(a.shape)}"
            )
        if a.shape[2] != self.config.action_dim:
            raise ValueError(
                f"actions have width {a.shape[2]}, expected "
                f"{self.config.action_dim}"
            )
        return a

    def _as_target_tensor(self, targets: ArrayLike) -> torch.Tensor:
        t = self._tensor(targets)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.dim() != 4:
            raise ValueError(
                f"expected (B, K, context_len, state_dim) targets, got "
                f"{tuple(t.shape)}"
            )
        return t
