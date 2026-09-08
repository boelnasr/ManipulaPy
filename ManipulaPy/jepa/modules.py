#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JEPA network modules - ManipulaPy

The three Torch modules a manipulator JEPA is built from:

- :class:`MLPTrunk` - the shared residual-free MLP used by both halves.
- :class:`StateWindowEncoder` - embeds a window of featurized robot states,
  with a learned mask token so a partially observed window is a first-class
  input rather than a hole filled with zeros.
- :class:`LatentPredictor` - advances an embedding one step per action, purely
  in latent space. Rolling it ``K`` times is the model's forward simulation.

This module imports Torch at the top level; it is reached only through
``ManipulaPy.jepa``, which raises a helpful ImportError when Torch is absent.

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

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MLPTrunk", "StateWindowEncoder", "LatentPredictor"]


class MLPTrunk(nn.Module):
    """LayerNorm-GELU MLP.

    Args:
        in_dim: Input width.
        hidden: Hidden widths, one entry per hidden layer.
        out_dim: Output width.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list = []
        width = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(width, int(h)))
            layers.append(nn.LayerNorm(int(h)))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            width = int(h)
        layers.append(nn.Linear(width, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the trunk to ``x`` of shape ``(..., in_dim)``."""
        return self.net(x)


class StateWindowEncoder(nn.Module):
    """Embed a window of featurized robot states into one latent vector.

    The window is flattened rather than attended over: with a context of a
    handful of steps, position in the window is unambiguous after flattening,
    and the parameter count stays low enough to train on the modest datasets a
    single manipulator produces.

    Masked entries are replaced by a learned per-feature mask token and the
    binary mask itself is concatenated to the input, so the encoder can
    distinguish "this joint was hidden" from "this joint reads zero" - the
    distinction that makes masked-context prediction a usable training signal.

    Args:
        state_dim: Width of one featurized state.
        context_len: Number of states per window.
        embed_dim: Width of the produced embedding.
        hidden: Hidden widths of the trunk.
        dropout: Dropout probability inside the trunk.
        normalize: L2-normalize the output embedding.
    """

    def __init__(
        self,
        state_dim: int,
        context_len: int,
        embed_dim: int,
        hidden: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_len = int(context_len)
        self.embed_dim = int(embed_dim)
        self.normalize = bool(normalize)

        self.mask_token = nn.Parameter(torch.zeros(self.state_dim))
        self.trunk = MLPTrunk(
            in_dim=self.context_len * self.state_dim * 2,
            hidden=hidden,
            out_dim=self.embed_dim,
            dropout=dropout,
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed a batch of state windows.

        Args:
            states: ``(B, context_len, state_dim)`` featurized states.
            mask: Optional boolean tensor broadcastable to ``states``. True
                marks an entry as hidden from the encoder.

        Returns:
            ``(B, embed_dim)`` embeddings.
        """
        if states.dim() != 3:
            raise ValueError(
                f"states must be (B, context_len, state_dim), got "
                f"{tuple(states.shape)}"
            )
        if states.shape[1] != self.context_len or (states.shape[2] != self.state_dim):
            raise ValueError(
                f"states shape {tuple(states.shape)} does not match encoder "
                f"(context_len={self.context_len}, state_dim={self.state_dim})"
            )

        if mask is None:
            mask_f = torch.zeros_like(states)
            visible = states
        else:
            mask_b = mask.to(dtype=torch.bool).expand_as(states)
            mask_f = mask_b.to(states.dtype)
            token = self.mask_token.to(states.dtype).expand_as(states)
            visible = torch.where(mask_b, token, states)

        x = torch.cat([visible, mask_f], dim=-1).flatten(start_dim=1)
        z = self.out_norm(self.trunk(x))
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z


class LatentPredictor(nn.Module):
    """Action-conditioned latent transition model.

    One call advances the embedding by a single control step::

        z_{k+1} = z_k + f(z_k, a_k, k)

    The residual form matters: at initialization the predictor is close to the
    identity, so early training compares an embedding against a nearby
    embedding instead of against noise, which is what keeps the EMA target from
    collapsing before the predictor has learned anything.

    Args:
        embed_dim: Width of the latent embedding.
        action_dim: Width of one action vector.
        horizon: Number of distinct step embeddings to learn. Rollouts longer
            than this reuse the last one (see :meth:`rollout`).
        hidden: Hidden widths of the trunk.
        dropout: Dropout probability inside the trunk.
        normalize: L2-normalize each produced embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        horizon: int,
        hidden: Sequence[int] = (256, 256),
        dropout: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.normalize = bool(normalize)

        self.step_embedding = nn.Embedding(self.horizon, self.embed_dim)
        nn.init.zeros_(self.step_embedding.weight)
        self.trunk = MLPTrunk(
            in_dim=2 * self.embed_dim + self.action_dim,
            hidden=hidden,
            out_dim=self.embed_dim,
            dropout=dropout,
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)
        # Start as the identity map: the residual branch contributes nothing
        # until training moves the final layer off zero.
        nn.init.zeros_(self.trunk.net[-1].weight)
        nn.init.zeros_(self.trunk.net[-1].bias)

    def step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        step_index: int = 0,
    ) -> torch.Tensor:
        """Advance ``z`` by one action.

        Args:
            z: ``(B, embed_dim)`` current embedding.
            action: ``(B, action_dim)`` action applied at this step.
            step_index: Index into the learned step embeddings, clamped into
                range so rollouts may exceed the trained horizon.

        Returns:
            ``(B, embed_dim)`` next embedding.
        """
        idx = min(max(int(step_index), 0), self.horizon - 1)
        step = self.step_embedding.weight[idx].to(z.dtype)
        step = step.unsqueeze(0).expand(z.shape[0], -1)
        delta = self.trunk(torch.cat([z, action, step], dim=-1))
        z_next = self.out_norm(z + delta)
        if self.normalize:
            z_next = F.normalize(z_next, dim=-1)
        return z_next

    def forward(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`rollout`."""
        return self.rollout(z0, actions)

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll the latent state forward over an action sequence.

        Args:
            z0: ``(B, embed_dim)`` starting embedding.
            actions: ``(B, K, action_dim)`` action sequence. ``K`` may exceed
                the trained horizon; extra steps reuse the last step
                embedding.

        Returns:
            ``(B, K, embed_dim)`` predicted embeddings, one per action.
        """
        if actions.dim() != 3:
            raise ValueError(
                f"actions must be (B, K, action_dim), got {tuple(actions.shape)}"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions have width {actions.shape[-1]}, expected "
                f"{self.action_dim}"
            )
        if actions.shape[0] != z0.shape[0]:
            raise ValueError(
                f"batch mismatch: z0 has {z0.shape[0]} rows, actions have "
                f"{actions.shape[0]}"
            )

        z = z0
        out = []
        for k in range(actions.shape[1]):
            z = self.step(z, actions[:, k], step_index=k)
            out.append(z)
        return torch.stack(out, dim=1)
