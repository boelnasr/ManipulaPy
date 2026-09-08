#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Context masking for JEPA - ManipulaPy

I-JEPA hides blocks of an image and asks the predictor to recover their
embeddings. The equivalent structure for a manipulator is not a rectangle of
pixels but a *joint*: hiding joint 3 across the whole context window forces the
encoder to infer that joint's contribution from the coupled dynamics of the
others, which is exactly the correlation a world model should carry.

Two mask shapes are provided and can be combined:

- :func:`sample_joint_mask` - hide whole joints (their angle and velocity
  features), optionally for the entire window at once.
- :func:`sample_time_mask` - hide a contiguous span of timesteps, the temporal
  analogue of an I-JEPA block.

Masks are boolean tensors shaped like the state window, ``(B, C, S)``, where
True marks an entry the encoder may not see. Features outside any joint's
ownership - the end-effector block - are never masked by
:func:`sample_joint_mask`, because they are a function of *all* joints and
hiding them alongside a single joint would confound the two signals.

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

from typing import Optional

import torch

__all__ = [
    "joint_mask_to_feature_mask",
    "sample_joint_mask",
    "sample_time_mask",
]


def joint_mask_to_feature_mask(
    joint_mask: torch.Tensor,
    num_joints: int,
    state_dim: int,
) -> torch.Tensor:
    """Expand a per-joint mask to the feature layout.

    Mirrors :meth:`RobotStateFeaturizer.joint_feature_indices`: joint ``j``
    owns features ``j`` (cosine), ``n + j`` (sine) and ``2n + j`` (velocity).

    Args:
        joint_mask: ``(..., num_joints)`` boolean tensor, True to hide.
        num_joints: Degrees of freedom.
        state_dim: Width of one featurized state.

    Returns:
        ``(..., state_dim)`` boolean tensor.
    """
    if joint_mask.shape[-1] != num_joints:
        raise ValueError(
            f"joint_mask last dim is {joint_mask.shape[-1]}, expected " f"{num_joints}"
        )
    if state_dim < 3 * num_joints:
        raise ValueError(f"state_dim {state_dim} is too small for {num_joints} joints")

    joint_mask = joint_mask.to(dtype=torch.bool)
    out = torch.zeros(
        (*joint_mask.shape[:-1], state_dim),
        dtype=torch.bool,
        device=joint_mask.device,
    )
    out[..., 0:num_joints] = joint_mask
    out[..., num_joints : 2 * num_joints] = joint_mask
    out[..., 2 * num_joints : 3 * num_joints] = joint_mask
    return out


def sample_joint_mask(
    batch_size: int,
    context_len: int,
    num_joints: int,
    state_dim: int,
    prob: float,
    *,
    share_across_time: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample a per-joint context mask.

    Args:
        batch_size: Number of windows.
        context_len: Steps per window.
        num_joints: Degrees of freedom.
        state_dim: Width of one featurized state.
        prob: Probability of hiding any given joint. ``0.0`` returns an
            all-False mask.
        share_across_time: Hide a chosen joint for the whole window rather
            than resampling per timestep. On by default: a joint that flickers
            in and out of view is trivially interpolated and teaches nothing.
        generator: Optional RNG for reproducible masks.
        device: Device for the returned mask.

    Returns:
        ``(batch_size, context_len, state_dim)`` boolean mask.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")

    if prob == 0.0:
        return torch.zeros(
            (batch_size, context_len, state_dim),
            dtype=torch.bool,
            device=device,
        )

    shape = (
        (batch_size, 1, num_joints)
        if share_across_time
        else (batch_size, context_len, num_joints)
    )
    draws = torch.rand(shape, generator=generator, device=device)
    joint_mask = draws < prob
    if share_across_time:
        joint_mask = joint_mask.expand(batch_size, context_len, num_joints)
    return joint_mask_to_feature_mask(joint_mask, num_joints, state_dim)


def sample_time_mask(
    batch_size: int,
    context_len: int,
    state_dim: int,
    span: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample a contiguous block of hidden timesteps.

    Args:
        batch_size: Number of windows.
        context_len: Steps per window.
        state_dim: Width of one featurized state.
        span: Number of consecutive steps to hide. Clamped to
            ``context_len - 1`` so at least one step stays visible; ``0``
            returns an all-False mask.
        generator: Optional RNG for reproducible masks.
        device: Device for the returned mask.

    Returns:
        ``(batch_size, context_len, state_dim)`` boolean mask.
    """
    if span < 0:
        raise ValueError(f"span must be non-negative, got {span}")

    mask = torch.zeros(
        (batch_size, context_len, state_dim), dtype=torch.bool, device=device
    )
    span = min(int(span), context_len - 1)
    if span <= 0:
        return mask

    starts = torch.randint(
        0,
        context_len - span + 1,
        (batch_size,),
        generator=generator,
        device=device,
    )
    steps = torch.arange(context_len, device=device).unsqueeze(0)
    hidden = (steps >= starts.unsqueeze(1)) & (steps < (starts + span).unsqueeze(1))
    return mask | hidden.unsqueeze(-1)
