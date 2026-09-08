#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JEPA objective terms - ManipulaPy

The loss a joint-embedding predictive architecture is trained with, split into
its three independent pieces:

- :func:`prediction_loss` - regression of predicted embeddings onto the
  stop-gradient target embeddings, discounted over the horizon.
- :func:`variance_loss` / :func:`covariance_loss` - the VICReg regularizers.
  Without them (or an equivalent asymmetry) a joint-embedding objective has a
  trivial optimum: map every input to the same point. The EMA target encoder
  already fights that; these terms make the guarantee explicit and cheap.
- :func:`jepa_loss` - the weighted sum, returned with its components so a
  training run can be diagnosed rather than just watched.

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

from typing import Dict

import torch
import torch.nn.functional as F

__all__ = [
    "prediction_loss",
    "variance_loss",
    "covariance_loss",
    "jepa_loss",
]


def prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    kind: str = "smooth_l1",
    discount: float = 1.0,
) -> torch.Tensor:
    """Latent regression loss between predicted and target embeddings.

    Args:
        predicted: ``(B, K, D)`` embeddings produced by the predictor.
        target: ``(B, K, D)`` embeddings from the target encoder. Detached
            internally, so callers need not remember to.
        kind: ``"smooth_l1"``, ``"mse"`` or ``"cosine"``.
        discount: Per-step weight ``discount ** k``, normalized to sum to one
            so the loss scale does not depend on the horizon.

    Returns:
        Scalar loss.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"shape mismatch: predicted {tuple(predicted.shape)} vs target "
            f"{tuple(target.shape)}"
        )
    if predicted.dim() != 3:
        raise ValueError(f"expected (B, K, D) tensors, got {tuple(predicted.shape)}")

    target = target.detach()
    if kind == "smooth_l1":
        per_step = F.smooth_l1_loss(predicted, target, reduction="none").mean(
            dim=(0, 2)
        )
    elif kind == "mse":
        per_step = F.mse_loss(predicted, target, reduction="none").mean(dim=(0, 2))
    elif kind == "cosine":
        per_step = (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean(dim=0)
    else:
        raise ValueError(
            f"unknown prediction loss {kind!r}; expected 'smooth_l1', 'mse' "
            "or 'cosine'"
        )

    horizon = per_step.shape[0]
    weights = torch.pow(
        torch.as_tensor(float(discount), device=per_step.device, dtype=per_step.dtype),
        torch.arange(horizon, device=per_step.device, dtype=per_step.dtype),
    )
    weights = weights / weights.sum()
    return (per_step * weights).sum()


def variance_loss(embeddings: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg hinge-variance term.

    Penalizes any embedding dimension whose standard deviation across the
    batch falls below one, which is the signature of a collapsing encoder.

    Args:
        embeddings: ``(N, D)`` or ``(B, K, D)`` embeddings.
        eps: Added under the square root for numerical stability.

    Returns:
        Scalar loss, zero when every dimension has unit or greater deviation.
    """
    x = embeddings.reshape(-1, embeddings.shape[-1])
    if x.shape[0] < 2:
        return x.new_zeros(())
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return F.relu(1.0 - std).mean()


def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """VICReg off-diagonal covariance term.

    Drives the off-diagonal entries of the embedding covariance toward zero so
    dimensions carry distinct information instead of duplicating one another.

    Args:
        embeddings: ``(N, D)`` or ``(B, K, D)`` embeddings.

    Returns:
        Scalar loss.
    """
    x = embeddings.reshape(-1, embeddings.shape[-1])
    n, d = x.shape
    if n < 2:
        return x.new_zeros(())
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (n - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    kind: str = "smooth_l1",
    discount: float = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """Full JEPA objective.

    Args:
        predicted: ``(B, K, D)`` predicted embeddings.
        target: ``(B, K, D)`` target-encoder embeddings.
        kind: Regression loss passed to :func:`prediction_loss`.
        discount: Horizon discount passed to :func:`prediction_loss`.
        variance_weight: Weight of :func:`variance_loss`.
        covariance_weight: Weight of :func:`covariance_loss`.

    Returns:
        Dict with the scalar ``"loss"`` plus its ``"prediction"``,
        ``"variance"`` and ``"covariance"`` components. The regularizers are
        applied to the *predicted* embeddings; the targets carry no gradient,
        so regularizing them would be a no-op.
    """
    pred_term = prediction_loss(predicted, target, kind=kind, discount=discount)
    var_term = (
        variance_loss(predicted) if variance_weight > 0.0 else predicted.new_zeros(())
    )
    cov_term = (
        covariance_loss(predicted)
        if covariance_weight > 0.0
        else predicted.new_zeros(())
    )
    total = pred_term + variance_weight * var_term + covariance_weight * cov_term
    return {
        "loss": total,
        "prediction": pred_term,
        "variance": var_term,
        "covariance": cov_term,
    }
