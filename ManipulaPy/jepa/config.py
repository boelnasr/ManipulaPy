#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JEPA configuration - ManipulaPy

Declarative hyper-parameters for :class:`~ManipulaPy.jepa.model.ManipulatorJEPA`.
:class:`JEPAConfig` is a plain dataclass (no Torch import) so it can be built,
serialized and inspected on a NumPy-only install; the model consumes it.

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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

__all__ = ["JEPAConfig"]


@dataclass
class JEPAConfig:
    """Hyper-parameters for a manipulator state-space JEPA.

    The architecture follows the joint-embedding predictive recipe: a context
    encoder embeds a short window of past robot states, an action-conditioned
    predictor advances that embedding ``horizon`` steps in latent space, and an
    EMA copy of the encoder (the *target* encoder) embeds the states that were
    actually reached. Nothing is decoded back to state space, so the objective
    never spends capacity on unpredictable detail.

    Attributes:
        num_joints: Degrees of freedom of the manipulator.
        state_dim: Width of one featurized state vector. Must match the
            featurizer that produced the training data
            (:class:`~ManipulaPy.jepa.features.RobotStateFeaturizer`).
        action_dim: Width of one action vector (joint torques by default).
        embed_dim: Width of the latent embedding the predictor operates in.
        context_len: Number of past states the encoder sees per view.
        horizon: Number of latent steps the predictor is trained to roll out.
        encoder_hidden: Hidden widths of the encoder trunk.
        predictor_hidden: Hidden widths of the per-step predictor trunk.
        dropout: Dropout probability inside both trunks.
        joint_mask_prob: Probability of masking a joint's features in the
            context view during training. ``0.0`` disables masking; a positive
            value gives the I-JEPA-style "predict from a partial view" signal.
        ema_base: Target-encoder EMA momentum at the first training step.
        ema_final: Target-encoder EMA momentum at the last training step. The
            momentum is annealed linearly between the two, as in I-JEPA.
        pred_loss: Latent regression loss, one of ``"smooth_l1"``, ``"mse"``
            or ``"cosine"``.
        variance_weight: Weight of the VICReg hinge-variance term that keeps
            each embedding dimension from collapsing to a constant.
        covariance_weight: Weight of the VICReg off-diagonal covariance term
            that keeps embedding dimensions from becoming redundant.
        discount: Per-step weight ``discount ** k`` applied to the latent
            regression loss, so near-term prediction dominates.
        normalize_embeddings: If True, L2-normalize encoder outputs. Useful
            with ``pred_loss="cosine"``; leave off for the regression losses.
        seed: Seed used for parameter initialization when the model builds its
            own generator. ``None`` leaves Torch's global RNG untouched.
    """

    num_joints: int
    state_dim: int
    action_dim: int
    embed_dim: int = 128
    context_len: int = 4
    horizon: int = 8
    encoder_hidden: Tuple[int, ...] = (256, 256)
    predictor_hidden: Tuple[int, ...] = (256, 256)
    dropout: float = 0.0
    joint_mask_prob: float = 0.15
    ema_base: float = 0.996
    ema_final: float = 1.0
    pred_loss: str = "smooth_l1"
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    discount: float = 1.0
    normalize_embeddings: bool = False
    seed: Any = None
    #: Free-form provenance (robot name, URDF path, featurizer settings, ...).
    #: Carried through checkpoints so a saved model documents its own data.
    metadata: Dict[str, Any] = field(default_factory=dict)

    _LOSSES = ("smooth_l1", "mse", "cosine")

    def __post_init__(self) -> None:
        for name in ("num_joints", "state_dim", "action_dim", "embed_dim"):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            setattr(self, name, int(value))
        for name in ("context_len", "horizon"):
            value = getattr(self, name)
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            setattr(self, name, int(value))

        self.encoder_hidden = tuple(int(h) for h in self.encoder_hidden)
        self.predictor_hidden = tuple(int(h) for h in self.predictor_hidden)
        if any(h <= 0 for h in self.encoder_hidden + self.predictor_hidden):
            raise ValueError("hidden widths must be positive")

        if self.pred_loss not in self._LOSSES:
            raise ValueError(
                f"pred_loss must be one of {self._LOSSES}, got {self.pred_loss!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.joint_mask_prob <= 1.0:
            raise ValueError(
                f"joint_mask_prob must be in [0, 1], got {self.joint_mask_prob}"
            )
        for name in ("ema_base", "ema_final"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.discount <= 0.0:
            raise ValueError(f"discount must be positive, got {self.discount}")
        if self.variance_weight < 0.0 or self.covariance_weight < 0.0:
            raise ValueError("VICReg weights must be non-negative")

    @property
    def encoder_input_dim(self) -> int:
        """Flattened encoder input width.

        Each of the ``context_len`` steps contributes its ``state_dim``
        features plus a same-width binary channel marking which of them were
        masked out, so the encoder can tell a masked value from a real zero.
        """
        return self.context_len * self.state_dim * 2

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy of this configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JEPAConfig":
        """Rebuild a configuration from :meth:`to_dict` output."""
        payload = dict(data)
        payload.pop("_LOSSES", None)
        return cls(**payload)
