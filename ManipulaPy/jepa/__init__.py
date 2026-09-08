#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JEPA Module - ManipulaPy

A Joint-Embedding Predictive Architecture for manipulator dynamics.

A JEPA learns to predict *representations* of future states rather than the
states themselves. Applied to a robot arm, that is the difference between a
model asked to reproduce every joint angle exactly - including the parts of the
future no controller can know - and one free to keep only what is actually
predictable. The pieces here mirror the I-JEPA / V-JEPA recipe, specialized to
articulated-body state:

============================  ==============================================
Piece                         Role
============================  ==============================================
:class:`RobotStateFeaturizer` Exact features from ManipulaPy kinematics:
                              trigonometric joint angles, velocities, and the
                              product-of-exponentials end-effector pose.
:func:`generate_rollouts`     Ground-truth training data by integrating
                              ManipulaPy's forward dynamics under smooth
                              random torques - no simulator required.
:class:`ManipulatorJEPA`      Context encoder, EMA target encoder, and an
                              action-conditioned latent predictor.
:class:`JEPATrainer`          Training loop with EMA annealing and a
                              representation-collapse diagnostic.
:class:`LatentMPCPlanner`     Cross-entropy-method planning entirely inside
                              the learned latent space.
============================  ==============================================

Quick start::

    from ManipulaPy.jepa import (
        JEPAConfig, JEPATrainer, LatentMPCPlanner, ManipulatorJEPA,
        RobotStateFeaturizer, TrajectoryWindowDataset, generate_rollouts,
        make_dataloaders,
    )

    featurizer = RobotStateFeaturizer(robot)
    rollouts = generate_rollouts(dynamics, featurizer, num_rollouts=32)
    dataset = TrajectoryWindowDataset(rollouts, context_len=4, horizon=8)

    model = ManipulatorJEPA(JEPAConfig(
        num_joints=featurizer.num_joints,
        state_dim=featurizer.state_dim,
        action_dim=dataset.action_dim,
    ))
    model.fit_normalizer(dataset.all_states(), dataset.all_actions())

    train_loader, val_loader = make_dataloaders(dataset)
    JEPATrainer(model).fit(train_loader, epochs=20, val_loader=val_loader)

This module requires PyTorch, which is an optional ManipulaPy dependency::

    pip install "ManipulaPy[jepa]"

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

import importlib.util

# The configuration and the featurizer are pure NumPy, so they stay importable
# on a Torch-free install; everything below them needs Torch. Probe rather than
# try/except around the import so a broken Torch installation still surfaces
# its own error instead of being reported as "PyTorch is not installed".
from .config import JEPAConfig
from .features import RobotStateFeaturizer

if importlib.util.find_spec("torch") is None:  # pragma: no cover - env guard
    raise ImportError(
        "ManipulaPy.jepa requires PyTorch. Install it with "
        '`pip install "ManipulaPy[jepa]"` or `pip install torch`.'
    )

from .data import (  # noqa: E402
    Rollout,
    TrajectoryWindowDataset,
    generate_rollouts,
    rollouts_from_trajectories,
)
from .losses import (  # noqa: E402
    covariance_loss,
    jepa_loss,
    prediction_loss,
    variance_loss,
)
from .masking import (  # noqa: E402
    joint_mask_to_feature_mask,
    sample_joint_mask,
    sample_time_mask,
)
from .model import ManipulatorJEPA  # noqa: E402
from .modules import LatentPredictor, MLPTrunk, StateWindowEncoder  # noqa: E402
from .planning import LatentMPCPlanner  # noqa: E402
from .training import JEPATrainer, ema_momentum, make_dataloaders  # noqa: E402

__all__ = [
    # configuration and features
    "JEPAConfig",
    "RobotStateFeaturizer",
    # data
    "Rollout",
    "TrajectoryWindowDataset",
    "generate_rollouts",
    "rollouts_from_trajectories",
    # model
    "ManipulatorJEPA",
    "MLPTrunk",
    "StateWindowEncoder",
    "LatentPredictor",
    # objective
    "jepa_loss",
    "prediction_loss",
    "variance_loss",
    "covariance_loss",
    # masking
    "sample_joint_mask",
    "sample_time_mask",
    "joint_mask_to_feature_mask",
    # training and planning
    "JEPATrainer",
    "make_dataloaders",
    "ema_momentum",
    "LatentMPCPlanner",
]
