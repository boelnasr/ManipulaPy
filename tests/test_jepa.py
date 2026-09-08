#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for the JEPA world-model module - ManipulaPy.

Covers the four things that can silently be wrong in a joint-embedding
predictive architecture and would not show up as an exception:

1. **Window alignment.** ``targets[k]`` must really be the window ending at
   ``t + 1 + k``. An off-by-one here trains the model to predict the present.
2. **Target-encoder isolation.** The target encoder must carry no gradient and
   must move only through the EMA update.
3. **Anti-collapse.** The VICReg terms must actually punish a constant
   embedding, and a short real training run must not collapse.
4. **Kinematic grounding.** The featurizer's end-effector block must match
   ManipulaPy's own forward kinematics, not an approximation of it.

Copyright (c) 2025 Mohamed Aboelnasr
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
"""

import types

import numpy as np
import pytest

from ManipulaPy.dynamics import ManipulatorDynamics
from ManipulaPy.jepa.config import JEPAConfig
from ManipulaPy.jepa.features import RobotStateFeaturizer


def _real_torch_available() -> bool:
    """True only when the *real* PyTorch package is importable.

    The suite's conftest installs a lightweight ``torch`` stand-in in
    ``sys.modules`` on a base install, so ``importorskip`` would wrongly
    succeed and then fail inside the first real tensor op. Matching the
    predicate used by ``tests/test_gradient_contract.py``.
    """
    try:
        import torch
    except Exception:
        return False
    return isinstance(torch, types.ModuleType)


requires_torch = pytest.mark.skipif(
    not _real_torch_available(), reason="requires a real PyTorch installation"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dynamics() -> ManipulatorDynamics:
    """A small 6-DOF arm, matching the fixture used by tests/test_dynamics.py."""
    s_list = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, -1, 0, -0.089, 0, 0],
            [0, -1, 0, -0.089, 0, 0.425],
            [0, -1, 0, -0.089, 0, 0.817],
            [1, 0, 0, 0, 0.109, 0],
            [0, -1, 0, -0.089, 0, 0.817],
        ]
    ).T
    m_list = np.array([[1, 0, 0, 0.817], [0, 1, 0, 0], [0, 0, 1, 0.191], [0, 0, 0, 1]])
    glist = np.stack([np.eye(6) * (1.0 + 0.1 * i) for i in range(6)])
    return ManipulatorDynamics(
        M_list=m_list,
        omega_list=None,
        r_list=None,
        b_list=None,
        S_list=s_list,
        B_list=np.copy(s_list),
        Glist=glist,
    )


@pytest.fixture(scope="module")
def featurizer(dynamics: ManipulatorDynamics) -> RobotStateFeaturizer:
    return RobotStateFeaturizer(dynamics, velocity_scale=0.2)


@pytest.fixture(scope="module")
def rollouts(dynamics, featurizer):
    """A small but genuine dataset from ManipulaPy's own forward dynamics."""
    from ManipulaPy.jepa.data import generate_rollouts

    return generate_rollouts(
        dynamics,
        featurizer,
        num_rollouts=3,
        steps=24,
        dt=0.02,
        substeps=1,
        torque_scale=2.0,
        seed=7,
    )


def _small_config(state_dim: int, action_dim: int, **overrides) -> JEPAConfig:
    params = dict(
        num_joints=6,
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=16,
        encoder_hidden=(32,),
        predictor_hidden=(32,),
        context_len=3,
        horizon=4,
        seed=0,
    )
    params.update(overrides)
    return JEPAConfig(**params)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestJEPAConfig:
    def test_encoder_input_dim_accounts_for_the_mask_channel(self) -> None:
        cfg = JEPAConfig(num_joints=6, state_dim=27, action_dim=6, context_len=4)
        assert cfg.encoder_input_dim == 4 * 27 * 2

    def test_round_trips_through_a_dict(self) -> None:
        cfg = JEPAConfig(num_joints=6, state_dim=27, action_dim=6, horizon=5)
        assert JEPAConfig.from_dict(cfg.to_dict()) == cfg

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_joints": 0},
            {"horizon": -1},
            {"pred_loss": "huber"},
            {"dropout": 1.0},
            {"joint_mask_prob": 1.5},
            {"ema_base": 1.5},
            {"discount": 0.0},
            {"variance_weight": -1.0},
        ],
    )
    def test_rejects_out_of_range_settings(self, kwargs) -> None:
        base = dict(num_joints=6, state_dim=27, action_dim=6)
        base.update(kwargs)
        with pytest.raises(ValueError):
            JEPAConfig(**base)


# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------


class TestRobotStateFeaturizer:
    def test_state_dim_matches_the_produced_vector(self, featurizer) -> None:
        theta = np.linspace(-0.5, 0.5, featurizer.num_joints)
        assert featurizer.featurize(theta).shape == (featurizer.state_dim,)
        assert featurizer.state_dim == 3 * featurizer.num_joints + 9

    def test_ee_block_equals_manipulapy_forward_kinematics(
        self, featurizer, dynamics
    ) -> None:
        """The geometric features must be FK itself, not a stand-in for it."""
        theta = np.array([0.3, -0.4, 0.5, 0.1, -0.2, 0.6])
        features = featurizer.featurize(theta)
        expected = np.asarray(dynamics.forward_kinematics(theta, frame="space"))

        n = featurizer.num_joints
        np.testing.assert_allclose(features[3 * n : 3 * n + 3], expected[:3, 3])
        np.testing.assert_allclose(
            features[3 * n + 3 : 3 * n + 9],
            expected[:3, :2].reshape(-1, order="F"),
        )

    def test_angles_are_continuous_across_the_pi_wrap(self, featurizer) -> None:
        """+pi and -pi are the same configuration and must featurize alike."""
        below = np.full(featurizer.num_joints, np.pi - 1e-9)
        above = np.full(featurizer.num_joints, -np.pi + 1e-9)
        np.testing.assert_allclose(
            featurizer.featurize(below), featurizer.featurize(above), atol=1e-7
        )

    def test_velocity_scale_is_applied(self, dynamics) -> None:
        feat = RobotStateFeaturizer(dynamics, velocity_scale=0.25)
        n = feat.num_joints
        dtheta = np.arange(1.0, n + 1.0)
        out = feat.featurize(np.zeros(n), dtheta)
        np.testing.assert_allclose(out[2 * n : 3 * n], dtheta * 0.25)

    def test_batch_matches_per_sample(self, featurizer) -> None:
        thetas = np.random.default_rng(0).normal(size=(5, featurizer.num_joints))
        batch = featurizer.featurize_batch(thetas)
        for i in range(thetas.shape[0]):
            np.testing.assert_allclose(batch[i], featurizer.featurize(thetas[i]))

    def test_joint_feature_indices_are_disjoint_and_in_range(self, featurizer) -> None:
        seen: set = set()
        for j in range(featurizer.num_joints):
            idx = featurizer.joint_feature_indices(j)
            assert len(idx) == 3
            assert not seen & set(idx)
            seen.update(idx)
            assert all(0 <= i < featurizer.state_dim for i in idx)

    def test_ee_pose_requires_a_manipulator(self) -> None:
        with pytest.raises(ValueError, match="include_ee_pose"):
            RobotStateFeaturizer(None, num_joints=6, include_ee_pose=True)

    def test_works_without_a_manipulator_when_ee_pose_is_off(self) -> None:
        feat = RobotStateFeaturizer(None, num_joints=4, include_ee_pose=False)
        assert feat.state_dim == 12
        assert feat.featurize(np.zeros(4)).shape == (12,)

    def test_rejects_a_wrong_length_state(self, featurizer) -> None:
        with pytest.raises(ValueError, match="expected"):
            featurizer.featurize(np.zeros(featurizer.num_joints + 1))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


@requires_torch
class TestRollouts:
    def test_rollouts_are_finite_and_move(self, rollouts) -> None:
        assert len(rollouts) == 3
        for rollout in rollouts:
            assert np.all(np.isfinite(rollout.states))
            assert rollout.states.shape[0] == rollout.actions.shape[0]
            # A rollout that never leaves its initial pose teaches nothing.
            assert np.ptp(rollout.thetas, axis=0).max() > 1e-6

    def test_generation_is_reproducible_under_a_seed(
        self, dynamics, featurizer
    ) -> None:
        from ManipulaPy.jepa.data import generate_rollouts

        kwargs = dict(num_rollouts=1, steps=8, dt=0.02, substeps=1, torque_scale=1.0)
        a = generate_rollouts(dynamics, featurizer, seed=3, **kwargs)
        b = generate_rollouts(dynamics, featurizer, seed=3, **kwargs)
        np.testing.assert_allclose(a[0].states, b[0].states)
        np.testing.assert_allclose(a[0].actions, b[0].actions)

    def test_joint_limits_are_respected(self, dynamics, featurizer) -> None:
        from ManipulaPy.jepa.data import generate_rollouts

        limits = [(-0.2, 0.2)] * featurizer.num_joints
        out = generate_rollouts(
            dynamics,
            featurizer,
            num_rollouts=2,
            steps=16,
            dt=0.02,
            substeps=1,
            torque_scale=25.0,
            joint_limits=limits,
            seed=1,
        )
        for rollout in out:
            assert rollout.thetas.min() >= -0.2 - 1e-9
            assert rollout.thetas.max() <= 0.2 + 1e-9

    def test_velocity_limit_is_respected(self, dynamics, featurizer) -> None:
        from ManipulaPy.jepa.data import generate_rollouts

        out = generate_rollouts(
            dynamics,
            featurizer,
            num_rollouts=1,
            steps=16,
            dt=0.02,
            substeps=1,
            torque_scale=40.0,
            velocity_limit=0.5,
            seed=2,
        )
        assert np.abs(out[0].dthetas).max() <= 0.5 + 1e-9

    def test_rollouts_from_trajectories_accepts_planner_output(
        self, featurizer
    ) -> None:
        from ManipulaPy.jepa.data import rollouts_from_trajectories

        n = featurizer.num_joints
        traj = {
            "positions": np.zeros((10, n)),
            "velocities": np.ones((10, n)) * 0.1,
            "accelerations": np.ones((10, n)) * 0.01,
        }
        out = rollouts_from_trajectories([traj], featurizer, dt=0.05)
        assert len(out) == 1
        assert out[0].states.shape == (10, featurizer.state_dim)
        # No torques present, so accelerations stand in as the action.
        np.testing.assert_allclose(out[0].actions, traj["accelerations"])

    def test_rollouts_from_trajectories_reports_a_missing_key(self, featurizer) -> None:
        from ManipulaPy.jepa.data import rollouts_from_trajectories

        with pytest.raises(KeyError, match="positions"):
            rollouts_from_trajectories(
                [{"velocities": np.zeros((4, 6))}], featurizer, dt=0.01
            )


@requires_torch
class TestTrajectoryWindowDataset:
    def test_window_alignment(self, rollouts) -> None:
        """targets[k] must be the window ending at t + 1 + k, exactly.

        This is the test the whole module rests on: if the alignment slips by
        one the model learns to predict the state it was already given, the
        loss drops beautifully, and the world model is worthless.
        """
        from ManipulaPy.jepa.data import TrajectoryWindowDataset

        c, k = 3, 4
        ds = TrajectoryWindowDataset(rollouts, context_len=c, horizon=k)
        rollout = rollouts[0]
        t = c - 1  # the first anchor of the first rollout
        sample = ds[0]

        np.testing.assert_allclose(
            sample["context"].numpy(), rollout.states[t - c + 1 : t + 1], rtol=1e-6
        )
        np.testing.assert_allclose(
            sample["actions"].numpy(), rollout.actions[t : t + k], rtol=1e-6
        )
        for j in range(k):
            np.testing.assert_allclose(
                sample["targets"][j].numpy(),
                rollout.states[t + 2 + j - c : t + 2 + j],
                rtol=1e-6,
            )
            # The last row of target j is the state reached at t + 1 + j.
            np.testing.assert_allclose(
                sample["targets"][j][-1].numpy(),
                rollout.states[t + 1 + j],
                rtol=1e-6,
            )

    def test_length_matches_the_available_anchors(self, rollouts) -> None:
        from ManipulaPy.jepa.data import TrajectoryWindowDataset

        c, k = 3, 4
        ds = TrajectoryWindowDataset(rollouts, context_len=c, horizon=k)
        expected = sum(max(len(r) - k - c + 1, 0) for r in rollouts)
        assert len(ds) == expected

    def test_never_reads_past_the_end_of_a_rollout(self, rollouts) -> None:
        from ManipulaPy.jepa.data import TrajectoryWindowDataset

        ds = TrajectoryWindowDataset(rollouts, context_len=3, horizon=4)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["targets"].shape == (4, 3, ds.state_dim)
            assert bool(sample["targets"].isfinite().all())

    def test_rejects_rollouts_that_are_too_short(self, rollouts) -> None:
        from ManipulaPy.jepa.data import TrajectoryWindowDataset

        with pytest.raises(ValueError, match="long enough"):
            TrajectoryWindowDataset(rollouts, context_len=8, horizon=64)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@requires_torch
class TestMasking:
    def test_joint_mask_hides_exactly_that_joint(self) -> None:
        import torch

        from ManipulaPy.jepa.masking import joint_mask_to_feature_mask

        joint_mask = torch.zeros(1, 6, dtype=torch.bool)
        joint_mask[0, 2] = True
        out = joint_mask_to_feature_mask(joint_mask, num_joints=6, state_dim=27)
        assert out.shape == (1, 27)
        assert out[0].sum().item() == 3
        assert bool(out[0, 2]) and bool(out[0, 8]) and bool(out[0, 14])

    def test_ee_features_are_never_masked_by_joint_masks(self) -> None:
        from ManipulaPy.jepa.masking import sample_joint_mask

        mask = sample_joint_mask(64, 3, 6, 27, prob=1.0, generator=None)
        assert bool(mask[..., 18:].any()) is False
        assert bool(mask[..., :18].all()) is True

    def test_zero_probability_masks_nothing(self) -> None:
        from ManipulaPy.jepa.masking import sample_joint_mask

        mask = sample_joint_mask(8, 3, 6, 27, prob=0.0)
        assert mask.shape == (8, 3, 27)
        assert not bool(mask.any())

    def test_shared_masks_are_constant_across_the_window(self) -> None:
        import torch

        from ManipulaPy.jepa.masking import sample_joint_mask

        mask = sample_joint_mask(
            32,
            4,
            6,
            27,
            prob=0.5,
            share_across_time=True,
            generator=torch.Generator().manual_seed(0),
        )
        for step in range(1, 4):
            assert bool(torch.equal(mask[:, 0], mask[:, step]))

    def test_time_mask_hides_a_contiguous_span(self) -> None:
        import torch

        from ManipulaPy.jepa.masking import sample_time_mask

        mask = sample_time_mask(
            16, 5, 27, span=2, generator=torch.Generator().manual_seed(0)
        )
        per_step = mask.any(dim=-1)
        assert bool((per_step.sum(dim=1) == 2).all())
        for row in per_step:
            hidden = row.nonzero().flatten()
            assert int(hidden[-1] - hidden[0]) == len(hidden) - 1

    def test_time_mask_always_leaves_a_visible_step(self) -> None:
        from ManipulaPy.jepa.masking import sample_time_mask

        mask = sample_time_mask(4, 3, 27, span=99)
        assert bool((~mask.any(dim=-1)).any(dim=1).all())


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


@requires_torch
class TestLosses:
    def test_perfect_prediction_costs_nothing(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import prediction_loss

        z = torch.randn(4, 3, 8)
        for kind in ("smooth_l1", "mse", "cosine"):
            assert float(prediction_loss(z, z.clone(), kind=kind)) == pytest.approx(
                0.0, abs=1e-6
            )

    def test_discount_shifts_weight_toward_the_near_term(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import prediction_loss

        target = torch.zeros(2, 3, 4)
        near = torch.zeros(2, 3, 4)
        near[:, 0] = 1.0  # error at the first step
        far = torch.zeros(2, 3, 4)
        far[:, -1] = 1.0  # same error, at the last step

        undiscounted_near = float(prediction_loss(near, target, discount=1.0))
        undiscounted_far = float(prediction_loss(far, target, discount=1.0))
        assert undiscounted_near == pytest.approx(undiscounted_far)

        assert float(prediction_loss(near, target, discount=0.5)) > float(
            prediction_loss(far, target, discount=0.5)
        )

    def test_targets_receive_no_gradient(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import prediction_loss

        predicted = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 4, requires_grad=True)
        prediction_loss(predicted, target).backward()
        assert predicted.grad is not None
        assert target.grad is None

    def test_variance_loss_punishes_a_constant_embedding(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import variance_loss

        collapsed = torch.ones(64, 8)
        spread = torch.randn(64, 8) * 3.0
        # A constant embedding has zero variance, so the hinge saturates at
        # 1 - sqrt(eps) rather than at exactly 1.
        assert float(variance_loss(collapsed, eps=1e-4)) == pytest.approx(
            1.0 - 1e-2, abs=1e-5
        )
        assert float(variance_loss(spread)) == pytest.approx(0.0, abs=1e-6)

    def test_covariance_loss_punishes_duplicated_dimensions(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import covariance_loss

        base = torch.randn(256, 1)
        duplicated = base.repeat(1, 6)
        independent = torch.randn(256, 6)
        assert float(covariance_loss(duplicated)) > float(covariance_loss(independent))

    def test_jepa_loss_reports_its_components(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import jepa_loss

        out = jepa_loss(torch.randn(4, 3, 8), torch.randn(4, 3, 8))
        assert set(out) == {"loss", "prediction", "variance", "covariance"}
        assert float(out["loss"]) == pytest.approx(
            float(out["prediction"])
            + 1.0 * float(out["variance"])
            + 0.04 * float(out["covariance"]),
            rel=1e-5,
        )

    def test_shape_mismatch_is_reported(self) -> None:
        import torch

        from ManipulaPy.jepa.losses import prediction_loss

        with pytest.raises(ValueError, match="shape mismatch"):
            prediction_loss(torch.randn(2, 3, 4), torch.randn(2, 3, 5))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@requires_torch
class TestManipulatorJEPA:
    def _model(self, **overrides):
        from ManipulaPy.jepa.model import ManipulatorJEPA

        return ManipulatorJEPA(_small_config(27, 6, **overrides))

    def test_predictor_starts_as_the_identity(self) -> None:
        """A zero-initialized residual keeps the first steps well conditioned."""
        import torch

        model = self._model()
        z0 = torch.randn(4, model.config.embed_dim)
        actions = torch.randn(4, 2, 6)
        rolled = model.predictor.rollout(z0, actions)
        expected = torch.nn.functional.layer_norm(z0, (model.config.embed_dim,))
        torch.testing.assert_close(rolled[:, 0], expected, atol=1e-5, rtol=1e-5)

    def test_target_encoder_is_frozen(self) -> None:
        import torch

        model = self._model()
        assert not any(p.requires_grad for p in model.target_encoder.parameters())

        before = [p.detach().clone() for p in model.target_encoder.parameters()]
        out = model(
            torch.randn(4, 3, 27), torch.randn(4, 4, 6), torch.randn(4, 4, 3, 27)
        )
        out["loss"].backward()
        assert all(p.grad is None for p in model.target_encoder.parameters())
        for old, new in zip(before, model.target_encoder.parameters()):
            torch.testing.assert_close(old, new.detach())

    def test_ema_update_moves_the_target_toward_the_context_encoder(self) -> None:
        import torch

        model = self._model()
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p))

        online = [p.detach().clone() for p in model.context_encoder.parameters()]
        target = [p.detach().clone() for p in model.target_encoder.parameters()]
        model.update_target(0.9)
        for o, t_old, t_new in zip(online, target, model.target_encoder.parameters()):
            torch.testing.assert_close(
                t_new.detach(), 0.9 * t_old + 0.1 * o, atol=1e-6, rtol=1e-5
            )

    def test_ema_momentum_zero_copies_the_context_encoder(self) -> None:
        import torch

        model = self._model()
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p))
        model.update_target(0.0)
        for o, t in zip(
            model.context_encoder.parameters(), model.target_encoder.parameters()
        ):
            torch.testing.assert_close(o.detach(), t.detach())

    def test_rejects_an_out_of_range_momentum(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="momentum"):
            model.update_target(1.5)

    def test_masked_and_unmasked_contexts_embed_differently(self) -> None:
        import torch

        model = self._model().eval()
        states = torch.randn(2, 3, 27)
        mask = torch.zeros(2, 3, 27, dtype=torch.bool)
        mask[:, :, 0] = True
        assert not torch.allclose(model.encode(states), model.encode(states, mask))

    def test_mask_hides_the_value_it_covers(self) -> None:
        """Two states differing only under the mask must embed identically."""
        import torch

        model = self._model().eval()
        a = torch.randn(1, 3, 27)
        b = a.clone()
        mask = torch.zeros(1, 3, 27, dtype=torch.bool)
        mask[:, :, 4] = True
        b[:, :, 4] = 99.0
        torch.testing.assert_close(
            model.encode(a, mask), model.encode(b, mask), atol=1e-6, rtol=1e-5
        )

    def test_normalizer_standardizes_the_fitted_data(self) -> None:
        import torch

        model = self._model()
        rng = np.random.default_rng(0)
        states = rng.normal(loc=5.0, scale=3.0, size=(500, 27))
        model.fit_normalizer(states)
        normalized = model.normalize_states(
            torch.as_tensor(states, dtype=torch.float32)
        )
        assert float(normalized.mean().abs()) < 1e-4
        assert float(normalized.std()) == pytest.approx(1.0, abs=1e-2)

    def test_normalizer_survives_a_constant_feature(self) -> None:
        model = self._model()
        states = np.zeros((32, 27))
        model.fit_normalizer(states)
        assert bool(np.all(np.isfinite(model.state_std.numpy())))
        assert float(model.state_std.min()) > 0.0

    def test_predict_accepts_a_horizon_beyond_the_trained_one(self) -> None:
        import torch

        model = self._model().eval()
        long_actions = torch.randn(2, model.config.horizon + 5, 6)
        out = model.predict(torch.randn(2, 3, 27), long_actions)
        assert out.shape == (2, model.config.horizon + 5, model.config.embed_dim)

    def test_single_window_is_treated_as_a_batch_of_one(self) -> None:
        import torch

        model = self._model().eval()
        assert model.encode(torch.randn(3, 27)).shape == (
            1,
            model.config.embed_dim,
        )

    def test_reports_a_shape_that_contradicts_the_config(self) -> None:
        import torch

        model = self._model()
        with pytest.raises(ValueError, match="does not match config"):
            model.encode(torch.randn(2, 9, 27))

    def test_save_and_load_round_trip(self, tmp_path) -> None:
        import torch

        from ManipulaPy.jepa.model import ManipulatorJEPA

        model = self._model().eval()
        model.fit_normalizer(np.random.default_rng(0).normal(size=(64, 27)))
        path = tmp_path / "jepa.pt"
        model.save(str(path))

        restored = ManipulatorJEPA.load(str(path))
        assert restored.config == model.config
        states = torch.randn(2, 3, 27)
        torch.testing.assert_close(model.encode(states), restored.encode(states))

    def test_masking_is_off_in_eval_mode(self) -> None:
        """Evaluation must see the whole context, or its numbers are noise."""
        import torch

        model = self._model(joint_mask_prob=1.0).eval()
        ctx = torch.randn(4, 3, 27)
        act = torch.randn(4, 4, 6)
        tgt = torch.randn(4, 4, 3, 27)
        with torch.no_grad():
            a = model(ctx, act, tgt)["predicted"]
            b = model(ctx, act, tgt, apply_random_mask=False)["predicted"]
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Training and planning
# ---------------------------------------------------------------------------


@requires_torch
class TestTraining:
    def test_ema_schedule_anneals_between_its_endpoints(self) -> None:
        from ManipulaPy.jepa.training import ema_momentum

        assert ema_momentum(0, 100, 0.99, 1.0) == pytest.approx(0.99)
        assert ema_momentum(99, 100, 0.99, 1.0) == pytest.approx(1.0)
        mid = ema_momentum(50, 101, 0.99, 1.0)
        assert 0.99 < mid < 1.0
        # Degenerate schedules fall back to the final value rather than divide
        # by zero.
        assert ema_momentum(0, 1, 0.9, 1.0) == pytest.approx(1.0)

    def test_training_reduces_the_prediction_loss_without_collapsing(
        self, rollouts
    ) -> None:
        """The two things that must hold together: it learns, and it doesn't
        collapse. Either alone is easy and meaningless."""
        from ManipulaPy.jepa.data import TrajectoryWindowDataset
        from ManipulaPy.jepa.model import ManipulatorJEPA
        from ManipulaPy.jepa.training import JEPATrainer, make_dataloaders

        ds = TrajectoryWindowDataset(rollouts, context_len=3, horizon=4)
        model = ManipulatorJEPA(_small_config(ds.state_dim, ds.action_dim))
        model.fit_normalizer(ds.all_states(), ds.all_actions())
        train_loader, val_loader = make_dataloaders(
            ds, batch_size=16, val_fraction=0.2, seed=0
        )

        trainer = JEPATrainer(model, lr=3e-3, device="cpu")
        history = trainer.fit(train_loader, epochs=12, val_loader=val_loader)

        assert history["train_prediction"][-1] < history["train_prediction"][0]
        assert len(history["val_loss"]) == 12
        verdict = trainer.collapse_check()
        assert verdict["collapsed"] is False
        assert verdict["embedding_std"] > 1e-2

    def test_trainer_round_trips_through_a_checkpoint(self, rollouts, tmp_path) -> None:
        import torch

        from ManipulaPy.jepa.data import TrajectoryWindowDataset
        from ManipulaPy.jepa.model import ManipulatorJEPA
        from ManipulaPy.jepa.training import JEPATrainer, make_dataloaders

        ds = TrajectoryWindowDataset(rollouts, context_len=3, horizon=4)
        cfg = _small_config(ds.state_dim, ds.action_dim)
        model = ManipulatorJEPA(cfg)
        loader, _ = make_dataloaders(ds, batch_size=16, val_fraction=0.0)

        trainer = JEPATrainer(model, device="cpu")
        trainer.fit(loader, epochs=1)
        path = tmp_path / "trainer.pt"
        trainer.save(str(path))

        fresh = JEPATrainer(ManipulatorJEPA(cfg), device="cpu")
        fresh.load(str(path))
        assert fresh.global_step == trainer.global_step
        states = torch.randn(2, 3, ds.state_dim)
        torch.testing.assert_close(
            trainer.model.encode(states), fresh.model.encode(states)
        )

    def test_make_dataloaders_without_validation(self, rollouts) -> None:
        from ManipulaPy.jepa.data import TrajectoryWindowDataset
        from ManipulaPy.jepa.training import make_dataloaders

        ds = TrajectoryWindowDataset(rollouts, context_len=3, horizon=4)
        train_loader, val_loader = make_dataloaders(ds, val_fraction=0.0)
        assert val_loader is None
        assert len(train_loader.dataset) == len(ds)


@requires_torch
class TestLatentMPCPlanner:
    def _planner_and_model(self, rollouts, **overrides):
        from ManipulaPy.jepa.data import TrajectoryWindowDataset
        from ManipulaPy.jepa.model import ManipulatorJEPA
        from ManipulaPy.jepa.planning import LatentMPCPlanner
        from ManipulaPy.jepa.training import JEPATrainer, make_dataloaders

        ds = TrajectoryWindowDataset(rollouts, context_len=3, horizon=4)
        model = ManipulatorJEPA(_small_config(ds.state_dim, ds.action_dim))
        model.fit_normalizer(ds.all_states(), ds.all_actions())
        loader, _ = make_dataloaders(ds, batch_size=16, val_fraction=0.0)
        JEPATrainer(model, lr=3e-3, device="cpu").fit(loader, epochs=6)

        params = dict(
            horizon=4,
            num_samples=64,
            num_elites=8,
            num_iterations=3,
            action_low=-2.0,
            action_high=2.0,
            seed=0,
        )
        params.update(overrides)
        return LatentMPCPlanner(model, **params), model, ds

    def test_plan_shape_and_bounds(self, rollouts) -> None:
        planner, _, ds = self._planner_and_model(rollouts)
        plan = planner.plan(rollouts[0].states[:3], rollouts[1].states[5:8])
        assert plan["actions"].shape == (4, ds.action_dim)
        assert plan["actions"].dtype == np.float64
        assert plan["actions"].min() >= -2.0 - 1e-6
        assert plan["actions"].max() <= 2.0 + 1e-6
        assert plan["predicted_embeddings"].shape == (4, 16)
        assert np.isfinite(plan["cost"])

    def test_plan_beats_random_sampling(self, rollouts) -> None:
        """CEM must actually search: its plan should reach the goal embedding
        more closely than the mean random sequence it started from."""
        import torch

        planner, model, _ = self._planner_and_model(rollouts)
        context = rollouts[0].states[:3]
        goal = rollouts[1].states[5:8]
        plan = planner.plan(context, goal)

        z0 = model.encode(
            torch.as_tensor(context, dtype=torch.float32), use_target=True
        )
        z_goal = planner.goal_embedding(goal)
        rng = np.random.default_rng(0)
        random_actions = torch.as_tensor(
            rng.uniform(-2.0, 2.0, size=(200, 4, model.config.action_dim)),
            dtype=torch.float32,
        )
        with torch.no_grad():
            rolled = model.predictor.rollout(
                z0.expand(200, -1), model.normalize_actions(random_actions)
            )
        random_distance = torch.linalg.vector_norm(rolled[:, -1] - z_goal, dim=-1)
        assert plan["goal_distance"] < float(random_distance.mean())

    def test_planning_from_a_goal_embedding_matches_a_goal_window(
        self, rollouts
    ) -> None:
        planner, model, _ = self._planner_and_model(rollouts)
        goal_window = rollouts[1].states[5:8]
        embedding = planner.goal_embedding(goal_window)
        assert embedding.shape == (1, model.config.embed_dim)
        again = planner.goal_embedding(embedding[0])
        assert bool((embedding == again).all())

    def test_reset_clears_the_warm_start(self, rollouts) -> None:
        planner, _, _ = self._planner_and_model(rollouts)
        planner.plan(rollouts[0].states[:3], rollouts[1].states[5:8])
        assert planner._last_mean is not None
        planner.reset()
        assert planner._last_mean is None

    def test_rejects_more_elites_than_samples(self, rollouts) -> None:
        with pytest.raises(ValueError, match="num_elites"):
            self._planner_and_model(rollouts, num_samples=8, num_elites=16)

    def test_rejects_inverted_action_bounds(self, rollouts) -> None:
        with pytest.raises(ValueError, match="action_low"):
            self._planner_and_model(rollouts, action_low=3.0, action_high=-3.0)


# ---------------------------------------------------------------------------
# Package integration
# ---------------------------------------------------------------------------


@requires_torch
def test_module_is_reachable_from_the_package_namespace() -> None:
    import ManipulaPy

    assert ManipulaPy.ManipulatorJEPA is ManipulaPy.jepa.ManipulatorJEPA
    assert ManipulaPy.JEPAConfig is JEPAConfig
    assert "jepa" in dir(ManipulaPy)


@requires_torch
def test_public_names_are_exported() -> None:
    import ManipulaPy.jepa as jepa

    for name in jepa.__all__:
        assert hasattr(jepa, name), f"{name} is in __all__ but not exported"


def test_jepa_is_a_declared_feature() -> None:
    """The feature flag must exist whether or not Torch is installed."""
    import ManipulaPy

    assert "jepa" in ManipulaPy.check_dependencies(verbose=False)
