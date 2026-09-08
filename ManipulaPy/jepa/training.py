#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
JEPA training loop - ManipulaPy

:class:`JEPATrainer` runs the optimizer, anneals the target-encoder momentum,
and reports the diagnostics that matter for a non-contrastive objective. The
loss going down is *not* sufficient evidence that a JEPA is learning: a
collapsed encoder that maps everything to one point also drives the prediction
term to zero. The trainer therefore tracks ``embedding_std`` every epoch, and
:meth:`JEPATrainer.collapse_check` turns it into a verdict.

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
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .model import ManipulatorJEPA

__all__ = ["JEPATrainer", "make_dataloaders", "ema_momentum"]

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("loss", "prediction", "variance", "covariance", "embedding_std")


def ema_momentum(step: int, total_steps: int, base: float, final: float) -> float:
    """Target-encoder momentum for a given training step.

    Annealed linearly from ``base`` to ``final``, as in I-JEPA: early on the
    target should track the online encoder closely enough to stay a useful
    regression goal, and late in training it should be nearly frozen so the
    predictor converges against a stationary target rather than chasing one.

    Args:
        step: Zero-based optimizer step.
        total_steps: Planned total number of steps. Values below one collapse
            the schedule to ``final``.
        base: Momentum at step zero.
        final: Momentum at the last step.

    Returns:
        Momentum in ``[min(base, final), max(base, final)]``.
    """
    if total_steps <= 1:
        return float(final)
    frac = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return float(base + (final - base) * frac)


def make_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    *,
    seed: Optional[int] = 0,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Split a dataset and wrap both halves in loaders.

    Args:
        dataset: Typically a
            :class:`~ManipulaPy.jepa.data.TrajectoryWindowDataset`.
        batch_size: Batch size for both loaders.
        val_fraction: Fraction held out for validation. ``0`` returns
            ``(train_loader, None)``.
        seed: Seed for the split. None uses Torch's global RNG.
        num_workers: DataLoader worker processes.
        shuffle: Shuffle the training loader.

    Returns:
        ``(train_loader, val_loader_or_None)``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    n_total = len(dataset)  # type: ignore[arg-type]
    n_val = int(round(n_total * val_fraction))
    if n_val == 0 or n_total - n_val < 1:
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            None,
        )

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    train_set, val_set = random_split(
        dataset, [n_total - n_val, n_val], generator=generator
    )
    return (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        ),
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


class JEPATrainer:
    """Optimize a :class:`~ManipulaPy.jepa.model.ManipulatorJEPA`.

    Args:
        model: The model to train. Moved to ``device``.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay. Applied to the context encoder and
            predictor only - the target encoder carries no gradient.
        grad_clip: Global gradient-norm clip, or None to disable.
        device: Torch device. Defaults to CUDA when available.
        cosine_schedule: Decay the learning rate along a cosine to zero over
            the planned step count.
        log_every: Log a line every N optimizer steps. ``0`` disables.

    Example:
        >>> trainer = JEPATrainer(model)                    # doctest: +SKIP
        >>> history = trainer.fit(train_loader, epochs=20)  # doctest: +SKIP
        >>> trainer.collapse_check()                        # doctest: +SKIP
    """

    def __init__(
        self,
        model: ManipulatorJEPA,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: Optional[float] = 1.0,
        device: Optional[Any] = None,
        cosine_schedule: bool = True,
        log_every: int = 0,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.grad_clip = grad_clip
        self.cosine_schedule = bool(cosine_schedule)
        self.log_every = int(log_every)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.lr, weight_decay=weight_decay
        )
        self.global_step = 0
        self.history: Dict[str, List[float]] = {}

    # -- training --------------------------------------------------------

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        epochs: int = 10,
        val_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        *,
        total_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Train for ``epochs`` passes over ``train_loader``.

        Args:
            train_loader: Yields batches with ``context``, ``actions`` and
                ``targets`` keys.
            epochs: Number of passes.
            val_loader: Optional held-out loader, evaluated after each epoch
                with masking and dropout off.
            total_steps: Planned optimizer steps, used by the momentum and
                learning-rate schedules. Inferred from ``len(train_loader) *
                epochs`` when the loader has a length.
            verbose: Print one summary line per epoch.

        Returns:
            The accumulated history: one entry per epoch under ``train_*`` and
            (when validating) ``val_*`` keys.
        """
        if total_steps is None:
            try:
                total_steps = len(train_loader) * int(epochs)  # type: ignore
            except TypeError:
                total_steps = max(int(epochs), 1)
        total_steps = max(int(total_steps), 1)

        for epoch in range(int(epochs)):
            train_metrics = self._run_epoch(
                train_loader, total_steps=total_steps, train=True
            )
            self._record("train", train_metrics)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    val_loader, total_steps=total_steps, train=False
                )
                self._record("val", val_metrics)

            if verbose:
                line = (
                    f"epoch {epoch + 1}/{epochs}  "
                    f"train_loss={train_metrics['loss']:.5f}  "
                    f"pred={train_metrics['prediction']:.5f}  "
                    f"emb_std={train_metrics['embedding_std']:.4f}"
                )
                if val_metrics is not None:
                    line += f"  val_loss={val_metrics['loss']:.5f}"
                logger.info(line)
                print(line)

        return self.history

    def _run_epoch(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        total_steps: int,
        train: bool,
    ) -> Dict[str, float]:
        """One pass over ``loader``; returns sample-weighted mean metrics."""
        self.model.train(train)
        totals = {key: 0.0 for key in _METRIC_KEYS}
        seen = 0

        for batch in loader:
            context = batch["context"].to(self.device)
            actions = batch["actions"].to(self.device)
            targets = batch["targets"].to(self.device)
            size = context.shape[0]

            if train:
                out = self.model(context, actions, targets)
                self.optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self._apply_lr(total_steps)
                self.optimizer.step()
                self.model.update_target(
                    ema_momentum(
                        self.global_step,
                        total_steps,
                        self.model.config.ema_base,
                        self.model.config.ema_final,
                    )
                )
                self.global_step += 1
                if self.log_every and self.global_step % self.log_every == 0:
                    logger.info(
                        "step %d  loss=%.5f",
                        self.global_step,
                        out["loss"].detach().item(),
                    )
            else:
                with torch.no_grad():
                    out = self.model(context, actions, targets, apply_random_mask=False)

            for key in _METRIC_KEYS:
                totals[key] += out[key].detach().item() * size
            seen += size

        if seen == 0:
            raise ValueError("the data loader yielded no batches")
        return {key: totals[key] / seen for key in _METRIC_KEYS}

    def _apply_lr(self, total_steps: int) -> None:
        """Set the current learning rate from the cosine schedule."""
        if not self.cosine_schedule:
            return
        frac = min(max(self.global_step / float(total_steps), 0.0), 1.0)
        lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _record(self, split: str, metrics: Dict[str, float]) -> None:
        """Append one epoch's metrics to the history."""
        for key, value in metrics.items():
            self.history.setdefault(f"{split}_{key}", []).append(value)

    # -- evaluation ------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Compute the metrics on ``loader`` without training."""
        return self._run_epoch(loader, total_steps=1, train=False)

    def collapse_check(self, threshold: float = 1e-2) -> Dict[str, Any]:
        """Report whether the representation has collapsed.

        A joint-embedding objective has a degenerate optimum - map everything
        to a constant - so a falling loss is only good news if the embeddings
        still vary. This compares the last recorded per-dimension standard
        deviation of the target embeddings against ``threshold``.

        Args:
            threshold: Standard deviation below which the representation is
                called collapsed.

        Returns:
            Dict with ``collapsed`` (bool), ``embedding_std`` (float or None
            when nothing has been trained yet) and ``threshold``.
        """
        values = self.history.get("train_embedding_std", [])
        latest = values[-1] if values else None
        return {
            "collapsed": bool(latest is not None and latest < threshold),
            "embedding_std": latest,
            "threshold": float(threshold),
        }

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Write model, optimizer and schedule state to ``path``."""
        torch.save(
            {
                "config": self.model.config.to_dict(),
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "history": self.history,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore state written by :meth:`save` into this trainer."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        self.history = payload.get("history", {})
