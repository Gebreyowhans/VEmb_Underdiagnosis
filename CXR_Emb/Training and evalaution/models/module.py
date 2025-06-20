"""Classification LightningModule wrapper.

This module cleans styling and comments of an existing implementation while
keeping its public interface and behaviour intact.
"""

import os
import warnings
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from metrics.metrics import calculate_roc_auc, calculate_fpr_fnr, find_best_threshold


class CLS(L.LightningModule):
    """A generic PyTorch‑Lightning module for classification tasks.

    Args:
        model: Backbone network that returns *logits*.
        criterion: Loss function (defaults to :class:`torch.nn.CrossEntropyLoss`).
        lr: Learning‑rate for ``Adam`` optimiser.
        weight_decay: Weight‑decay for ``Adam`` optimiser.
        prediction_on: Dataset split used by ``predict_step``.
        save_probabilities_path: Optional directory where probabilities & labels
            are persisted at the end of ``predict``.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        prediction_on: str = "test",
        save_probabilities_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        print("CLS init", "*" * 50)

        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.prediction_on = prediction_on
        self.save_probabilities_path = save_probabilities_path

        # Buffers for per‑epoch statistics
        self.stages: Dict[str, Dict[str, List]] = {
            "train": {"loss": [], "labels": [], "probabilities": []},
            "val": {"loss": [], "labels": [], "probabilities": []},
            "test": {"loss": [], "labels": [], "probabilities": []},
        }

    # ---------------------------------------------------------------------
    # Mini‑batch steps
    # ---------------------------------------------------------------------
    def forward(self, x):  # noqa: D401, N802
        """Delegate to the wrapped *model*."""
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: D401, N802
        """Single optimisation step on *train* split."""
        data, target = batch["data"].to(self.device), batch["labels"].to(self.device)
        output = self.model(data.float().squeeze())
        loss = self.criterion(output, target)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):  # noqa: D401, N802
        """Forward‑pass and metric accumulation for *val* split."""
        loss = self._shared_eval(batch, stage="val")
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):  # noqa: D401, N802
        """Forward‑pass and metric accumulation for *test* split."""
        loss = self._shared_eval(batch, stage="test")
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):  # noqa: D401, N802
        """Store probabilities & labels for use after ``trainer.predict``."""
        data, target = batch["data"].to(self.device), batch["labels"].to(self.device)
        output = self.model(data.float().squeeze())
        loss = self.criterion(output, target)
        self._append_data(output, target, loss, stage=self.prediction_on)

    # ---------------------------------------------------------------------
    # Epoch‑end hooks
    # ---------------------------------------------------------------------
    def on_predict_epoch_end(self):  # noqa: D401, N802
        """Aggregate losses & optionally save probabilities after *predict*."""
        mean_loss = np.mean([t.cpu().numpy() for t in self.stages[self.prediction_on]["loss"]])
        print(f"loss on {self.prediction_on} set = {mean_loss}")

        # Persist results if requested
        if self.save_probabilities_path:
            self._save_probabilities_and_labels(
                self.stages[self.prediction_on]["probabilities"],
                self.stages[self.prediction_on]["labels"],
                self.save_probabilities_path,
                stage=self.prediction_on,
            )

        # Reset buffers
        self.stages[self.prediction_on] = {"loss": [], "labels": [], "probabilities": []}

    def on_test_epoch_end(self):  # noqa: D401, N802
        """Compute & report metrics for the *test* split."""
        avg_loss = np.mean(self.stages["test"]["loss"])
        _, _, class_auc = calculate_roc_auc(
            self.stages["test"]["probabilities"], self.stages["test"]["labels"]
        )
        average_auc = sum(class_auc.values()) / len(class_auc)

        print("-" * 50)
        print(class_auc)
        print(f"Test loss = {avg_loss}")
        print(f"Average AUC = {average_auc}")
        self.stages["test"] = {"loss": [], "labels": [], "probabilities": []}

    def on_validation_epoch_end(self):  # noqa: D401, N802
        """Compute & report metrics for the *validation* split."""
        avg_loss = np.mean(self.stages["val"]["loss"])
        _, _, class_auc = calculate_roc_auc(
            self.stages["val"]["probabilities"], self.stages["val"]["labels"]
        )
        average_auc = sum(class_auc.values()) / len(class_auc)

        print("-" * 50)
        print(class_auc)
        print(f"Validation loss = {avg_loss}")
        print(f"Average AUC = {average_auc}")
        self.stages["val"] = {"loss": [], "labels": [], "probabilities": []}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _shared_eval(self, batch, stage: str = "val"):  # noqa: D401, N802
        """Evaluation logic shared by *validation* and *test* loops."""
        data, target = batch["data"].to(self.device), batch["labels"].to(self.device)
        output = self.model(data.float())
        loss = self.criterion(output, target).item()
        self._append_data(output, target, loss, stage=stage)
        return loss

    def _append_data(self, output, target, loss, stage: str = "test"):  # noqa: D401, N802
        """Accumulate batch‑level statistics in stage buffer."""
        stage_data = self.stages.get(stage)
        if stage_data is None:
            warnings.warn(f"Invalid stage: {stage}. No data will be appended.")
            return

        stage_data["loss"].append(loss)
        stage_data["labels"].extend(target.cpu().numpy())

        if isinstance(self.criterion, nn.CrossEntropyLoss):
            probs = torch.softmax(output, dim=1)
        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            probs = torch.sigmoid(output)
        else:
            warnings.warn(
                f"{self.criterion.__class__.__name__} not explicitly supported; "
                "returning raw outputs as probabilities."
            )
            probs = output

        stage_data["probabilities"].extend(probs.cpu().numpy())

    def _save_probabilities_and_labels(
        self,
        probabilities,
        labels,
        save_path: str,
        *,
        stage: str = "train",
    ) -> None:  # noqa: D401, N802
        """Serialize probabilities & labels for downstream analysis."""
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"probabilities_{stage}.npy"), np.array(probabilities))
        np.save(os.path.join(save_path, f"labels_{stage}.npy"), np.array(labels))

    # ---------------------------------------------------------------------
    # Optimiser
    # ---------------------------------------------------------------------
    def configure_optimizers(self):  # noqa: D401, N802
        """Instantiate optimiser used by PyTorch‑Lightning trainer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
