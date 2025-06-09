import lightning as L
import torch.nn
import numpy as np
import torch.nn as nn
import warnings
import torch.nn.functional as F
from metrics.metrics import calculate_roc_auc, calculate_fpr_fnr, find_best_threshold
import os


class CLS(L.LightningModule):

    def __init__(self, model: nn.Module,
                 criterion=nn.CrossEntropyLoss(),
                 lr=0.0005,
                 weight_decay=0.0005,
                 prediction_on="test",
                 save_probabilities_path=None):
        super().__init__()
        print("CLS init", "*"*50)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.prediction_on = prediction_on
        self.save_probabilities_path = save_probabilities_path

        self.criterion = criterion

        self.stages = {"test": {"loss": [], "labels": [], "probabilities": []},
                       "val": {"loss": [], "labels": [], "probabilities": []},
                       "train": {"loss": [], "labels": [], "probabilities": []}}

    def training_step(self, batch, batch_idx):

        data, target = batch['data'].to(
            self.device), batch['labels'].to(self.device)

        output = self.model(data.float().squeeze())
        loss = self.criterion(output, target)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        loss = self.share_val_test(batch, stage="val")
        self.log('val_loss', loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):

        loss = self.share_val_test(batch, stage="test")
        self.log('test_loss', loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):

        data, target = batch['data'].to(
            self.device), batch['labels'].to(self.device)
        output = self.model(data.float().squeeze())
        loss = self.criterion(output, target)
        self.append_data(output, target, loss, stage=self.prediction_on)

    def on_predict_epoch_end(self, results):

        # Assuming self.stages[self.prediction_on]["loss"] is a PyTorch tensor on GPU
        # Assuming self.stages[self.prediction_on]["loss"] is a list of PyTorch tensors on GPU
        loss_list = self.stages[self.prediction_on]["loss"]

        # Move each tensor to CPU and convert to NumPy array
        loss_numpy_list = [tensor.cpu().numpy() for tensor in loss_list]

        # Compute the mean of the NumPy array
        mean_loss = np.mean(loss_numpy_list)

        print("loss on {} set = {}".format(self.prediction_on, mean_loss))
        # save probabilities and labels
        if self.save_probabilities_path is not None:
            self.save_probabilities_and_labels(self.stages[self.prediction_on]["probabilities"],
                                               self.stages[self.prediction_on]["labels"], self.save_probabilities_path, stage=self.prediction_on)

        # clean stage data
        self.stages[self.prediction_on] = {
            "loss": [], "labels": [], "probabilities": []}

    def on_test_epoch_end(self):

        avg_loss = np.mean(self.stages["test"]["loss"])

        _, _, class_auc = calculate_roc_auc(
            self.stages["test"]["probabilities"], self.stages["test"]["labels"])
        average_auc = sum(class_auc.values()) / len(class_auc)

        print("-"*50)
        print(class_auc)
        print("Test loss = {}".format(avg_loss))
        print("Average AUC = {}".format(average_auc))

        # clean stage data
        self.stages["test"] = {"loss": [], "labels": [], "probabilities": []}

    def on_validation_epoch_end(self):

        avg_loss = np.mean(self.stages["val"]["loss"])

        _, _, class_auc = calculate_roc_auc(
            self.stages["val"]["probabilities"], self.stages["val"]["labels"])
        print(class_auc)
        # thresholds_test = find_best_threshold(self.probabilities_val_test, self.labels_val_test)
        # print("Best thresholds for validation set: {}".format(thresholds_test))
        # fpr_numbers_test, fnr_numbers_test = calculate_fpr_fnr(self.probabilities_val_test, self.labels_val_test, thresholds_test)

        # # | Class | FPR    | FNR    |
        # # |-------|--------|--------|
        # # |   0   |  0.00  |  0.00  |

        # print("| Class | FPR    | FNR    |")
        # print("|-------|--------|--------|")
        # for i in range(len(fpr_numbers_test)):

        #     print("|   {}   |   {:.3f}  |   {:.3f}  |".format(i, fpr_numbers_test[i], fnr_numbers_test[i]))

        average_auc = sum(class_auc.values()) / len(class_auc)

        print("-"*50)
        print("Validation loss = {}".format(avg_loss))
        print("Average AUC = {}".format(average_auc))

        # clean stage data
        self.stages["val"] = {"loss": [], "labels": [], "probabilities": []}

    def share_val_test(self, batch, stage="val"):

        data, target = batch['data'].to(
            self.device), batch['labels'].to(self.device)
        output = self.model(data.float())

        loss = self.criterion(output, target).item()
        self.append_data(output, target, loss, stage=stage)

        return loss

    def append_data(self, output, target, loss, stage="test"):
        stage_data = self.stages.get(stage)
        if stage_data:
            stage_data["loss"].append(loss)
            stage_data["labels"].extend(target.data.cpu().numpy())

            if isinstance(self.criterion, nn.CrossEntropyLoss):
                stage_data["probabilities"].extend(
                    torch.softmax(output, dim=1).data.cpu().numpy())
            elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
                stage_data["probabilities"].extend(
                    torch.sigmoid(output).data.cpu().numpy())
            else:
                warnings.warn(
                    f"Loss is not supported for {stage} step. No probabilities will be returned.")
                stage_data["probabilities"].extend(output.data.cpu().numpy())
        else:
            warnings.warn(f"Invalid stage: {stage}. No data will be appended.")

    def save_probabilities_and_labels(self, probabilities, labels, save_path, stage="train"):
        os.makedirs(save_path, exist_ok=True)

        # Convert to numpy and save
        probabilities_np = np.array(probabilities)
        labels_np = np.array(labels)

        np.save(os.path.join(save_path, "probabilities_" +
                stage+".npy"), probabilities_np)
        np.save(os.path.join(save_path, "labels_"+stage+".npy"), labels_np)

    def configure_optimizers(self):

        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
