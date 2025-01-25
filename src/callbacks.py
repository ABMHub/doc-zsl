import os
import torch

class Callback:
    def on_epoch_end(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

class ModelCheckpoint(Callback):
    def __init__(self, model_save_path: os.PathLike):
        self.model_save_path = model_save_path
        self.best_model_epoch = -1
        self.best_model_loss = 1e9

    def on_epoch_end(self, val_loss: float, epoch: int, model: torch.nn.Module, **kwargs):
        if self.best_model_loss > val_loss:
            self.best_model_loss = val_loss
            self.best_model_epoch = epoch
            torch.save(model, self.model_save_path)
            print("Best Model Saved")

    def on_train_end(self, **kwargs):
        print(f"Best model on epoch {self.best_model_epoch}")
