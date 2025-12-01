import os
import torch
import threading

class Callback:
    def on_epoch_end(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

class ModelCheckpoint(Callback):
    def __init__(self, last_model_save_path: os.PathLike, best_model_save_path: os.PathLike):
        self.last_model_save_path = last_model_save_path
        self.best_model_save_path = best_model_save_path
        self.best_model_epoch = -1
        self.best_model_loss = 1e9

    def on_epoch_end(self, val_loss: float, epoch: int, model: torch.nn.Module, **kwargs):
        self.model_save(model, self.last_model_save_path)
        print("Last Model Saving on background")
        if self.best_model_loss > val_loss:
            self.best_model_loss = val_loss
            self.best_model_epoch = epoch
            self.model_save(model, self.best_model_save_path)
            print("Best Model Saving on background")

    def on_train_end(self, **kwargs):
        print(f"Best model on epoch {self.best_model_epoch}")

    @staticmethod
    def model_save(model, model_path):
        def save_model_background(model, path):
            torch.save(model, path)
            print("Model Saved")

        save_thread = threading.Thread(target=save_model_background, args=(model, model_path))
        save_thread.start()