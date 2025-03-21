import torch

class Config:
  def __init__(
    self,
    criterion,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    learning_rate: float = 1e-3,
    scheduler_patience: int = 7,
    momentum: float = 0,
    decay: float = 0,
    **kwargs
  ):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.criterion = criterion
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.kwargs = kwargs

  def config_dict(self):
    ret = {
      "learning_rate": self.learning_rate,
      "optimizer": self.optimizer.__class__.__name__,
      "criterion": self.criterion.__name__,
      "architecture": self.model.name,
      "scheduler": self.scheduler.__class__.__name__
    }
    ret.update(self.kwargs)
    return ret
