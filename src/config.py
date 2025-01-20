import torch

class Config:
  def __init__(
    self,
    criterion,
    model,
    optimizer,
    epochs: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    learning_rate = 1e-3,
    **kwargs
  ):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.criterion = criterion
    self.model = model
    self.optimizer = optimizer(model.parameters(), lr=self.learning_rate)
    self.scheduler = scheduler(self.optimizer, verbose=True, patience=3)
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
