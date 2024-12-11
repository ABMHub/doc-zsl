import torch

class Config:
  def __init__(
    self,
    batch_size: int = 16,
    epochs: int = 1
  ):
    self.batch_size = batch_size
    self.optimizer = torch.optim.AdamW
    self.epochs = epochs
    self.lr = 1e-3
