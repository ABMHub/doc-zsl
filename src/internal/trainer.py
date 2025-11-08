import torch
from tqdm.auto import tqdm
import typing
from collections.abc import Callable
import numpy as np

from internal.loss import ContrastiveLoss
from internal.log import Log
from internal.callbacks import Callback
from internal.dataloader import DataLoader

TRAIN, VAL = True, False

class EarlyStopping:
  def __init__(self, patience: int):
    self.patience = patience
    self.best_loss = 1e8
    self.best_loss_epoch = None

  def should_we_stop(self, val_loss: float, current_epoch: int):
    if val_loss < self.best_loss:
      self.best_loss = val_loss
      self.best_loss_epoch = current_epoch

    elif (current_epoch - self.best_loss_epoch) > self.patience:
      print("Early Stopping Break")
      return True
    
    return False

def pretty_print_dict(d: dict):
  for elem in d:
    print(f"\t{elem} - {d[elem]}")

def train_epoch(epoch, data_loader, model, criterion, optimizer, device, log : Log, scheduler):
  model.train()
  # accelerator = Accelerator()
  # model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
  loop = tqdm(data_loader, postfix={"Loss": 0}, leave=False)

  for batch in loop:
    (x1, x2), y = batch
    x1 = x1.to(device)
    x2 = x2.to(device)
    x = (x1, x2)
    y = y.to(device)

    # process
    outputs = model(*x)
    loss = criterion(*outputs, y) # revisar

    out_cpu = [elem.detach().cpu().numpy() for elem in outputs]

    optimizer.zero_grad()
    loss.backward()

    try:
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    except RuntimeError:
      continue

    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    log.update_metric("loss", train=TRAIN, value=loss.detach().cpu())
    log.update_all_metrics(train=TRAIN, y_true=y.detach().cpu().numpy(), y_pred=out_cpu)
    loop.set_postfix(log.create_metrics_dict(TRAIN))

  print(f"Train epoch {epoch}")
  pretty_print_dict(log.create_metrics_dict(TRAIN))

# Function for the validation data loader
def val_epoch(
    epoch: int,
    data_loader: DataLoader,
    model: torch.nn.Module,
    criterion: Callable,
    device: str,
    log: Log
  ):
  model.eval()
  y_pred_batch = []
  y_true = []

  with torch.no_grad():
    loop = tqdm(data_loader, total=len(data_loader), leave=False)
    for batch in loop:
      (x1, x2), y = batch
      x1 = x1.to(device)
      x2 = x2.to(device)
      x = (x1, x2)
      y = y.to(device)

      # process
      outputs = model(*x)
      out_cpu = [elem.detach().cpu().numpy() for elem in outputs]

      y_pred_batch.append(out_cpu)
      y_true += y.detach().cpu().numpy().tolist()

      loss = criterion(*outputs, y)
      log.update_metric("loss", VAL, value=loss.detach().cpu())
      
  log.update_all_metrics(train=VAL, y_true=y_true, y_pred=y_pred_batch, df=data_loader.dataset.dataset.df)
  # loop.set_postfix(log.create_metrics_dict(VAL))

  print(f"Val epoch {epoch}")
  pretty_print_dict(log.create_metrics_dict(VAL))

def test_epoch(
    data_loader: DataLoader,
    model: torch.nn.Module,
    criterion: Callable,
    device: str,
    log: Log
  ):
  model.eval()
  y_pred_batch = []
  y_true = []
  test_loss = []

  with torch.no_grad():
    loop = tqdm(data_loader, total=len(data_loader), leave=False)
    for batch in loop:
      (x1, x2), y = batch
      x1 = x1.to(device)
      x2 = x2.to(device)
      x = (x1, x2)
      y = y.to(device)

      # process
      outputs = model(*x)
      out_cpu = [elem.detach().cpu().numpy() for elem in outputs]

      y_pred_batch.append(out_cpu)
      y_true += y.detach().cpu().numpy().tolist()

      loss = criterion(*outputs, y)
      test_loss.append(loss.detach().cpu())

  metrics = log.compute_all_metrics(y_true=y_true, y_pred=y_pred_batch, df=data_loader.dataset.dataset.df)
  metrics["loss"] = np.mean(test_loss)

  print("Test epoch done")
  pretty_print_dict(metrics)

  return metrics

def train(
    config,
    train_dataloader,
    val_dataloader,
    model: torch.nn.Module,
    device,
    log: Log,
    patience: int = None,
    callbacks: typing.List[Callback] = [],
    model_save_path: str = None,
    distance_metric: str = None
  ):
  ea = EarlyStopping(patience)

  epochs = config.epochs
  optimizer = config.optimizer
  scheduler = config.scheduler
  model.to(device)

  loss_f = ContrastiveLoss(margin=1, cosine_distance=distance_metric=="cosine")

  try:
    for i in range(epochs):
      train_epoch(
        epoch=i,
        data_loader=train_dataloader,
        model=model,
        criterion=loss_f,
        optimizer=optimizer,
        device=device,
        log=log,
        scheduler=scheduler
      )
      
      val_epoch(
        epoch=i,
        data_loader=val_dataloader,
        model=model,
        criterion=loss_f,
        device=device,
        log=log
      )

      val_loss = log.val_metrics["loss"].get_current_info()

      log.end_epoch()
      train_dataloader.on_epoch_end()
      val_dataloader.on_epoch_end()

      for callback in callbacks:
        callback.on_epoch_end(
          val_loss=val_loss,
          epoch=i,
          model=model,
        )

      if ea.should_we_stop(val_loss, i):
        break

    for callback in callbacks:
      callback.on_train_end(
        val_loss=val_loss,
        epoch=i,
        model=model,
      )

  except KeyboardInterrupt:
    print("Salvando modelo antes de encerrar...")
    raise

  finally:
    if model_save_path is not None:
      torch.save(model, model_save_path)
      print("Modelo salvo.")
