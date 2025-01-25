## Dependencies

# import pytesseract
# import torchmetrics
from accelerate import Accelerator
import numpy as np
import torch
import numpy as np
from tqdm.auto import tqdm
from loss import ContrastiveLoss
from log import Log

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

def train_epoch(epoch, data_loader, model, criterion, optimizer, device, log : Log):
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

    log.update_metric("loss", train=TRAIN, value=loss.detach().cpu())
    log.update_all_metrics(train=TRAIN, y_true=y.detach().cpu().numpy(), y_pred=out_cpu)
    loop.set_postfix(log.create_metrics_dict(TRAIN))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Train epoch {epoch}")
  pretty_print_dict(log.create_metrics_dict(TRAIN))

# Function for the validation data loader
def val_epoch(epoch, data_loader, model, criterion, device, log: Log):
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
      
  log.update_all_metrics(train=VAL, y_true=y_true, y_pred=y_pred_batch)
  # loop.set_postfix(log.create_metrics_dict(VAL))

  print(f"Val epoch {epoch}")
  pretty_print_dict(log.create_metrics_dict(VAL))

def train(
    config,
    train_dataloader,
    val_dataloader,
    model,
    device,
    log: Log,
    patience: int = None
  ):
  ea = EarlyStopping(patience)

  epochs = config.epochs
  optimizer = config.optimizer
  scheduler = config.scheduler
  model.to(device)

  try:
    for i in range(epochs):
      train_epoch(
        epoch=i,
        data_loader=train_dataloader,
        model=model,
        criterion=ContrastiveLoss,
        optimizer=optimizer,
        device=device,
        log=log
      )
      
      val_epoch(
        epoch=i,
        data_loader=val_dataloader,
        model=model,
        criterion=ContrastiveLoss,
        device=device,
        log=log
      )

      val_loss = log.val_metrics["loss"].get_current_info()

      if scheduler is not None:
        scheduler.step(val_loss)

      log.end_epoch()
      train_dataloader.on_epoch_end()
      val_dataloader.on_epoch_end()

      if ea.should_we_stop(val_loss, i):
        break

  except KeyboardInterrupt:
    print("Salvando modelo antes de encerrar...")
    raise

  finally:
    torch.save(model, "./model.pt")
    print("Modelo salvo.")
