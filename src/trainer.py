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

def pretty_print_dict(d: dict):
  for elem in d:
    print(f"\t{elem} - {d[elem]}")

def train_epoch(epoch, data_loader, model, criterion, optimizer, device, log : Log, scheduler=None):
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

      loss = criterion(*outputs, y)
      log.update_metric("loss", VAL, value=loss.detach().cpu())
      log.update_all_metrics(train=VAL, y_true=y.detach().cpu().numpy(), y_pred=out_cpu)  
      loop.set_postfix(log.create_metrics_dict(VAL))

  print(f"Val epoch {epoch}")
  pretty_print_dict(log.create_metrics_dict(VAL))

def train(
    config,
    train_dataloader,
    val_dataloader,
    model,
    device,
    log: Log
  ):

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
        log=log,
        scheduler=scheduler
      )
      
      val_epoch(
        epoch=i,
        data_loader=val_dataloader,
        model=model,
        criterion=ContrastiveLoss,
        device=device,
        log=log
      )

      if scheduler is not None:
        scheduler.step(log.val_metrics["loss"].get_current_info())

      log.end_epoch()
      train_dataloader.on_epoch_end()
      val_dataloader.on_epoch_end()
      
  except KeyboardInterrupt:
    torch.save(model, "./model.pt")
    raise
