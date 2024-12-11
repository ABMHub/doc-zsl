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


def train_epoch(data_loader, model, criterion, optimizer, device, log : Log, scheduler=None):
  model.train()
  accelerator = Accelerator()
  model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
  loop = tqdm(data_loader, postfix={"Loss": 0}, leave=True)

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

    log.update_metric("loss", train=True, value=loss.detach().cpu())
    log.update_all_metrics(train=True, y_true=y.detach().cpu().numpy(), y_pred=out_cpu)
    loop.set_postfix(log.create_metrics_dict(True))

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    if scheduler is not None:
      scheduler.step()

# Function for the validation data loader
def val_epoch(data_loader, model, criterion, device, log: Log):
  model.eval()
  epoch_loss = []

  y_pred = []
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
      out_cpu = [elem.detach().cpu().numpy() for elem in outputs] # resolver isso aqui.  ta dando problema no shape nas metricas
      y_pred.append(out_cpu)
      y_true.append(y.detach().cpu().numpy())

    loss = criterion(*outputs, y)
    log.update_metric("loss", False, value=loss.detach().cpu())


  # log.update_all_metrics(train=False, y_pred=y_pred, y_true=y_true)      

def train(config, train_dataloader, val_dataloader, model, device, log):
  epochs = config.epochs

  optimizer = config.optimizer(model.parameters(), lr=config.lr)

  for _ in range(epochs):
    print("Training the model.....")
    train_epoch(
      data_loader=train_dataloader,
      model=model,
      criterion=ContrastiveLoss,
      optimizer=optimizer,
      device=device,
      log=log,
      scheduler=None
    )
    
    print("Validating the model.....")
    val_epoch(
      data_loader=val_dataloader,
      model=model,
      criterion=ContrastiveLoss,
      device=device,
      log=log
    )

    log.end_epoch()

  torch.save(model, "./model.pt")
