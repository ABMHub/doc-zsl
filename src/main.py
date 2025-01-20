from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER, LR
from loss import ContrastiveLoss
import torch

img_shape = (224, 224)
out_dim = 64
batch_size = 16
shuffle_loader = True
learning_rate = 1e-3
patience = 7

model = CCT(out_dim=out_dim, img_shape=img_shape)
model = SiameseModel(model)

config = Config(
  batch_size=batch_size,
  criterion=ContrastiveLoss,
  model = model,
  shuffle_loader = True,
  epochs=1000,
  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
  learning_rate=learning_rate,
  img_shape = img_shape,
  optimizer=torch.optim.Adam,
  patience=patience
)

csv_path = "./splits.csv"

train_loader = DocDataset(csv_path, train=True, load_in_ram=True, img_shape=img_shape)
val_loader = DocDataset(csv_path, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std)

train_loader = ContrastivePairLoader(train_loader)
val_loader = ContrastivePairLoader(val_loader)

train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
val_loader = DataLoader(val_loader, batch_size, shuffle=False)

wandb_args = {
  "project": "mestrado-comparadora",
  "name": "CCT R31 5k",
  "notes": "Primeiro modelo a ser treinado.",
  "config": config.config_dict()
}

log = Log(wandb_flag=True, wandb_args=wandb_args)
log.create_metric("eer", EER(), True)
log.create_metric("eer", EER(), False)
log.create_metric("lr", LR(config.scheduler), True)

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log,
  patience = patience,
)