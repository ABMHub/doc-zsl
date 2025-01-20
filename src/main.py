from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER
from loss import ContrastiveLoss
import torch

img_shape = (100, 150)
out_dim = 10
batch_size = 16
shuffle_loader = True

model = CCT(out_dim=out_dim, img_shape=img_shape)
model = SiameseModel(model)

config = Config(
  batch_size=batch_size,
  criterion=ContrastiveLoss,
  model = model,
  shuffle_loader = True,
  epochs=1000,
  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
  learning_rate=1e-3,
  img_shape = img_shape,
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
  "name": "CCT R29 5k",
  "notes": "Primeiro modelo a ser treinado.",
  "config": config.config_dict()
}

log = Log(wandb_flag=True, wandb_args=wandb_args)
log.create_metric("eer", EER(), True)
log.create_metric("eer", EER(), False)

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log
)