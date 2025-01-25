from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT, EfficientNet
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER, LR
from loss import ContrastiveLoss
import torch

dataset = "rvl_zsl_5k"
split = "zsl"  # overlap, zsl, gzsl

wandb_flag = True

img_shape = (256, 256)
out_dim = 64
batch_size = 16
shuffle_loader = True
learning_rate = 1e-5
patience = 9
n_channels = 3
model_version = 1

# model = CCT(out_dim=out_dim, img_shape=img_shape, model_version=model_version, n_input_channels=n_channels)
model = EfficientNet(64, model_version=model_version)
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
  patience=patience,
  n_channels=n_channels,
  model_version=model_version,
  dataset=dataset,
  split=split
)

csv_path = "./splits.csv"

train_loader = DocDataset(csv_path, train=True, load_in_ram=True, img_shape=img_shape, n_channels=n_channels)
val_loader = DocDataset(csv_path, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels)

train_loader = ContrastivePairLoader(train_loader)
val_loader = ContrastivePairLoader(val_loader)

train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
val_loader = DataLoader(val_loader, batch_size, shuffle=False)

wandb_args = {
  "project": "mestrado-comparadora",
  # "name": "EfficientNet R4 5k",
  # "name": f"CCT_{model_version} R1 5k ZSL",
  "name": f"EfficientNet_b{model_version} R2 5k ZSL",
  "notes": "Teste sobre versões da rede.",
  "config": config.config_dict()
}

log = Log(wandb_flag=wandb_flag, wandb_args=wandb_args)
log.create_metric("eer", EER, True)
log.create_metric("eer", EER, False)
log.create_metric("lr", LR, True, scheduler=config.scheduler)

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log,
  patience = patience,
)