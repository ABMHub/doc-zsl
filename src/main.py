from internal.trainer import train
from internal.config import Config
from internal.architecture import Vit, SiameseModel, CCT, EfficientNet, ResNet, DiT
from internal.dataloader import DocDataset, DataLoader, ContrastivePairLoader
from internal.log import Log
from internal.metrics import EER, LR, Identification
from internal.loss import ContrastiveLoss
from internal.callbacks import ModelCheckpoint

import torch
import pandas as pd
import os

dataset = "rvl_zsl_5k"
split_mode = "zsl"  # overlap, zsl, gzsl
split_number = 0

wandb_flag = False
load_in_ram = True

img_shape = (224, 224)
out_dim = 16
batch_size = 16
shuffle_loader = True
learning_rate = 1e-4
patience = 1000
n_channels = 3
model_version = 1
epochs = 80
momentum = 0.9
decay = 0

def mkdir(folder: os.PathLike):
  """Creates a new folder. Catches the `FileExistsError` from `os.mkdir`.

  Args:
      folder (os.PathLike): new folder path
  """
  try:
    os.mkdir(folder)
  except FileExistsError:
    pass

project_name = f"resnet_cluster"

# choose a model from src/architecture
model = ResNet(out_dim=64, model_version=34, pretrained=True)

# hyperparameters are stored in each model's class
learning_rate = model.learning_rate
decay = model.weight_decay
momentum = model.momentum
optimizer = model.optimizer
scheduler = model.scheduler
lr_gamma = model.lr_gamma

# if using imagenet pretrained weights, use (224, 224)
# otherwise, use whatever you feel like
img_shape = (model.im_shape, model.im_shape)
model = SiameseModel(model)

csv_path = f"./dataset/splits/train_{split_mode}.csv"  # csv containg the train split for zsl or gzsl
df = pd.read_csv(csv_path, index_col=0)

# test becomes 1, train becomes 0
# gambiarra
df.loc[df["split"] == split_number, "split"] = -1
df.loc[df["split"] > 0, "split"] = 0
df.loc[df["split"] == -1, "split"] = 1

# load the eval protocol
protocol_path = "./dataset/protocols/val_protocol.csv"
protocol = pd.read_csv(protocol_path)
protocol = protocol[(protocol["split_mode"] == f"{split_mode}_split") & (protocol["split_number"] == split_number)]

# import transformers

# processor = transformers.AutoImageProcessor.from_pretrained("microsoft/dit-base")
processor = None

# create file loaders
train_loader = DocDataset(df, train=True, load_in_ram=True, img_shape=img_shape, n_channels=n_channels, preprocessor=processor)
val_loader = DocDataset(df, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels, preprocessor=processor)

# adapt loaders to a contrastive loader schema
train_loader = ContrastivePairLoader(train_loader, None)
val_loader = ContrastivePairLoader(val_loader, protocol)

# uses torch dataloader class to create batches
train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
val_loader = DataLoader(val_loader, batch_size, shuffle=False)

config = Config(
  batch_size=batch_size,
  criterion=ContrastiveLoss,
  model = model,
  shuffle_loader = True,
  epochs=epochs,
  # scheduler=scheduler(optimizer, len(train_loader)*20, len(train_loader)*200),
  scheduler=scheduler(optimizer, len(train_loader)*epochs, lr_gamma),
  # scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(), {"T_max": len(train_loader) * epochs}),
  learning_rate=learning_rate,
  img_width = img_shape[0],
  img_height = img_shape[1],
  optimizer=optimizer,
  momentum=momentum,
  decay=decay
)

# wandb_args = {
#   "project": "mestrado-comparadora",
#   "name": "CCT R29 5k",
#   "notes": "Primeiro modelo a ser treinado.",
#   "config": config.config_dict()
# }

log = Log(wandb_flag=wandb_flag)
log.create_metric("eer", EER, True)
log.create_metric("eer", EER, False)
log.create_metric("lr", LR, True, scheduler=config.scheduler)
# log.create_metric("ident", Identification, False)

# run_name = wandb.run.name

models_folder = "trained_models"
mkdir(models_folder)
mkdir(f"./{models_folder}/{project_name}")
mc = ModelCheckpoint(f"./{models_folder}/{project_name}/resnet_cluster_best.pt")

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log,
  patience = patience,
  callbacks=[mc],
  model_save_path=f"./{models_folder}/{project_name}/resnet_cluster_last.pt"
)