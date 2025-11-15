from internal.trainer import train, test_epoch
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
import gc

import wandb

dataset = "rvl_zsl_5k"
split_mode = "zsl"  # overlap, zsl, gzsl
split_number = 0

wandb_flag = True
load_in_ram = True

img_shape = (224, 224)
out_dim = 16
batch_size = 16
shuffle_loader = True
patience = 1000
n_channels = 3
model_version = 1
epochs = 90

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
model = ResNet(out_dim=256, model_version=18, pretrained=True)

# hyperparameters are stored in each model's class
learning_rate = model.learning_rate
decay = model.weight_decay
momentum = model.momentum
optimizer = model.optimizer
scheduler = model.scheduler
lr_gamma = model.lr_gamma
scheduler_step = 0.1

# if using imagenet pretrained weights, use (224, 224)
# otherwise, use whatever you feel like
img_shape = (model.im_shape, model.im_shape)
model = SiameseModel(model)

csv_path = f"./train_{split_mode}_active.csv"  # csv containg the train split for zsl or gzsl
df = pd.read_csv(csv_path, index_col=0)

# test becomes 1, train becomes 0
# gambiarra
df.loc[df["split"] == split_number, "split"] = -1
df.loc[df["split"] > 0, "split"] = 0
df.loc[df["split"] == -1, "split"] = 1

# load the eval protocol
protocol_path = "./train_protocol_active.csv"
protocol = pd.read_csv(protocol_path)
protocol = protocol[(protocol["split_mode"] == f"{split_mode}_split") & (protocol["split_number"] == split_number)]

test_csv_path = f"./dataset/splits/test_{split_mode}.csv"
test_df = pd.read_csv(test_csv_path, index_col=0)
test_df.insert(len(test_df.columns), "split", 1)
test_protocol_path = "./dataset/protocols/test_protocol.csv"
test_protocol = pd.read_csv(test_protocol_path)
test_protocol = test_protocol[(test_protocol["split_mode"] == f"{split_mode}_split")]

# import transformers

# processor = transformers.AutoImageProcessor.from_pretrained("microsoft/dit-base")
processor = None

# create file loaders
train_loader = DocDataset(df, train=True, load_in_ram=True, img_shape=img_shape, n_channels=n_channels)
val_loader = DocDataset(df, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels)
test_loader = DocDataset(test_df, train=False, load_in_ram=False, img_shape=img_shape, n_channels=n_channels, mean=train_loader.mean, std=train_loader.std)

train_loader = ContrastivePairLoader(train_loader, None)
val_loader = ContrastivePairLoader(val_loader, protocol)
test_loader = ContrastivePairLoader(test_loader, test_protocol)

train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
val_loader = DataLoader(val_loader, batch_size, shuffle=False)
test_loader = DataLoader(test_loader, batch_size, shuffle=False)

config = Config(
  batch_size=batch_size,
  criterion=ContrastiveLoss,
  model = model,
  shuffle_loader = True,
  epochs=epochs,
  # scheduler=scheduler(optimizer, len(train_loader)*20, len(train_loader)*200),
  scheduler=scheduler(optimizer, len(train_loader) * scheduler_step, lr_gamma),
  # scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(), {"T_max": len(train_loader) * epochs}),
  learning_rate=learning_rate,
  img_width = img_shape[0],
  img_height = img_shape[1],
  optimizer=optimizer,
  momentum=momentum,
  decay=decay
)

run_name = "ResNet Test 1"

wandb_args = {
  "project": "active-learning",
  "name": run_name,
  "notes": "Primeiro modelo a ser treinado.",
  "config": config.config_dict()
}

log = Log(wandb_flag=wandb_flag, wandb_args=wandb_args)
log.create_metric("eer", EER, True)
log.create_metric("eer", EER, False)
log.create_metric("lr", LR, True, scheduler=config.scheduler)
# log.create_metric("ident", Identification, False)

models_folder = "trained_models"
mkdir(models_folder)
mkdir(f"./{models_folder}/{project_name}")
mc = ModelCheckpoint(f"./{models_folder}/{project_name}/{run_name}_best.pt")

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log,
  patience = patience,
  callbacks=[mc],
  model_save_path=f"./{models_folder}/{project_name}/{run_name}_last.pt",
  distance_metric="cosine"
)

test_last_scores = test_epoch(
  test_loader, 
  model, 
  ContrastiveLoss(margin=1, cosine_distance=True),
  "cuda",
  log
)

torch.cuda.empty_cache()
gc.collect()

model = torch.load(f"./{models_folder}/{project_name}/{run_name}_best.pt")

test_best_scores = test_epoch(
  test_loader, 
  model, 
  ContrastiveLoss(margin=1, cosine_distance=True),
  "cuda",
  log
)

for key in list(test_last_scores.keys()):
  test_last_scores["last-last" + key] = test_last_scores.pop(key)

for key in list(test_best_scores.keys()):
  test_best_scores["best-test" + key] = test_best_scores.pop(key)

test_last_scores.update(test_best_scores)
wandb.run.summary.update(test_last_scores)