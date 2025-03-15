from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT, EfficientNet, ResNet
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER, LR, Identification
from loss import ContrastiveLoss
import torch
from callbacks import ModelCheckpoint
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

def mkdir(folder):
  try:
    os.mkdir(folder)
  except FileExistsError as e:
    print(e)

project_name = f"vit_test"

# model = CCT(out_dim=out_dim, img_shape=img_shape, model_version=model_version, n_input_channels=n_channels)
# model = EfficientNet(out_dim, model_version=model_version)
model = Vit(out_dim=64, model_version="b", pretrained=True)
# model = Vit(out_dim=64, model_version="b", pretrained=False)
img_shape = (model.im_shape, model.im_shape)
model = SiameseModel(model)

csv_path = f"./train_{split_mode}.csv"
df = pd.read_csv(csv_path, index_col=0)
# split_string = "split"
# df = df.rename(columns={split_string: "train"})
#gambiarra
df.loc[df["split"] == split_number, "split"] = -1
df.loc[df["split"] > 0, "split"] = 0
df.loc[df["split"] == -1, "split"] = 1

protocol_path = "./train_protocol.csv"
protocol = pd.read_csv(protocol_path)
protocol = protocol[(protocol["split_mode"] == f"{split_mode}_split") & (protocol["split_number"] == split_number)]

train_loader = DocDataset(df, train=True, load_in_ram=True, img_shape=img_shape, n_channels=n_channels)
val_loader = DocDataset(df, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels)

train_loader = ContrastivePairLoader(train_loader, None)
val_loader = ContrastivePairLoader(val_loader, protocol)

train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
val_loader = DataLoader(val_loader, batch_size, shuffle=False)

optimizers = {
  "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay),
  "adam": lambda: torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay),
  "sgd": lambda: torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
}

config = Config(
  batch_size=batch_size,
  criterion=ContrastiveLoss,
  model = model,
  shuffle_loader = True,
  epochs=int(epochs),
  scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": len(train_loader) * epochs}),
  learning_rate=learning_rate,
  img_width = img_shape[0],
  img_height = img_shape[1],
  optimizer=optimizers['adamw'](),
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
mc = ModelCheckpoint(f"./{models_folder}/{project_name}/vit1e-3_best.pt")

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log,
  patience = patience,
  callbacks=[mc],
  model_save_path=f"./{models_folder}/{project_name}/vit1e-3_last.pt"
)