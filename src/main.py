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

dataset = "rvl_zsl_5k"
split = "zsl"  # overlap, zsl, gzsl
split_number = 0

wandb_flag = False
load_in_ram = True

img_shape = (224, 224)
out_dim = 16
batch_size = 16
shuffle_loader = True
learning_rate = 1e-3
patience = 9
n_channels = 3
model_version = 2

project_name = f"EfficientNet_b{model_version} R2 5k ZSL"

model = CCT(out_dim=out_dim, img_shape=img_shape, model_version=model_version, n_input_channels=n_channels)
# model = EfficientNet(out_dim, model_version=model_version)
# model = Vit(out_dim=64, model_version="b", pretrained=False)
# model = Vit(out_dim=64, model_version="b", pretrained=False)
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
  optimizer=torch.optim.SGD,
  patience=patience,
  n_channels=n_channels,
  model_version=model_version,
  dataset=dataset,
  split=split
)

csv_path = "./splits.csv"
df = pd.read_csv(csv_path)
split_string = f"{split}_split"
df = df.rename(columns={split_string: "train"})
#gambiarra
df.loc[df["train"] == split_number, "train"] = -1
df.loc[df["train"] > 0, "train"] = 0
df.loc[df["train"] == -1, "train"] = 1

protocol_path = "./protocol.csv"
protocol = pd.read_csv(protocol_path)
protocol = protocol[(protocol["split_mode"] == split_string) & (protocol["split_number"] == split_number)]

train_loader = DocDataset(df, train=True, load_in_ram=load_in_ram, img_shape=img_shape, n_channels=n_channels)
val_loader = DocDataset(df, train=False, load_in_ram=load_in_ram, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels)

train_loader = ContrastivePairLoader(train_loader, None)
val_loader = ContrastivePairLoader(val_loader, protocol)

train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
val_loader = DataLoader(val_loader, batch_size, shuffle=False)

wandb_args = {
  "project": "mestrado-comparadora",
  # "name": "EfficientNet R4 5k",
  # "name": f"CCT_{model_version} R1 5k ZSL",
  "name": project_name,
  "notes": "Teste sobre versões da rede.",
  "config": config.config_dict()
}

log = Log(wandb_flag=wandb_flag, wandb_args=wandb_args)
log.create_metric("eer", EER, True)
log.create_metric("eer", EER, False)
log.create_metric("lr", LR, True, scheduler=config.scheduler)
log.create_metric("ident", Identification, False)

mc = ModelCheckpoint(f"./{project_name}_best.pt")

train(
  config = config,
  train_dataloader = train_loader,
  val_dataloader = val_loader,
  model = model,
  device = "cuda",
  log = log,
  patience = patience,
  callbacks=[mc],
  model_save_path=f"{project_name}_last.pt"
)