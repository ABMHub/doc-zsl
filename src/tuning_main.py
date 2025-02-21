from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT, EfficientNet, ResNet
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER, LR, Identification
from loss import ContrastiveLoss
import torch
import wandb
import math
from callbacks import ModelCheckpoint
import pandas as pd
import gc

metric = {
  'name': 'val-loss',
  'goal': 'minimize'
}

model_version = 34

parameters_dict = {
  "pre_trained": {"value": True},
  'optimizer': {'value': 'sgd'},
  'learning_rate': {
    'value': 1e-2
  },
  "out_dim": {
    'value': 64
  },
  "momentum": {
    'value': 0.97,
  },
  "w_decay": {
    "values": [1e-4, 1e-5, 1e-6, 1e-7]
  },
  "batch_size": {"value": 16},
  'epochs': {"value": 100},
  'n_channels': {"value": 3},
  'patience': {"value": 200},
  "model_version": {"value": 1},
  "split_mode": {"value": "zsl"},
  "split_number": {"values": list(range(5))}
}

sweep_config = {
  'method': 'grid',
  'metric': metric,
  "name": f"EfficientNet_weight_decay_cosine_tune_1e-2",
  # "name": f"EfficientNet_b{model_version}_torchvision",
  "parameters": parameters_dict
}

def main(config=None):
  with wandb.init(config=config):
    wdb_config = wandb.config
    out_dim = int(wdb_config.out_dim)
    batch_size = int(wdb_config.batch_size)
    n_channels = int(wdb_config.n_channels)
    patience = int(wdb_config.patience)
    model_version = wdb_config.model_version
    momentum = float(wdb_config.momentum)
    decay = float(wdb_config.w_decay)
    pre_trained = bool(wdb_config.pre_trained)
    split_number = int(wdb_config.split_number)
    split_mode = str(wdb_config.split_mode)
    epochs = int(wdb_config.epochs)

    optimizers = {
      "adamw": torch.optim.AdamW,
      "adam": torch.optim.Adam,
      "sgd": torch.optim.SGD
    }

    shuffle_loader = True

    model = EfficientNet(out_dim=out_dim, model_version=model_version, pretrained=pre_trained)
    img_shape = (model.im_shape, model.im_shape)
    # model = CCT(out_dim=out_dim, img_shape=img_shape)
    # model = ResNet(out_dim=out_dim, model_version=model_version, pretrained=pre_trained)
    model = SiameseModel(model)

    csv_path = "./splits.csv"
    df = pd.read_csv(csv_path)
    split_string = f"{split_mode}_split"
    df = df.rename(columns={split_string: "train"})
    #gambiarra
    df.loc[df["train"] == split_number, "train"] = -1
    df.loc[df["train"] > 0, "train"] = 0
    df.loc[df["train"] == -1, "train"] = 1

    protocol_path = "./protocol.csv"
    protocol = pd.read_csv(protocol_path)
    protocol = protocol[(protocol["split_mode"] == split_string) & (protocol["split_number"] == split_number)]

    train_loader = DocDataset(df, train=True, load_in_ram=True, img_shape=img_shape, n_channels=n_channels)
    val_loader = DocDataset(df, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels)

    train_loader = ContrastivePairLoader(train_loader, None)
    val_loader = ContrastivePairLoader(val_loader, protocol)

    train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
    val_loader = DataLoader(val_loader, batch_size, shuffle=False)

    config = Config(
      batch_size=batch_size,
      criterion=ContrastiveLoss,
      model = model,
      shuffle_loader = True,
      epochs=int(wdb_config.epochs),
      scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": len(train_loader) * epochs}),
      learning_rate=wdb_config.learning_rate,
      img_width = img_shape[0],
      img_height = img_shape[1],
      optimizer=optimizers[wdb_config.optimizer],
      momentum=momentum,
      decay=decay
    )

    # wandb_args = {
    #   "project": "mestrado-comparadora",
    #   "name": "CCT R29 5k",
    #   "notes": "Primeiro modelo a ser treinado.",
    #   "config": config.config_dict()
    # }

    log = Log(wandb_flag=True)
    log.create_metric("eer", EER, True)
    log.create_metric("eer", EER, False)
    log.create_metric("lr", LR, True, scheduler=config.scheduler)
    # log.create_metric("ident", Identification, False)

    project_name = wandb.run.name

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

    torch.cuda.empty_cache()
    gc.collect()

# sweep_id = wandb.sweep(sweep_config, project="mestrado-comparadora")
# exit()

# wandb.agent(sweep_id, function=main, count=None)
wandb.agent("dlretyut", function=main, count=None, project="mestrado-comparadora")