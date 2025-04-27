from internal.trainer import train
from internal.config import Config
from internal.architecture import Vit, SiameseModel, CCT, EfficientNet, ResNet, ModelUnit, DenseNet, AlexNet, VGG, EfficientNetV2, MobileNetV3, ConvNext
from internal.dataloader import DocDataset, DataLoader, ContrastivePairLoader
from internal.log import Log
from internal.metrics import EER, LR, Identification
from internal.loss import ContrastiveLoss
from internal.callbacks import ModelCheckpoint

import torch
import wandb
import math
import pandas as pd
import gc
import os

metric = {
  'name': 'val-loss',
  'goal': 'minimize'
}

parameters_dict = {
  "pre_trained":        {"value": True},
  "out_dim":            {'value': 64},
  "batch_size":         {"value": 16},
  'epochs':             {"value": 90},
  'n_channels':         {"value": 3},
  'patience':           {"value": 1e5},
  "scheduler_step":     {"value": 90},

  "aaa_model_version":  {"values": ["tiny", "small", "base", "large"]},
  "split_mode":         {"values": ["zsl", "gzsl"]},
  "split_number":       {"values": list(range(5))}
}

project_name = "ConvNext"

sweep_config = {
  'method': 'grid',
  'metric': metric,
  "name": project_name,
  # "name": f"EfficientNet_b{model_version}_torchvision",
  "parameters": parameters_dict
}

def mkdir(folder):
  try:
    os.mkdir(folder)
  except FileExistsError as e:
    print(e)

def main(config=None):
  with wandb.init(config=config):
    wdb_config = wandb.config
    out_dim = int(wdb_config.out_dim)
    batch_size = int(wdb_config.batch_size)
    n_channels = int(wdb_config.n_channels)
    patience = int(wdb_config.patience)
    model_version = wdb_config.aaa_model_version
    pre_trained = bool(wdb_config.pre_trained)
    split_number = int(wdb_config.split_number)
    split_mode = str(wdb_config.split_mode)
    epochs = int(wdb_config.epochs)
    scheduler_step = wdb_config.scheduler_step

    model_dict = {
      "ak55hy5u": (ResNet, "ResNet T2"),
      "d2cfk3o7": (Vit, "ViT"),
      "k151sfrd": (DenseNet, "DenseNet"),
      "3yzu0li3": (AlexNet, "AlexNet"),
      "vo5zn89m": (VGG, "VGG"),
      "emb7e2fs": (EfficientNetV2, "EfficientNetV2"),
      "l2jrbgze": (MobileNetV3, "MobileNetV3"),
      "nbxz2o7j": (EfficientNet, "EfficientNet"),
      "xm0u9xr6": (ConvNext, "ConvNext"),
    }

    model: ModelUnit
    model, project_name = model_dict[wandb.run.sweep_id]
    model = model(out_dim=out_dim, model_version=model_version, pretrained=pre_trained)

    learning_rate = model.learning_rate
    decay = model.weight_decay
    momentum = model.momentum
    optimizer = model.optimizer
    scheduler = model.scheduler
    lr_gamma = model.lr_gamma
    
    shuffle_loader = True

    # model = EfficientNet(out_dim=out_dim, model_version=model_version, pretrained=pre_trained)
    # model = CCT(out_dim=out_dim, img_shape=img_shape)

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

    config = Config(
      batch_size=batch_size,
      criterion=ContrastiveLoss,
      model = model,
      shuffle_loader = True,
      epochs=epochs,
      scheduler=scheduler(optimizer, len(train_loader) * scheduler_step, lr_gamma),
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

    log = Log(wandb_flag=True)
    log.create_metric("eer", EER, True)
    log.create_metric("eer", EER, False)
    log.create_metric("lr", LR, True, scheduler=config.scheduler)
    # log.create_metric("ident", Identification, False)

    run_name = wandb.run.name

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
      model_save_path=f"./{models_folder}/{project_name}/{run_name}_last.pt"
    )

    torch.cuda.empty_cache()
    gc.collect()

# sweep_id = wandb.sweep(sweep_config, project="icdar-experiments")
# exit()

# wandb.agent(sweep_id, function=main, count=None)
# wandb.agent("k151sfrd", function=main, count=None, project="icdar-experiments") # densenet
# wandb.agent("3yzu0li3", function=main, count=None, project="icdar-experiments") # Alexnet
# wandb.agent("ak55hy5u", function=main, count=None, project="icdar-experiments") # resnet t2
# wandb.agent("vo5zn89m", function=main, count=None, project="icdar-experiments") # vgg
# wandb.agent("l2jrbgze", function=main, count=None, project="icdar-experiments") # mobilenet v3
# wandb.agent("emb7e2fs", function=main, count=None, project="icdar-experiments") # efficientnet v2
# wandb.agent("nbxz2o7j", function=main, count=None, project="icdar-experiments") # efficientnet
# wandb.agent("d2cfk3o7", function=main, count=None, project="icdar-experiments") # vit
wandb.agent("xm0u9xr6", function=main, count=None, project="icdar-experiments") # convnext