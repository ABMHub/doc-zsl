from internal.trainer import train, test_epoch
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
import json

def mkdir(folder):
  try:
    os.mkdir(folder)
  except FileExistsError as e:
    print(e)

def main(config=None):
  with wandb.init(config=config) as run:
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
    distance_metric = wdb_config.distance_metric
    scheduler_step = wdb_config.scheduler_step

    model_dict = {
      "i7po2o08": (ResNet, "ResNet T2"),
      "fovpkr1a": (ResNet, "ResNet 512"),
      "ur9z0s0w": (AlexNet, "AlexNet"),
      "28w1flxs": (AlexNet, "AlexNet 512"),
      "ezexg7tm": (Vit, "ViT"),
      "z2sl3ctn": (Vit, "ViT 512"),
      "b5jgnxma": (EfficientNet, "EfficientNet"),
      "qdyxxs2q": (MobileNetV3, "MobileNet"),
      "tc3u5buc": (VGG, "VGG"),
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

    csv_path = f"./dataset/splits/train_{split_mode}.csv"
    df = pd.read_csv(csv_path, index_col=0)
    # split_string = "split"
    # df = df.rename(columns={split_string: "train"})
    #gambiarra
    df.loc[df["split"] == split_number, "split"] = -1
    df.loc[df["split"] > 0, "split"] = 0
    df.loc[df["split"] == -1, "split"] = 1

    protocol_path = "./dataset/protocols/val_protocol.csv"
    protocol = pd.read_csv(protocol_path)
    protocol = protocol[(protocol["split_mode"] == f"{split_mode}_split") & (protocol["split_number"] == split_number)]

    test_csv_path = f"./dataset/splits/test_{split_mode}.csv"
    test_df = pd.read_csv(test_csv_path, index_col=0)
    test_df.insert(len(test_df.columns), "split", 1)
    test_protocol_path = "./dataset/protocols/test_protocol.csv"
    test_protocol = pd.read_csv(test_protocol_path)
    test_protocol = test_protocol[(test_protocol["split_mode"] == f"{split_mode}_split")]

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

    log = Log(
      minimize_loss=False,
      wandb_flag=True,
      wandb_args=None,
    )
    log.create_metric("eer", EER, True, cosine=distance_metric=="cosine")
    log.create_metric("eer", EER, False, cosine=distance_metric=="cosine")
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
      model_save_path=f"./{models_folder}/{project_name}/{run_name}_last.pt",
      distance_metric=distance_metric
    )

    test_last_scores = test_epoch(
      test_loader, 
      model, 
      ContrastiveLoss(margin=1, cosine_distance=distance_metric=="cosine"),
      "cuda",
      log
    )

    torch.cuda.empty_cache()
    gc.collect()

    model = torch.load(f"./{models_folder}/{project_name}/{run_name}_best.pt")

    test_best_scores = test_epoch(
      test_loader, 
      model, 
      ContrastiveLoss(margin=1, cosine_distance=distance_metric=="cosine"),
      "cuda",
      log
    )

    for key in list(test_last_scores.keys()):
      test_last_scores["last-last" + key] = test_last_scores.pop(key)

    for key in list(test_best_scores.keys()):
      test_best_scores["best-test" + key] = test_best_scores.pop(key)

    test_last_scores.update(test_best_scores)
    run.summary.update(test_last_scores)

# metric = {
#   'name': 'val-loss',
#   'goal': 'minimize'
# }

# with open("./parameters_sweep/tuning/vgg.json", "r") as f:
#   parameters_dict = json.load(f)

# project_name = "VGG"

# sweep_config = {
#   'method': 'grid',
#   'metric': metric,
#   "name": project_name,
#   # "name": f"EfficientNet_b{model_version}_torchvision",
#   "parameters": parameters_dict
# }

# sweep_id = wandb.sweep(sweep_config, project="defesa-mestrado")
# exit()

# wandb.agent(sweep_id, function=main, count=None)
wandb.agent("z2sl3ctn", function=main, count=None, project="defesa-mestrado") # vit 512
wandb.agent("i7po2o08", function=main, count=None, project="defesa-mestrado") # resnet
wandb.agent("fovpkr1a", function=main, count=None, project="defesa-mestrado") # resnet 512
wandb.agent("tc3u5buc", function=main, count=None, project="defesa-mestrado") # vgg
wandb.agent("b5jgnxma", function=main, count=None, project="defesa-mestrado") # efficientnet
wandb.agent("qdyxxs2q", function=main, count=None, project="defesa-mestrado") # mobilenet

# wandb.agent("ezexg7tm", function=main, count=None, project="defesa-mestrado") # vit
# wandb.agent("28w1flxs", function=main, count=None, project="defesa-mestrado") # alexnet 512
# wandb.agent("ur9z0s0w", function=main, count=None, project="defesa-mestrado") # alexnet