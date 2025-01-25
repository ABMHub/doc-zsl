from trainer import train
from config import Config
from architecture import Vit, SiameseModel, CCT, EfficientNet
from dataloader import DocDataset, DataLoader, ContrastivePairLoader
from log import Log
from metrics import EER, LR
from loss import ContrastiveLoss
import torch
import wandb

metric = {
    'name': 'val-loss',
    'goal': 'minimize'
}

parameters_dict = {
    # "img_side_size": {'values': [50, 100, 150, 200, 250]},
    "img_side_size": {'value': 224},
    'optimizer': {'value': 'sgd'},
    'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
    "out_dim": {'values': [16, 32, 64, 128]},
    "batch_size": {"value": 16},
    'epochs': {"value": 1000},
    'n_channels': {"value": 3},
    'patience': {"value": 10},
    "model_version": {"value": 1}
}

sweep_config = {
    'method': 'bayes',
    'metric': metric,
    "parameters": parameters_dict
}

# sweep_id = wandb.sweep(sweep_config, project="mestrado-comparadora")

def main(config=None):
  with wandb.init(config=config):
    wdb_config = wandb.config
    img_shape = (int(wdb_config.img_side_size), int(wdb_config.img_side_size))
    out_dim = int(wdb_config.out_dim)
    batch_size = int(wdb_config.batch_size)
    n_channels = int(wdb_config.n_channels)
    patience = int(wdb_config.patience)
    model_version = int(wdb_config.model_version)

    optimizers = {
      "adamw": torch.optim.AdamW,
      "adam": torch.optim.Adam,
      "sgd": torch.optim.SGD
    }

    shuffle_loader = True

    model = EfficientNet(out_dim=out_dim, model_version=model_version)
    # model = CCT(out_dim=out_dim, img_shape=img_shape)
    model = SiameseModel(model)

    config = Config(
      batch_size=batch_size,
      criterion=ContrastiveLoss,
      model = model,
      shuffle_loader = True,
      epochs=int(wdb_config.epochs),
      scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
      learning_rate=wdb_config.learning_rate,
      img_width = img_shape[0],
      img_height = img_shape[1],
      optimizer=optimizers[wdb_config.optimizer]
    )

    csv_path = "./splits.csv"

    train_loader = DocDataset(csv_path, train=True, load_in_ram=True, img_shape=img_shape, n_channels=n_channels)
    val_loader = DocDataset(csv_path, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std, n_channels=n_channels)

    train_loader = ContrastivePairLoader(train_loader)
    val_loader = ContrastivePairLoader(val_loader)

    train_loader = DataLoader(train_loader, batch_size, shuffle=shuffle_loader)
    val_loader = DataLoader(val_loader, batch_size, shuffle=False)


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

    train(
      config = config,
      train_dataloader = train_loader,
      val_dataloader = val_loader,
      model = model,
      device = "cuda",
      log = log,
      patience = patience,
    )

# wandb.agent(sweep_id, function=main, count=None)
wandb.agent("uqrs008s", function=main, count=None, project="mestrado-comparadora")