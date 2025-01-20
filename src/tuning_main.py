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
    "img_side_size": {'values': [50, 100, 150, 200, 250]},
    # "img_side_size": {'value': 224},
    'optimizer': {'values': ['adamw', 'sgd', 'adam']},
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'log_uniform',
        'min': -10,
        'max': -1
    },
    "out_dim": {
        'distribution': 'q_uniform',
        "q": 8,
        'min': 8,
        'max': 64
    },
    "batch_size": {"values": [8, 16, 32, 64]},
    'epochs': {"value": 80}
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

    optimizers = {
      "adamw": torch.optim.AdamW,
      "adam": torch.optim.Adam,
      "sgd": torch.optim.SGD
    }

    shuffle_loader = True

    # model = EfficientNet(out_dim=out_dim)
    model = CCT(out_dim=out_dim, img_shape=img_shape)
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

    train_loader = DocDataset(csv_path, train=True, load_in_ram=True, img_shape=img_shape)
    val_loader = DocDataset(csv_path, train=False, load_in_ram=True, img_shape=img_shape, mean=train_loader.mean, std=train_loader.std)

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
    log.create_metric("eer", EER(), True)
    log.create_metric("eer", EER(), False)
    log.create_metric("lr", LR(config.scheduler), True)

    train(
      config = config,
      train_dataloader = train_loader,
      val_dataloader = val_loader,
      model = model,
      device = "cuda",
      log = log
    )

# wandb.agent(sweep_id, function=main, count=None)
wandb.agent("3qzddqf3", function=main, count=None, project="mestrado-comparadora")