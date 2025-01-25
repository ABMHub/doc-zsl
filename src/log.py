import wandb
import numpy as np
from metrics import Metric, Loss
from typing import Type

class Log:
  def __init__(self, minimize_loss = True, wandb_flag: bool = True, wandb_args: dict = None):
    self.train_metrics: dict[str, Metric] = dict()
    self.val_metrics: dict[str, Metric] = dict()
    self.create_metric("loss", Loss, True)
    self.create_metric("loss", Loss, False)

    self.wandb_flag = wandb_flag

    if self.wandb_flag and wandb.run is None:
      wandb.init(**wandb_args)

  def create_metric(self, name, metric: Type[Metric], train = True, **kwargs):
    obj = metric(train=train, **kwargs)
    if train:
      self.train_metrics[name] = obj

    else:
      self.val_metrics[name] = obj

  def update_metric(self, metric_name, train, **kwargs):
    dic = self.train_metrics if train else self.val_metrics

    dic[metric_name].add_data(**kwargs)

  def update_all_metrics(self, train: bool, **kwargs):  # except loss
    try:
      dic = self.train_metrics if train else self.val_metrics
      for k in dic.keys():
        if k != "loss":
          dic[k].add_data(**kwargs)
    except:
      pass

  def end_epoch(self):
    if self.wandb_flag:
      self.update_wandb()

    for mode in [self.train_metrics, self.val_metrics]:
      for metric in mode.keys():
        mode[metric].end_epoch()

  def create_metrics_dict(self, train: bool = True, show_mode: bool = False):
    dic = self.train_metrics if train else self.val_metrics
    ret = {}
    mode_string = ""
    if show_mode:
      mode_string = "train-" if train else "val-"

    for k in dic.keys():
      ret[f"{mode_string}{k}"] = dic[k].get_current_info()

    return ret
  
  def update_wandb(self):
    complete_log = self.create_metrics_dict(train=True, show_mode=True)
    complete_log.update(self.create_metrics_dict(train=False, show_mode=True))

    wandb.log(complete_log)
