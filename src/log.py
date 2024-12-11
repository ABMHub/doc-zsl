import wandb
import numpy as np
from metrics import Metric, Loss

class Log:
  def __init__(self, minimize_loss = True):
    self.train_metrics: dict[str, Metric] = dict()
    self.val_metrics: dict[str, Metric] = dict()
    self.create_metric("loss", Loss(), True)
    self.create_metric("loss", Loss(), False)

  def create_metric(self, name, metric_object: Metric, train = True):
    if train:
      self.train_metrics[name] = metric_object

    else:
      self.val_metrics[name] = metric_object

  def update_metric(self, metric_name, train, **kwargs):
    dic = self.train_metrics if train else self.val_metrics

    dic[metric_name].add_data(**kwargs)

  def update_all_metrics(self, train: bool, **kwargs):  # except loss
    dic = self.train_metrics if train else self.val_metrics
    for k in dic.keys():
      if k != "loss":
        dic[k].add_data(**kwargs)

  def end_epoch(self):
    for mode in [self.train_metrics, self.val_metrics]:
      for metric in mode.keys():
        mode[metric].end_epoch()

  def create_metrics_dict(self, train: bool = True):
    dic = self.train_metrics if train else self.val_metrics
    ret = {}

    for k in dic.keys():
      ret[k] = dic[k].get_current_info()

    return ret
