import wandb
import numpy as np
from internal.metrics import Metric, Loss
from typing import Type
import threading

class Log:
  """Stores the logs of every step or epoch
  """
  def __init__(
      self,
      minimize_loss = True,
      wandb_flag: bool = True,
      wandb_args: dict = None
    ):
    self.train_metrics: dict[str, Metric] = dict()
    self.val_metrics: dict[str, Metric] = dict()
    self.create_metric("loss", Loss, True)
    self.create_metric("loss", Loss, False)

    self.wandb_flag = wandb_flag
    self.wandb_run = wandb.run

    self.best_metrics = None
    self.current_epoch = 0

    if self.wandb_flag and self.wandb_run is None:
      self.wandb_run = wandb.init(**wandb_args)

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
    dic = self.train_metrics if train else self.val_metrics
    for k in dic.keys():
      try:
        if k != "loss":
          dic[k].add_data(**kwargs)
      except:
        print(f"Erro na atualização das métricas - {str(k)}")

  def compute_all_metrics(self, **kwargs):
    return_dict = {}
    try:
      for k in self.val_metrics.keys():
        if k != "loss":
          return_dict[k] = self.val_metrics[k].compute_metric(**kwargs)
    except:
      pass

    return return_dict

  def end_epoch(self):
    if self.wandb_flag:
      self.update_wandb()

    self.update_best_metrics()
    self.current_epoch += 1

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
  
  def create_complete_metrics_dict(self):
    complete_log = self.create_metrics_dict(train=True, show_mode=True)
    complete_log.update(self.create_metrics_dict(train=False, show_mode=True))

    return complete_log

  def update_best_metrics(self):
    complete_log = self.create_complete_metrics_dict()

    if self.best_metrics is None or self.best_metrics["best-val-loss"] > complete_log["val-loss"]:
      self.best_metrics = complete_log
      self.best_metrics["epoch"] = self.current_epoch

      for key in list(self.best_metrics.keys()):
        self.best_metrics["best-" + str(key)] = self.best_metrics.pop(key)

      self.wandb_run.summary.update(self.best_metrics)

  def update_wandb(self):
    complete_log = self.create_complete_metrics_dict()

    save_thread = threading.Thread(target=wandb.log, args=(complete_log,))
    save_thread.start()
