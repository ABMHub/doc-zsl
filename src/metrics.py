from sklearn.metrics import roc_curve
import numpy as np

class Metric:
  def __init__(self, name, minimize = True, train = True):
    self.minimize = minimize
    self.name = name
    self.current_epoch = []
    self.epoch_history = []
    self.train = train

  def add_data(self, *args, **kwargs):
    raise NotImplementedError

  def end_epoch(self):
    self.epoch_history.append(self.current_epoch)
    self.current_epoch = []

  def get_current_info(self):
    if len(self.current_epoch) == 0:
      return None
    
    return np.mean(self.current_epoch)
  
  def batches_to_list(self, pred_batches):
    n_outs = len(pred_batches[0])
    ret = [[] for _ in range(n_outs)]

    for elem in pred_batches:
      for i in range(n_outs):
        ret[i] += elem[i].tolist()

    return np.array(ret)

class Loss(Metric):
  def __init__(self, **kwargs):
    super().__init__("loss", **kwargs)

  def add_data(self, value, **kwargs):
    self.current_epoch.append(value)

class EER(Metric):
  def __init__(self, **kwargs):
    super().__init__("eer", **kwargs)
    self.dists_pred = []
    self.y = []

  def end_epoch(self):
    self.dists_pred = []
    self.y = []
    super().end_epoch()

  def add_data(self, y_true, y_pred, **kwargs):
    if not self.train:
      y_pred = self.batches_to_list(y_pred)

    y_pred = [self.euclidian_distance(*elem) for elem in list(zip(*y_pred))]

    self.dists_pred += y_pred
    self.y += list(y_true)

    batch_eer, _ = self.calculate_eer()
    self.current_epoch = [batch_eer] # pra resolver o np.mean

    return batch_eer

  @staticmethod
  def euclidian_distance(x1, x2):
    return np.linalg.norm(x1-x2)

  def calculate_eer(self) -> tuple[float, float]:
    fpr, tpr, threshold = roc_curve(self.y, self.dists_pred, pos_label=0)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer, eer_threshold

class LR(Metric):
  def __init__(self, scheduler, **kwargs):
    super().__init__("learning_rate", **kwargs)
    self.scheduler = scheduler

  def add_data(self, *args, **kwargs):
    pass

  def get_current_info(self):
    return self.scheduler.get_last_lr()[0]
