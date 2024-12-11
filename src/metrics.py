from sklearn.metrics import roc_curve
import numpy as np

class Metric:
  def __init__(self, name, minimize = True):
    self.minimize = minimize
    self.name = name
    self.current_epoch = []
    self.epoch_history = []

  def add_data(self, *args, **kwargs):
    raise NotImplementedError

  def end_epoch(self):
    self.epoch_history.append(self.current_epoch)
    self.current_epoch = []

  def get_current_info(self):
    if len(self.current_epoch) == 0:
      return None
    
    return np.mean(self.current_epoch)

class Loss(Metric):
  def __init__(self, minimize=True):
    super().__init__("loss", minimize=minimize)

  def add_data(self, value, **kwargs):
    self.current_epoch.append(value)

class EER(Metric):
  def __init__(self, minimize=True):
    super().__init__("eer", minimize=minimize)

  def add_data(self, y_true, y_pred, **kwargs):
    y_pred = [self.euclidian_distance(*elem) for elem in list(zip(*y_pred))]
    batch_eer, _ = self.calculate_eer(y_true, y_pred)
    self.current_epoch.append(batch_eer)

    return batch_eer

  @staticmethod
  def euclidian_distance(x1, x2):
    return np.linalg.norm(x1-x2)

  @staticmethod
  def calculate_eer(y_true, y_pred) -> tuple[float, float]:
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer, eer_threshold
