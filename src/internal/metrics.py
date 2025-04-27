from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd

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
    """_summary_

    Args:
        pred_batches (_type_): _description_

    Returns:
        _type_: [modal, data]
    """
    n_outs = len(pred_batches[0])
    ret = [[] for _ in range(n_outs)]

    for elem in pred_batches:
      for i in range(n_outs):
        ret[i] += elem[i].tolist()

    return np.array(ret)

  @staticmethod
  def euclidian_distance(x1, x2):
    return np.linalg.norm(x1-x2)

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

class Identification(Metric):
  def __init__(self, **kwargs):
    super().__init__("identification", **kwargs)

  def add_data(self, y_pred, df: pd.DataFrame, **kwargs):
    if self.train:
      raise ValueError("Não implementado para o cenário de treino")
    
    y_pred = self.batches_to_list(y_pred)[0]
    classes = df["class_index"]
    unique_classes = classes.unique()
    correct = 0
    total = 0

    for i in range(len(y_pred)):
      dists = []
      elem_class = classes[i]
      elem_features = y_pred[i]
      elem_same_class = classes[(classes == elem_class) & (classes.index != i)]
      if len(elem_same_class) == 0: continue

      same_class_sample_idx = elem_same_class.sample(n=1).index[0]
      same_class_dist = self.euclidian_distance(y_pred[same_class_sample_idx], elem_features)

      for j in range(len(unique_classes)):
        current_class = unique_classes[j]
        if current_class != elem_class:
          current_series = classes[classes == current_class]
          idx = current_series.sample(n=1).index[0]
          compare_features = y_pred[idx]

          dists.append(self.euclidian_distance(elem_features, compare_features))

      total += 1
      if np.min(dists) > same_class_dist:
        correct += 1

    acc = correct/total
    self.current_epoch = [acc]
    return acc
