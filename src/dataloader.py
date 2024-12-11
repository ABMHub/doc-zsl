import os
import random
import operator

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.transforms import ToTensor

class DataLoader(TorchDataLoader):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def on_epoch_end(self, *args, **kwargs):
    self.dataset.on_epoch_end(*args, **kwargs)

class DatasetTemplate:
  def on_epoch_end(self):
    pass

class DocDataset(TorchDataset, DatasetTemplate):
  def __init__(
    self,
    csv_path: os.PathLike,
    train: bool,
    load_in_ram : bool = False
  ):
    super().__init__()
    df = pd.read_csv(csv_path, index_col=False)
    self.df = df[(df["train"] == int(train))].sample(frac=1).reset_index(drop=True)
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    row = self.df.iloc[index]
    path, class_index = row["file_path"], row["class_index"]

    im = Image.open(path)
    im : Image.Image

    im = im.resize((224, 224)).convert("RGB")

    return ToTensor()(im), class_index
  
class ContrastivePairLoader(TorchDataset, DatasetTemplate):
  def __init__(self, dataset: DocDataset):
    super(ContrastivePairLoader, self).__init__()
    self.dataset = dataset
    self.randomize_pairs()

  def randomize_pairs(self):
    dic = {
      "x1": list(range(len(self.dataset))),
      "x2": [],
      "y": []
    }

    ds_df = self.dataset.df

    for i in range(len(self.dataset)):
      file_path = ds_df.iloc[i]["file_path"]
      class_index = ds_df.iloc[i]["class_index"]
      y = random.getrandbits(1)
      op = operator.eq if y else operator.ne

      # se y == 1, escolhe um aleatorio da mesma classe
      # se y == 0, escolhe um aleatorio de outra classe
      data_filter = op(ds_df["class_index"], class_index)
      data_filter = data_filter & (ds_df["file_path"] != file_path)

      index = ds_df[data_filter].sample(n=1).index[0]

      dic["x2"].append(index)
      dic["y"].append(y)

    self.df = pd.DataFrame(dic)

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, index):
    row = self.df.iloc[index]

    x1 = self.dataset[row["x1"]][0]
    x2 = self.dataset[row["x2"]][0]
    y = row["y"]

    return (x1, x2), y
  
  def on_epoch_end(self):
    return self.randomize_pairs()
