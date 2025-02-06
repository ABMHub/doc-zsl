import os
import random
import operator

from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
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
    dataframe: pd.DataFrame,
    train: bool,
    img_shape: tuple = (224, 224),
    load_in_ram : bool = False,
    mean: float = 0.9402,
    std: float = 0.1724,
    n_channels: int = 3
  ):
    super().__init__()
    self.df = dataframe[(dataframe["train"] == int(not train))]
    if train:
      self.df = self.df.sample(frac=1).reset_index(drop=True)
    self.train = train
    self.img_shape = img_shape
    self.ram = load_in_ram
    self.mean, self.std = mean, std
    self.n_channels = n_channels
    if self.ram:
      print("Preprocessing dataset")
      self.processed_ds = [self.process_image(self.df.iloc[i]["file_path"]) for i in tqdm(range(len(self.df)))]
      # print(torch.std_mean(torch.stack(self.processed_ds, dim=0), dim=None))
    
  def __len__(self):
    return len(self.df)
  
  def process_image(self, file_path):
    im = Image.open(file_path)
    im : Image.Image

    im = im.resize(self.img_shape)
    im = im.convert("L") if self.n_channels == 1 else im.convert("RGB")

    return (ToTensor()(im) - self.mean) / self.std

  def __getitem__(self, index):
    row = self.df.iloc[index]
    path, class_index = row["file_path"], row["class_index"]
    if self.ram:
      return self.processed_ds[index], class_index
    # else
    return self.process_image(path), class_index
  
class ContrastivePairLoader(TorchDataset, DatasetTemplate):
  def __init__(self, dataset: DocDataset, protocol: pd.DataFrame = None):
    super(ContrastivePairLoader, self).__init__()
    self.dataset = dataset
    self.train = self.dataset.train
    if protocol is None:
      self.randomize_pairs()

    else:
      self.prepare_protocol(protocol)

  def prepare_protocol(self, protocol):
    dic ={
      "x1": [self.dataset.df.index.get_loc(elem) for elem in protocol["file_a_idx"].values],
      "x2": [self.dataset.df.index.get_loc(elem) for elem in protocol["file_b_idx"].values],
      "y": list(protocol["is_equal"].values),
    }

    self.df = pd.DataFrame(dic)
    1+1

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
      filtered_data = ds_df[data_filter]
      if len(filtered_data) < 1:
        data_filter = (ds_df["class_index"] != class_index) & (ds_df["file_path"] != file_path)
        filtered_data = ds_df[data_filter]

      index = filtered_data.sample(n=1).index[0]

      dic["x2"].append(index)
      dic["y"].append(y)

    self.df = pd.DataFrame(dic)

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    row = self.df.iloc[index]

    x1 = self.dataset[row["x1"]][0]
    x2 = self.dataset[row["x2"]][0]
    y = row["y"]

    return (x1, x2), y
  
  def on_epoch_end(self):
    if self.train:
      self.randomize_pairs()
