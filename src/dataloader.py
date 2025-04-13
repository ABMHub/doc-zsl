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

from typing import Tuple

class DataLoader(TorchDataLoader):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def on_epoch_end(self, *args, **kwargs):
    self.dataset.on_epoch_end(*args, **kwargs)

class DatasetTemplate:
  def on_epoch_end(self):
    pass

class DocDataset(TorchDataset, DatasetTemplate):
  """Class to load data from the dataset.
  The `__getitem__` method returns an element and its class
  Must be used with a ContrastiveLoader to train a document matching model
  """
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
    """_summary_

    Args:
        dataframe (pd.DataFrame): the dataframe containing the dataset
        train (bool): true if train, false if test
        img_shape (tuple, optional): image height and width. Defaults to (224, 224).
        load_in_ram (bool, optional): if true, load the entire dataset in ram. Defaults to False, with lazy image load.
        mean (float, optional): mean value of the dataset. Defaults to 0.9402.
        std (float, optional): standard deviation of the dataset. Defaults to 0.1724.
        n_channels (int, optional): number of channels (RGB or grayscale). Defaults to 3.
    """
    super().__init__()
    self.df = dataframe[(dataframe["split"] == int(not train))]
    if train:
      self.df = self.df.sample(frac=1).reset_index(drop=True)
    self.train = train
    self.img_shape = img_shape
    self.ram = load_in_ram
    self.mean, self.std = mean, std
    self.n_channels = n_channels
    if self.ram:
      print("Preprocessing dataset")
      self.processed_ds = [self.process_image(self.df.iloc[i]["doc_path"]) for i in tqdm(range(len(self.df)))]
      # print(torch.std_mean(torch.stack(self.processed_ds, dim=0), dim=None))
    
  def __len__(self):
    return len(self.df)
  
  def process_image(self, file_path: os.PathLike) -> torch.Tensor:
    """Preprocess image
    - Resize to desired shape
    - Convert to either RGB or grayscale
    - Standardization following given mean and std

    Args:
        file_path (os.PathLike): path containing a image (png, jpg, tiff...)

    Returns:
        torch.Tensor: preprocessed image matrix
    """
    im = Image.open(file_path)
    im : Image.Image

    im = im.resize(self.img_shape)
    im = im.convert("L") if self.n_channels == 1 else im.convert("RGB")

    return (ToTensor()(im) - self.mean) / self.std

  def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
    """Get `index` image and it's class number

    Args:
        index (int): the index on the dataframe

    Returns:
        Tuple[torch.tensor, int]: x and y
    """
    row = self.df.iloc[index]
    path, class_number = row["doc_path"], row["class_number"]
    # if the dataset is already in ram, just retrieve the data
    if self.ram:
      return self.processed_ds[index], class_number
    # otherwise, process the image
    return self.process_image(path), class_number
  
class ContrastivePairLoader(TorchDataset, DatasetTemplate):
  """Adapts a classification loader into a contrastive learning data loader
  Fixes the pairs on the test set, while maintaining training random
  """
  def __init__(self, dataset: DocDataset, protocol: pd.DataFrame = None):
    super(ContrastivePairLoader, self).__init__()
    self.dataset = dataset
    self.train = self.dataset.train
    # if there is no protocol, randomly gather pair (probably train loader)
    if protocol is None:
      self.randomize_pairs()

    # if there is a protocol, just load it
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
      file_path = ds_df.iloc[i]["doc_path"]
      class_number = ds_df.iloc[i]["class_number"]
      y = random.getrandbits(1)
      op = operator.eq if y else operator.ne

      # se y == 1, escolhe um aleatorio da mesma classe
      # se y == 0, escolhe um aleatorio de outra classe
      data_filter = op(ds_df["class_number"], class_number)
      data_filter = data_filter & (ds_df["doc_path"] != file_path)
      filtered_data = ds_df[data_filter]
      if len(filtered_data) < 1:
        data_filter = (ds_df["class_number"] != class_number) & (ds_df["doc_path"] != file_path)
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
    # if this is a train loader, we want a new set of pairs every epoch
    if self.train:
      self.randomize_pairs()
