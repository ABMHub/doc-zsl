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
from collections.abc import Callable

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
    n_channels: int = 3,
    preprocessor: Callable[[Image.Image], torch.Tensor] = None
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
        preprocessor: a callable to overwrite the entire default preprocessing pipeline
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
    if preprocessor is not None:
      self.process_image = lambda path: preprocessor(images=Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"][0]
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

  def prepare_protocol(self, protocol: pd.DataFrame) -> None:
    """Converts the dataframe protocol into image pairs in the dataloader

    Args:
        protocol (pd.DataFrame): the testing protocol
    """
    dic ={
      "x1": [self.dataset.df.index.get_loc(elem) for elem in protocol["file_a_idx"].values],
      "x2": [self.dataset.df.index.get_loc(elem) for elem in protocol["file_b_idx"].values],
      "y": list(protocol["is_equal"].values),
    }

    self.df = pd.DataFrame(dic)

  def randomize_pairs(self) -> None:
      """Create a random list of image pairs. Likely to be used in training.
      """
      ds_df = self.dataset.df
      n_samples = len(self.dataset)
      
      # Pre-compute class groupings
      class_to_indices = {}
      for idx, class_num in enumerate(ds_df["class_number"]):
          if class_num not in class_to_indices:
              class_to_indices[class_num] = []
          class_to_indices[class_num].append(idx)
      
      # Convert to numpy arrays for faster access
      classes = ds_df["class_number"].values
      all_classes = list(class_to_indices.keys())
      
      x1_list = list(range(n_samples))
      x2_list = []
      y_list = []
      
      for i in range(n_samples):
          current_class = classes[i]
          y = random.getrandbits(1)
          
          if y == 1:  # Same class
              same_class_indices = class_to_indices[current_class]
              # Filter out the current index
              candidates = [idx for idx in same_class_indices if idx != i]
              
              if len(candidates) < 1:
                  # Fallback: choose different class
                  other_classes = [c for c in all_classes if c != current_class]
                  chosen_class = random.choice(other_classes)
                  candidates = class_to_indices[chosen_class]
              
              x2_idx = random.choice(candidates)
          else:  # Different class
              other_classes = [c for c in all_classes if c != current_class]
              chosen_class = random.choice(other_classes)
              x2_idx = random.choice(class_to_indices[chosen_class])
          
          x2_list.append(x2_idx)
          y_list.append(y)
      
      self.df = pd.DataFrame({
          "x1": x1_list,
          "x2": x2_list,
          "y": y_list
      })

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
