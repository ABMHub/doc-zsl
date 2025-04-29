from internal.architecture import *
from internal.dataloader import DocDataset

from PIL import Image
from torchvision.transforms import ToTensor

import os
import torch
import numpy as np
import pandas as pd
import tqdm

model_path = "./trained_models/resnet_cluster/resnet_cluster_best.pt"
dataset_path = "/home/lucasabm/datasets/rvl-cdip/images/specification"
cuda = True

# from dataloader.py
def process_image(
    file_path: os.PathLike,
    img_shape: Tuple[int, int] = (224, 224),
    n_channels: int = 3,
    mean: float = 0.9402,
    std: float = 0.1724,
) -> torch.Tensor:
    """Preprocess image
    - Resize to desired shape
    - Convert to either RGB or grayscale
    - Standardization following given mean and std

    Args:
        file_path (os.PathLike): path containing a image (png, jpg, tiff...)

    Returns:
        torch.Tensor: preprocessed image matrix
    """
    im: Image.Image = Image.open(file_path)

    im = im.resize(img_shape)
    im = im.convert("L") if n_channels == 1 else im.convert("RGB")

    return (ToTensor()(im) - mean) / std

cuda = "cuda" if cuda else "cpu"

model: SiameseModel = torch.load(model_path, cuda)
model = model.model

data_list_path = os.listdir(dataset_path)
data_list_path = [os.path.join(dataset_path, elem) for elem in data_list_path]

d = {
    'file_name': [],
    'features': [],
}

for elem in tqdm.tqdm(data_list_path):
    tensor = process_image(elem)
    batch = tensor.unsqueeze(0) # pura preguiça sinceramente
    batch = batch.to(cuda)
    feature_space: np.ndarray = model(batch)[0].detach().cpu().numpy()

    d["features"].append(feature_space.tolist())
    d["file_name"].append(elem)

class_name = dataset_path.split("/")[-1]

df = pd.DataFrame(d)
df.to_csv(os.path.join(dataset_path, '..', f"{class_name}.csv"), index=False)
