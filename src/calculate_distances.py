from internal.architecture import *
from internal.dataloader import DocDataset

from PIL import Image
from torchvision.transforms import ToTensor

import os
import torch
import numpy as np
import pandas as pd
import tqdm
import argparse

from absl import app
from absl.flags import argparse_flags


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Pretrained model (Resnet).",
        default="./dataset_creation/resnet_cluster_best.pt",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to a subset of rvl-cdip dataset.",
        default="~/datasets/rvl-cdip/images/specification",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device used by Pytorch computation.",
        default="cuda",
    )
    args = parser.parse_args(argv[1:])
    return args


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


def main(args):
    device = "cuda" if args.device == "cuda" else "cpu"

    model: SiameseModel = torch.load(args.model_path, device)
    model = model.model

    data_list_path = os.listdir(args.dataset_path)
    data_list_path = [
        os.path.join(args.dataset_path, elem) for elem in data_list_path
    ]

    d = {
        "file_name": [],
        "features": [],
    }

    for elem in tqdm.tqdm(data_list_path):
        tensor = process_image(elem)
        batch = tensor.unsqueeze(0)  # should be a proper batch
        batch = batch.to(device)
        feature_space: np.ndarray = model(batch)[0].detach().cpu().numpy()

        d["features"].append(feature_space.tolist())
        d["file_name"].append(elem)

    class_name = args.dataset_path.split("/")[-1]

    df = pd.DataFrame(d)
    df.to_csv(
        os.path.join(args.dataset_path, "..", f"{class_name}.csv"), index=False
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
