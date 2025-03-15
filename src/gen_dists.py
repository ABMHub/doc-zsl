import pandas as pd
import os
from architecture import *
from dataloader import *
from trainer import val_epoch
from loss import ContrastiveLoss
from log import Log
from metrics import EER
import torch
import numpy as np
from tqdm import tqdm
# os.chdir("..")

csvs_path = "./results/csvs"
models_path = "./results/models"
dataset_path = "./separacao-rvl_cdip"
models = os.listdir(csvs_path)

df_prot = pd.read_csv("./deepseek_zsl_checkpoint.csv")

def process_image(file_path, im_shape):
    im = Image.open(file_path)
    im : Image.Image

    mean = 0.9402
    std = 0.1724

    im = im.resize(im_shape)
    im = im.convert("RGB")

    return (ToTensor()(im) - mean) / std

for arch in models:
    arch_results_path = os.path.join(csvs_path, arch)
    df = pd.read_csv(arch_results_path)
    for model_name in df["aaa_model_version"].unique():
        split = "zsl"
        for split_number in df["split_number"].unique():
            temp_df = df[(df["aaa_model_version"] == model_name) & (df["split_mode"] == split) & (df["split_number"] == split_number)]["Name"]
            temp_df = temp_df.reset_index(drop=True)
            try:
                model_name_temp = temp_df.iloc[0]

                model_full_path = os.path.join(models_path, arch.split(".")[0], f"{model_name_temp}_best.pt")
                
                # load model
                model = torch.load(model_full_path, map_location="cuda")
                im_shape = (model.model.im_shape, model.model.im_shape)
                model = SiameseModel(model.model)

                print(im_shape)
                print(f"{arch}_{model_name}_{str(split_number)}")

                dists = []
                for elem in tqdm(df_prot.iloc):
                    file1, file2 = elem["file1"], elem["file2"]
                    file1_fullpath = os.path.join(dataset_path, file1)
                    file2_fullpath = os.path.join(dataset_path, file2)

                    im1_processed, im2_processed = process_image(file1_fullpath, im_shape), process_image(file2_fullpath, im_shape)
                    im1_processed, im2_processed = im1_processed.to("cuda"), im2_processed.to("cuda")
                    y1, y2 = model(im1_processed.unsqueeze(0), im2_processed.unsqueeze(0))
                    dist = np.linalg.norm(y1.detach().cpu().numpy()-y2.detach().cpu().numpy())
                    dists.append(dist)

                df_prot.insert(len(df_prot.columns), f"{arch.split(".")[0]}_{model_name}_{str(split_number)}", dists)
                df_prot.to_csv("./distances_vision_models.csv")

            except KeyboardInterrupt:
                raise

            except Exception:
                raise
