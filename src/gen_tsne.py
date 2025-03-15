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
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
# os.chdir("..")

csvs_path = "./results/csvs"
models_path = "./results/models"
dataset_path = "./"
models = os.listdir(csvs_path)

df_prot = pd.read_csv("./test_zsl.csv")

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
                model = model.model

                print(im_shape)
                print(f"{arch}_{model_name}_{str(split_number)}")

                spaces = []
                for elem in tqdm(df_prot.iloc):
                    file1 = elem["doc_path"]
                    file1_fullpath = os.path.join(dataset_path, file1)

                    im1_processed = process_image(file1_fullpath, im_shape)
                    im1_processed = im1_processed.to("cuda")
                    y1 = model(im1_processed.unsqueeze(0))
                    spaces.append(y1.detach().cpu().numpy()[0])

                tsne = TSNE(n_components=2, random_state=42)
                X_embedded = tsne.fit_transform(np.array(spaces))

                print(df_prot["class_number"].values)
                ys = np.array([str(i) for i in df_prot["class_number"].values])
                sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=ys, legend=False, s=50)
                title = f"{arch.split(".")[0].replace("T2", "")} {model_name}"
                plt.title(title, fontdict={"fontsize": 20})
                plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
                plt.savefig(f"plot/{arch}_{model_name}_{str(split_number)}.pdf", dpi=300, bbox_inches="tight")
                plt.close()

            except KeyboardInterrupt:
                raise

            except Exception:
                raise
