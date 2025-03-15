import pandas as pd
import os
from architecture import *
from dataloader import *
from trainer import val_epoch
from loss import ContrastiveLoss
from log import Log
from metrics import EER
import torch
# os.chdir("..")

csvs_path = "./results/csvs"
models_path = "./results/models"
models = os.listdir(csvs_path)

res = {
    "arquitetura": [],
    "versao": [],
    "split": [],
    "numero_split": [],
    "eer val": [],
    "eer test": []
}

for arch in models:
    arch_results_path = os.path.join(csvs_path, arch)
    df = pd.read_csv(arch_results_path)
    for model_name in df["aaa_model_version"].unique():
        for split in df["split_mode"].unique():
            for split_number in df["split_number"].unique():
                temp_df = df[(df["aaa_model_version"] == model_name) & (df["split_mode"] == split) & (df["split_number"] == split_number)]["Name"]
                temp_df = temp_df.reset_index(drop=True)
                try:
                    model_name_temp = temp_df.iloc[0]

                    model_full_path = os.path.join(models_path, arch.split(".")[0], f"{model_name_temp}_best.pt")
                    
                    # load model
                    model = torch.load(model_full_path, map_location="cuda")
                    model = SiameseModel(model.model)

                    csv_path = f"./train_{split}.csv"
                    df_split = pd.read_csv(csv_path, index_col=0)
                    df_split.loc[df_split["split"] == split_number, "split"] = -1
                    df_split.loc[df_split["split"] > 0, "split"] = 0
                    df_split.loc[df_split["split"] == -1, "split"] = 1

                    protocol_path = "./train_protocol.csv"
                    protocol = pd.read_csv(protocol_path)
                    protocol = protocol[(protocol["split_mode"] == f"{split}_split") & (protocol["split_number"] == split_number)]

                    val_loader = DocDataset(df_split, train=False, load_in_ram=False, img_shape=(224, 224), n_channels=3)
                    val_loader = ContrastivePairLoader(val_loader, protocol)
                    val_loader = DataLoader(val_loader, 16, shuffle=False)

                    log = Log(wandb_flag=False)
                    log.create_metric("eer", EER, False)

                    val_epoch(0, val_loader, model, ContrastiveLoss, "cuda", log)

                    eer_val = log.create_metrics_dict(False)["eer"]

                    csv_path = f"./test_{split}.csv"
                    df_split = pd.read_csv(csv_path, index_col=0)
                    df_split.insert(1, "split", 1)
                    protocol_path = "./test_protocol.csv"
                    protocol = pd.read_csv(protocol_path)
                    protocol = protocol[(protocol["split_mode"] == f"{split}_split")]
                    val_loader = DocDataset(df_split, train=False, load_in_ram=False, img_shape=(224, 224), n_channels=3)
                    val_loader = ContrastivePairLoader(val_loader, protocol)
                    val_loader = DataLoader(val_loader, 16, shuffle=False)

                    log = Log(wandb_flag=False)
                    log.create_metric("eer", EER, False)

                    val_epoch(0, val_loader, model, ContrastiveLoss, "cuda", log)

                    eer_test = log.create_metrics_dict(False)["eer"]
                except KeyboardInterrupt:
                    raise

                except Exception:
                    continue

                res["arquitetura"].append(arch)
                res["versao"].append(model_name)
                res["split"].append(split)
                res["numero_split"].append(split_number)
                res["eer val"].append(eer_val)
                res["eer test"].append(eer_test)
            
res = pd.DataFrame(res)
res.to_csv("report.csv")

means_df = {
    "arq": [],
    "ver": [],
    "split": [],
    "mean eer val": [],
    "mean eer test": []
}

for arq in res["arquitetura"].unique():
    arq_df = res[res["arquitetura"] == arq]
    for versao in arq_df["versao"].unique():
        versao_df = arq_df[arq_df["versao"] == versao]
        for split in versao_df["split"].unique():
            split_df : pd.DataFrame = versao_df[versao_df["split"] == split]

            means_df["arq"].append(arq)
            means_df["ver"].append(versao)
            means_df["split"].append(split)
            means_df["mean eer val"].append(split_df["eer val"].mean())
            means_df["mean eer test"].append(split_df["eer test"].mean())

pd.DataFrame(means_df).to_csv("./means.csv")
            