import pandas as pd
import os
from internal.architecture import *
from internal.dataloader import *
from internal.trainer import val_epoch
from internal.loss import ContrastiveLoss
from internal.log import Log
from internal.metrics import EER
import torch
from transformers import BeitImageProcessor, BeitForMaskedImageModeling
# os.chdir("..")

# processor = BeitImageProcessor.from_pretrained("microsoft/dit-base")
model = DiT()

SPLIT_FOLDER = "dataset/splits"
PROTOCOL_FOLDER = "dataset/protocols"

# load model
model.cuda()
model = SiameseModel(model)
split = "zsl"
split_number = 0
# csv_path = f"./train_{split}.csv"
# df_split = pd.read_csv(csv_path, index_col=0)
# df_split.loc[df_split["split"] == split_number, "split"] = -1
# df_split.loc[df_split["split"] > 0, "split"] = 0
# df_split.loc[df_split["split"] == -1, "split"] = 1

# protocol_path = "./train_protocol.csv"
# protocol = pd.read_csv(protocol_path)
# protocol = protocol[(protocol["split_mode"] == f"{split}_split") & (protocol["split_number"] == split_number)]

# val_loader = DocDataset(df_split, train=False, load_in_ram=False, img_shape=(224, 224), n_channels=3)
# val_loader = ContrastivePairLoader(val_loader, protocol)
# val_loader = DataLoader(val_loader, 16, shuffle=False)

# log = Log(wandb_flag=False)
# log.create_metric("eer", EER, False)

# val_epoch(0, val_loader, model, ContrastiveLoss, "cuda", log)

# eer_val = log.create_metrics_dict(False)["eer"]

csv_path = os.path.join(SPLIT_FOLDER, f"./test_{split}.csv")
df_split = pd.read_csv(csv_path, index_col=0)
df_split.insert(1, "split", 1)
protocol_path = os.path.join(PROTOCOL_FOLDER, "./test_protocol.csv")
protocol = pd.read_csv(protocol_path)
protocol = protocol[(protocol["split_mode"] == f"{split}_split")]
val_loader = DocDataset(df_split, train=False, load_in_ram=False, img_shape=(224, 224), n_channels=3)
val_loader = ContrastivePairLoader(val_loader, protocol)
val_loader = DataLoader(val_loader, 16, shuffle=False)

log = Log(wandb_flag=False)
log.create_metric("eer", EER, False)

val_epoch(0, val_loader, model, ContrastiveLoss, "cuda", log)

eer_test = log.create_metrics_dict(False)["eer"]
print(eer_test)
            