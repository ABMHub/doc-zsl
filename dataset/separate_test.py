import pandas as pd
import os

source_splits_path = "./dataset/active_labeling/loop4/splits_loop4.csv"
source_protocols_path = "./dataset/active_labeling/loop4/protocols_loop4.csv"
dest_base_path = "./dataset/active_labeling/loop4/"
dest_splits_path = "./dataset/active_labeling/loop4/splits"
dest_protocols_path = "./dataset/active_labeling/loop4/protocols"

suffix = 'loop4'

df = pd.read_csv(source_splits_path)

test_df_zsl = df[df["zsl_split"] == 0]
test_df_zsl = test_df_zsl.drop(columns=["zsl_split", "gzsl_split"])
test_df_zsl = test_df_zsl.rename(columns={"zsl_split": "split"})

test_df_gzsl = df[df["gzsl_split"] == 0]
test_df_gzsl = test_df_gzsl.drop(columns=["zsl_split", "gzsl_split"])
test_df_gzsl = test_df_gzsl.rename(columns={"gzsl_split": "split"})

train_df_zsl = df[df["zsl_split"] != 0]
train_df_zsl.loc[train_df_zsl["zsl_split"] == 5, "zsl_split"]  = 0
train_df_zsl = train_df_zsl.drop(columns=["gzsl_split"])
train_df_zsl = train_df_zsl.rename(columns={"zsl_split": "split"})

train_df_gzsl = df[df["gzsl_split"] != 0]
train_df_gzsl.loc[train_df_gzsl["gzsl_split"] == 5, "gzsl_split"] = 0
train_df_gzsl = train_df_gzsl.drop(columns=["zsl_split"])
train_df_gzsl = train_df_gzsl.rename(columns={"gzsl_split": "split"})

test_df_zsl.to_csv(os.path.join(dest_splits_path, f"./test_zsl_{suffix}.csv"))
test_df_gzsl.to_csv(os.path.join(dest_splits_path, f"./test_gzsl_{suffix}.csv"))
train_df_zsl.to_csv(os.path.join(dest_splits_path, f"./train_zsl_{suffix}.csv"))
train_df_gzsl.to_csv(os.path.join(dest_splits_path, f"./train_gzsl_{suffix}.csv"))

df = pd.read_csv(source_protocols_path)
df = df.drop(df.columns[0], axis=1)

test_prot = df[df["split_number"] == 0]
test_prot = test_prot.drop(columns=["split_number"])

train_prot = df[df["split_number"] != 0]
train_prot.loc[train_prot["split_number"] == 5, "split_number"] = 0

test_prot.to_csv(os.path.join(dest_protocols_path, f"./test_protocol_{suffix}.csv"), index=False)
train_prot.to_csv(os.path.join(dest_protocols_path, f"./train_protocol_{suffix}.csv"), index=False)
