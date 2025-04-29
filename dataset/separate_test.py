import pandas as pd

df = pd.read_csv("./splits.csv")

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

test_df_zsl.to_csv("./test_zsl.csv")
test_df_gzsl.to_csv("./test_gzsl.csv")
train_df_zsl.to_csv("./train_zsl.csv")
train_df_gzsl.to_csv("./train_gzsl.csv")

df = pd.read_csv("./protocol.csv")
df = df.drop(df.columns[0], axis=1)

test_prot = df[df["split_number"] == 0]
test_prot = test_prot.drop(columns=["split_number"])

train_prot = df[df["split_number"] != 0]
train_prot.loc[train_prot["split_number"] == 5, "split_number"] = 0

test_prot.to_csv("./test_protocol.csv", index=False)
train_prot.to_csv("./train_protocol.csv", index=False)
