import os
import random
import pandas as pd

train_frac, val_frac = 0.8, 0.2

dataset_path = "separacao-rvl_cdip"
csv_dest_path = "./"

dic = {
    "class_name": [],
    "class_number": [],
    "doc_path": [],
    "doc_id": [],
    # "random_split": [],
    # "zsl_split":[],
    # "gzsl_split": [],
}

classes = os.listdir(dataset_path)
for i, class_ in enumerate(classes):
    class_folder_path = os.path.join(dataset_path, class_)
    docs = os.listdir(class_folder_path)
    for doc in docs:
        dic["doc_id"].append(doc)
        dic["doc_path"].append(os.path.join(class_folder_path, doc))
        dic["class_name"].append(class_)
        dic["class_number"].append(i)

df = pd.DataFrame(dic)
l_df = len(df)
l_classes = len(classes)

random_sample = [0]*round(train_frac*l_df) + [1]*round(val_frac*l_df)# + [2]*round(test_frac*l_df)
zsl_sample = [0]*round(train_frac*l_classes) + [1]*round(val_frac*l_classes)

gzsl_sample_c = [0]*round((train_frac/2)*l_classes) + [1]*round((val_frac/2)*l_classes)
gzsl_sample_d = [0]*round((train_frac/2)*l_df) + [1]*round((val_frac/2)*l_df)

# random split
df = df.sample(frac=1)
df.insert(len(df.columns), "random_split", random_sample)

# zsl split
class_id_list = list(range(len(classes)))
random.shuffle(class_id_list)
zsl_split_indexes = {a: b for a, b in zip(class_id_list, zsl_sample)}
zsl_split = df["class_number"].apply(lambda a: zsl_split_indexes[a])
df.insert(len(df.columns), "zsl_split", zsl_split)

df = df.sort_index()
df.to_csv("./splits.csv", index=False)

# random.shuffle(class_id_list)
# gzsl_c_split_indexes = {a: b for a, b in zip(class_id_list, gzsl_sample_c)}