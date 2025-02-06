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

n_cross_val = 5

zsl_sample = []
gzsl_sample = []

df = df.sample(frac=1)

remain = l_classes
n_cross_val_temp = n_cross_val
for i in range(n_cross_val):
    split_amount = int(remain//n_cross_val_temp)
    remain -= split_amount
    n_cross_val_temp -= 1

    zsl_sample += [i]*split_amount
    gzsl_sample += [i]*(split_amount//2)

# zsl split
class_id_list = list(range(len(classes)))
random.shuffle(class_id_list)
zsl_split_indexes = {a: b for a, b in zip(class_id_list, zsl_sample)}
zsl_split = df["class_number"].apply(lambda a: zsl_split_indexes[a])
df.insert(len(df.columns), "zsl_split", zsl_split)

df = df.sample(frac=1)

class_id_list = list(range(len(classes)))
random.shuffle(class_id_list)
gzsl_split_indexes = {a: b for a, b in zip(class_id_list, gzsl_sample)}
gzsl_split = df["class_number"].apply(lambda a: gzsl_split_indexes.get(a, -1))

remain = l_df
n_cross_val_temp = n_cross_val
for i in range(n_cross_val):
    already_in_class = (gzsl_split==i).sum()
    need_more = (remain//n_cross_val_temp) - already_in_class
    remain -= already_in_class + need_more

    j = 0
    while need_more > 0:
        if gzsl_split.iloc[j] == -1:
            gzsl_split.iloc[j] = i
            need_more -= 1
        j += 1

    n_cross_val_temp -= 1

df.insert(len(df.columns), "gzsl_split", gzsl_split)
df = df.sort_index()
df.to_csv("./splits.csv", index=False)
# random.shuffle(class_id_list)
# gzsl_c_split_indexes = {a: b for a, b in zip(class_id_list, gzsl_sample_c)}