import os
import pandas as pd
import scipy
import numpy as np
import shutil
import tqdm

# sh_function = shutil.copy
sh_function = shutil.move

def move_or_copy(a, b):
    try:
        sh_function(a, b)
    except:
        pass

clusters_folder = "clusters/" # pasta com os csvs
images_folder = "/mnt/c/Users/lucas/Datasets/rvl-cdip" # pasta com as imagens
clusters_dest_folder = "/mnt/c/Users/lucas/Datasets/rvl-cdip/clusters" # pasta destino

def mkdir(folder):
    try:
        os.mkdir(folder)
    except FileExistsError as e:
        print(e)

mkdir(clusters_dest_folder)

for i, elem in tqdm.tqdm(enumerate(os.listdir(clusters_folder)), "CSVs"):
    if i == 0: continue
    class_path = os.path.join(clusters_dest_folder, str(i))
    csv_path = os.path.join(clusters_folder, elem)
    mkdir(class_path)
    df = pd.read_csv(f"./clusters/distances_{i}.csv", index_col=0)
    df_clusters = df["label"].unique()
    cluster_rank = {"cluster": [], "mean_distance": [], "cluster_size": []}

    for cluster in df_clusters:
        cluster_elements = df[df["label"] == cluster]
        arrays = [np.fromstring(elem[1:-1], sep=" ") for elem in list(cluster_elements["features"].values)]
        md = np.mean(scipy.spatial.distance.cdist(arrays, arrays))
        cluster_rank["mean_distance"].append(md)
        cluster_rank['cluster'].append(cluster)
        cluster_rank["cluster_size"].append(len(cluster_elements))

    cluster_rank = pd.DataFrame(cluster_rank)

    for group, f in tqdm.tqdm([
        ("entre 5 e 10", lambda x: (x > 5) & (x <= 10)),
        ("entre 10 e 20", lambda x: (x > 10) & (x <= 20)),
        ("mais de 20", lambda x: x > 20),
        ("5 ou menos", lambda x: x <= 5),
    ], 'group'):
        group_folder = os.path.join(class_path, group)
        mkdir(group_folder)

        group_clusters = cluster_rank[f(cluster_rank["cluster_size"])]
        ordered_clusters = group_clusters.sort_values(by=["mean_distance"])

        for j, row in tqdm.tqdm(enumerate(ordered_clusters.iloc), "cluster"):
            cluster_index = row["cluster"]
            cluster_elements = df[df["label"] == cluster_index]

            cluster_folder = os.path.join(group_folder, f"{str(j)} - c{cluster_index}")
            mkdir(cluster_folder)

            for file_id in cluster_elements["name"].values:
                file_path = os.path.join(images_folder, file_id)
                move_or_copy(file_path, cluster_folder)
