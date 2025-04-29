import os
import pandas as pd
import scipy
import numpy as np
import shutil
import tqdm

sh_function = shutil.copy
# sh_function = shutil.move

def move_or_copy(a, b):
    try:
        sh_function(a, b)
    except:
        pass

clusters_csv = "/home/lucasabm/datasets/rvl-cdip/images/specification.csv" # pasta com os csvs
images_folder = "/home/lucasabm/datasets/rvl-cdip/images/specification" # pasta com as imagens
clusters_dest_folder = "/home/lucasabm/datasets/rvl-cdip/clusters/specification" # pasta destino

def mkdir(folder):
    try:
        os.mkdir(folder)
    except FileExistsError as e:
        print(e)

mkdir(clusters_dest_folder)

df = pd.read_csv(clusters_csv)
df_clusters = df["cluster_index"].unique()
cluster_rank = {"cluster": [], "mean_distance": [], "cluster_size": []}

for cluster in df_clusters:
    cluster_elements = df[df["cluster_index"] == cluster]
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
    group_folder = os.path.join(clusters_dest_folder, group)
    mkdir(group_folder)

    group_clusters = cluster_rank[f(cluster_rank["cluster_size"])]
    ordered_clusters = group_clusters.sort_values(by=["mean_distance"])

    for j, row in tqdm.tqdm(enumerate(ordered_clusters.iloc), "cluster"):
        cluster_index = row["cluster"]
        cluster_elements = df[df["cluster_index"] == cluster_index]

        cluster_folder = os.path.join(group_folder, f"{str(j)} - c{cluster_index}")
        mkdir(cluster_folder)

        for file_path in cluster_elements["file_name"].values:
            move_or_copy(file_path, cluster_folder)
