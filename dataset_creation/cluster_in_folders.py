import os
import pandas as pd
import scipy
import numpy as np
import shutil
import tqdm
import argparse

from absl import app
from absl.flags import argparse_flags


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--clusters_csv",
        type=str,
        help="Directory which contains the clusters csvs.",
        default="~/datasets/rvl-cdip/images/specification_clusters.csv",
    )
    parser.add_argument(
        "--clusters_dest_folder",
        type=str,
        help="Output directory with results",
        default="~/datasets/rvl-cdip/clusters/specification",
    )
    args = parser.parse_args(argv[1:])
    return args


sh_function = shutil.copy
# sh_function = shutil.move

def move_or_copy(a, b):
    try:
        sh_function(a, b)
    except:
        pass


def mkdir(folder):
    try:
        os.makedirs(folder, exist_ok=True)
    except FileExistsError as e:
        print(e)


def main(args):
    mkdir(args.clusters_dest_folder)

    df = pd.read_csv(args.clusters_csv)
    df_clusters = df["cluster_index"].unique()
    cluster_rank = {"cluster": [], "mean_distance": [], "cluster_size": []}

    for cluster in df_clusters:
        cluster_elements = df[df["cluster_index"] == cluster]
        arrays = [
            np.fromstring(elem[1:-1], sep=" ")
            for elem in list(cluster_elements["features"].values)
        ]
        md = np.mean(scipy.spatial.distance.cdist(arrays, arrays))
        cluster_rank["mean_distance"].append(md)
        cluster_rank["cluster"].append(cluster)
        cluster_rank["cluster_size"].append(len(cluster_elements))

    cluster_rank = pd.DataFrame(cluster_rank)

    for group, f in tqdm.tqdm(
        [
            ("between 5 and 10", lambda x: (x > 5) & (x <= 10)),
            ("between 10 and 20", lambda x: (x > 10) & (x <= 20)),
            ("more than 20", lambda x: x > 20),
            ("5 or less", lambda x: x <= 5),
        ],
        "group",
    ):
        group_folder = os.path.join(args.clusters_dest_folder, group)
        mkdir(group_folder)

        group_clusters = cluster_rank[f(cluster_rank["cluster_size"])]
        ordered_clusters = group_clusters.sort_values(by=["mean_distance"])

        for j, row in tqdm.tqdm(enumerate(ordered_clusters.iloc), "cluster"):
            cluster_index = row["cluster"]
            cluster_elements = df[df["cluster_index"] == cluster_index]

            cluster_folder = os.path.join(
                group_folder, f"{str(j)} - c{cluster_index}"
            )
            mkdir(cluster_folder)

            for file_path in cluster_elements["file_name"].values:
                move_or_copy(file_path, cluster_folder)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
