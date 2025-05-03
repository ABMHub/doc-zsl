from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import argparse

from absl import app
from absl.flags import argparse_flags

def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dists_csv",
        type=str,
        help="Directory which contains the distance csvs.",
        default="~/datasets/rvl-cdip/images/specification.csv",
    )
    parser.add_argument(
        "--clusters_dest_csv",
        type=str,
        help="Path for the clusters csv.",
        default="~/datasets/rvl-cdip/images/specification_clusters.csv",
    )
    args = parser.parse_args(argv[1:])
    return args

def main(args):
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)

    df = pd.read_csv(args.dists_csv)
    df["features"] = df["features"].apply(eval)

    labels = agg.fit_predict(list(df["features"]))
    column_name = "cluster_index"
    if column_name in df.columns:
        df.drop(column_name, axis=1, inplace=True)
    df.insert(2, "cluster_index", labels)
    df.to_csv(args.clusters_dest_csv, index=False)

if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
