from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

clusters_csv_path = "/home/lucasabm/datasets/rvl-cdip/images/specification.csv"
clusters_dest = ""

agg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)

df = pd.read_csv(clusters_csv_path)
df["features"] = df["features"].apply(eval)

labels = agg.fit_predict(list(df["features"]))
column_name = "cluster_index"
if column_name in df.columns:
    df.drop(column_name, axis=1, inplace=True)
df.insert(2, "cluster_index", labels)
df.to_csv(clusters_csv_path, index=False)
