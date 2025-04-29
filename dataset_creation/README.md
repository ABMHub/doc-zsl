# How to improve the dataset

The dataset is, as of right now, entirely based on RVL-CDIP. Therefore, the first step to improve the LA-CDIP dataset is to [download](https://huggingface.co/datasets/aharley/rvl_cdip/blob/main/data/rvl-cdip.tar.gz) the whole dataset.

Afterwards, to save your own time, you should use a model to put each image in a feature space, and use a clustering algorithm to group them together. This should produce a impure, but helpful separation, and taking the manual work from this step should make the labeling process easier and more efficient.

From there, the work is manual and there are two main concerns you should be paying attention at: inter-class and intra-class separation. Each cluster needs to be pure, that means, no two different types of document layout in the same class. And each cluster needs to be unique, that means, there cannot exist two classes with the same layout pattern.

Remembering every class currently separated gets exponentially harder as the dataset grows. Therefore, you should use the RVL-CDIP original class labels at your favor, and assume that no layout will be seen in two different RVL-CDIP classes. This allows you to label documents independently across every one of the 16 RVL-CDIP classes. Remembering existing classes is a problem I encorage you to create strategies to solve, or use labeling tools to solve this problem for you somehow.

## Avaible Scripts

To get to the point where you need to manually separate the data, you need to follow a few steps:

1. [Download RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip/blob/main/data/rvl-cdip.tar.gz)
2. Use the `dataset_creation/class_to_folder.py` script to move each RVL-CDIP class to its respective folder
3. (Optional) Train a model with the avaible dataset. You may use one already trained.
4. Use the `src/calculate_distances.py` script to use the trained model to generate the feature-value for each document in a RVL-CDIP class.
5. Use the `dataset_creation/create_clusters.py` to aggregate the documents into clusters.
6. Use the `dataset_creation/cluster_in_folders.py` to assing each cluster to a folder, making it easy to do the manual labeling work.

Do these steps in each RVL-CDIP class independently, as there should not be many overlaps of document layouts between the classes.

In the end, each class should be separated into 4 categories:
- clusters with more than 20 elements
- clusters between 10 and 20 elements
- clusters between 5 and 10 elements
- clusters with at most 5 elements

This should optimize the time spent, as the clusters with more than 20 elements should be more valuable to look. To further optimize the job, the clusters so the first cluster in the folder has a lower mean distance, meaning the data is closer together, and the documents look similar.
