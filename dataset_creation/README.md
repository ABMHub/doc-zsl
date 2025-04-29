# How to improve the dataset

The dataset is, as of right now, entirely based on RVL-CDIP. Therefore, the first step to improve the LA-CDIP dataset is to [download](https://huggingface.co/datasets/aharley/rvl_cdip/blob/main/data/rvl-cdip.tar.gz) the whole dataset.

Afterwards, to save your own time, you should use a model to put each image in a feature space, and use a clustering algorithm to group them together. This should produce a impure, but helpful separation, and taking the manual work from this step should make the labeling process easier and more efficient.

From there, the work is manual and there are two main concerns you should be paying attention at: inter-class and intra-class separation. Each cluster needs to be pure, that means, no two different types of document layout in the same class. And each cluster needs to be unique, that means, there cannot exist two classes with the same layout pattern.

Remembering every class currently separated gets exponentially harder as the dataset grows. Therefore, you should use the RVL-CDIP original class labels at your favor, and assume that no layout will be seen in two different RVL-CDIP classes. This allows you to label documents independently across every one of the 16 RVL-CDIP classes. Remembering existing classes is a problem I encorage you to create strategies to solve, or use labeling tools to solve this problem for you somehow.