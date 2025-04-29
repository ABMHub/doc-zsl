import os
import pandas as pd
import shutil
import tqdm
# nessa pasta devem estar os arquivos train, test e val.txt
labels_path = "/home/lucasabm/datasets/rvl-cdip/labels"

images_path = "/home/lucasabm/datasets/rvl-cdip/images"

# from https://huggingface.co/datasets/aharley/rvl_cdip
class_map = {
  "0": "letter",
  "1": "form",
  "2": "email",
  "3": "handwritten",
  "4": "advertisement",
  "5": "scientific report",
  "6": "scientific publication",
  "7": "specification",
  "8": "file folder",
  "9": "news article",
  "10": "budget",
  "11": "invoice",
  "12": "presentation",
  "13": "questionnaire",
  "14": "resume",
  "15": "memo"
}

def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

splits_path = [os.path.join(labels_path, elem) for elem in os.listdir(labels_path)]

d = {
    'image_path': [],
    'class': []
}        
for split in splits_path:
    with open(split, "r") as f:
        lines: list[str] = f.readlines()

        for line in lines:
            p, c = line.split(" ")
            d["image_path"].append(os.path.join(images_path, p.strip()))
            d["class"].append(c.strip())

df = pd.DataFrame(d)
df["class"] = df["class"].apply(lambda a: class_map[a])

l = df["class"].unique()

[mkdir(os.path.join(images_path, elem)) for elem in l]

for elem in tqdm.tqdm(df.iloc):
    image_path = elem["image_path"]
    class_name = elem["class"]

    try:
        shutil.move(image_path, os.path.join(images_path, class_name))
    except:
        pass
