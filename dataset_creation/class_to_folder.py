import os
import pandas as pd
import shutil
import tqdm
import argparse

from absl import app
from absl.flags import argparse_flags


def get_class_map():
    # from https://huggingface.co/datasets/aharley/rvl_cdip
    return {
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
        "15": "memo",
    }


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        help="Train.txt, test.txt, and val.txt from rvl-cdip dataset.",
        default="~/datasets/rvl-cdip/labels",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        help="Path directories of rvl-cdip dataset.",
        default="~/datasets/rvl-cdip/images",
    )
    args = parser.parse_args(argv[1:])
    return args


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def main(args):
    splits_path = [
        os.path.join(args.labels_path, elem)
        for elem in os.listdir(args.labels_path)
    ]

    d = {"image_path": [], "class": []}
    for split in splits_path:
        with open(split, "r") as f:
            lines: list[str] = f.readlines()

            for line in lines:
                p, c = line.split(" ")
                path = os.path.join(args.images_path, p.strip())
                d["image_path"].append(path)
                d["class"].append(c.strip())

    df = pd.DataFrame(d)
    class_map = get_class_map()
    df["class"] = df["class"].apply(lambda a: class_map[a])

    l = df["class"].unique()

    for elem in l:
        mkdir(os.path.join(args.images_path, elem))

    for elem in tqdm.tqdm(df.iloc):
        image_path = elem["image_path"]
        class_name = elem["class"]

        try:
            shutil.move(image_path, os.path.join(args.images_path, class_name))
        except:
            pass


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
