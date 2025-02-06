import pandas as pd

splits = "./splits.csv"

modals = ["zsl_split", "gzsl_split"]

df = pd.read_csv(splits)
out = {
    "split_mode": [],
    "split_number": [],
    "file_a_idx": [],
    "file_a_name": [],
    "file_b_idx": [],
    "file_b_name": [],
    "is_equal": []
}

for modal in modals:
    splits = df[modal].unique()
    for split in splits:
        test_split = df[df[modal] == split]
        test_elems = test_split["class_number"]
        # test_classes = test_elems.unique()

        for idx in test_elems.index:
            elem = test_elems.loc[idx]
            same = test_elems[(test_elems == elem) & (test_elems.index != idx)]
            diff = test_elems[test_elems != elem]

            if len(same) > 0:
                same_comp = same.sample(n=1)

                out["split_mode"].append(modal)
                out["split_number"].append(split)
                out["file_a_idx"].append(idx)
                out["file_a_name"].append(test_split.loc[idx]["doc_id"])
                out["file_b_idx"].append(same_comp.index[0])
                out["file_b_name"].append(test_split.loc[same_comp.index[0]]["doc_id"])
                out["is_equal"].append(1)

            diff_comp = diff.sample(n=1)

            out["split_mode"].append(modal)
            out["split_number"].append(split)
            out["file_a_idx"].append(idx)
            out["file_a_name"].append(test_split.loc[idx]["doc_id"])
            out["file_b_idx"].append(diff_comp.index[0])
            out["file_b_name"].append(test_split.loc[diff_comp.index[0]]["doc_id"])
            out["is_equal"].append(0)

protocol = pd.DataFrame(out)
protocol.to_csv("./protocol.csv")
