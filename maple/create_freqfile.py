import os
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from easydict import EasyDict as edict
from utils import timer, save_pickle, load_pickle

"""
For mllt resampled loss to work, we need to create a frequency file for each dataset.
"""

DATA_ROOT = Path("nete_format_data")


def read_aspect(path: os.PathLike):
    df = pd.read_csv(path)
    cat2idx = {c: i for i, c in enumerate(df["category"])}
    return cat2idx


def read_train_index(index_dir: os.PathLike):
    with open(os.path.join(index_dir, "train.index"), "r") as f:
        train_index = [int(x) for x in f.readline().split(" ")]
    return train_index


@timer
def main(args):
    INDEX_NUM = 5

    if os.path.exists(args.freq_file):
        print(f"file {args.freq_file} exists; overwriting!!")

    data = load_pickle(args.data_path)
    cat2idx = read_aspect(args.aspect_path)

    print(f"cat2idx: {cat2idx}")
    class_freq = [np.zeros(len(cat2idx)) for _ in range(INDEX_NUM)]
    neg_class_freq = [np.zeros(len(cat2idx)) for _ in range(INDEX_NUM)]
    train_index = [None for _ in range(INDEX_NUM)]
    for index in range(INDEX_NUM):
        index_dir = args.data_path.parent / str(index + 1)
        train_index[index] = set(read_train_index(index_dir))

    for idx, d in enumerate(tqdm(data)):
        for index in range(INDEX_NUM):
            if idx in train_index[index]:
                category = d["category"]
                if category not in cat2idx:
                    continue
                cat_idx = cat2idx[category]
                class_freq[index][cat_idx] += 1

    # save the file as a dictionary pickled with 2 keys
    for index in range(INDEX_NUM):
        # check validd data total
        data_size = sum(class_freq[index])
        for c in range(len(cat2idx)):
            neg_class_freq[index][c] = data_size - class_freq[index][c]

        index_dir = args.data_path.parent / str(index + 1)
        out_dict = {
            "class_freq": class_freq[index],
            "neg_class_freq": neg_class_freq[index],
        }
        freq_filepath = os.path.join(index_dir, args.freq_file)
        print("out_dict:", out_dict)
        save_pickle(freq_filepath, out_dict)
        print("finished saving frequency file into {}".format(freq_filepath))


if __name__ == "__main__":
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-a", "--auto_arg_by_dataset", type=str)
    parser.add_argument("--freq_file", type=str, default="freq_file.pkl")
    args = parser.parse_args()
    if args.auto_arg_by_dataset == "yelp":
        dargs = edict(
            data_path=DATA_ROOT / "yelp/reviews.pickle",
            aspect_path=DATA_ROOT / "yelp/aspect_category_index.csv",
        )
    elif args.auto_arg_by_dataset == "yelp23":
        dargs = edict(
            data_path=DATA_ROOT / "yelp23/reviews.pickle",
            aspect_path=DATA_ROOT / "yelp23/aspect_category_index.csv",
        )
    else:
        raise ValueError(f"Invalid dataset: {args.auto_arg_by_dataset}")

    args = edict(vars(args))
    args.update(dargs)
    main(args)
