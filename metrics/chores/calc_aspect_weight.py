import os
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from utils import load_pickle, save_pickle


def to_recursive_dict(d: defaultdict) -> dict:
    """
    Convert a dictionary to a recursive dictionary
    """
    if isinstance(d, defaultdict):
        d = {k: to_recursive_dict(v) for k, v in d.items()}
    return d


def load_index(index_dir: os.PathLike):
    assert os.path.exists(index_dir)
    with open(os.path.join(index_dir, "train.index"), "r") as f:
        train_index = [int(x) for x in f.readline().split(" ")]
    with open(os.path.join(index_dir, "validation.index"), "r") as f:
        valid_index = [int(x) for x in f.readline().split(" ")]
    with open(os.path.join(index_dir, "test_full.index"), "r") as f:
        test_index = [int(x) for x in f.readline().split(" ")]
    return train_index, valid_index, test_index


def save_index(path: str, index: list):
    with open(path, "w") as f:
        for i, idx in enumerate(index):
            if i == len(index) - 1:
                f.write(f"{idx}")
            else:
                f.write(f"{idx} ")
    return


def calc_aspect_weight(args):
    data = load_pickle(args.data_path)
    train_index, _, test_index = load_index(args.index_dir)
    # aspect_dict = pd.read_csv(args.aspect_path)
    if os.path.exists(args.prob_path):
        print(f"Loading from pre-computed {args.prob_path}")
        train_aspect_weight = load_pickle(args.prob_path)
    else:
        train_aspect_weight = defaultdict(int)  # (u,i): {aspect: weight}
        # load train set only
        for i in train_index:
            d = data[i]
            user = d["user"]
            item = d["item"]
            cat = d["category"]
            train_aspect_weight[cat] += 1

        # normalize and take the reciprocal
        total = sum(train_aspect_weight.values())
        train_aspect_weight = {k: total / v for k, v in train_aspect_weight.items()}

        # save the result in the same directory as data_path
        print(f"Saved to {args.prob_path}")
        # turn to normal dict
        train_aspect_weight = to_recursive_dict(train_aspect_weight)
        save_pickle(args.prob_path, train_aspect_weight)

    # in test_index, randomize a category based on the reciprocal of the weight
    # in test_filepath, leave only the randomized category
    # * Rewrite TEST INDEX *
    test_ui_aspect = defaultdict(list)  # (u,i): [cat]
    for i in test_index:
        d = data[i]
        user = d["user"]
        item = d["item"]
        cat = d["category"]
        test_ui_aspect[(user, item)].append((i, cat))

    selected_test_index = []
    ui_select = {}
    for k, v in test_ui_aspect.items():
        probs = [train_aspect_weight[c] for i, c in v]  # need to be in order
        norm_probs = [p / sum(probs) for p in probs]
        # sum prob to 1
        # if all the probs are 0, then set to equal probability
        if sum(probs) == 0:
            probs = [1 / len(v)] * len(v)
        # randomize a cat based on the weight
        # choice = np.random.choice(aspect_dict["category"], p=user_item_aspect_weight[(user, item)])
        x = np.random.choice(np.arange(len(v)), p=norm_probs)
        choice_i, choice_cat = v[x]
        selected_test_index.append(choice_i)
        ui_select[k] = choice_cat

    # write out the selected test index
    select_test_path = Path(args.index_dir) / "test.index"
    print("Number of test data:", len(selected_test_index))
    save_index(select_test_path, selected_test_index)
    print(f"Saved to {select_test_path}")

    # * TEST DATA select *
    # load test set and then use the weight to choose only one user-item data
    # try:
    #     test_data = load_json(args.test_filepath)
    # except:
    #     test_data = load_jsonl(args.test_filepath)
    # final_data = []
    # other_keys = {
    #     k for k in test_data[0].keys() if k not in aspect_dict["category"].values
    # }
    # for d in test_data:
    #     temp_d = {}
    #     user = d["user"]
    #     item = d["item"]
    #     choice = ui_select[(user, item)]
    #     d["category"] = choice
    #     for k in other_keys:
    #         temp_d[k] = d[k]
    #     temp_d["category"] = choice
    #     temp_d["result"] = d[choice]
    #     final_data.append(temp_d)
    #     test_aspect_select[(user, item)] = choice

    # # save the selected aspect for each user-item pair
    # # and rewrite test.index
    # save_pickle(Path(args.index_dir) / "test_aspect_select.pickle", test_aspect_select)

    # # write out the final_data
    # test_filepath = Path(args.test_filepath)
    # name = test_filepath.name
    # out_testfile_path = test_filepath.parent / f"{name.split('.')[0]}_select.json"
    # save_json(out_testfile_path, final_data)
    # print(f"Saved to {out_testfile_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--auto_arg_by_dataset",
        type=str,
        default="default_value",
        help="automatically decide args by dataset; accepts yelp23, yelp, gest",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="",
    )
    args = parser.parse_args()
    if args.auto_arg_by_dataset == "yelp23":
        dargs = edict(
            data_path=f"nete_format_data/yelp23/reviews.pickle",
            aspect_path=f"nete_format_data/yelp23/aspect_category_index.csv",
            prob_path=f"nete_format_data/yelp23/train_aspect_prob.pickle",
        )

    elif args.auto_arg_by_dataset == "yelp":
        dargs = edict(
            data_path=f"/nete_format_data/yelp/reviews.pickle",
            aspect_path=f"/nete_format_data/yelp/aspect_category_index.csv",
            prob_path=f"/nete_format_data/yelp/train_aspect_prob.pickle",
        )
    else:
        raise ValueError(f"Invalid dataset: {args.auto_arg_by_dataset}")
    args = edict(vars(args))
    args.update(dargs)
    calc_aspect_weight(args)
