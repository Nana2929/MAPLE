"""
@File    :   run_feat_metrics.py
@Time    :   2024/08/16 11:11:07
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Compute "Factuality" (iFMR) and "Aspect-wise explainability" (iFCR, GT-FMR) metrics.
"""

import os
import re
from pathlib import Path

import pandas as pd
from tools.feat_metrics import FeatureDictionary, FeatureScorer
from utils import find_outputdir, load_index, load_json_or_jsonl, load_pickle

DATA_ROOT = "nete_format_data"
# ERRA_CKPT_DIR = "model"
# OTHER_CKPT_DIR = "/home/P76114511/projects/aspect_retriever/baselines"
# PEPLER_CKPT_DIR = "/home/P76114511/PEPLER/outputs"
# MONO_CKPT_DIR = "/home/P76114511/projects/aspect_retriever/baselines/mono"


dummy_words = [
    "and",
    "very",
    "the",
    "is",
    "a",
    "an",
    "it",
    "this",
    "that",
    "of",
    "in",
    "that",
    "are",
    "were",
    "was",
    "food",
]


def clean_func(text: str):
    # before the first <eos>
    text = text.split("<eos>")[0]
    # remove <bos>, <pad>, <feat> ...
    text = text.replace("<bos>", "").replace("<pad>", "").replace("<feat>", "")
    return text


def get_output_fname(model: str, input_filename: str):
    if model == "maple":
        pattern = r"generated_(?P<strategy>[^_]+)_k=(?P<topk>\d+)\.jsonl"
        match = re.search(pattern, input_filename)
        if match:
            strategy = match.group("strategy")
            topk = match.group("topk")
            return f"{model}_{strategy}_k={topk}.csv"
    else:
        return f"{model}.csv"


def get_input_fpath(args, index=1):
    model = args.model
    if model == "maple":
        return (
            Path(args.maple_ckpt_dir)
            / args.auto_arg_by_dataset
            / f"{index}"
            / args.input_filename
        )
    else:
        raise ValueError(f"Model {model} not supported.")
    # elif model == "pepler":
    #     return (
    #         Path(PEPLER_CKPT_DIR)
    #         / f"{index}"
    #         / args.auto_arg_by_dataset
    #         / f"{args.auto_arg_by_dataset}mf"
    #         / args.input_filename
    #     )
    # elif model == "mono":
    #     return (
    #         Path(MONO_CKPT_DIR)
    #         / args.auto_arg_by_dataset
    #         / f"{index}"
    #         / "generated.jsonl"
    #     )
    # elif model == "erra":
    #     return (
    #         Path(ERRA_CKPT_DIR)
    #         / args.auto_arg_by_dataset
    #         / f"{index}"
    #         / args.input_filename
    #     )
    # else:
    #     return (
    #         Path(OTHER_CKPT_DIR)
    #         / model
    #         / args.auto_arg_by_dataset
    #         / f"{index}"
    #         / args.input_filename
    #     )


get_longest_feature_triplet = lambda triplets: max(triplets, key=lambda x: len(x[0]))


def main(args):
    data_path = Path(DATA_ROOT) / args.auto_arg_by_dataset / "reviews.pickle"
    reviews = load_pickle(data_path)
    train_item2feature = {
        i: FeatureDictionary(dummy_words, 4) for i in range(1, args.fold_num + 1)
    }
    print(f"Processing model {args.model} on dataset {args.auto_arg_by_dataset}")
    for index in range(1, args.fold_num + 1):
        print(f"Processing fold {index}...")
        index_dir = Path(DATA_ROOT) / args.auto_arg_by_dataset / f"{index}"
        train_index = load_index(index_dir / "train.index")
        train_index = set(train_index)
        # ==========================
        # iterate through the reviews to collect features for each index
        # ==========================
        for idx, review in enumerate(reviews):
            if idx not in train_index:
                continue
            item_id = review["item"]
            fea, _, _, _, _ = review["template"]
            train_item2feature[index].add(key=item_id, feature=fea)
    # ==========================
    # for each fold, calculate the iFMR, iFCR, gt-FMR
    # ==========================
    for index in range(1, args.fold_num + 1):
        rows = []
        ifname = get_input_fpath(args, index)
        print(f"input filepath: {ifname}")
        ofname = get_output_fname(args.model, args.input_filename)
        print(f"output filename: {ofname}")
        index_dir = Path(DATA_ROOT) / args.auto_arg_by_dataset / f"{index}"
        test_index = load_index(index_dir / "test.index")
        # ==========================
        # load the prediction file and preprocess the texts
        # ==========================
        data = load_json_or_jsonl(ifname)
        data = data[: args.max_samples]
        print(f"test index size: {len(test_index)}")
        print(f"test data size: {len(data)}")
        for idx, d in enumerate(data):
            d["fake"] = clean_func(d["fake"])
            review = reviews[test_index[idx]]
            d["gt_features"] = set([fea for fea, _, _, _, _ in review["triplets"]])
        # ==========================
        # instantiate the FeatureScorer and calculate the metrics
        # ==========================
        fscorer = FeatureScorer(
            testdata=data, train_item2feature=train_item2feature[index]
        )
        out = fscorer.calculate()
        rows.append(out)
        df = pd.DataFrame(rows)
        # op = os.path.join(
        #     args.output_dir, args.model, args.auto_arg_by_dataset, str(index), ofname
        # )
        output_dir = find_outputdir(ifname)
        output_path = os.path.join(output_dir, ofname)
        print(f"Saving aspect-wise explainability metrics to {output_path}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--maple_ckpt_dir", type=str, default="checkpoints/reproduce")
    parser.add_argument("--auto_arg_by_dataset", type=str, default="yelp")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--model", type=str, default="maple")
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument(
        "--input_filename", type=str, default="generated_supervised_k=3.jsonl"
    )  # maple: generated_{STRATEGY}_k={TOPK}.jsonl", others: generated_wid.jsonl
    args = parser.parse_args()
    assert args.fold_num > 0 and args.fold_num <= 5
    assert args.max_samples > 0

    main(args)
