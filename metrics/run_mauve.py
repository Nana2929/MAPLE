"""
@File    :   run_mauve.py
@Time    :   2024/08/16 11:12:34
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Compute MAUVE metric.
@Credit  :   Updated from https://github.com/zhouhanxie/PRAG/tree/main/evaluations/mauve
"""

import mauve
import pandas as pd
from utils import load_index, load_pickle, load_json_or_jsonl, find_outputdir
import os
import argparse
from easydict import EasyDict as edict
from pathlib import Path

DATA_PATH = Path("nete_format_data")


def clean_func(text: str):
    # before the first <eos>
    text = text.split("<eos>")[0]
    # remove <bos>, <pad>, <feat> ...
    text = text.replace("<bos>", "").replace("<pad>", "").replace("<feat>", "")
    return text


def get_filename(input_file: str):
    return os.path.basename(input_file).split(".")[0]  # remove extension


select_longest_triplet = lambda triplets: max(triplets, key=lambda x: len(x[0]))


def main(args):
    generated = load_json_or_jsonl(args.input_file)
    generated = [clean_func(g["fake"]) for g in generated]
    reviews = load_pickle(args.data_path)
    testindex = load_index(args.index_dir / "test.index")
    # Clipping
    generated = generated[: args.max_samples]
    testindex = testindex[: args.max_samples]

    gold = []
    if args.auto_arg_by_dataset == "yelp":
        for idx in testindex:
            review = reviews[idx]
            text = review["template"][2]
            gold.append(text)
    elif args.auto_arg_by_dataset == "yelp23":
        for idx in testindex:
            review = reviews[idx]
            triplets = review["triplets"]
            text = select_longest_triplet(triplets)[2]
            # text = " ".join([i["text"] for i in triplets])
            gold.append(text)

    out = mauve.compute_mauve(
        p_text=generated,
        q_text=gold,
        device_id=0,
        max_text_length=60,
        verbose=True,
    )
    output_dir = find_outputdir(args.input_file)
    input_filename = get_filename(args.input_file)
    outname = f"mauve_{input_filename}.log"

    mauve_file = os.path.join(output_dir, outname)
    print(f"Saving mauve results to {mauve_file}")
    with open(mauve_file, "w") as f:
        f.write(str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MAUVE metric for generated text"
    )
    parser.add_argument(
        "-a", "--auto_arg_by_dataset", type=str, default=None, help="auto argument"
    )
    parser.add_argument(
        "-i", "--index", type=int, default=1, help="index of the dataset"
    )
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--max_samples", type=int, default=10000, help="max samples")
    args = parser.parse_args()
    index = args.index

    assert args.auto_arg_by_dataset in ("yelp", "yelp23", "tripadvisor")
    dargs = edict(
        dict(
            data_path=DATA_PATH / f"{args.auto_arg_by_dataset}/reviews.pickle",
            index_dir=DATA_PATH / f"{args.auto_arg_by_dataset}/{index}",
        )
    )

    args = vars(args)
    args.update(dargs)
    args = edict(args)

    main(args)
