'''
@File    :   postprocess_uie.py
@Time    :   2023/12/25 16:38:13
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Copied from notebooks/6.0-gest(yelp)-absa-eda.ipynb, and modified to be a script.
'''
import logging
import os
from pathlib import Path
from pprint import pprint

import fire
from easydict import EasyDict as edict
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from common import io, utils
from common.data_model import UIE_TRIPLET

MATCH = " and "


def generate_combined_range(set1: set, set2: set) -> range:
    """
    util function to postprocess the uie results (combine the aspect and opinion token sets and return the
    range of the combined set, so as to recover the original text span of the aspect and opinion)
    """

    if not set1 or not set2:
        raise ValueError("Input lists must not be empty")

    start_range = min(min(set1), min(set2))
    end_range = max(max(set1), max(set2)) + 1

    return range(start_range, end_range)

def _process(entries: list, wnl: WordNetLemmatizer)->list:
    for entry in tqdm(entries):
        # user = entry['user_id']
        # bus = entry['business_id']
        tokens = entry['tokens']
        uie_absa = entry['uie_absa']
        # no label
        if not 'relation' in uie_absa:
            continue
        raw_triplet_offsets = uie_absa['relation']['offset']
        raw_triplet_strings = uie_absa['relation']['string']

        triplets = []
        for offset, string in zip(raw_triplet_offsets, raw_triplet_strings):
            sentiment, _, aspect_token_set, _, opinion_token_set = offset
            sentiment, _, aspect_term, _, opinion_term = string
            aspect = aspect_term # aka ' '.join([tokens[i] for i in aspect_token_set])
            opinion = opinion_term # aka ' '.join([tokens[i] for i in opinion_token_set])
            rng = generate_combined_range(aspect_token_set, opinion_token_set)
            text = ' '.join([tokens[i] for i in rng])

            lemm_opinion = utils.text_normalize(opinion, lemmatizer=wnl)

            # if MATCH not in aspect, the single triplet is added
            # (.split() returns a list of one element)
            sep_aspects = aspect.split(MATCH)
            for sep_aspect in sep_aspects:
                # LEMMATIZE + PUNCTS REMOVAL
                lemm_sep_aspect = utils.text_normalize(text=sep_aspect, lemmatizer=wnl)
                triplets.append(UIE_TRIPLET(aspect=lemm_sep_aspect, opinion=lemm_opinion,
                                    text=text, sentiment=sentiment))
                        # add an extra key to place the processed triplets
        entry['uie_absa']['triplet'] = triplets
    return entries



def process(args: edict)->None:
    """
    Postprocess the raw UIE data and save the processed data to the output directory.
    """
    wnl = WordNetLemmatizer()
    os.makedirs(args.output_dir, exist_ok=True)
    # if args.dataset_name.lower() == "gest":
    #     train_data = io.load_uie_files(args.data_root, "train")
    #     test_data = io.load_uie_files(args.data_root, "test")
    #     val_data = io.load_uie_files(args.data_root, "val")
    #     data = {"train": train_data, "test": test_data, "val": val_data}
    #     for split_name in ['train', 'test', 'val']:
    #         data[split_name] = _process(data[split_name], wnl)
    #     for split_name in ['train', 'test', 'val']:
    #         save_path = Path(args.output_dir) / f"{args.dataset_name}_{split_name}_reviews_uie.pkl"
    #         logging.info(f"Peeking at {split_name} data:")
    #         pprint(data[split_name][0])
    #         io.save_pickle(data[split_name], save_path)
    #         logging.info(f"Saved {split_name} (#:{len(data[split_name])}) data to {save_path}")

    if args.dataset_name.lower() == "yelp23":
        data = io.load_uie_files(args.data_root)
        data = _process(data, wnl)
        save_path = Path(args.output_dir) / f"{args.dataset_name}_reviews_uie.pkl"
        io.save_pickle(data, save_path)
        logging.info(f"Saved data (#:{len(data)}) to {save_path}")



def determine_args(dataset_name: str):
    # if dataset_name.lower() == "gest":
    #     data_root = "/workspace/P76114511/Gest"
    #     output_dir = "../Gest/uie_annotated_data_postprocessed"
    #     args = edict({
    #         "data_root": data_root,
    #         "output_dir": output_dir,
    #         "dataset_name": "gest"
    #     })
    if dataset_name.lower() == "yelp23":
        data_root = "/workspace/P76114511/yelp23_uie"
        output_dir = "../yelp_2023/uie_annotated_data_postprocessed"
        args = edict({
            "data_root": data_root,
            "output_dir": output_dir,
            "dataset_name": "yelp23"
        })
    return args

def main(auto_arg_by_dataset: str):
    args = determine_args(auto_arg_by_dataset)
    process(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

