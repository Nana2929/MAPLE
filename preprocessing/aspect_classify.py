"""
@File    :   aspect_classify.py
@Time    :   2023/11/22 10:57:10
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   0-shot aspect classification
            Save aspect2category dict to save_dir/aspect2category.pkl
            for later use in aspect ranking models.
            Updated: 2023/12/25
            Integrate with aspect_classify_2stage.py.
"""

import logging
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch
from easydict import EasyDict as edict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.data_model import UIE_TRIPLET
from common.io import read_pickle, save_pickle
from common.utils import create_dual_logger
from resources import predefined_aspect_categories as ac

# COMMON_LABELS = ["ambience", "service"]


def classify(
    sentence: str,
    label_list: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
    # down_labels: list[str],
    # down_threshold: float = 0.8,
) -> int:
    sequences = [sentence for _ in label_list]
    hypotheses = [f"This example is about {label}." for label in label_list]
    batch_encoding = tokenizer(
        sequences,
        hypotheses,
        return_tensors="pt",
        # truncation_strategy="only_first",
        padding=True,
    )
    model.to(device)
    logits = model(**batch_encoding.to(device)).logits
    entailment_logits = logits[:, 2]
    probs = entailment_logits.softmax(dim=0)
    max_prob_index = probs.argmax().item()
    return max_prob_index
    # get argmax of probs as predicted label
    # if (
    #     label_list[max_prob_index] in down_labels
    #     and probs[max_prob_index] > down_threshold
    # ):
    #     return max_prob_index
    # else:
    #     probs[max_prob_index] = 0
    #     next_max_prob_index = probs.argmax().item()
    #     return next_max_prob_index


def eda(aspect2category: dict):
    # print out the number of aspects in each category
    category2count = defaultdict(int)
    for _, category in aspect2category.items():
        category2count[category] += 1
    logger.info("Category statistics:")
    for cat, count in category2count.items():
        logger.info(f"\t{cat}: {count}")
    return


# read in reviews pkl
# if data_path is a directory, read in all pkl files in the directory
def recursive_read_pkl(data_path: str) -> list:
    if Path(data_path).is_dir():
        logger.info(f"DIRECTORY DETECTED. Reading in pkl files in {data_path}")
        reviews = []
        data_path = Path(data_path)
        for pkl_path in data_path.glob("*.pkl"):
            this_pkl_reviews = recursive_read_pkl(pkl_path)
            logger.info(
                f"Reading in #:{len(this_pkl_reviews)} reviews from `{pkl_path}`"
            )
            reviews.extend(this_pkl_reviews)
        return reviews
    elif Path(data_path).with_suffix(".pkl") or Path(data_path).with_suffix(".pickle"):
        return read_pickle(data_path)


def main(
    args: edict,
) -> None:
    global logger
    # create save_dir if not exists
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    logger = create_dual_logger(save_dir / "0shot_classify.log", verbose=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_path = Path(args.data_path)
    try:
        reviews = recursive_read_pkl(data_path)
    except:  # if data_path is a file
        reviews = read_pickle(data_path)
    logger.info(f"Total reviews #: {len(reviews)}")

    # check if aspect2category dict is cached,
    # if yes, load it and only zero-shot classify the remaining aspects
    if (
        args.aspect2category_path is not None
        and Path(args.aspect2category_path).exists()
    ):
        logger.info(
            f"Cached detected. Loading aspect2category dict from {args.aspect2category_path}"
        )
        cached_aspect2category = read_pickle(args.aspect2category_path)
        # check how many aspects are already classified
        logger.info(f"Total aspects # stored in cache: {len(cached_aspect2category)}")
    else:
        cached_aspect2category = {}
    aspect2category = {}

    aspects = set()
    # if args.auto_arg_by_dataset.lower() in ["gest", "yelp23"]:
    for review in reviews:

        if args.auto_arg_by_dataset.lower() in ["yelp23", "yelp"]:
        #     # yelp23 initial scripts
        #     for t in review["triplets"]:
        #         aspects.add(t[0])
        # else:
        #     # yelp-relabeled feature
            triplets = review["triplets"]
            for triplet in triplets:
                # print(triplet)
                aspects.add(triplet[0])
                # print('size of aspects:', len(aspects))

    aspects = sorted(list(aspects))
    logger.info(f"Total distinct aspects #: {len(aspects)}")
    # peek feature set
    logger.info(f"Peek 10 aspects: {aspects[:10]}")

    mix_list = ac.get_simplified_list()
    # logger.info(f"Loading yelp label list of length {len(y_coarse_label_list)}")
    # logger.info(f"Loading semeval16 label list of length {len(semeval16_coarse_label_list)}")
    logger.info(f"Loading label list of length {len(mix_list)}, {mix_list}")
    # !! These categories are easily overused by bart model
    assert "ambience" not in mix_list
    assert "miscellaneous" not in mix_list
    cached_aspects = set(cached_aspect2category.keys())
    # ============== MAIN LOOP ==================
    for i, aspect in enumerate(aspects):
        if aspect in cached_aspects:
            aspect2category[aspect] = cached_aspect2category[aspect]
            continue
        try:
            ac_index = classify(
                sentence=aspect,
                label_list=mix_list,  # Tex & Mex, Italian, ... (cuisine), service, location
                model=model,
                tokenizer=tokenizer,
                device=args.device,
            )
            final_label = mix_list[ac_index]
        except Exception as e:
            logger.info(e)
            final_label = random.choice(mix_list)
        aspect2category[aspect] = final_label
        save_dict_path = save_dir / f"aspect2category_#{i}.pkl"

        if (i + 1) % args.save_period == 0:
            logger.info(f"Saving aspect2category dict at aspect #{i}")
            save_pickle(aspect2category, save_dict_path)
            # peek 1 aspect
            logger.info(
                f"Peek 1 aspect over save_period: {aspect}: {aspect2category[aspect]}"
            )
    # final save
    save_pickle(aspect2category, save_dir / f"aspect2category.pkl")
    # clean up the temp files
    temps = list(save_dir.glob(f"aspect2category_*.pkl"))
    for temp in temps:
        temp.unlink()
    # eda
    eda(aspect2category)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--save_period", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--auto_arg_by_dataset", type=str, default="yelp23")
    args = parser.parse_args()
    DATA_ROOT = Path("/workspace/P76114511/projects/nete_format_data")

    if args.auto_arg_by_dataset.lower() == "yelp23":
        data_args = edict(
            save_dir="resources/yelp23",
            aspect2category_path="resources/yelp23/aspect2category.pkl",
            data_path=DATA_ROOT / "yelp23/reviews.pickle",
        )
    # ========= Li et al. =============
    elif args.auto_arg_by_dataset == "yelp":
        data_args = edict(
            save_dir="resources/yelp",
            aspect2category_path="resources/yelp/aspect2category.pkl",
            data_path="../nlg4rec_data/Yelp/uie_annotated_data/reviews_uie.pkl",
        )
    elif args.auto_arg_by_dataset == "tripadvisor":
        data_args = edict(
            save_dir="resources/tripadvisor",
            aspect2category_path=None,
            data_path="../nlg4rec_data/TripAdvisor/reviews.pickle",
        )
    elif args.auto_arg_by_dataset == "movies_and_tv":
        data_args = edict(
            save_dir="resources/movies_and_tv",
            aspect2category_path=None,
            data_path="../nlg4rec_data/Amazon/MoviesAndTV/reviews.pickle",
        )
    elif args.auto_arg_by_dataset == "clothing":
        data_args = edict(
            save_dir="resources/clothing",
            aspect2category_path=None,
            data_path="../nlg4rec_data/Amazon/ClothingShoesAndJewelry/reviews.pickle",
        )
    args = edict(vars(args))
    args.update(data_args)

    main(args)
