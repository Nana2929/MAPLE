"""
@File    :   aspect_classify_sentence_level.py
@Time    :   2024.05.08
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :
"""

import logging
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.data_model import UIE_TRIPLET
from common.io import read_pickle, save_pickle
from common.utils import create_dual_logger
from resources import predefined_aspect_categories as ac


def separate_classify(
    sentence: str,
    label_list: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
) -> int:
    """used when all labels can not be put into a single batch"""
    half_len = len(label_list) // 2
    sequences1 = [sentence for _ in label_list[:half_len]]
    hypotheses1 = [f"This example is {label}." for label in label_list[:half_len]]
    sequences2 = [sentence for _ in label_list[half_len:]]
    hypotheses2 = [f"This example is {label}." for label in label_list[half_len:]]
    batch_encoding1 = tokenizer(
        sequences1,
        hypotheses1,
        return_tensors="pt",
        truncation_strategy="only_first",
        padding=True,
    )
    batch_encoding2 = tokenizer(
        sequences2,
        hypotheses2,
        return_tensors="pt",
        truncation_strategy="only_first",
        padding=True,
    )
    model.to(device)
    logits1 = model(**batch_encoding1.to(device)).logits
    logits2 = model(**batch_encoding2.to(device)).logits
    entailment_logits1 = logits1[:, 2]
    entailment_logits2 = logits2[:, 2]
    probs1 = entailment_logits1.softmax(dim=0)
    probs2 = entailment_logits2.softmax(dim=0)
    # get argmax of probs as predicted label
    return torch.cat((probs1, probs2)).argmax().item()


def classify(
    sentence: str,
    label_list: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str,
) -> int:
    sequences = [sentence for _ in label_list]
    hypotheses = [f"This example is {label}." for label in label_list]
    batch_encoding = tokenizer(
        sequences,
        hypotheses,
        return_tensors="pt",
        truncation_strategy="only_first",
        padding=True,
    )
    model.to(device)
    logits = model(**batch_encoding.to(device)).logits
    entailment_logits = logits[:, 2]
    probs = entailment_logits.softmax(dim=0)
    # get argmax of probs as predicted label
    return probs.argmax().item()


# read in reviews pkl
# if data_path is a directory, read in all pkl files in the directory
def read_pickle(data_path: Path) -> list[dict]:
    import pickle

    with open(data_path, "rb") as f:
        reviews = pickle.load(f)
    return reviews


def main(
    args: edict,
) -> None:
    global logger
    # create save_dir if not exists
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    logger = create_dual_logger(
        save_dir / "0shot_classify_sentence_level.log", verbose=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data_path = Path(args.data_path)
    logger.info(f"Loading reviews from {data_path}")
    reviews = read_pickle(data_path)
    logger.info(f"Total reviews #: {len(reviews)}")
    #!!sentence-level: instead of fea, use text to do classify
    #!!2024.01.15
    #!!2024.05.08: fixing the mixed list by removing miscellaneous
    mix_list = ac.get_mixed_list()
    # logger.info(f"Loading yelp label list of length {len(y_coarse_label_list)}")
    # logger.info(f"Loading semeval16 label list of length {len(semeval16_coarse_label_list)}")
    logger.info(f"Loading mixed label list of length {len(mix_list)}, {mix_list}")

    # ============== MAIN LOOP ==================
    aspect_counter = defaultdict(int)
    relabeled_count = len(reviews)
    for i, review in enumerate(tqdm(reviews)):
        (fea, adj, text, sentiment, pre_label) = review["template"]
        # if pre_label is not misc or service, continue
        if pre_label not in ["miscellaneous", "service"]:
            relabeled_count -= 1
            final_label = pre_label
        else:
            try:
                final_label = None
                ac_index = classify(
                    sentence=text,
                    label_list=mix_list,  # Tex & Mex, Italian, ... (cuisine), ambience, service, location
                    model=model,
                    tokenizer=tokenizer,
                    device=args.device,
                )
                final_label = mix_list[ac_index]
            except Exception as e:
                print(e)
                # problematic aspect presents: randomize an aspect category for it (consider removing it later)
                # eg. #481219 怎么还没有留影呢？我跟我的中国朋友们常常去那里吃一点台湾菜，正好地道台湾菜的餐厅离我的大学那么近。阿姨非常热心，菜很快出去了。中
                # 文的菜单非常好，英文的还可以，算普通的美国化中国菜一样。价钱低，可是只收了金钱一点不方便，如果接受卡的话我一定会更多来。
                final_label = random.choice(mix_list)
        aspect_counter[final_label] += 1
        review["template"] = (fea, adj, text, sentiment, final_label)
        temp_reviews_path = save_dir / f"{args.auto_arg_by_dataset}_reviews_#{i}.pickle"

        if i % args.save_period == 0:
            logger.info(f"Saving re-labeled reviews at aspect #{i}")
            save_pickle(reviews[: i + 1], temp_reviews_path)
            # peek 1 aspect
            logger.info(f"Peek 1 aspect over save_period: {text}: {final_label}")
            logger.info(f"Aspect counter: {aspect_counter}")

    # final save
    logger.info(f"Saving final re-labeled reviews")
    logger.info(f"Relabeled count: {relabeled_count}")
    save_pickle(reviews, save_dir / f"reviews.pickle")
    logger.info(f"Final Aspect counter: {aspect_counter}")
    # remove all temp files
    temps = list(save_dir.glob(f"{args.auto_arg_by_dataset}_reviews_*.pickle"))
    for temp in temps:
        temp.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--save_period", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--auto_arg_by_dataset", type=str, default="yelp")
    args = parser.parse_args()

    # YELP_REVIEWS_PATH = (
    #     "../yelp_2023/uie_annotated_data_postprocessed/yelp23_reviews_uie.pkl"
    # )
    # elif args.auto_arg_by_dataset.lower() == "yelp23":
    #     data_args = edict(
    #         save_dir = "resources/yelp23",
    #         aspect2category_path = "../yelp_2023/yelp_2023_pruned/aspect_classify/aspect2category_#481222.pkl",
    #         data_path = YELP_REVIEWS_PATH,
    #     )
    # ========= Li et al. =============
    if args.auto_arg_by_dataset == "yelp":
        data_args = edict(
            save_dir="resources/sentence_level_yelp",
            # data_path="../nlg4rec_data/Yelp/reviews.pickle", # when aspect_level_yelp
            data_path="/home/P76114511/projects/nete_format_data/yelp/reviews.pickle",
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
