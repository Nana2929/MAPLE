import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import pandas as pd
from easydict import EasyDict as edict

from common import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiChecker:
    def __init__(self, args):
        self.reviews = io.read_pickle(args.data_path)
        self.aspect_dict = io.read_pickle(args.aspect_dict_path)
        self.check_aspect_dict(self.aspect_dict)

        self.new_user_set = set()
        self.new_item_set = set()
        self.user_set = set()
        self.item_set = set()

    def check_aspect_dict(self, aspect_dict: dict[str, str]):
        """
        sort aspect category index by category
        """
        apc_index = set(aspect_dict.values())
        sorted_apc_index = sorted(apc_index)
        # check aspect_dict distribution
        feat_counter = pd.Series(aspect_dict.values()).value_counts()
        logger.info(f"how many features belong to each category: {feat_counter}")

    def check_reviews(self):
        review_count_without_misc = 0
        for review in self.reviews:
            (fea, adj, text, sentiment) = review["template"]
            aspect_category = self.aspect_dict[fea]
            review["template"] = (fea, adj, text, sentiment, aspect_category)
            review["triplets"] = [(fea, adj, text, sentiment, aspect_category)]
            review["category"] = aspect_category
            assert set(review.keys()).issuperset(
                {"user", "item", "rating", "template", "triplets", "category"}
            )
            self.user_set.add(review["user"])
            self.item_set.add(review["item"])

            if aspect_category != "miscellaneous":
                self.new_user_set.add(review["user"])
                self.new_item_set.add(review["item"])
                review_count_without_misc += 1
        logger.info(f"num of users: {len(self.user_set)}")
        logger.info(f"num of items: {len(self.item_set)}")
        logger.info(f"num of exclude-misc users: {len(self.new_user_set)}")
        logger.info(f"num of exclude-misc items: {len(self.new_item_set)}")
        # io.save_pickle(self.reviews, self.out_root / "reviews.pickle")
        # logger.info(f"aspect counter: {self._aspect_counter}")
        logger.info(f"Num of reviews: {len(self.reviews)}")
        logger.info(
            f"Num of reviews exclude-miscellaneous: {review_count_without_misc}"
        )
        logger.info(f"Peek a review: {self.reviews[0]}")

        # cp the index directories to the out_root
        # for index_dir in range(1, 6):
        #     dir = Path(args.data_path).parent / f"{index_dir}"
        #     os.system(f"cp -r {dir} {self.out_root}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--auto_arg_by_dataset",
        type=str,
        default="yelp",
        help="yelp, clothing, movies_and_tv, tripadvisor",
    )
    args = parser.parse_args()

    main_dir = Path("../nlg4rec_data")
    # out_dir = Path("check_data")
    auto_arg_by_dataset = args.auto_arg_by_dataset
    if args.auto_arg_by_dataset == "yelp":
        args = edict(
            data_path=main_dir / "Yelp/reviews.pickle",
            aspect_dict_path="resources/yelp/aspect2category.pkl",
            # out_root=out_dir / "yelp",
        )
    elif args.auto_arg_by_dataset == "clothing":
        args = edict(
            data_path=main_dir / "Amazon/ClothingShoesAndJewelry/reviews.pickle",
            aspect_dict_path="resources/clothing/aspect2category.pkl",
            # out_root=out_dir / "clothing",
        )
    elif args.auto_arg_by_dataset == "movies_and_tv":
        args = edict(
            data_path=main_dir / "Amazon/MoviesAndTV/reviews.pickle",
            aspect_dict_path="resources/movies_and_tv/aspect2category.pkl",
            # out_root=out_dir / "movies_and_tv",
        )
    elif args.auto_arg_by_dataset == "tripadvisor":
        args = edict(
            data_path=main_dir / "TripAdvisor/reviews.pickle",
            aspect_dict_path="resources/tripadvisor/aspect2category.pkl",
            # out_root=out_dir / "tripadvisor",
        )

    checker = LiChecker(args)
    checker.check_reviews()
