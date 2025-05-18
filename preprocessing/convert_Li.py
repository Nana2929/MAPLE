import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import pandas as pd
from easydict import EasyDict as edict

from common import io
from common.utils import create_dual_logger


class LiConverter:
    def __init__(self, args):
        self.reviews = io.read_pickle(args.data_path)
        self.out_root = args.out_root
        logger.info(f"aspect_dict_path: {args.aspect_dict_path}")
        self.aspect_dict = io.read_pickle(args.aspect_dict_path)

        self._sort_and_save_aspect_category_index(self.aspect_dict)

        self.user_set = set()
        self.item_set = set()
        for review in self.reviews:
            self.user_set.add(review["user"])
            self.item_set.add(review["item"])
        logger.info(f"num of users: {len(self.user_set)}")
        logger.info(f"num of items: {len(self.item_set)}")

    def _sort_and_save_aspect_category_index(self, aspect_dict: dict[str, str]):
        """
        sort aspect category index by category
        """
        apc_index = set(aspect_dict.values())
        sorted_apc_index = sorted(apc_index)
        pd.DataFrame.from_dict({"category": sorted_apc_index}).to_csv(
            self.out_root / "aspect_category_index.csv"
        )

    def convert_and_save(self):
        if os.path.exists(self.out_root / "reviews.pickle"):
            logger.warning(
                f"Output already exists. To not overwrite {self.out_root / 'reviews.pickle'} please remove or rename it first."
            )
            return
        for review in self.reviews:

            new_triplets = []
            # choose the longest feature as the template (the trained text)
            fea_max_len, fea_max_idx = 0, 0
            if len(review["triplets"]) == 0:
                review["triplets"] = [review["template"]]
            _, _, text, _ = review["template.old"] = review.pop("template")

            for idx, triplet in enumerate(review["triplets"]):
                fea, _, _, _ = triplet
                aspect_category = self.aspect_dict[fea]
                new_triplets.append((*triplet, aspect_category))
                if len(fea) > fea_max_len:
                    fea_max_len, fea_max_idx = len(fea), idx
            fea, adj, shorter_text, sentiment = review["triplets"][fea_max_idx]
            triplet = (fea, adj, text, sentiment, aspect_category)
            review["template"] = triplet
            review["category"] = aspect_category = self.aspect_dict[
                review["template"][0]
            ]
            review["triplets"] = new_triplets
            assert set(review.keys()).issuperset(
                {"user", "item", "rating", "template", "triplets", "category"}
            )

        io.save_pickle(self.reviews, self.out_root / "reviews.pickle")
        logger.info(f"saving to {self.out_root / 'reviews.pickle'}")
        logger.info(f"num of reviews: {len(self.reviews)}")
        # peek
        for i in range(10):
            template = self.reviews[i]["template"]
            old_template = self.reviews[i]["template.old"]
            logger.info(
                f"example {i} | template: {template}, old_template: {old_template}"
            )
        # cp the index directories to the out_root
        # for index_dir in range(1, 6):
        #     dir = Path(args.data_path).parent / f"{index_dir}"
        #     os.system(f"cp -r {dir} {self.out_root}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--auto_arg_by_dataset",
        type=str,
        default="yelp",
        help="yelp, clothing, movies_and_tv, tripadvisor",
    )
    args = parser.parse_args()

    main_dir = Path("../nlg4rec_data")
    out_dir = Path("nete_format_data")
    auto_arg_by_dataset = args.auto_arg_by_dataset
    if args.auto_arg_by_dataset == "yelp":
        args = edict(
            data_path="nlg4rec_data/Yelp/uie_annotated_data/reviews_uie.pkl",  # UIE labeled reviews
            aspect_dict_path="resources/yelp/aspect2category.pkl",
            out_root=out_dir / "yelp",
        )
    elif args.auto_arg_by_dataset == "clothing":
        args = edict(
            data_path=main_dir / "Amazon/ClothingShoesAndJewelry/reviews.pickle",
            aspect_dict_path="resources/clothing/aspect2category.pkl",
            out_root=out_dir / "clothing",
        )
    elif args.auto_arg_by_dataset == "movies_and_tv":
        args = edict(
            data_path=main_dir / "Amazon/MoviesAndTV/reviews.pickle",
            aspect_dict_path="resources/movies_and_tv/aspect2category.pkl",
            out_root=out_dir / "movies_and_tv",
        )
    elif args.auto_arg_by_dataset == "tripadvisor":
        args = edict(
            data_path=main_dir / "TripAdvisor/reviews.pickle",
            aspect_dict_path="resources/tripadvisor/aspect2category.pkl",
            out_root=out_dir / "tripadvisor",
        )
    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)
    logger = create_dual_logger(
        out_dir / f"convert_{auto_arg_by_dataset}.log", verbose=True
    )
    converter = LiConverter(args)
    converter.convert_and_save()
