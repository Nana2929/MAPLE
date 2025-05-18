"""
@File    :   convert_yelp23_nete.py
@Time    :   2024/01/16 00:43:30
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Refer to ./convert_gest_nete.py
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from easydict import EasyDict as edict

from common import io
from common.data_model import UIE_TRIPLET
from common.utils import RsTrainTestSplitter, create_dual_logger
import sys

SPECIAL_DELIMITER = "|"


class YelpNeteConverter:
    def __init__(self, args: edict):
        self.out_root = args.out_root
        self.aspect_dict = io.read_pickle(args.aspect_dict_path)

        # for converting user_id and item_id from string to int
        # ========== READ DATA ==========
        self._data = io.read_pickle(args.data_path)
        self.user_id_map = set()
        self.item_id_map = set()

        for review in self._data:
            self.user_id_map.add(review["user_id"])
            self.item_id_map.add(review["business_id"])
        logger.info(f"num of users: {len(self.user_id_map)}")
        logger.info(f"num of items: {len(self.item_id_map)}")

        # =========== SAVING =============
        aspect2category_records = [
            {"aspect": k, "category": v} for k, v in self.aspect_dict.items()
        ]
        pd.DataFrame.from_records(aspect2category_records).to_csv(
            self.out_root / "aspect2category.csv"
        )
        self._sort_and_save_aspect_category_index(self.aspect_dict)

    def _sort_and_save_aspect_category_index(self, aspect_dict: dict[str, str]):
        """
        sort aspect category index by category
        """
        apc_index = set(aspect_dict.values())
        sorted_apc_index = sorted(apc_index)
        pd.DataFrame.from_dict({"category": sorted_apc_index}).to_csv(
            self.out_root / "aspect_category_index.csv"
        )

    def _train_test_split(self) -> None:

        # * 2023/01/22 * train test split in the level of user-item
        tts_executor = RsTrainTestSplitter(
            data=[
                edict(
                    user=xs[0]["user"],
                    item=xs[0]["item"],
                    unique_id=f"{xs[0]['user']}{SPECIAL_DELIMITER}{xs[0]['item']}",
                )
                for uid, xs in self.data.items()
            ],
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
            kfold=args.kfold,
        )()

        for kf_idx, index_list in enumerate(tts_executor, start=1):
            INDEX_DIR_NAME = f"{kf_idx}"
            indices = {"train": [], "val": [], "test": []}
            logger.info(f"======= {INDEX_DIR_NAME} =======")

            counter = 0
            for split, uids in index_list.items():
                for uid in uids:
                    # add all aspect-specfic review segments into the index
                    user, item = uid.split(SPECIAL_DELIMITER)
                    uid_reviews = self.data[uid]
                    indices[split].extend(
                        set(range(counter, counter + len(uid_reviews)))
                    )
                    counter += len(uid_reviews)

                indices[split].sort()
                logger.info(f"\t{split} u-i-a segments: {len(indices[split])}")

            # write the indices into text files
            INDEX_DIR = self.out_root / INDEX_DIR_NAME
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            io.save_index(path=INDEX_DIR / "train.index", index=indices["train"])
            io.save_index(path=INDEX_DIR / "validation.index", index=indices["val"])
            io.save_index(path=INDEX_DIR / "test.index", index=indices["test"])
        # * when saving the data, flatten so that each is u-i-a specific
        # flatten the self.data
        self.data = [review for uid in self.data for review in self.data[uid]]
        # check counter size is equal to len(self.data)
        assert counter == len(self.data)
        io.save_pickle(path=self.out_root / "reviews.pickle", obj=self.data)

    def _normalize_review(self, review: dict) -> dict:
        # """convert 1 review"""
        triplets = self._make_triplets(review["uie_absa"]["triplet"])
        if len(triplets) == 0:
            return None

        groups = defaultdict(list)

        returns = []
        for triplet in review["uie_absa"]["triplet"]:
            category = self.aspect_dict[triplet.aspect]
            groups[category].append(triplet)
        for category in groups:
            # choose the longest feat in the category as the template
            triplet = max(groups[category], key=lambda x: len(x.text))
            n_review = {
                "user": review["user_id"],
                "item": review["business_id"],
                "category": category,  # 'Italian', 'ambience', etc.
                "rating": review["stars"],
                "template": triplet,
                "triplets": triplets,
                "text": review["text"],  # !!2024.02.08 update, merged_text,
                "tokens": review["tokens"],
            }
            if "review_id" in review:
                # not an unique identifier anymore
                n_review["_review_id"] = review["review_id"]
            returns.append(n_review)
            print(n_review)
            sys.exit()
        return returns

    def _make_triplets(self, triplets: list[UIE_TRIPLET]) -> tuple[int]:
        # make aspect tag col
        # (fea, adj, text, sco, `aspect_category`)
        new_triplets = []
        for triplet in triplets:
            ac = self.aspect_dict.get(triplet.aspect, None)
            if len(triplet.text) == 0 or len(triplet.aspect) == 0:
                continue
            structure = (
                triplet.aspect,
                triplet.opinion,
                triplet.text,
                triplet.sentiment,
                ac,
            )
            new_triplets.append(structure)
        return new_triplets

    def convert_and_save(self) -> None:
        """main function"""
        uid2reviews = defaultdict(list)  # ui_id: [reviews]
        for review in self._data:
            n_reviews = self._normalize_review(review)  # None or a generator
            uid = f"{review['user_id']}{SPECIAL_DELIMITER}{review['business_id']}"
            if n_reviews is None:
                continue
            uid2reviews[uid] = n_reviews

        self.data = uid2reviews
        logger.info(f"num of total ui-pairs: {len(self.data)}")
        self._train_test_split()


if __name__ == "__main__":
    main_dir = Path(
        "/workspace/P76114511/data/raw_yelp_2023/uie_annotated_data_postprocessed"
    )
    args = edict(
        data_path=main_dir / "yelp23_reviews_uie.pkl",
        aspect_dict_path="resources/yelp23/aspect2category.pkl",
        kfold=5,
        random_state=42,
        val_ratio=0.1,
        test_ratio=0.1,
        out_root=Path("nete_format_data/yelp23"),
    )

    # out_dir.mkdir(parents=True, exist_ok=True) # warning if exists
    if not args.out_root.exists():
        args.out_root.mkdir(parents=True)
    logger = create_dual_logger(args.out_root / "convert_yelp23.log", verbose=True)
    converter = YelpNeteConverter(args)
    converter.convert_and_save()
