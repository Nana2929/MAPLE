import os
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .base_data_initializer import BaseDataInitializer
from .dclass import Strategy
from .strategy_preputils import heuristic_load_test_data, simple_load_test_data
from .utils import EntityDictionary

random.seed(42)


select_random_feat = lambda features: random.choice(features)
select_longest_feat = lambda features: max(
    features, key=lambda x: (len(x[0]), len(x[2]))
)  # fea length; tem length
select_feat_map = {
    "random": select_random_feat,
    "longest": select_longest_feat,
}


class AspectDataInitializer(BaseDataInitializer):

    def __init__(
        self,
        data_path: os.PathLike,
        aspect_path: os.PathLike,
        index_dir: os.PathLike,
        max_test_aspect_tokens: int,
        test_strategy: str,
        feat_select_strategy: str,
        seq_len: int,
        tokenizer,
        logger=None,
        *args,
        **kwargs,
    ):
        """Prepare data before loading it into separate Dataset objs

        Parameters
        ----------
        data_path : os.PathLike
        aspect_path : os.PathLike
        index_dir : os.PathLike
        max_test_aspect_tokens : int
            (K in the paper)
            This argument takes effect only if test_strategy is 'random', 'heuristic', 'heu_tfidf', 'gt' or 'supervised'.
            If using 'supervised', then the aspect token is inferred from the model.
        seq_len : int, optional
            maximum sequenence length for generated text, by default None
        test_stratgy : str, optional
            Which strategy to use for testing/inference stage, by default "heu"
            Must be one of ['gt', 'heuristic', 'supervised'].
            the last one is model-based. The model should be trained to infer the aspect probs before use.
        """

        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.logger = logger
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        if "user_dict" in kwargs:
            print("Loading pre-built user_dict from kwargs")
            self.user_dict = kwargs["user_dict"]
        if "item_dict" in kwargs:
            print("Loading pre-built item_dict from kwargs")
            self.item_dict = kwargs["item_dict"]
        # self.max_rating = float("-inf")
        # self.min_rating = float("inf")

        # aspect2idx is inverted index of aspect list
        self.aspect_list = []
        self.aspect2idx = {}
        self.feature_set = set()
        self.max_test_aspect_tokens = max_test_aspect_tokens
        self.feat_select_fn = select_feat_map.get(feat_select_strategy)
        self.data_path = data_path
        self.aspect_path = aspect_path
        self.initialize(data_path)
        self.initialize_aspect(aspect_path)
        (
            self.train,
            self.valid,
            self.test,
            self.tag_train,
            self.tag_valid,
            self.user2feature,
            self.item2feature,
        ) = self.load_data(self.data_path, index_dir, test_strategy=test_strategy)

    def initialize_aspect(self, aspect_path) -> None:
        assert os.path.exists(aspect_path)
        aspects = pd.read_csv(aspect_path)
        self.aspect_list = aspects["category"].tolist()
        self.aspect2idx = {aspect: idx for idx, aspect in enumerate(self.aspect_list)}

    def initialize(self, data_path) -> None:
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, "rb"))

        for review in reviews:

            self.user_dict.add_entity(review["user"])
            self.item_dict.add_entity(review["item"])
            feat = review["template"][0]
            if isinstance(feat, list):
                self.feature_set.update(feat)
            else:  # isinstance(feat, str):
                self.feature_set.add(feat)

        print("feature set size: {}".format(len(self.feature_set)))
        if self.logger is not None:
            self.logger.info("feature set size: {}".format(len(self.feature_set)))

    def load_data(
        self, data_path: os.PathLike, index_dir: os.PathLike, test_strategy: str
    ) -> tuple[list, list, list, dict, dict]:
        """_summary_

        Parameters
        ----------
        data_path : os.PathLike
            path to the data file (reviews.pickle)
        index_dir : os.PathLike
            path to the index directory (k-fold cross validation)
        test_strategy : str
            how to prepare aspect tokens in testing stage
        Returns
        -------
        train : list, list of reviews for training
        valid : list, list of reviews for validation
        test : list, list of reviews for testing
        user2feature : dict, user to a list of features that he/she uses
        item2feature : dict, item to a list of features that it is described upon
        """
        reviews = pickle.load(open(data_path, "rb"))
        ui_data = defaultdict(list)
        aspect_scores = defaultdict(lambda: np.zeros(len(self.aspect_list)))
        data = [None for _ in range(len(reviews))]

        for idx, review in enumerate(reviews):
            # from triplets, find those with correct category
            category = review["template"][-1]  # category
            sub_triplets = [t for t in review["triplets"] if t[-1] == category]
            sub_triplets.append(review["template"])
            fea, adj, tem, sco, category = self.feat_select_fn(
                sub_triplets
            )  # 2 examples have "" as feature

            tokens = self.tokenizer(tem)["input_ids"]
            text = self.tokenizer.decode(
                tokens[: self.seq_len]
            )  # keep seq_len tokens at most
            review_text = review["text"] if "text" in review else text
            category_idx = self.aspect2idx[category]

            # tokens = self.tokenizer(text)["input_ids"]
            # text = self.tokenizer.decode(
            #     tokens[: self.seq_len]
            # )  # keep seq_len tokens at most
            ui_id = f'{review["user"]}_{review["item"]}'
            sample = {
                "user": self.user_dict.entity2idx[review["user"]],
                "item": self.item_dict.entity2idx[review["item"]],
                "triplets": review["triplets"],
                "rating": sco,
                "text": text,
                "feature": fea,
                "overall_rating": review["rating"],
                "review_text": review_text,
                "category": category_idx,
                "category_name": category,
                "idx": idx,
            }
            ui_data[ui_id].append(sample)
            aspect_scores[ui_id][category_idx] += 1
        # flattening the review segments
        for ui_id, segments in ui_data.items():
            category_idxes = []
            category_names = []
            for seg_id, seg in enumerate(segments):

                # mark only the first seg to be counted into tag prediction task data to avoid training
                # on the same user-item pair multiple times in one epoch
                if seg_id == 0:
                    seg["ui_representative"] = True
                else:
                    seg["ui_representative"] = False
                seg["aspect_score"] = np.array(aspect_scores[ui_id])
                assert seg["aspect_score"].shape[0] == len(
                    self.aspect_list
                )  # (n_aspect,)
                assert np.sum(seg["aspect_score"]) <= len(
                    self.aspect_list
                )  # at most n_aspect
                # something like: [0,1,0,0,0,1..., 0]
                # seg["idx"] records the index of the review in the original data
                data[seg["idx"]] = seg
                cat = seg["category"]
                category_idxes.append(cat)
                category_names.append(self.aspect_list[cat])
            # end of for seg_id, seg in enumerate(segments)
            for seg_id, seg in enumerate(segments):
                seg["category_idxes"] = category_idxes
                seg["category_names"] = category_names

        del ui_data
        self.data = data
        # load train, test, valid index
        train_index, valid_index, test_index = self.load_index(index_dir)
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        # text-gen training/validation data
        train, valid = [], []
        # text-gen + aspect distribution prediction data
        # in tag_train, tag_valid, we only keep one data for each user-item pair to prevent over-train on
        # same aspect distribution
        tag_train, tag_valid = [], []
        self.user2feature, self.item2feature = {}, {}

        self.user2cat, self.item2cat = defaultdict(set), defaultdict(set)

        for idx in train_index:
            review = data[idx]

            if review["ui_representative"]:
                tag_train.append(review)

            train.append(review)
            u = review["user"]
            i = review["item"]
            f = review["feature"]
            if u in self.user2feature:
                (
                    self.user2feature[u].append(f)
                    if isinstance(f, str)
                    else self.user2feature[u].extend(f)
                )
            else:
                self.user2feature[u] = [f] if isinstance(f, str) else f
            if i in self.item2feature:
                (
                    self.item2feature[i].append(f)
                    if isinstance(f, str)
                    else self.item2feature[i].extend(f)
                )
            else:
                self.item2feature[i] = [f] if isinstance(f, str) else f
            self.user2cat[u].add(review["category"])
            self.item2cat[i].add(review["category"])

        for idx in valid_index:
            review = data[idx]
            valid.append(review)
            if data[idx]["ui_representative"]:
                tag_valid.append(review)

        test = self.load_test_data(test_strategy)

        return (
            train,
            valid,
            test,
            tag_train,
            tag_valid,
            self.user2feature,
            self.item2feature,
        )

    def get_tagtrain_index_dic(self):
        self.cls_data_list = [[] for _ in range(len(self.aspect_list))]
        for i, d in enumerate(self.tag_train):
            category_idxes = d["category_idxes"]
            for c in category_idxes:
                # make index_dic/cls_data_list needed by classAwareSampler
                self.cls_data_list[c].append(i)
        return self.cls_data_list

    def load_test_data(self, strategy):
        # with open(os.path.join(index_dir, "test.index"), "r") as f:
        #    test_index = [int(x) for x in f.readline().split(" ")]
        print(f"Loading test data with strategy: {strategy}")

        if strategy == Strategy.HEURISTIC:
            test_data = heuristic_load_test_data(
                data=self.data,
                test_index=self.test_index,
                user2cat=self.user2cat,
                item2cat=self.item2cat,
                aspect2idx=self.aspect2idx,
                aspect_list=self.aspect_list,
                topk=self.max_test_aspect_tokens,
            )
        elif strategy == Strategy.SUPERVISED or strategy == Strategy.GT:
            test_data = simple_load_test_data(
                data=self.data, test_index=self.test_index
            )
        else:
            raise ValueError(f"Invalid test strategy: {strategy}")

        test_catids = [set() for _ in range(len(self.test_index))]
        test_size = len(self.test_index)
        for i in range(test_size):
            idx = self.test_index[i]
            triplets = self.data[idx]["triplets"]
            for t in triplets:
                cat = self.aspect2idx[t[-1]]
                test_catids[i].add(cat)
        self.test_catids = test_catids
        return test_data


class PureTextInitializer(BaseDataInitializer):

    def __init__(
        self,
        data_path: os.PathLike,
        aspect_path: os.PathLike,
        index_dir: os.PathLike,
        feat_select_strategy: str,
        seq_len: int,
        logger=None,
        *args,
        **kwargs,
    ):
        """Prepare data before loading it into separate Dataset objs

        Parameters
        ----------
        data_path : os.PathLike
        aspect_path : os.PathLike
        index_dir : os.PathLike
        max_test_aspect_tokens : int
            This argument takes effect only if test_strategy is 'random', 'heuristic', 'heu_tfidf', 'gt' or 'supervised'.
            If using 'supervised', then the aspect token is inferred from the model.
        seq_len : int, optional
            maximum sequenence length for generated text, by default None
        test_stratgy : str, optional
            Which strategy to use for testing/inference stage, by default "heu"
            Must be one of ['gt', 'heuristic', 'heu_tfidf', 'supervised']. The first 2 are heuristics/statisitcal-based,
            the last one is model-based. The model should be trained to infer the aspect tokens.
        """

        # self.user_dict = EntityDictionary()
        # self.item_dict = EntityDictionary()
        self.logger = logger
        self.seq_len = seq_len
        # if "user_dict" in kwargs:
        #     print("Loading pre-built user_dict from kwargs")
        #     self.user_dict = kwargs["user_dict"]
        # if "item_dict" in kwargs:
        #     print("Loading pre-built item_dict from kwargs")
        #     self.item_dict = kwargs["item_dict"]
        # self.max_rating = float("-inf")
        # self.min_rating = float("inf")
        # aspect2idx is inverted index of aspect list
        self.aspect_list = []
        self.aspect2idx = {}
        self.feat_select_fn = select_feat_map.get(feat_select_strategy)
        self.data_path = data_path
        self.aspect_path = aspect_path
        self.initialize_aspect(aspect_path)
        (
            self.train,
            self.valid,
            self.test,
        ) = self.load_data(self.data_path, index_dir)

    def initialize_aspect(self, aspect_path) -> None:
        assert os.path.exists(aspect_path)
        aspects = pd.read_csv(aspect_path)
        self.aspect_list = aspects["category"].tolist()
        self.aspect2idx = {aspect: idx for idx, aspect in enumerate(self.aspect_list)}

    def load_data(
        self, data_path: os.PathLike, index_dir: os.PathLike
    ) -> tuple[list, list, list, dict, dict]:
        """_summary_

        Parameters
        ----------
        data_path : os.PathLike
            path to the data file (reviews.pickle)
        index_dir : os.PathLike
            path to the index directory (k-fold cross validation)
        -------
        train : list, list of reviews for training
        valid : list, list of reviews for validation
        test : list, list of reviews for testing
        """
        reviews = pickle.load(open(data_path, "rb"))
        aspect_scores = defaultdict(lambda: np.zeros(len(self.aspect_list)))
        data = []

        for idx, review in enumerate(reviews):
            # from triplets, find those with correct category
            category = review["template"][-1]  # category
            sub_triplets = [t for t in review["triplets"] if t[-1] == category]
            sub_triplets.append(review["template"])
            fea, adj, tem, sco, category = self.feat_select_fn(
                sub_triplets
            )  # 2 examples have "" as feature
            review_text = review["text"] if "text" in review else tem
            category_idx = self.aspect2idx[category]
            data.append(
                {
                    "user_id": review["user"],
                    "item_id": review["item"],
                    "triplets": review["triplets"],
                    "aspect_rating": sco,  # see enrich_rec_dataset/convert_{}_nete.py:merge_triplets()
                    "text": review_text,
                    "feature": fea,
                    "overall_rating": review["rating"],
                    "category": category_idx,
                    "category_name": category,
                }
            )
        print("data size: {}".format(len(data)))
        train_index, valid_index, test_index = self.load_index(index_dir)
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        # text-gen training/validation data
        train, valid, test = [], [], []
        # text-gen + aspect distribution prediction data
        # in tag_train, tag_valid, we only keep one data for each user-item pair to prevent over-train on
        # same aspect distribution

        for idx in train_index:
            review = data[idx]
            train.append(review)
        for idx in valid_index:
            review = data[idx]
            valid.append(review)
        for idx in test_index:
            review = data[idx]
            test.append(review)

        return (
            train,
            valid,
            test,
        )
