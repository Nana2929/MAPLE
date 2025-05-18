"""
@File    :   metrics/tools/feat_metrics.py
@Time    :   2024/04/18 16:15:14
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw

"""

import logging
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from spacy.lang.en import English
from tqdm import tqdm

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from tools.bart_score import BARTScorer
from utils import (
    unique_sentence_percent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# constants

QUERY_OUT: Final[str] = "fake"

PRED_CAT: Final[str] = "fake_category"
GT_CAT: Final[str] = "all_gt_category"  # "all_gt_category" !!typo!! in retrieve.py

RET_KEY: Final[str] = "item_retrieved"
EPS: Final[float] = 1e-6

TOP_K: Final[int] = 20
DEVICE: Final[str] = "cuda"
BARTCKPT: Final[str] = "facebook/bart-large-cnn"
BERTCKPT: Final[str] = "bert-base-uncased"
BATCH_SIZE: Final[int] = 256


def safe_divide(a, b):
    if abs(b) < 1e-6:
        return 0
    else:
        return a / b


def unique_sentence_ratio_in_user_and_item(data):
    """
    Calculate the unique sentence ratio in user and item
    """
    user_sentences = defaultdict(list)
    item_sentences = defaultdict(list)

    for d in data:
        user = d["user_id"]
        item = d["item_id"]
        user_sentences[user].append(d[QUERY_OUT])
        item_sentences[item].append(d[QUERY_OUT])
    for k in user_sentences.keys():
        user_sentences[k], _ = unique_sentence_percent(user_sentences[k])
    for k in item_sentences.keys():
        item_sentences[k], _ = unique_sentence_percent(item_sentences[k])

    usr_user = np.mean(list(user_sentences.values()))
    usr_item = np.mean(list(item_sentences.values()))
    return usr_user, usr_item


class FeatureDictionary:
    """
    A dictionary that stores features for each item or user key
    key: item ID or user ID, here we use item ID only
    usage:
    ```
    fd = FeatureDictionary(dummy_words)
    in this process of add(), we filter out the dummy words
    fd.add(key="iaxnd119u92", feature="ice cream")
    fd.add(key="iaxnd119u92", feature=["chocolate", "vanilla"]) # add multiple features
    fd["iaxnd119u92"] # get the features for the item, crediting to __getitem__
    fd.items() # get all items and their features as a list of tuples
    ```
    """

    def __init__(self, filter_words: list[str], minlen: int = 4):
        self.key2set = defaultdict(lambda: set())
        self.minlen = minlen
        self.filter_words = filter_words

    def add(self, key: int, feature: str | list):
        if isinstance(feature, str):
            if len(feature) >= self.minlen and feature not in self.filter_words:
                self.key2set[key].add(feature)
        elif isinstance(feature, list):
            for f in feature:
                if len(f) >= self.minlen and f not in self.filter_words:
                    self.key2set[key].add(f)

    def items(self):
        return self.key2set.items()

    def __getitem__(self, key):
        return self.key2set[key]


class SubFeatureScorer:
    def __init__(
        self,
        testdata: list,
        item2feature: FeatureDictionary,
        feat2cat: dict,
    ):
        self.testdata = testdata
        self.feat2cat = feat2cat
        self.item2feature = item2feature
        # collect only features from the item in the test set && belong the legal_cats

    def initialize(self, legal_cats):
        print(f"only counting features from {legal_cats}, {len(legal_cats)}")
        self.feature_set = set()
        self.feature_predicted = set()
        for d in self.testdata:
            item_id = d["item_id"]
            for fea in self.item2feature[item_id]:
                if self.feat2cat[fea] in legal_cats:
                    self.feature_set.add(fea)
        for d in self.testdata:
            text = d["fake"]
            for fea in self.feature_set:
                if fea in text:
                    self.feature_predicted.add(fea)
        print(f"feature set size: {len(self.feature_set)}")
        print(f"predicted feature size: {len(self.feature_predicted)}")

    def calculate(self):
        return self.feature_coverage_ratio()

    def feature_coverage_ratio(self):
        return safe_divide(len(self.feature_predicted), len(self.feature_set))


class FeatureScorer:
    def __init__(
        self,
        testdata,
        train_item2feature,
    ):
        self.testdata = testdata
        self.train_item2feature = train_item2feature
        self.feature_set = set()
        self.test_feature_set = set()

        for k, v in self.train_item2feature.items():
            # print(f"item {k} has {len(v)} features: {v}")
            self.feature_set.update(v)

        ifm = defaultdict(set)
        test_itemids = set()
        for d in self.testdata:

            item_id = d["item_id"]
            test_itemids.add(item_id)
            item_feature = self.train_item2feature[item_id]
            text = d[QUERY_OUT]
            for ifea in item_feature:
                if ifea in text:
                    ifm[item_id].add(ifea)
        # check how many features are matched for item in the testset
        self.ifm = ifm
        self.test_itemids = test_itemids

    def calculate(self):
        """
        Calculate all the feature-based metrics except for FCR and subset FCR.
        """
        # train_ifcr: for each item in the testset, we calculate |predicted_feats|/|all_feats for the item in train set| and take the average of all ratios
        # testitem fcr: |all_feats predicted in the testset|/|all_feats in the testset items|
        # fcr: |all_feats predicted in the testset|/|all_feats in the trainset|

        item_fcr = self.itemwise_feature_coverage_ratio(
            self.ifm, self.train_item2feature, self.test_itemids
        )  # archive
        gt_fmr = self.gtfeature_matching_ratio()
        item_fmr = self.itemfeature_matching_ratio()

        return {
            "item-wise fmr": item_fmr,
            "item-wise fcr": item_fcr,
            "gt_fmr": gt_fmr,
        }

    def itemfeature_matching_ratio(self):
        count = 0
        for d in self.testdata:
            item_id = d["item_id"]
            menu = self.train_item2feature[item_id]
            text = d[QUERY_OUT]
            # if any of the feature is mentioned in the text, count 1
            for ifea in menu:
                if ifea in text:
                    count += 1
                    break
        print(f"iFMR: count: {count}, len: {len(self.testdata)}")
        return safe_divide(count, len(self.testdata))

    def gtfeature_matching_ratio(self):
        count = 0
        for d in self.testdata:
            text = d[QUERY_OUT]
            gt_features = d["gt_features"]
            # if any of the ground-truth feature (UIE labeled feature) is mentioned in the text, count 1
            for fea in gt_features:
                if fea in text:
                    count += 1
                    break
        print(f"gtFMR: count: {count}, len: {len(self.testdata)}")
        return safe_divide(count, len(self.testdata))

    def itemwise_feature_coverage_ratio(self, ifm, item2feature, test_itemids):
        """
        ifm[item_id] = matched features in testset for item_id
        item2feature[item_id] = all features in trainset for item_id
        for each item in the testset (each restaurant in the testset),
        this score measures the ratio of "features that are matched out in the testset" to "all features in the trainset"
        """

        x = []
        for item_id in test_itemids:
            matched_ifea = ifm[item_id]
            all_ifea = item2feature[item_id]
            x.append(safe_divide(len(matched_ifea), len(all_ifea)))
        return np.mean(x)

    # def test_feature_coverage_ratio(self, ifm, item2feature, test_itemids):
    #     numerator = set()
    #     denominator = set()

    #     for item_id in test_itemids:
    #         matched_ifea = ifm[item_id]
    #         all_ifea = item2feature[item_id]
    #         numerator.update(matched_ifea)
    #         denominator.update(all_ifea)

    #     return safe_divide(len(numerator), len(denominator))
