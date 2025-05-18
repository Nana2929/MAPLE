import datetime
import json
import math
import os
import pickle
import random
import re
from collections import defaultdict
import torch

try:
    from tools.bleu import compute_bleu
    from tools.rouge import rouge
except:
    from .tools.bleu import compute_bleu
    from .tools.rouge import rouge

from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.probability import FreqDist


def find_outputdir(input_filepath: os.PathLike):
    # input.parent score_outputs
    return Path(input_filepath).parent / "score_outputs"


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(
        sentences
    )


def get_diversity(corpus):
    """
    coprus:
     :list[str]
    return:
        none
    prints:
        d-1,2,3; ENTR 1,2,3
    """

    entr_scores = {}
    distinct_scores = {}
    for gram_size in [1, 2, 3]:
        all_grams = []
        for sent in corpus:
            all_grams += list(ngrams(sent.split(), gram_size))
        fdist = FreqDist(Counter(all_grams))
        entr = 0
        for x in fdist.keys():
            p_x = fdist.freq(x)
            entr += p_x * np.log2(p_x)
        entr_scores[gram_size] = -entr
        distinct_scores[gram_size] = distinct_n_corpus_level(corpus, gram_size)
    return entr_scores, distinct_scores


def feature_matching_ratio(predict_texts, test_features):
    count = 0

    for predict_text, fea in zip(predict_texts, test_features):
        # print()
        assert isinstance(predict_text, str)
        if isinstance(fea, list):
            for f in fea:  # a list of features
                if f in predict_text:
                    count += 1
                    break

        else:  # single feature
            if fea in predict_text:
                count += 1
    return count / len(predict_texts)


def evaluate_hit_ratio(user2items_test, user2items_top, top_k=None):
    hits = 0
    for label, predict in zip(user2items_test, user2items_top):

        if top_k is not None:
            try:
                predict = predict[
                    :top_k
                ]  # assuming the predict is ordered by rank score
            except:
                print("label", label)
                print("predict", predict)

        rank_list = set(predict)
        test_list = set(label)
        if len(rank_list & test_list) > 0:
            hits += 1
    return hits / len(user2items_test)


def evaluate_ndcg(user2items_test, user2items_top, top_k=20):
    # https://github.com/REASONER2023/reasoner2023.github.io/blob/main/metrics/metrics.py#L62
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]

    ndcg = 0
    for i in range(len(user2items_test)):
        rank_list = user2items_top[i]
        test_list = user2items_test[i]
        dcg_u = 0
        for idx, item in enumerate(rank_list):
            if idx >= top_k:
                break
            if item in test_list:
                dcg_u += dcgs[idx]

        ndcg += dcg_u

    return ndcg / (sum(dcgs) * len(user2items_test))


def load_index(index_path):
    with open(index_path, "r") as f:
        index = [int(x) for x in f.readline().split(" ")]
    return index


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_test_reviews(args):
    """get full-level review from user-item pairs"""

    data = load_pickle(args.data_path)
    index_dir = Path(args.index_dir)
    test_index = load_index(index_dir / "test.index")
    get_text = lambda x: x["text"] if "text" in x else x["template"][2]
    test_reviews = [get_text(data[i]) for i in test_index]
    return test_reviews[: args.max_samples]


def __load_test_one_segment(args):
    """get only 1 review segment from user-item pairs (archived)"""
    data = load_pickle(args.data_path)
    index_dir = Path(args.index_dir)
    test_index = load_index(index_dir / "test.index")
    get_segment = lambda x: x["template"][2]
    test_segments = [get_segment(data[i]) for i in test_index]
    return test_segments[: args.max_samples]


def load_test_segments_features(args):
    """get all review segments from user-item pairs"""

    data = load_pickle(args.data_path)
    index_dir = Path(args.index_dir)
    test_index = load_index(index_dir / "test.index")
    test_segments = []
    test_features = []
    for i in test_index:
        d = data[i]
        if "triplets" in d:
            segments = [t[2] for t in d["triplets"]]
            features = [t[0] for t in d["triplets"]]
        else:  # yelp
            segments = [d["text"]]
            features = d["template"][0]
        test_segments.append(segments)
        test_features.append(features)
        if len(test_segments) == args.max_samples:
            break
    return test_segments, test_features


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_json_or_jsonl(path):
    try:
        return load_json(path)
    except:
        return load_jsonl(path)


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle(filepath: str, data: dict):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def save_json(filepath: str, data: dict):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    """
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    """
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    if not isinstance(references[0], list):
        formatted_ref = [[ref] for ref in references]
    else:  # a list of lists
        formatted_ref = references
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):

    unique_seq = set()
    for seq in sequence_batch:
        unique_seq.add(seq)
    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for r, p in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub**2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    def __init__(self, data_path, index_dir, tokenizer, seq_len):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float("-inf")
        self.min_rating = float("inf")
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train, self.valid, self.test, self.user2feature, self.item2feature = (
            self.load_data(data_path, index_dir)
        )

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, "rb"))
        for review in reviews:
            self.user_dict.add_entity(review["user"])
            self.item_dict.add_entity(review["item"])
            rating = review["rating"]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, "rb"))
        for review in reviews:
            try:
                (fea, adj, tem, sco) = review["template"]
            except:
                (fea, adj, tem, sco, cat) = review["template"]
            tokens = self.tokenizer(tem)["input_ids"]
            text = self.tokenizer.decode(
                tokens[: self.seq_len]
            )  # keep seq_len tokens at most
            data.append(
                {
                    "user": self.user_dict.entity2idx[review["user"]],
                    "item": self.item_dict.entity2idx[review["item"]],
                    "rating": review["rating"],
                    "text": text,
                    "feature": fea,
                }
            )
            try:
                self.feature_set.add(fea)
            except:
                self.feature_set = self.feature_set | set(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review["user"]
            i = review["item"]
            f = review["feature"]
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, "train.index"), "r") as f:
            train_index = [int(x) for x in f.readline().split(" ")]
        with open(os.path.join(index_dir, "validation.index"), "r") as f:
            valid_index = [int(x) for x in f.readline().split(" ")]
        with open(os.path.join(index_dir, "test.index"), "r") as f:
            test_index = [int(x) for x in f.readline().split(" ")]
        return train_index, valid_index, test_index


def now_time():
    return "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") + "]: "


def postprocessing(string):
    """
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub("'s", " 's", string)
    string = re.sub("'m", " 'm", string)
    string = re.sub("'ve", " 've", string)
    string = re.sub("n't", " n't", string)
    string = re.sub("'re", " 're", string)
    string = re.sub("'d", " 'd", string)
    string = re.sub("'ll", " 'll", string)
    string = re.sub("\(", " ( ", string)
    string = re.sub("\)", " ) ", string)
    string = re.sub(",+", " , ", string)
    string = re.sub(":+", " , ", string)
    string = re.sub(";+", " . ", string)
    string = re.sub("\.+", " . ", string)
    string = re.sub("!+", " ! ", string)
    string = re.sub("\?+", " ? ", string)
    string = re.sub(" +", " ", string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens
