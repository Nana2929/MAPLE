import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import scipy
import torch


@dataclass
class AspectDataBatch:
    user: torch.LongTensor
    item: torch.LongTensor
    aspect: torch.LongTensor
    aspect_score: torch.FloatTensor
    seq: torch.LongTensor
    mask: torch.LongTensor
    rating: torch.FloatTensor = None

    def __len__(self):
        return self.user.size(0)


# ===================
# Feature Category fidf
# ===================


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result

    return wrapper


@timer
def run_tfidf(id2feature, vectorizer):
    return vectorizer.fit_transform(id2feature)


@timer
def get_tfidf_score(id, tfidf, topk=None, return_score=False):
    feat_index = tfidf[id, :].nonzero()[1]
    # get feat_names
    ffidf_scores = zip(feat_index, [tfidf[id, x] for x in feat_index])
    # sort by score (descending)
    ffidf_scores = sorted(ffidf_scores, key=lambda x: -x[1])
    if topk is not None:
        ffidf_scores = ffidf_scores[:topk]
    # no score
    if not return_score:
        return [f for (f, _) in ffidf_scores]
    return ffidf_scores


def identity_tokenizer(text: list[str]):
    assert isinstance(text, list)
    return text


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path):
    with open(path, "rb") as f:
        model = torch.load(f)
    model.eval()
    return model


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def get_entity(self, idx):  # get original data's identifier
        return self.idx2entity[idx]

    def get_idx(self, entity):  # get the 0-nuser or 0-nitem index
        return self.entity2idx[entity]

    def __len__(self):
        return len(self.idx2entity)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result

    return wrapper


class FfidfStore:
    def __init__(self, cache_dir: os.PathLike):
        self.cache_dir = Path(cache_dir)
        assert self.cache_dir.exists(), f"{self.cache_dir} does not exist"
        print(f"Loading precomputed ffidf values from {self.cache_dir}")
        self.user_ffidf = load_pickle(self.cache_dir / "user_ffidf.pickle")
        self.item_ffidf = load_pickle(self.cache_dir / "item_ffidf.pickle")
        self.user_feat_names = load_pickle(self.cache_dir / "user_feat_names.pickle")
        self.item_feat_names = load_pickle(self.cache_dir / "item_feat_names.pickle")
        self.idx2user = load_pickle(self.cache_dir / "idx2user.pickle")
        self.idx2item = load_pickle(self.cache_dir / "idx2item.pickle")

        # revert the idx2user and idx2item
        self.user2idx = {v: k for k, v in self.idx2user.items()}
        self.item2idx = {v: k for k, v in self.idx2item.items()}
        print(f"Loaded {len(self.user2idx)} users and {len(self.item2idx)} items.")

        print(f"Loading completed.")

    def _get_ffidf(
        self,
        id: str,
        feat_names: list[str],
        ffidf: scipy.sparse.csr_matrix,
        topk=10,
        return_score=False,
    ):
        feat_index = ffidf[id, :].nonzero()[1]
        # get feat_names
        feat_names = [feat_names[i] for i in feat_index]
        ffidf_scores = zip(feat_names, [ffidf[id, x] for x in feat_index])
        # sort by score (descending)
        ffidf_scores = sorted(ffidf_scores, key=lambda x: -x[1])
        if return_score:
            return ffidf_scores[:topk]
        else:
            return [f for (f, _) in ffidf_scores[:topk]]

    def get_user_ffidf(self, user_id: str, topk: int, return_score: bool = False):
        if topk < 0:
            topk = len(self.user_feat_names)
        return self._get_ffidf(
            self.user2idx[user_id],
            self.user_feat_names,
            self.user_ffidf,
            topk,
            return_score,
        )

    def get_item_ffidf(self, item_id: str, topk: int, return_score: bool = False):
        if topk < 0:
            topk = len(self.item_feat_names)
        return self._get_ffidf(
            self.item2idx[item_id],
            self.item_feat_names,
            self.item_ffidf,
            topk,
            return_score,
        )
