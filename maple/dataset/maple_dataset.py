import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, WeightedRandomSampler

from .utils import AspectDataBatch

"""
{'review_id': '110922565534353024196_604a5d09d863393130a413fa',
'user': '110922565534353024196',
'item': '604a5d09d863393130a413fa',
'rating': 4,
'template': ('ramen', 'good', 'good ramen', 'positive', 'Japanese'),
'triplets':
    [
        ('ramen', 'good', 'good ramen', 'positive', 'Japanese'),
        ('broth', 'flavorful', 'flavorful broth', 'positive', 'ambience')
    ],
'text': 'Very good ramen with flavorful broth.',
'tokens': ['Very', 'good', 'ramen', 'with', 'flavorful', 'broth', '.'],
'pics': ['AF1QipPaM_gz9dcOcO4HGD7OrPIwZqhX_be1HE1PIptm']}

"""


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result

    return wrapper


class MultiAspectTestset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        bos,
        eos,
        max_samples=None,
        *args,
        **kwargs,
    ):
        if max_samples:
            data = data[:max_samples]
            print("Sampling {} test samples".format(max_samples))
        self.data = data
        self.tokenizer = tokenizer
        u, i, a, t, aspect_scores, category_idxes, feature = [], [], [], [], [], [], []
        fake_category_idxes = []
        # pending: make (offset_mapping = (start, end)) for each user-item pair (may have multiple aspect triplets)
        # because shuffling will change the order of user-item pairs
        for x in data:
            u.append(x["user"])
            i.append(x["item"])
            a.append(torch.tensor(x["fake_categories"]))
            fake_category_idxes.append(x["fake_categories"])
            t.append("{} {} {}".format(bos, x["text"], eos))
            category_idxes.append(x["category_idxes"])
            feature.append(x["feature"])
            aspect_scores.append(x["aspect_score"])
        encoded_inputs = tokenizer(t, padding=True, return_tensors="pt")
        # * uia prompt tuning (uia to aspect text span prediction) *
        self.seq = encoded_inputs["input_ids"].contiguous()
        self.mask = encoded_inputs["attention_mask"].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.gt_labels = aspect_scores
        # untensorized gt
        self.gt_labels = aspect_scores

        # untensorized prediction
        self.fake_category_idxes = fake_category_idxes
        self.aspect = a
        aspect_scores = np.array(
            aspect_scores
        )  # exploiting the fast conversion from np to tensor
        self.aspect_score = torch.tensor(aspect_scores, dtype=torch.float).contiguous()
        # https://stackoverflow.com/questions/69352980/normalize-an-array-of-floats-into-a-certain-range-with-keeping-sign-in-python

        self.feature = feature
        self.category_idxes = category_idxes

    def collate_fn(self, batch):
        # check https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
        users, items, aspects, aspect_scores, seqs, masks = ([], [], [], [], [], [])
        for u, i, a, d, t, m in batch:
            users.append(u)
            items.append(i)
            aspects.append(a)
            aspect_scores.append(d)
            seqs.append(t)
            masks.append(m)
        return AspectDataBatch(
            user=torch.stack(users),
            item=torch.stack(items),
            aspect=pad_sequence(
                aspects, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ),
            aspect_score=torch.stack(aspect_scores),
            seq=torch.stack(seqs),
            mask=torch.stack(masks),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        aspect = self.aspect[index]
        aspect_score = self.aspect_score[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        return user, item, aspect, aspect_score, seq, mask


class AspectDataset(Dataset):

    def __init__(self, data, tokenizer, bos, eos, max_len, *args, **kwargs):
        u, i, a, r, t, category, category_idxes, feature = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.max_len = max_len
        aspect_scores = []
        # pending: make (offset_mapping = (start, end)) for each user-item pair (may have multiple aspect triplets)
        # because shuffling will change the order of user-item pairs

        if "max_samples" in kwargs:
            data = data[: kwargs["max_samples"]]
        self.data = data

        for x in data:
            u.append(x["user"])
            i.append(x["item"])
            a.append(x["category"])
            t.append("{} {} {}".format(bos, x["text"], eos))

            category.append(x["category"])
            category_idxes.append(x["category_idxes"])
            feature.append(x["feature"])
            aspect_scores.append(x["aspect_score"])

        encoded_inputs = tokenizer(
            t, padding=True, return_tensors="pt", max_length=max_len, truncation=True
        )
        # * uia prompt tuning (uia to aspect text span prediction) *
        self.seq = encoded_inputs["input_ids"].contiguous()
        self.mask = encoded_inputs["attention_mask"].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.aspect = torch.tensor(a, dtype=torch.int64).contiguous()
        aspect_scores = np.array(
            aspect_scores
        )  # exploiting the fast conversion from np to tensor
        aspect_score = torch.tensor(aspect_scores, dtype=torch.float).contiguous()
        # 1. normalize to [-1,1] and 2. keep the sign (positive, negative)
        # https://stackoverflow.com/questions/69352980/normalize-an-array-of-floats-into-a-certain-range-with-keeping-sign-in-python
        print(
            "Original aspect score with score range: [{}, {}]".format(
                aspect_score.min(), aspect_score.max()
            )
        )
        # untensorized gt
        self.gt_labels = aspect_scores
        self.category = category
        self.aspect_score = aspect_score
        self.min_aspect_score = -1
        self.max_aspect_score = 1

        self.category = category
        self.feature = feature

    def get_weighted_sampler(self):
        # https://www.kaggle.com/code/mineshjethva/pytorch-weighted-sampler-for-imbalance-classifier
        cnt = Counter(self.category)
        class_sample_count = np.array([cnt[i] for i in range(len(cnt))])
        weights = 1.0 / class_sample_count
        samples_weights = weights[self.category]
        print("Aspect class weight (data sampler): ", weights)
        sampler = WeightedRandomSampler(
            samples_weights,
            len(samples_weights),
            replacement=True,
        )
        return sampler

    def collate_fn(self, batch):
        # check https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
        users, items, aspects, aspect_scores, aspect_scores = (
            [],
            [],
            [],
            [],
            [],
        )
        seqs, masks = [], []
        for u, i, a, d, t, m in batch:
            users.append(u)
            items.append(i)
            aspects.append(a)
            aspect_scores.append(d)
            seqs.append(t)
            masks.append(m)
        return AspectDataBatch(
            user=torch.stack(users),
            item=torch.stack(items),
            aspect=torch.stack(aspects),
            aspect_score=torch.stack(aspect_scores),
            seq=torch.stack(seqs),
            mask=torch.stack(masks),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        aspect = self.aspect[index]
        aspect_score = self.aspect_score[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        return user, item, aspect, aspect_score, seq, mask
