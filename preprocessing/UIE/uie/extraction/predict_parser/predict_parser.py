#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import Counter, List, Tuple


class PredictParser:
    def __init__(self, label_constraint=None):
        self.spot_set = label_constraint.type_list if label_constraint else list()
        self.role_set = label_constraint.role_list if label_constraint else list()

    def decode(self, gold_list, pred_list, text_list=None, raw_list=None) -> Tuple[List, Counter]:
        pass
