"""
以 PRAG retrieved 的前K(K=3)個評論內，對應的feature-category拿出來
作為 PRAG 預測的主題
和 gt 的 aspect category 計算 hit ratio, f1, ndcg
"""

from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import load_pickle, load_json_or_jsonl, evaluate_hit_ratio, evaluate_ndcg
from collections import Counter
import pandas as pd
import numpy as np

dataset = "yelp"
fold = 1
K = 3
prag_filepath = f"prag_checkpoints/{dataset}/{fold}/{dataset}_generated.jsonl"
reviews_path = f"nete_format_data/{dataset}/reviews.pickle"
aspect_path = f"nete_format_data/{dataset}/aspect_category_index.csv"
df = pd.read_csv(aspect_path)
index2cat = df["category"]
cat2index = {cat: idx for idx, cat in enumerate(index2cat)}


prag_prediction = load_json_or_jsonl(prag_filepath)
reviews = load_pickle(reviews_path)
# collect user-item aspect
ui2cat = {}

print(f"Processing reviews: {len(reviews)}")
for review in tqdm(reviews):
    user = review["user"]
    item = review["item"]
    _, _, _, _, cat = review["template"]
    ui2cat[(user, item)] = cat


user2items_test = []
user2items_top = []
naspect = 18
user2items_test_multihot = [np.zeros(naspect) for _ in range(len(prag_prediction))]
user2items_top_multihot = [np.zeros(naspect) for _ in range(len(prag_prediction))]
for idx, pred in enumerate(prag_prediction):
    user = pred["user_id"]
    item = pred["item_id"]
    gt_cat = ui2cat[(user, item)]
    topk_ret = pred["retrieved"][:K]
    pred_cat = []
    user2items_test_multihot[idx][cat2index[gt_cat]] = 1

    for ret in topk_ret:
        ret_user, ret_item = ret["user_id"], ret["item_id"]
        c = ui2cat[(ret_user, ret_item)]
        pred_cat.append(c)
        user2items_top_multihot[idx][cat2index[c]] = 1
    user2items_test.append([gt_cat])
    user2items_top.append(pred_cat)
# calc hit ratio, f1, ndcg
print("sample: ", user2items_test[:5], user2items_top[:5])
hit_ratio = evaluate_hit_ratio(user2items_test, user2items_top, top_k=K)
f1 = f1_score(user2items_test_multihot, user2items_top_multihot, average="micro")
ndcg = evaluate_ndcg(user2items_test, user2items_top, top_k=K)
print(f"hit ratio: {hit_ratio}, f1: {f1}, ndcg: {ndcg}")
