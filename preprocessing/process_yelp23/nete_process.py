'''
@File    :   nete_process.py
@Time    :   2023/10/27 20:54:48
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   About 40G memory is required to run this script
            Yelp is a sparse dataset,
            so a better approach is to not prune the graph at each iteration but to build a new one from scratch.

'''
#%%
import logging
import os
from collections import defaultdict
from pathlib import Path
import fire
import utils
from logger import create_logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.data_model import yelp_business as Business
from common.data_model import yelp_review as Review
from common.data_model import yelp_tip as Tip
from common.data_model import yelp_user as User
# yelp_2023_raw/yelp_academic_dataset_user.json


FILE_PREFIX = "yelp_academic_dataset"
MIN_DEGREE = 20 # NETE setting
log_file_path = os.path.join(os.path.dirname(__file__), "nete_process_yelp_2.log")
lg = create_logger(log_file_path=log_file_path, logger_name="nete_process_yelp")



class UBGraph:

    def __init__(self, users: list[User], businesses: list[Business], reviews: list[Review]):
        self.users = users
        self.businesses = businesses
        self.reviews = reviews
        self._build_graph()

    def _build_graph(self):
        self.neighbors = defaultdict(set)
        for review in self.reviews:
            uid = review.user_id
            bid = review.business_id
            assert not uid.endswith("_u")
            assert not bid.endswith("_b")
            self.neighbors[uid+"_u"].add(bid+"_b")
            self.neighbors[bid+"_b"].add(uid+"_u")
        self.neighbors = dict(self.neighbors)

    def _get_neighbor_count(self, uid: str):
        return len(self.neighbors[uid])

    @utils.timer
    def _prune_node_by_degree(self, min_degree: int) -> int:
        add_node_set = set()
        for node in self.neighbors:
            if self._get_neighbor_count(node) >= min_degree:
                add_node_set.add(node)
        lg.info(f"Collecting {len(add_node_set)} nodes to keep, pruning the rest")


        # build the graph
        new_neighbors = defaultdict(set)
        old_node_count = len(self.neighbors)
        for node in self.neighbors:
            if node not in add_node_set:
                continue
            for neighbor in self.neighbors[node]:
                if neighbor in add_node_set:
                    new_neighbors[node].add(neighbor)
        lg.info(f"Building new graph with {len(new_neighbors)} nodes")
        self.neighbors = dict(new_neighbors)
        new_node_count = len(self.neighbors)
        return old_node_count - new_node_count

    @utils.timer
    def _cleanup(self):
        remaining_user_ids= set()
        remaining_business_ids = set()
        for node in self.neighbors:
            if node.endswith("_u"):
                remaining_user_ids.add(node.rstrip("_u"))
            elif node.endswith("_b"):
                remaining_business_ids.add(node.rstrip("_b"))
        # prune self.users & self.businesses
        self.users = [user for user in self.users if user.user_id in remaining_user_ids]
        self.businesses = [business for business in self.businesses if business.business_id in remaining_business_ids]
        # prune self.reviews
        new_reviews = []
        for review in self.reviews:
            if review.user_id in remaining_user_ids and review.business_id in remaining_business_ids:
                new_reviews.append(review)
        self.reviews = new_reviews


    def prune(self, min_degree: int):
        round = 0
        while True:
            round += 1
            prune_count = self._prune_node_by_degree(min_degree)
            lg.info(f"[Round {round}] Pruned {prune_count} nodes")
            if prune_count == 0:
                break
        self._cleanup()
        lg.info(f"Remaining: {len(self.users)} users, {len(self.businesses)} businesses, {len(self.reviews)} reviews")

def main(yelp_dir: str = "../yelp_2023/yelp_2023_raw"):
    yelp_dir = Path(yelp_dir) # the downloaded data from https://www.yelp.com/dataset/download should contain yelp_academic_dataset_*.json
    users = utils.read_jsonl(yelp_dir / f"{FILE_PREFIX}_user.json")
    businesses = utils.read_jsonl(yelp_dir / f"{FILE_PREFIX}_business.json")
    review_path = yelp_dir /  "yelp_academic_dataset_review.json" #f"{FILE_PREFIX}_review_pruned_uie/yelp_reviews_uie_&.pkl"
    # reviews = utils.read_jsonl(yelp_dir / f"{FILE_PREFIX}_review.json")
    reviews = utils.read_pickle(review_path)

    lg.info(f"Loaded {len(users)} users, {len(businesses)} businesses, {len(reviews)} reviews")
    users = [User(**user) for user in users]
    businesses = [Business(**business) for business in businesses]
    reviews = [Review(**review) for review in reviews]
    graph = UBGraph(users, businesses, reviews)
    lg.info(f"Built graph with {len(graph.neighbors)} nodes")
    lg.info(f"Pruning nodes with degree < {MIN_DEGREE}")
    graph.prune(min_degree=MIN_DEGREE)

    # test
    OUT_DIR = "/home/P76114511/projects/yelp_2023/yelp_2023_p2"
    OUT_DIR = Path(OUT_DIR)
    OUT_DIR.mkdir(exist_ok=True)
    utils.write_jsonl([user.to_dict() for user in graph.users], OUT_DIR / f"{FILE_PREFIX}_user_p2.json")
    utils.write_jsonl([business.to_dict() for business in graph.businesses], OUT_DIR/ f"{FILE_PREFIX}_business_p2.json")
    utils.save_pickle([review.to_dict() for review in graph.reviews], OUT_DIR / f"yelp_reviews_uie_&_p2.pkl")

    # save reviews.json, users.json, businesses.json

#%%
if __name__ == "__main__":
    fire.Fire(main)

