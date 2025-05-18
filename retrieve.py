"""
@File    :   retrieve.py
@Time    :   2024/04/08 19:54:48
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Logic for retrieval is defined here.
             It includes (1) inferencing MAPLE and
             (2) retrieval of reviews using MAPLE generated text.
"""

import logging
import math
import os
import sys
import pandas as pd
import torch
from collections import defaultdict

# ========= 3rd party libs =========
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from pathlib import Path
from tqdm import tqdm

# ===== self defined modules =====
from maple.dataset import (
    AspectDataInitializer,
    AspectDataset,
    MultiAspectTestset,
)
from maple.utils import (
    fbos,
    bos,
    eos,
    pad,
    load_pickle,
    save_jsonl,
    evaluate_hit_ratio,
    evaluate_ndcg,
)

from maple_retrieval_pipeline import load_corpus, MapleRetrievalPipeline
from common_utils import load_jsonl
from maple.dataset.dclass import Strategy
from common_utils import get_meta

sys.path.append("recommender/")
# for loading corpus.user_dict and corpus.item_dict, we need to let the recommender.dataset module
# to be accessible for pickle loading.


def get_keynames(dataset_name: str):
    if dataset_name == "yelp23":
        return "user_id", "business_id"
    elif dataset_name == "yelp":
        return "user", "item"
    else:
        return None, None


def main(args):

    # fname = f"retrieve_{args.aspect_select_strategy}.log"
    logging.basicConfig(
        # filename=args.checkpoint / fname,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        # filemode="w",
    )

    logger = logging.getLogger(__name__)

    ###############################################################################
    # Check arguments and display
    ###############################################################################
    assert args.data_path is not None, "data_path should be provided for loading data"
    assert (
        args.aspect_select_strategy in Strategy
    ), f"aspect_select_strategy should be one of {Strategy}"
    if args.data_path is None or os.path.exists(args.data_path) is False:
        raise ValueError("--data_path should be provided for loading data")
    if args.index_dir is None:
        raise ValueError("--index_dir should be provided for loading data splits")

    logger.info(f"========== ARGUMENTS ==========")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"========== end of ARGUMENTS ==========")

    ###############################################################################
    # Load data
    ###############################################################################
    prebuilt_user_dict = load_pickle(args.checkpoint / "user_dict.pickle")
    prebuilt_item_dict = load_pickle(args.checkpoint / "item_dict.pickle")
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        bos_token=bos,
        eos_token=eos,
        pad_token=pad,
    )
    initializer = AspectDataInitializer(
        data_path=args.data_path,
        aspect_path=args.aspect_path,
        index_dir=args.index_dir,
        max_test_aspect_tokens=args.max_test_aspect_tokens,
        user_dict=prebuilt_user_dict,
        item_dict=prebuilt_item_dict,
        test_strategy=args.aspect_select_strategy,
        feat_select_strategy="longest",
        seq_len=args.words,
        tokenizer=tokenizer,
        logger=logger,
    )

    if (
        args.aspect_select_strategy == Strategy.SUPERVISED
        or args.aspect_select_strategy == Strategy.GT
    ):
        test_data = AspectDataset(
            initializer.test,
            tokenizer,
            bos,
            eos,
            max_len=args.words,
            max_samples=args.max_samples,
        )
    else:
        logger.info(f"Testing data has multi-aspects")
        test_data = MultiAspectTestset(
            initializer.test, tokenizer, bos, eos, max_samples=args.max_samples
        )
    logger.info(f"Testing dataset type: {test_data.__class__}, size: {len(test_data)}")
    test_loader = DataLoader(
        test_data,
        batch_size=args.maple_batch_size,
        shuffle=False,
        collate_fn=test_data.collate_fn,
    )
    user_id_key, item_id_key = get_keynames(args.auto_arg_by_dataset)
    user_meta = (
        get_meta(args.user_meta_path, user_id_key) if args.user_meta_path else {}
    )
    item_meta = (
        get_meta(args.item_meta_path, item_id_key) if args.item_meta_path else {}
    )

    ###############################################################################
    # Build the model
    ###############################################################################
    if torch.cuda.is_available():
        if args.cpu:
            logger.warning(
                "WARNING: You have a CUDA device, so you should probably not run with --cpu"
            )

    nuser = len(initializer.user_dict)
    nitem = len(initializer.item_dict)
    naspect = len(initializer.aspect_list)

    device = torch.device("cpu" if args.cpu else "cuda")
    logger.info(f"Loading model from {args.checkpoint}")
    with open(args.checkpoint / "model.pt", "rb") as f:
        model = torch.load(f)
    logger.info(
        f"number of users: {nuser}, number of items: {nitem}, number of aspects: {naspect}"
    )
    model.to(device)

    ###############################################################################
    # RetrievalPipeline
    ###############################################################################
    train_index = set(initializer.train_index)
    train_item2docids = defaultdict(list)
    train_user2docids = defaultdict(list)
    corpus = load_corpus(
        args.corpus_path, max_docs_per_item=args.max_docs_per_item, logger=logger
    )
    for doc in corpus:
        if int(doc["id"]) not in train_index:
            continue
        train_item2docids[doc["item"]].append(doc["id"])
        train_user2docids[doc["user"]].append(doc["id"])
    # turn to fixed dict so that it raises KeyError when not found
    train_item2docids = {k: v for k, v in train_item2docids.items()}
    train_user2docids = {k: v for k, v in train_user2docids.items()}
    corpus_dict = {x["id"]: x for x in corpus}
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    args.device = device
    retrieval_pipeline = MapleRetrievalPipeline(
        args=args,
        logger=logger,
        corpus={},  # corpus,
        model=model,
        user_dict=initializer.user_dict,
        item_dict=initializer.item_dict,
        train_item2docids=train_item2docids,
        train_user2docids=train_user2docids,
    )
    ###############################################################################
    # 1. Query Generation (MAPLE inference)
    ###############################################################################

    if args.aspect_select_strategy == Strategy.SUPERVISED:
        out = retrieval_pipeline.inference(
            model,
            test_loader,
            tokenizer,
            aspect_select_strategy=args.aspect_select_strategy,
            use_filter=None,
            aspect_history=None,
            aspect_list=initializer.aspect_list,
        )

    elif args.aspect_select_strategy == Strategy.GT:
        out = retrieval_pipeline.inference(
            model,
            test_loader,
            tokenizer,
            aspect_select_strategy=args.aspect_select_strategy,
            multi_aspect_forward=False,
        )
    else:
        out = retrieval_pipeline.inference(
            model,
            test_loader,
            tokenizer,
            aspect_select_strategy=args.aspect_select_strategy,
            multi_aspect_forward=(args.max_test_aspect_tokens > 1),
        )
    tokens_predict = out["token_ids"]
    aspectids_predict = out.get("aspect_ids", None)
    aspect_predict = out.get("aspect_probs", None)
    ###############################################################################
    # 2. Retrieval
    ###############################################################################

    text_predict = [" ".join(xs) for xs in tokens_predict]

    item_retrieveds, user_retrieveds = retrieval_pipeline.retrieve_reviews(
        test_data=initializer.test,
        queries=text_predict,
    )

    ###############################################################################
    # Save the generated text and retrieval results
    ###############################################################################
    outs = []
    if args.aspect_select_strategy in [
        Strategy.HEURISTIC,
    ]:
        fake_category = test_data.fake_category_names
        fake_category_idx = test_data.category
        aspect_probs = [None] * len(test_data)
    elif args.aspect_select_strategy == Strategy.SUPERVISED:
        fake_category = [
            [initializer.aspect_list[x] for x in xs] for xs in aspectids_predict
        ]
        fake_category_idx = aspectids_predict
        aspect_probs = aspect_predict
    else:  # args.aspect_select_strategy == "gt"
        fake_category = None
        fake_category_idx = None

    if fake_category_idx:
        test_hit_ratio = evaluate_hit_ratio(
            top_k=args.max_test_aspect_tokens,
            user2items_test=initializer.test_catids[: args.max_samples],
            user2items_top=fake_category_idx,
        )
        test_ndcg = evaluate_ndcg(
            top_k=args.max_test_aspect_tokens,
            user2items_test=initializer.test_catids[: args.max_samples],
            user2items_top=fake_category_idx,
        )
    else:  # gt
        test_hit_ratio = 1
        test_ndcg = 1
    logger.info(
        f"dataset: {args.auto_arg_by_dataset}, index: {args.index_dir}, strategy: {args.aspect_select_strategy}"
    )
    logger.info(f"test samples: {len(test_data)}")
    logger.info(f"max_test_aspect_tokens: {args.max_test_aspect_tokens}")
    logger.info(f"test hit_ratio@{args.max_test_aspect_tokens}: {test_hit_ratio:.5f}")
    logger.info(f"test ndcg@{args.max_test_aspect_tokens}: {test_ndcg:.5f}")
    for i in range(len(test_data)):
        d = initializer.test[i]
        user_id, item_id = initializer.user_dict.get_entity(
            d["user"]
        ), initializer.item_dict.get_entity(d["item"])
        user_info = user_meta.get(user_id, {})
        item_info = item_meta.get(item_id, {})

        x = {
            "user": d["user"],  # the 0 - to (nuser-1) encoded index
            "user_id": user_id,  # the original user identifier in raw data
            "item": d["item"],
            "item_id": item_id,
            "user_name": user_info.get("name", user_id),
            "item_name": item_info.get("name", item_id),
            "all_gt_category": [
                initializer.aspect_list[x] for x in initializer.test_catids[i]
            ],
            "real": d["text"],
            "fake": text_predict[i],
            "fake_feat": None,
            "gt_category": d["category_name"],
            "fake_category": fake_category[i] if fake_category else None,
            "fake_category_probs": aspect_probs[i] if aspect_probs else None,
            "triplets": d["triplets"],
        }
        x["item_retrieved"] = [
            {
                "id": docid,
                "score": score,
                "contents": corpus_dict[docid]["contents"],
                "user": corpus_dict[docid]["user"],
            }
            for score, docid in item_retrieveds[i]
        ]
        x["user_retrieved"] = [
            {
                "id": docid,
                "score": score,
                "contents": corpus_dict[docid]["contents"],
                "item": corpus_dict[docid]["item"],
            }
            for score, docid in user_retrieveds[i]
        ]
        outs.append(x)
    prefix = "generated"
    if args.aspect_select_strategy == "gt":
        prediction_path = os.path.join(
            args.checkpoint,
            f"{prefix}_{args.aspect_select_strategy}.jsonl",
        )
    else:
        prediction_path = os.path.join(
            args.checkpoint,
            f"{prefix}_{args.aspect_select_strategy}_k={args.max_test_aspect_tokens}.jsonl",
        )
    save_jsonl(path=prediction_path, data=outs)
    logger.info("Generated text saved to ({})".format(prediction_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference for Maple",
    )
    parser.add_argument("--cpu", action="store_true", help="use cpu for inference")

    parser.add_argument("-i", "--index", type=int, default=1, help="index of the data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="directory for loading the checkpoint, user_dict, item_dict",
    )
    parser.add_argument(
        "--do_retrieval",
        action="store_true",
        help="whether to do retrieval for the generated text",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="max number of samples to be inferenced and retrieved reviews for",
    )
    # ======== Data ========
    parser.add_argument(
        "-a",
        "--auto_arg_by_dataset",
        type=str,
        default="yelp23",
        help="automatically set args by dataset name",
    )  # accepts yelp23, yelp

    # ========= Query generator: MAPLE =========
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="model name / model backbone",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=20,
        help="number of words to generate for each sample",
    )
    parser.add_argument(
        "-s",
        "--aspect_select_strategy",
        type=str,
        default="supervised",
        help=f"how to decide which aspects to condition upon; one of {Strategy}",
    )
    parser.add_argument(
        "--max_test_aspect_tokens",
        type=int,
        default=2,  # hyperparam chosen from validation set ndcg and hit ratio
        help="max number of aspect tokens to be averaged and conditioned on",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="temperature for sampling text tokens",
    )
    parser.add_argument(
        "--maple_batch_size",
        type=int,
        default=256,
        help="batch size of MAPLE inference",
    )
    # ======== Retrieval ========
    parser.add_argument(
        "--encoder",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="encoder model of queries, \
        note that the prepared corpus should be encoded by the same model!!",
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=512,
        help="batch size of encoding queries",
    )
    parser.add_argument(
        "--retrieve_topk",
        type=int,
        default=5,
        help="use topk retrieval results of searcher for augmentation",
    )
    parser.add_argument(
        "--max_docs_per_item",
        type=int,
        default=30,
        help="maximum number of documents per item",
    )
    parser.add_argument(
        "--ckptdir",
        type=str,
        help="checkpoint directory",
    )
    args = parser.parse_args()
    index = args.index

    ROOT = Path(__file__).parent.parent

    if args.auto_arg_by_dataset == "yelp23":
        dargs = edict(
            data_path=ROOT / "nete_format_data/yelp23/reviews.pickle",
            index_dir=ROOT / f"nete_format_data/yelp23/{index}",
            aspect_path=ROOT / "nete_format_data/yelp23/aspect_category_index.csv",
            checkpoint=ROOT / f"aspect_retriever/{args.ckptdir}/yelp23/{index}",
            ffidf_cache_dir=ROOT / f"ffidf_cache/yelp23/{index}",
            item_meta_path=ROOT
            / "nete_format_data/yelp23/yelp_academic_dataset_business.json",
            user_meta_path=ROOT
            / "nete_format_data/yelp23/yelp_academic_dataset_user.json",
            corpus_path=ROOT / "nete_format_data/retrieval/yelp23",
        )
    elif args.auto_arg_by_dataset == "yelp":
        dargs = edict(
            data_path=ROOT / "nete_format_data/yelp/reviews.pickle",
            index_dir=ROOT / f"nete_format_data/yelp/{index}",
            aspect_path=f"{ROOT}/nete_format_data/yelp/aspect_category_index.csv",
            checkpoint=ROOT / f"aspect_retriever/{args.ckptdir}/yelp/{index}",
            ffidf_cache_dir=ROOT / f"ffidf_cache/yelp/{index}",
            item_meta_path=ROOT / "nete_format_data/yelp/item.json",
            user_meta_path=ROOT / "nete_format_data/yelp/user.json",
            corpus_path=ROOT / "nete_format_data/retrieval/yelp",
        )
    args = edict(vars(args))
    args.update(dargs)
    main(args)
