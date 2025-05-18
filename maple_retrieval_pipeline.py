"""
@File    :   maple_retrieval_pipeline.py
@Time    :   2024/04/08 01:53:15
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Retrieval logic is defined in retrieve.py.
"""

import logging
import os
import gc
from tqdm import tqdm
from collections import defaultdict

# ========= 3rd party libs =========
from easydict import EasyDict as edict
from pathlib import Path

# ===== self defined modules =====
from maple.utils import (
    ids2tokens,
    select_and_generate,
    load_jsonl,
    save_jsonl,
    generate,
    eos,
)
# from retriever.modeling_searcher.search import DenseExactSearch
from maple.dataset.dclass import Strategy


def load_corpus(
    corpus_path: os.PathLike, max_docs_per_item: int, logger: logging.Logger
):
    """Load corpus from cached file,
    or merge raw_corpus(documents and metas) and encoded_corpus(encoded documents) on the fly

    Parameters
    ----------
    corpus_path : os.PathLike
        directory containing `corpus.jsonl` and `embeddings.jsonl`; or `cached_corpus.jsonl` if exists

    Returns
    -------
    dict
        corpus
    """
    corpus_path = Path(corpus_path)

    if os.path.exists(corpus_path / "cached_corpus.jsonl"):
        logger.info(f"Loading cached corpus from {corpus_path / 'cached_corpus.jsonl'}")
        return load_jsonl(corpus_path / "cached_corpus.jsonl")
    if os.path.isfile(corpus_path):
        logger.info(f"Loading cached corpus from {corpus_path}")
        return load_jsonl(corpus_path)
    logger.info(f"Merging raw_corpus and encoded_corpus on the fly")
    # list their sizes on disk in mega bytes
    logger.info(
        f"raw_corpus size: {os.path.getsize(Path(corpus_path) / 'corpus.jsonl') / 1024 / 1024} MB"
    )
    logger.info(
        f"encoded_corpus size: {os.path.getsize(Path(corpus_path) / 'embeddings.jsonl') / 1024 / 1024} MB"
    )
    raw_corpus = load_jsonl(Path(corpus_path) / "corpus.jsonl")
    encoded_corpus = load_jsonl(Path(corpus_path) / "embeddings.jsonl")

    assert len(raw_corpus) == len(
        encoded_corpus
    ), f"{len(raw_corpus)} != {len(encoded_corpus)}"
    cached_corpus = []
    # we use raw_corpus to decide whether to include the document or not
    # we limit each restaurant to be having at most 50 documents
    item_count = defaultdict(int)
    for r, e in zip(raw_corpus, encoded_corpus):
        item_count[r["item"]] += 1
        if item_count[r["item"]] > max_docs_per_item:
            continue
        assert str(r["id"]) == str(e["id"])
        cached_corpus.append({**r, **e})
    # check number of documents
    logger.info(f"cached_corpus (after trimmed) #: {len(cached_corpus)}")
    save_jsonl(cached_corpus, Path(corpus_path) / "cached_corpus.jsonl")
    # kill the copies and free the memory
    del raw_corpus, encoded_corpus
    gc.collect()
    return cached_corpus


class MapleRetrievalPipeline:

    def __init__(
        self,
        args,
        logger,
        corpus,
        model,
        user_dict,
        item_dict,
        train_item2docids,
        train_user2docids,
    ):
        self.args = args
        self.logger = logger
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.train_item2docids = train_item2docids
        self.train_user2docids = train_user2docids
        # self.searcher = DenseExactSearch(
        #     query_encoder=args.encoder,
        #     corpus=corpus,
        #     batch_size=args.encode_batch_size,
        #     device=args.device,
        # )

    def __inference(self, model, dataloader, tokenizer, multi_aspect_forward):
        idss_predicted = generate(
            model=model,
            dataloader=dataloader,
            pad_token_id=tokenizer.pad_token_id,
            max_len=self.args.words,
            multi_aspect_forward=multi_aspect_forward,
            logger=self.logger,
            device=self.args.device,
        )
        tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
        return {
            "token_ids": tokens_predict,
        }

    def __supervised_inference(
        self,
        model,
        dataloader,
        tokenizer,
        **kwargs,
    ):
        out_dict = select_and_generate(
            model=model,
            dataloader=dataloader,
            pad_token_id=tokenizer.pad_token_id,
            prefix_len=1,
            # tokenizer=tokenizer,
            config=edict(
                device=self.args.device,
                aspect_topk=self.args.max_test_aspect_tokens,  # decide multi_aspect_forward
                temperature=self.args.temperature,
            ),
            **kwargs,
        )
        idss_predicted = out_dict["token_ids"]
        aspectids_predicted = out_dict["aspect_ids"]
        aspect_probs = out_dict["aspect_probs"]
        tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
        return {
            "token_ids": tokens_predict,
            "aspect_ids": aspectids_predicted,
            "aspect_probs": aspect_probs,
        }

    def inference(self, model, dataloader, tokenizer, aspect_select_strategy, **kwargs):
        if aspect_select_strategy == Strategy.SUPERVISED:
            # needs to have `aspect_list` argument
            return self.__supervised_inference(model, dataloader, tokenizer, **kwargs)
        else:
            # needs to have `multi_aspect_forward` argument
            return self.__inference(model, dataloader, tokenizer, **kwargs)

    def retrieve_reviews(self, test_data, queries) -> list:
        raise NotImplementedError

        item_retrieveds = []
        user_retrieveds = []

        for i, d in enumerate(tqdm(test_data)):
            # len(queries) 為 args.max_samples
            if i >= len(queries):
                break
            user_id = self.user_dict.get_entity(d["user"])
            item_id = self.item_dict.get_entity(d["item"])
            predict_text = queries[i]
            item_retrieved = self.searcher.search(
                corpus_subset=self.train_item2docids.get(item_id, []),
                # with `vector` field, limited to under the same `item`
                queries=[predict_text],
                top_k=self.args.retrieve_topk,
                score_function="cos_sim",
                return_sorted=True,
            )
            user_retrieved = self.searcher.search(
                corpus_subset=self.train_user2docids.get(user_id, []),
                queries=[predict_text],
                top_k=self.args.retrieve_topk,
                score_function="cos_sim",
                return_sorted=True,
            )
            # squeeze the dimension
            item_retrieved = item_retrieved[0]
            user_retrieved = user_retrieved[0]
            # return the retrieved documents
            item_retrieveds.append(item_retrieved)
            user_retrieveds.append(user_retrieved)
        assert len(item_retrieveds) == len(queries)
        return item_retrieveds, user_retrieved
