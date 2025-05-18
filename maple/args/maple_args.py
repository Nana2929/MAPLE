import argparse
from pathlib import Path

from easydict import EasyDict as edict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Aspect Text Generation with Prompt-Tuning LEarning (MAPLE)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="path for loading the pickle data"
    )
    parser.add_argument("--index", type=int, default=1, help="fold of the dataset (1-5)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument(
        "--max_batch",
        type=int,
        default=None,
        help="max number of batches used for training in 1 epoch; once exceed this, the epoch will end",
    )

    parser.add_argument("--batch_size", type=int, default=196, help="batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient_accumulation_steps",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="ADAMW",
        help="optimizer type",
    )
    # parser.add_argument("--do_inference", action="store_true", help="do inference only")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use CPU instead of GPU (even if GPU is available)",
    )
    parser.add_argument("--log_interval", type=int, default=400, help="report interval")

    parser.add_argument(
        "--words",
        type=int,
        default=20,
        help="number of words to generate for each sample (including the <feat>... part if using feature-focused)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="max number of test samples to use for generating the testset",
    )

    # !! Legacy code !!
    # =========== ways to initialize soft prompt ===========
    parser.add_argument(
        "--use_pretrain_prompt",
        action="store_true",
        help="use gpt2 pretrained embeddings to initialize soft prompt for user, item, aspect, \
                        otherwise initialize them all as uniform_(-0.1, 0.1)",
    )
    parser.add_argument(
        "--prompt_tokens_maxlen",
        type=int,
        default=5,
        help="This argument takes effect only when --use_pretrain_prompt is True. \
                        number of tokens (NOT number of features; 1 feature can be a multi-token phrase) \
                        to initialize soft prompt for user, item, aspect",
    )
    parser.add_argument(
        "--ffidf_cache_dir",
        type=str,
        default=None,
        help="Legacy Code. This argument takes effect only when --use_pretrain_prompt is True. \
                        cache dir for ffidf values, for how to generate the ffidf_cache, refer to `ffidf/compute_ffidf.py`",
    )
    parser.add_argument(
        "--ffidf_topk",
        type=int,
        default=5,
        help="Legacy Code. This argument takes effect only when --use_pretrain_prompt is True. \
                        number of pre-extracted features to initialize the soft prompt for user and item",
    )
    # !! end of Legacy code !!
    # =========== use feature-focused prefix-tuning or not ===========
    parser.add_argument(
        "--feat_select_strategy",
        type=str,
        help="how to select feature for training data. For every u-i-a, we may have more than one triplet describing it.\
          We decide which one to use using strategy: random, longest",
        default="longest",
    )

    # =========== inference strategies ===========
    parser.add_argument(
        "--max_test_aspect_tokens",
        type=int,
        default=2,
        help="for testing, \
        at most how many aspect tokens can be used for averaging. \
            This is used for hit_ratio@ and ndcg@ calculation.",
    )
    parser.add_argument(
        "--aspect_select_strategy",
        type=str,
        default="supervised",
        help="how to select aspect tokens for inference, \
                        available: supervised, heuristic, and gt",
    )
    # =========== Model and Dataset Names ===========
    parser.add_argument(
        "-a",
        "--auto_arg_by_dataset",
        type=str,
        default=None,
        help="auto set some arguments by dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="base model name, currently only support gpt2",
    )
    # =========== multi-task learning ===========
    parser.add_argument(
        "--text_epochs",
        type=int,
        default=30,
        help="stage 1 epochs (the period when we only train the text generation task)",
    )
    parser.add_argument(
        "--text_endure_times",
        type=int,
        default=5,
        help="the maximum endure text-only training epochs of loss increasing on validation",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="stage 2 epochs (the period when we train both text-gen and aspect selection)",
    )  # tune all
    parser.add_argument(
        "--endure_times",
        type=int,
        default=2,
        help="the maximum endure epochs of loss increasing on validation",
    )
    parser.add_argument(
        "--text_reg",
        type=float,
        default=1.0,
        help="regularization on text generation task",
    )
    parser.add_argument(
        "--aspect_rating_reg",
        type=float,
        default=0.01,
        help="regularization on aspect selection task",
    )

    # =========== teacher forcing ===========
    # !! legacy code !!
    parser.add_argument(
        "--use_teacher_forcing",
        action="store_true",
        help="Whether to use teacher-forcing (predicted aspects) for text-gen in 2nd stage",
    )

    parser.add_argument(
        "--teafor_topk",
        type=int,
        default=1,
        help="Only takes effect when --use_teacher_forcing is True. \
                            topk predicted aspects to be conditioned on in teacher-forcing",
    )
    # !! end of legacy code !!
    # ============ checkpoint ============
    parser.add_argument(
        "--ckptdir",
        type=str,
        help="checkpoint dir",
    )
    args = parser.parse_args()
    # print args for debugging
    print(args)
    CKPT_DIR = args.ckptdir
    index = args.index
    # ========== Restaurant domain ==========

    if args.auto_arg_by_dataset == "yelp23":
        dargs = edict(
            data_path="nete_format_data/yelp23/reviews.pickle",
            index_dir=f"nete_format_data/yelp23/{index}",
            aspect_path="nete_format_data/yelp23/aspect_category_index.csv",
            freq_filepath=f"nete_format_data/yelp23/{index}/freq_file.pkl",
            checkpoint="checkpoints/{CKPT_DIR}/yelp23/{index}",
        )
    elif args.auto_arg_by_dataset == "yelp":
        dargs = edict(
            data_path="nete_format_data/yelp/reviews.pickle",
            index_dir=f"nete_format_data/yelp/{index}",
            aspect_path="nete_format_data/yelp/aspect_category_index.csv",
            freq_filepath=f"nete_format_data/yelp/{index}/freq_file.pkl",
            checkpoint=f"checkpoints/{CKPT_DIR}/yelp/{index}",
        )
    # ========== Non-restaurant domain ==========
    # elif args.auto_arg_by_dataset == "tripadvisor":
    #     dargs = edict(
    #         data_path=ROOT / "nete_format_data/tripadvisor/reviews.pickle",
    #         index_dir=ROOT / f"nete_format_data/tripadvisor/{index}",
    #         aspect_path=ROOT / "nete_format_data/tripadvisor/aspect_category_index.csv",
    #         checkpoint=ROOT / f"aspect_retriever/checkpoints/{CKPT_DIR}/tripadvisor",
    #     )
    # elif args.auto_arg_by_dataset == "clothing":
    #     dargs = edict(
    #         data_path=f"{ROOT}/nete_format_data/Amazon/clothing/reviews.pickle",
    #         index_dir=f"{ROOT}/nete_format_data/Amazon/clothing/{index}",
    #         aspect_path=f"{ROOT}/nete_format_data/Amazon/clothing/aspect_category_index.csv",
    #         checkpoint=f"{ROOT}/aspect_retriever/checkpoints/{CKPT_DIR}/clothing",
    #     )
    # elif args.auto_arg_by_dataset == "movies_and_tv":
    #     dargs = edict(
    #         data_path=f"{ROOT}/nete_format_data/Amazon/movies_and_tv/reviews.pickle",
    #         index_dir=f"{ROOT}/nete_format_data/Amazon/movies_and_tv/{index}",
    #         aspect_path=f"{ROOT}/nete_format_data/Amazon/movies_and_tv/aspect_category_index.csv",
    #         checkpoint=f"{ROOT}/aspect_retriever/checkpoints/{CKPT_DIR}/movies_and_tv",
    #     )
    args = edict(vars(args))
    args.update(dargs)
    return args
