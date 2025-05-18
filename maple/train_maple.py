"""
@File    :   train_maple.py
@Time    :   2024/04/07 16:59:13
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Training script for MAPLE model.
            For inferencing a trained model, see generate.py
"""

import logging
import math
import os
import random


# ======= 3rd party modules =======
import numpy as np
import torch
import wandb
from args.maple_args import parse_args

# ===== self defined modules =====
from dataset import AspectDataInitializer, AspectDataset
from dataset.dclass import Strategy
from easydict import EasyDict as edict
from mllt.datasets.loader import ClassAwareSampler
from modeling_maple import RecReg
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from utils import (
    MapleTrainer,
    bos,
    eos,
    ids2tokens,
    now_time,
    pad,
    save_jsonl,
    save_pickle,
    select_and_generate,
)


def get_model_path(checkpoint):
    model_path = os.path.join(checkpoint, "model.pt")
    # if os.path.exists(model_path):
    #     return model_path
    # dirs = os.listdir(checkpoint)
    # runs = [run for run in dirs if run.startswith("run_")]
    # if len(runs) > 0:
    #     runs.sort()
    #     model_path = os.path.join(checkpoint, runs[-1], "model.pt")
    # else:
    #     raise ValueError("No model.pt found in the checkpoint directory")
    return model_path


torch.cuda.empty_cache()
# deactivate wandb, see https://github.com/wandb/wandb/issues/764#issuecomment-726247199
# os.environ["WANDB_MODE"] = "disabled"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    run = wandb.init(
        dir="wandb",
        project="maple-{args.ckptdir}",
        notes=" (W/o ff) Attempt to train gpt2 on (1) user-item-aspect prompt nlg (2) user-item aspect prediction tag selection; with loss weight",
        config=args,
        name=f"maple-{args.auto_arg_by_dataset}-index={args.index}-{now_time()}",
    )
    print("Wandb Run ID: {}".format(wandb.run.id))
    print("Wandb Run Name: {}".format(wandb.run.name))

    if args.data_path is None or os.path.exists(args.data_path) is False:
        raise ValueError(
            f"--data_path {args.data_path} should be available for loading data"
        )
    if args.index_dir is None:
        raise ValueError(
            f"--index_dir {args.index_dir} should be available for loading data splits"
        )
    if args.epochs == 0 and args.text_epochs == 0:
        raise ValueError("Both text_epochs and epochs are 0, no training is performed")

    # log into a file of args.checkpoint
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.exists(os.path.join(args.checkpoint, "model.pt")):
        # checkpoint = create_run_dir(args.checkpoint)
        model_path = get_model_path(args.checkpoint)
        wandb.config.update({"checkpoint": args.checkpoint}, allow_val_change=True)
    else:
        print(
            "A model.pt file already exists in the checkpoint directory, continue executing will overwrite it!!"
        )
        model_path = get_model_path(args.checkpoint)
    fname = "train.log"
    logging.basicConfig(
        filename=os.path.join(args.checkpoint, fname),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",  # overwrite the original logs
    )
    logger = logging.getLogger(__name__)

    logger.info(
        f"Detected main directory: {args.checkpoint}, logging to {os.path.join(args.checkpoint, fname)}"
    )
    logger.info(f"========== ARGUMENTS ==========")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info(f"========== end of ARGUMENTS ==========")

    if torch.cuda.is_available():
        if args.cpu:
            logger.warning(
                "WARNING: You have a CUDA device, so you should probably not run with --cpu"
            )
    device = torch.device("cpu" if args.cpu else "cuda")
    ###############################################################################
    # Load data
    ###############################################################################

    logger.info(f"Loading data from {args.data_path}")
    if args.model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(
            args.model_name, bos_token=bos, eos_token=eos, pad_token=pad
        )
    else:
        raise NotImplementedError
    # print tokenizer special tokens
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"strategy: {args.aspect_select_strategy}")
    initializer = AspectDataInitializer(
        data_path=args.data_path,
        index_dir=args.index_dir,
        aspect_path=args.aspect_path,
        max_test_aspect_tokens=args.max_test_aspect_tokens,
        seq_len=args.words,
        tokenizer=tokenizer,
        feat_select_strategy=args.feat_select_strategy,
        test_strategy=args.aspect_select_strategy,
        logger=logger,
    )

    logger.info(f"Data initializer type: {initializer.__class__.__name__}")

    # feature_set = initializer.feature_set

    ###############################################################################
    # Build the model
    ###############################################################################

    nuser = len(initializer.user_dict)
    nitem = len(initializer.item_dict)
    naspect = len(initializer.aspect_list)
    ntoken = len(tokenizer)

    model = RecReg.from_pretrained(
        args.model_name,
        nuser,
        nitem,
        naspect,
        loss_cls=dict(
            type="ResampleLoss",
            use_sigmoid=True,
            reweight_func="rebalance",
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(
                init_bias=0.05, neg_scale=1.0
            ),  # neg_scale defaults to 1.0
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
            loss_weight=1.0,
            freq_file=args.freq_filepath,
        ),
        pad_token_id=tokenizer.pad_token_id,
    )

    logger.info(
        f"number of users: {nuser}, number of items: {nitem}, number of aspects: {naspect}"
    )
    logger.info(f"Building model ...")

    model.to(device)

    logger.info("Training from scratch ...")
    logger.info("Tokenizing the aspect list ...")

    # save the initializer.user_dict and initializer.item_dict
    save_pickle(
        data=initializer.user_dict,
        filepath=os.path.join(args.checkpoint, "user_dict.pickle"),
    )
    save_pickle(
        data=initializer.item_dict,
        filepath=os.path.join(args.checkpoint, "item_dict.pickle"),
    )
    logger.info("Saving user and item dictionary.")
    logger.info(f"Resizing the token embeddings to {ntoken}")
    model.resize_token_embeddings(
        ntoken
    )  # three tokens added, update embedding table to add <pad>, <bos>, <eos>
    # ========== Training data ==============
    train_data = AspectDataset(
        data=initializer.train,
        tokenizer=tokenizer,
        bos=bos,
        eos=eos,
        max_len=args.words,
    )
    # val_data = AspectDataset(
    #     data=initializer.valid,
    #     tokenizer=tokenizer,
    #     bos=bos,
    #     eos=eos,
    #     max_len=args.words,
    # )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
    )

    logger.info("u-i-a train data size: {}".format(len(train_data)))

    # ========= VALIDATION data =========

    tag_val_data = AspectDataset(
        initializer.tag_valid,
        tokenizer=tokenizer,
        bos=bos,
        eos=eos,
        max_len=args.words,
    )

    tag_val_loader = DataLoader(
        tag_val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tag_val_data.collate_fn,
    )
    # ========== TESTING data =========
    test_data = AspectDataset(
        data=initializer.test,  # == tag_test
        tokenizer=tokenizer,
        bos=bos,
        eos=eos,
        max_len=args.words,
        max_samples=args.max_samples,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_data.collate_fn,
    )
    logger.info("valid data size: {}".format(len(tag_val_data)))
    logger.info(
        "test data size: {}".format(len(test_data))
    )  # test_data == tag_test_data
    # =============== training ===============

    config = edict(
        optimizer=args.optimizer,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        model_path=model_path,
        lr=args.lr,
        max_batch=args.max_batch,
        pad_token_id=tokenizer.pad_token_id,
        prefix_len=1,  # <bos>
    )
    trainer = MapleTrainer(
        config=config,
        model=model,
        logger=logger,
        device=device,
        log_interval=args.log_interval,
        aspect_list=initializer.aspect_list,
    )
    ###############################################################################
    # Train Stage 1: Text-Prediction Only
    # Saving strategy: best text val loss (lowest)
    ###############################################################################
    if args.text_epochs > 0:

        model = trainer.train_loop(
            model=model,
            epochs=args.text_epochs,
            endure_times=args.text_endure_times,
            text_reg=args.text_reg,
            aspect_rating_reg=0,
            train_loader=train_loader,
            val_loader=tag_val_loader,
            topk=args.max_test_aspect_tokens,
            save_strategy="best_text",
        )  # load the best model saved during this train_loop
    ###############################################################################
    # Train Stage 2: Text-Prediction + Aspect Prediction Jointly
    # Saving strategy: best aspect tag prediction loss (lowest)
    ###############################################################################
    if args.epochs > 0:
        model = trainer.train_loop(
            model,
            epochs=args.epochs,
            endure_times=args.endure_times,
            text_reg=args.text_reg,
            aspect_rating_reg=args.aspect_rating_reg,
            train_loader=train_loader,
            val_loader=tag_val_loader,
            topk=args.max_test_aspect_tokens,
            save_strategy="best_tag",
        )
    if args.epochs == 0 and args.text_epochs == 0:
        raise ValueError("Both text_epochs and epochs are 0, no training is performed")
    val_t_loss, val_ra_loss, val_ndcg, val_hit_ratio = trainer.evaluate(
        model=model,
        dataloader=test_loader,
        topk=args.max_test_aspect_tokens,
    )
    logger.info("====================================")
    topk = args.max_test_aspect_tokens
    logger.info(
        "on validation (test): text ppl {:4.4f}|\
                | aspect rating loss {:4.4f}|\
                | ndcg@{} {:4.4f} | hit ratio@{} {:4.4f}".format(
            math.exp(val_t_loss), val_ra_loss, topk, val_ndcg, topk, val_hit_ratio
        )
    )
    ##########################################
    # Inference
    ##########################################
    # load the best model
    with open(model_path, "rb") as f:
        model = torch.load(f)
        model.to(device)

    out_dict = select_and_generate(
        model=model,
        dataloader=test_loader,
        pad_token_id=tokenizer.pad_token_id,
        prefix_len=1,
        config=edict(
            device=device,
            aspect_topk=args.max_test_aspect_tokens,
            temperature=None,
        ),
    )
    idss_predict = out_dict["token_ids"]
    aspectids_predict = out_dict["aspect_ids"]
    aspect_probs = out_dict["aspect_probs"]

    outs = []
    gt_tags = []
    text_fake = []
    for i in range(len(test_data)):
        d = initializer.test[i]
        if args.aspect_select_strategy == Strategy.HEURISTIC:
            fake_category = [
                initializer.aspect_list[x] for x in test_data.fake_category_idxes[i]
            ]
            aspect_probs[i] = [None] * len(initializer.aspect_list)
        elif args.aspect_select_strategy == Strategy.SUPERVISED:
            fake_category = [initializer.aspect_list[x] for x in aspectids_predict[i]]
        else:  # args.aspect_select_strategy == "gt"
            fake_category = None

        # <u> <i> <a> <bos> `tacos are delicious` <eos>
        fake = " ".join(
            ids2tokens(idss_predict[i], tokenizer, eos=tokenizer.eos_token),
        )
        gt_category = [x[-1] for x in d["triplets"]]
        gt_tags.append(gt_category)

        text_fake.append(fake)

        x = {
            "user": d["user"],  # the 0 - to nuser index
            "user_id": initializer.user_dict.get_entity(d["user"]),
            "item": d["item"],
            "item_id": initializer.item_dict.get_entity(d["item"]),
            "real": d["text"],
            "fake": fake,
            "fake_category": fake_category,
            "fake_category_probs": aspect_probs[i],
            "gt_category": d["category_name"],
            "all_gt_category": gt_category,
            "triplets": d["triplets"],
        }
        outs.append(x)
    model_parent = os.path.dirname(model_path)
    prediction_path = os.path.join(
        model_parent,
        f"generated_{args.aspect_select_strategy}_k={args.max_test_aspect_tokens}.jsonl",
    )
    save_jsonl(path=prediction_path, data=outs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
