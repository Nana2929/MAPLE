"""
@File    :   maple_trainer.oy
@Time    :   2024/01/31 17:11:05
@Author  :   Ching-Wen Yang
@Version :   1.0
@Contact :   P76114511@gs.ncku.edu.tw
@Desc    :   Maple Trainer
             Train text generation and aspect recommendation (multi-label classification) jointly.
"""

import random
import pandas as pd
import math
from logging import Logger
import os
import torch
from tqdm import tqdm
from collections import Counter
import numpy as np
import wandb
from transformers import get_linear_schedule_with_warmup
from .metrics import evaluate_ndcg, evaluate_hit_ratio
from .constants import SaveStrategy

###############################################################################
# Training, evaluate, generate functions
###############################################################################


OPTIM_MAP = {
    "ADAMAX": torch.optim.Adamax,
    "ADAM": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "ADAMW": torch.optim.AdamW,
}


def exp(x):
    return math.exp(x) if x < 1e2 else float("inf")


class MapleTrainer:

    def __init__(
        self,
        config,
        logger,
        device,
        log_interval,
        aspect_list,
        **kwargs,
    ):
        self.logger = logger
        self.device = device
        self.log_interval = log_interval  # 200
        self.max_batch = config.max_batch
        self.optimizer_name = config.optimizer
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # 1
        self.model_path = config.model_path
        self.lr = config.lr
        self.aspect_list = aspect_list
        self.pad_token_id = config.pad_token_id
        self.prefix_len = config.prefix_len

    @staticmethod
    def safe_multiloss(
        text_loss: float,
        aspect_rating_loss: float,
        text_reg: float,
        aspect_rating_reg: float,
    ):
        # if any of the loss is nan or inf, set it to 0
        if math.isnan(text_loss) or math.isinf(text_loss):
            text_loss = 0.0
        if math.isnan(aspect_rating_loss) or math.isinf(aspect_rating_loss):
            aspect_rating_loss = 0.0

        return text_reg * text_loss + aspect_rating_reg * aspect_rating_loss

    def get_scheduler(
        self, model, global_step: int
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        weight_decay = 1e-2
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = OPTIM_MAP[self.optimizer_name](
            params=optimizer_grouped_parameters,
            lr=self.lr,
        )
        num_warmup_steps = int(global_step * 0.1)
        self.logger.info(
            f"global_step: {global_step}, num_warmup_steps: {num_warmup_steps}"
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(global_step * 0.1),
            num_training_steps=global_step,
        )

        return (optimizer, lr_scheduler)

    def train(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        text_reg: float,
        aspect_rating_reg: float,
    ) -> None:
        # Turn on training mode which enables dropout.
        device = self.device
        text_loss = 0.0
        aspect_rating_loss = 0.0
        total_sample = 0

        for batch_idx, batch in enumerate(tqdm(dataloader), start=1):
            if self.max_batch is not None and batch_idx > self.max_batch:
                print(f"Early stopping at {batch_idx} batch")
                break
            user = batch.user.to(device)
            item = batch.item.to(device)
            aspect = batch.aspect.to(device)

            # Text Generation Ground-truth
            seq = batch.seq.to(device)  # (batch_size, seq_len)
            mask = batch.mask.to(device)
            # Aspect Prediction Ground-truth
            aspect_score = batch.aspect_score.to(device)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()
            outputs = model(
                user=user,
                item=item,
                aspect=aspect,
                text=seq,
                mask=mask,
                multi_aspect_forward=False,
                aspect_score=aspect_score,
            )
            # (text) NLL loss
            t_loss = outputs.text_loss
            # (aspect tag) BCE Loss
            ra_loss = outputs.tag_loss
            # Multi-task loss
            loss = self.safe_multiloss(
                text_loss=t_loss,
                aspect_rating_loss=ra_loss,
                text_reg=text_reg,
                aspect_rating_reg=aspect_rating_reg,
            )
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            batch_size = user.size(0)
            text_loss += batch_size * t_loss.item()
            aspect_rating_loss += batch_size * ra_loss.item()
            total_sample += batch_size

            if batch_idx % self.log_interval == 0 or batch == len(dataloader) - 1:
                cur_t_loss = text_loss / total_sample
                cur_ra_loss = aspect_rating_loss / total_sample
                self.logger.info(
                    f"text loss: {cur_t_loss} | aspect prediction loss: {cur_ra_loss} |\
                            {batch_idx}/{len(dataloader)} batches"
                )
                self.logger.info(
                    "text ppl {:4.4f} | {:5d}/{:5d} batches".format(
                        exp(cur_t_loss), batch_idx, len(dataloader)
                    )
                )
                wandb.log(
                    {
                        "step": batch_idx,
                        "train_t_loss": cur_t_loss,
                        "train_ra_loss": cur_ra_loss,
                        "train_t_ppl": exp(cur_t_loss),
                    }
                )

                text_loss = 0.0
                aspect_rating_loss = 0.0
                total_sample = 0

    def evaluate(
        self, model, dataloader, topk: int
    ) -> tuple[float, float, float, float]:
        # Turn on evaluation mode which disables dropout.

        device = self.device
        model.to(device)
        model.eval()
        text_loss = 0.0
        aspect_rating_loss = 0.0
        total_sample = 0

        with torch.no_grad():
            global_predicted_aspect_tags = []
            global_aspect_tags = []
            global_users = []
            global_items = []
            global_aspect_distributions = []
            global_predicted_aspect_distributions = []

            for batch_idx, batch in enumerate(tqdm(dataloader), start=1):
                user = batch.user.to(device)  # (batch_size,)
                item = batch.item.to(device)  # (batch_size,)
                global_users.extend(user.tolist())
                global_items.extend(item.tolist())
                aspect = batch.aspect.to(device)  # (batch_size,)
                aspect_score = batch.aspect_score.to(device)  # (batch_size, aspect_num)
                global_aspect_distributions.extend(aspect_score.tolist())
                seq = batch.seq.to(device)  # (batch_size, seq_len)
                mask = batch.mask.to(device)
                outputs = model(
                    user=user,
                    item=item,
                    aspect=aspect,
                    aspect_score=aspect_score,
                    text=seq,
                    mask=mask,
                    multi_aspect_forward=False,
                )
                t_loss = outputs.text_loss
                ra_loss = outputs.tag_loss
                predicted_aspect_rating = outputs.tag_rating

                global_predicted_aspect_distributions.extend(
                    predicted_aspect_rating.tolist()
                )
                for i in range(predicted_aspect_rating.size(0)):
                    topk_indices = torch.topk(
                        predicted_aspect_rating[i], topk, largest=True
                    ).indices

                    global_predicted_aspect_tags.append(topk_indices.tolist())
                    gt_aspects = aspect_score[i].nonzero().squeeze(dim=1).tolist()
                    assert isinstance(gt_aspects, list)
                    global_aspect_tags.append(gt_aspects)

                batch_size = user.size(0)
                text_loss += batch_size * t_loss.item()
                aspect_rating_loss += batch_size * ra_loss.item()
                total_sample += batch_size
        # end of with torch.no_grad()
        naspect = aspect_score.size(1)
        aspect_counter = Counter(
            [item for sublist in global_aspect_tags for item in sublist]
        )
        predict_aspect_counter = Counter(
            [item for sublist in global_predicted_aspect_tags for item in sublist]
        )
        aspect_ratio = [None for _ in range(naspect)]
        for i in range(naspect):
            aspect_ratio[i] = (predict_aspect_counter[i], aspect_counter[i])
        self.logger.info("# predict, # true occ: ")
        for i in range(naspect):
            aspect_name = self.aspect_list[i]
            self.logger.info(f"Aspect {i} {aspect_name}: {aspect_ratio[i]}")

        # evaluate metrics for aspect prediction
        ndcg = evaluate_ndcg(
            user2items_test=global_aspect_tags,
            user2items_top=global_predicted_aspect_tags,
        )
        hit_ratio = evaluate_hit_ratio(
            user2items_test=global_aspect_tags,
            user2items_top=global_predicted_aspect_tags,
        )

        return (
            text_loss / total_sample,
            aspect_rating_loss / total_sample,
            ndcg,
            hit_ratio,
        )

    @staticmethod
    def save_model(model, model_path):
        with open(model_path, "wb") as f:
            torch.save(model, f)

    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as f:
            model = torch.load(f)
        return model

    def train_loop(
        self,
        model,
        epochs: int,
        endure_times: int,
        text_reg: float,
        aspect_rating_reg: float,
        train_loader,
        val_loader,
        topk: int,
        save_strategy: str,
    ) -> torch.nn.Module:
        device = self.device
        global_steps = len(train_loader) * epochs
        if self.max_batch is not None:
            global_steps = min(global_steps, self.max_batch)
        self.optimizer, self.scheduler = self.get_scheduler(
            model=model, global_step=global_steps
        )
        self.logger.info(f"Optimizer: {self.optimizer}, Scheduler: {self.scheduler}")

        endure_count = 0
        epoch = 0
        if save_strategy == SaveStrategy.BEST_TAG:
            best_metrics = float("inf")
            monitor_metrics = "val_ra_loss"
        elif save_strategy == SaveStrategy.BEST_TEXT:
            best_metrics = float("inf")
            monitor_metrics = "val_t_loss"
        elif save_strategy == SaveStrategy.ALL:
            best_metrics = float("inf")
            monitor_metrics = "val_loss"
        else:
            raise ValueError(f"Unknown save strategy: {save_strategy}")

        self.logger.info(f"Save strategy: {save_strategy}")
        dirname = os.path.dirname(self.model_path)
        for epoch in range(1, epochs + 1):
            self.logger.info("epoch {}".format(epoch))
            self.train(
                model=model,
                dataloader=train_loader,
                text_reg=text_reg,
                aspect_rating_reg=aspect_rating_reg,
            )
            self.logger.info("Finish training epoch {}\nEvaluating...".format(epoch))
            val_t_loss, val_ra_loss, val_ndcg, val_hit_ratio = self.evaluate(
                model=model,
                dataloader=val_loader,
                topk=topk,
            )

            # idss_predict = generate(
            #     model=model,
            #     dataloader=val_loader,
            #     pad_token_id=self.pad_token_id,
            #     prefix_len=self.prefix_len,
            #     device=device,
            #     logger=self.logger,
            #     multi_aspect_forward=False,
            #     max_len=40,
            #     temperature=None,
            # )
            # self.logger.info("Generated text: ")
            # for i in range(DISPLAY_NUM):
            #     self.logger.info("-" + " ".join(idss_predict[i]))

            val_loss = self.safe_multiloss(
                text_loss=val_t_loss,
                aspect_rating_loss=val_ra_loss,
                text_reg=text_reg,
                aspect_rating_reg=aspect_rating_reg,
            )

            score_dict = {
                f"epoch": epoch,
                f"val_t_loss": val_t_loss,
                f"val_ra_loss": val_ra_loss,
                f"val_t_ppl": exp(val_t_loss),
                f"val_loss": val_loss,
                f"val_ndcg": val_ndcg,
                f"val_hit_ratio": val_hit_ratio,
            }
            prefix = f"[{save_strategy}-stage]"
            wandb.log({f"{prefix} {k}": v for k, v in score_dict.items()})

            self.logger.info(
                "text ppl {:4.4f} | valid t loss {:4.4f} | valid aspect prediction loss {:4.4f}| valid ndcg {:4.4f} | valid hit ratio {:4.4f}".format(
                    exp(val_t_loss), val_t_loss, val_ra_loss, val_ndcg, val_hit_ratio
                )
            )
            # # Save model for each epoch
            # epoch_path = os.path.join(dirname, f"model_stage={stage}_epoch={epoch}.pt")
            # self.save_model(model, epoch_path)
            # Save the model to specified best_model if the validation loss is the lowest we've seen so far.
            curr_metrics = score_dict.get(monitor_metrics)
            if monitor_metrics.endswith("loss"):
                # lower the better
                if curr_metrics < best_metrics:
                    best_metrics = curr_metrics
                    self.save_model(model, self.model_path)
                    self.logger.info(
                        f"Saving the best model with {monitor_metrics}:  {best_metrics:.4f}"
                    )
            else:
                if curr_metrics > best_metrics:
                    best_metrics = curr_metrics
                    self.save_model(model, self.model_path)
                    self.logger.info(
                        f"Saving the best model with {monitor_metrics}:  {best_metrics:.4f}"
                    )

            if curr_metrics != best_metrics:
                # means the model is not improving
                endure_count += 1
                self.logger.info("Endured {} time(s)".format(endure_count))
                if endure_count == endure_times:
                    print("Cannot endure it anymore | Exiting from early stop")
                    break
        self.logger.info(
            f"Tuning takes {epoch} epochs, reaching {endure_count} endure times"
        )
        # Load the best saved model.
        model = self.load_model(self.model_path)
        model = model.to(device)
        return model


def __select(predict_aspect_rating, topk, aspect_list) -> tuple[list, list]:

    topk_indices = []
    topk_probs = []

    CONFINED_TOPK = 5
    topk_ = torch.topk(predict_aspect_rating, CONFINED_TOPK, largest=True)
    temp_topk_indices = topk_.indices.cpu().tolist()
    temp_topk_probs = topk_.values.cpu().tolist()
    naspect = predict_aspect_rating.size(0)

    select_indices = random.choices(
        range(len(temp_topk_indices)),
        weights=temp_topk_probs,
        k=topk,
    )  # may repeat
    for idx in select_indices:
        topk_indices.append(temp_topk_indices[idx])
        topk_probs.append(temp_topk_probs[idx])
    return topk_indices, topk_probs


def select_and_generate(
    model,
    dataloader,
    pad_token_id,
    config,
    prefix_len=1,
    # use_filter="user",  # or "item" or None; legacy code
    # aspect_history=None,
    # aspect_list=None,
) -> dict:
    """Select <= 1 aspects with aspect-recommendation component and do uia-based text generation.
    prefix_len: int, the length of the prefix text to be used for first token generation
        if preifx is just `<bos>`, then prefix_len=1.
    """
    model.eval()
    # device used for model
    device = config.device
    # topk aspects to be used for text generation; multi-aspect forward if topk > 1
    topk = config.aspect_topk
    # temperature for sampling in text-generation
    temperature = config.temperature

    idss_predict = []
    aspectids_predict = []
    aspectprobs = []

    # print aspect topk

    with torch.no_grad():

        for batch in tqdm(dataloader):
            # * Aspect Selection *
            user = batch.user.to(device)  # (batch_size,)
            item = batch.item.to(device)  # (batch_size,)
            predicted_aspect_rating = model.forward_aspect(
                user=user,
                item=item,
            )
            # softmax
            predicted_aspect_rating = torch.sigmoid(predicted_aspect_rating)
            batch_aspects = []
            # !! Filtering with user, item history
            for i in range(predicted_aspect_rating.size(0)):
                topk_ = torch.topk(predicted_aspect_rating[i], topk, largest=True)
                topk_indices = topk_.indices.cpu().tolist()
                topk_probs = topk_.values.cpu().tolist()

                aspectids_predict.append(topk_indices)
                aspectprobs.append(topk_probs)
                tensorized_aspects = torch.tensor(topk_indices).to(device)
                batch_aspects.append(tensorized_aspects)
            batch_aspects = torch.stack(batch_aspects).to(device)
            # * Text Generation *
            aspect = batch_aspects.to(device)  # (batch_size, topk)
            aspect_score = batch.aspect_score.to(device)
            text = batch.seq[:, :prefix_len].to(device)  # (batch_size, 1)
            # <u> <i> <a> <feat> `tacos` <bos> `tacos are delicious` <eos>

            for _ in range(50):
                outputs = model(
                    user=user,
                    item=item,
                    aspect=aspect,
                    text=text,
                    mask=None,
                    aspect_score=aspect_score,
                    multi_aspect_forward=True if topk > 1 else False,
                    pad_token_id=pad_token_id,
                )
                if temperature is not None:
                    last_token = outputs.text_logits[:, -1, :] / (temperature + 1e-10)
                    # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(
                        last_token, dim=-1
                    )  # (batch_size, ntoken)
                    token = torch.multinomial(
                        word_prob, 1
                    )  # Sample from the multinomial distribution
                else:  # is None
                    last_token = outputs.text_logits[
                        :, -1, :
                    ]  # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(last_token, dim=-1)
                    token = torch.argmax(
                        word_prob, dim=1, keepdim=True
                    )  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, prefix_len:].tolist()  # remove prefix, (batch_size, seq_len)
            idss_predict.extend(ids)
    return {
        "token_ids": idss_predict,
        "aspect_ids": aspectids_predict,
        "aspect_probs": aspectprobs,
    }


def generate(
    model,
    dataloader,
    pad_token_id,
    device,
    logger,
    prefix_len=1,
    multi_aspect_forward=False,
    max_len=50,
    temperature=None,
) -> list:
    """Do aspect text generation and aspect rating prediction.
    The current version supports (1) gt aspect token inference (2) heuristic multi-aspect tokens
    Parameters
    ----------
    dataloader : DataLoader
    pad_token_id : int
    temperature : float, optional
    device : str
    logger : Logger
    multi_aspect_forward : bool, optional; use multi-aspect forward if your batch.aspect contains more than 1 aspect
                        per datapoint. Default is False because multi-aspect takes time for indexing and merging.

    Returns
    -------
    list of list of int (word indices)
    """
    logger.info("Temperature: {}".format(temperature))
    model.eval()  # Turn on evaluation mode which disables dropout.

    model.to(device)
    idss_predict = []
    aspect_rating_predict = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader), start=1):
            user = batch.user.to(device)  # (batch_size,)
            item = batch.item.to(device)
            aspect = batch.aspect.to(
                device
            )  # (batch_size,); (batch_size, topk) if multi-aspect forward
            text = batch.seq[:, :prefix_len].to(device)  # (batch_size, 1)
            # <u> <i> <a> <feat> `tacos` <bos> `tacos are delicious` <eos>
            for idx in range(max_len):
                # produce a word at each step
                outputs = model(
                    user=user,
                    item=item,
                    aspect=aspect,
                    text=text,
                    mask=None,
                    aspect_score=None,
                    multi_aspect_forward=multi_aspect_forward,
                    pad_token_id=pad_token_id,
                )
                if idx == 0:
                    aspect_rating_predict.extend(outputs.tag_rating.tolist())
                # last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                # word_prob = torch.softmax(last_token, dim=-1)
                # token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                if temperature is not None:
                    last_token = outputs.text_logits[:, -1, :] / (temperature + 1e-10)
                    # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(
                        last_token, dim=-1
                    )  # (batch_size, ntoken)
                    token = torch.multinomial(
                        word_prob, 1
                    )  # Sample from the multinomial distribution
                else:  # is None
                    last_token = outputs.text_logits[
                        :, -1, :
                    ]  # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(last_token, dim=-1)
                    token = torch.argmax(
                        word_prob, dim=1, keepdim=True
                    )  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, prefix_len:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)
    return idss_predict
