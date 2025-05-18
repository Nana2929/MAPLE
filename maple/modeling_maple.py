import torch
import torch.nn as nn

try:
    from modeling_maple_buildingblocks import (
        RecModelOutput,
        AspectMLP,
    )
except:
    from .modeling_maple_buildingblocks import (
        RecModelOutput,
        AspectMLP,
    )

from transformers import GPT2LMHeadModel
from mllt.models.losses import parse_losses

# =============================================================
# Continous Prompt Learning w/ recommendation as regularization
# =============================================================


class UIAPromptWithReg:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,  # gpt2
        nuser,
        nitem,
        naspect,
        loss_cls=None,
        pad_token_id=None,
        *args,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path)
        model.loss_cls = loss_cls
        model.init_modules(nuser, nitem, naspect)
        model.pad_token_id = pad_token_id

        return model

    def init_modules(self, nuser, nitem, naspect, hidden_size=400):
        # update all kwargs to self
        self.src_len = 3
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.aspect_embeddings = nn.Embedding(naspect, emsize)

        self.aspect_classifier = AspectMLP(
            naspect=naspect,
            emsize=emsize,
            hidden_size=hidden_size,
            loss_cls=self.loss_cls,
        )

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.aspect_embeddings.weight.data.uniform_(-initrange, initrange)
        self.aspect_classifier.init_weights()

    def forward_aspect(self, user, item):
        user = self.user_embeddings(user)
        item = self.item_embeddings(item)
        return self.aspect_classifier(user, item)

    def select_and_forward(
        self,
        user,
        item,
        text,
        mask,
        topk_aspect: int = 1,
        pad_token_id=None,
    ):
        # use predicted (or heuristic) aspect
        predicted_aspect_rating = self.forward_aspect(
            user=user,
            item=item,
        )
        predicted_batch_aspects = []
        for i in range(predicted_aspect_rating.size(0)):
            topk_indices = torch.topk(
                predicted_aspect_rating[i], topk_aspect, largest=True, sorted=True
            ).indices
            predicted_batch_aspects.append(topk_indices)

        predicted_batch_aspects = torch.stack(predicted_batch_aspects)
        # * Text Generation *
        out = self.forward(
            user=user,
            item=item,
            aspect=predicted_batch_aspects,
            text=text,
            mask=mask,
            multi_aspect_forward=(topk_aspect > 1),
            pad_token_id=pad_token_id,
            aspect_score=None,
            ignore_index=-100,
        )
        return out

    def forward(
        self,
        user,
        item,
        aspect,
        aspect_score,
        text,
        mask,
        multi_aspect_forward=False,
        pad_token_id=None,
        ignore_index=-100,
    ):
        # use `gt` aspect
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        # ======== Multi-forward =========
        if multi_aspect_forward:
            # inference: given several aspects, averaged them to make a prompt token
            a_src = []
            # for a_src, average on dimension 1 to (batch_size, emsize) and leaves the pad_token
            for a in aspect:
                a = a[a != pad_token_id]
                a = self.aspect_embeddings(a)
                a = torch.mean(a, dim=0).squeeze()
                a_src.append(a)
            a_src = torch.stack(a_src)  # (batch_size, emsize)
        else:  # training, use gt aspect token
            # aspect needs to be (batch_size, ) in this case, if not, we need to squeeze it
            if len(aspect.shape) > 1:
                aspect = aspect.squeeze(1)
            a_src = self.aspect_embeddings(aspect)
        # =================================
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat(
            [u_src.unsqueeze(1), i_src.unsqueeze(1), a_src.unsqueeze(1), w_src], 1
        )  # (batch_size, total_len, emsize)

        #  *Aspect Recommendation*
        outputs = super().forward(inputs_embeds=src)
        aspect_rating = self.aspect_classifier(u_src, i_src).squeeze()

        if mask is None or aspect_score is None:
            # auto-regressive generation
            # return super().forward(inputs_embeds=src), aspect_rating
            outputs = super().forward(inputs_embeds=src)
            return RecModelOutput(
                text_logits=outputs.logits,
                tag_rating=aspect_rating,
                lm_outputs=outputs,
            )
        else:
            # training
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(
                device
            )
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full(
                (batch_size, self.src_len), ignore_index, dtype=torch.int64
            ).to(
                device
            )  # (batch_size, src_len)
            pred_right = torch.where(
                mask == 1, text, torch.tensor(ignore_index).to(device)
            )  # (batch_size, src_len)
            # replace <pad> with ignore_index
            prediction = torch.cat(
                [pred_left, pred_right], 1
            )  # (batch_size, total_len)

            outputs = super().forward(
                attention_mask=pad_input, inputs_embeds=src, labels=prediction
            )

            aspect_loss_dict = self.aspect_classifier.loss(
                cls_score=aspect_rating, labels=aspect_score
            )
            aspect_loss, aspect_log_vars = parse_losses(aspect_loss_dict)
            return RecModelOutput(
                text_loss=outputs.loss,
                tag_loss=aspect_loss,
                text_logits=outputs.logits,
                tag_rating=aspect_rating,
                lm_outputs=outputs,
            )


class RecReg(UIAPromptWithReg, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
