from dataclasses import dataclass
from typing import Optional
import sys
import os

sys.path.append(os.getcwd())
from mllt.models.builder import build_loss

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


@dataclass
class RecModelOutput(ModelOutput):
    text_loss: Optional[torch.FloatTensor] = None
    text_logits: torch.FloatTensor = None
    tag_rating: torch.FloatTensor = None
    tag_loss: Optional[torch.FloatTensor] = None
    lm_outputs: any = None


class AspectMLP(nn.Module):
    # personalized attention weights to decide

    def __init__(
        self,
        naspect,
        emsize,
        hidden_size=256,
        loss_cls=dict(
            type="ResampleLoss",
            use_sigmoid=True,
            reweight_func="rebalance",
            focal=dict(focal=True, balance_param=2.0, gamma=2),
            logit_reg=dict(neg_scale=2.0, init_bias=0.05),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
            loss_weight=1.0,
            freq_file="freq_file.pkl",
        ),
    ) -> None:
        super(AspectMLP, self).__init__()
        relu = nn.LeakyReLU()
        self.first_layer = nn.Linear(emsize * 2, hidden_size)
        layer = nn.Linear(hidden_size, hidden_size)
        layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layers = nn.Sequential(layer, relu, layer2, relu)
        self.classifier = nn.Linear(hidden_size // 2, naspect)
        self.loss_cls = build_loss(loss_cls)

    def loss(self, cls_score, labels, weight=None, reduction_override=None) -> dict:
        losses = dict()
        losses["loss_cls"] = self.loss_cls(
            cls_score,
            labels,
            weight,
            avg_factor=None,
            reduction_override=reduction_override,
        )
        if isinstance(cls_score, list):
            cls_score = cls_score[0]
        return losses

    def init_weights(self) -> None:
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, user, item) -> torch.Tensor:
        """
        args
            user: embedding of dim (batch_size, emsize)
            item: embedding of dim (batch_size, emsize)
        return
            probs: aspect classification probability of dim (batch_size, naspect)
        """
        x = torch.cat([user, item], dim=1)  # (batch_size, emsize * 2)
        x = self.first_layer(x)  # (batch_size, hidden_size)
        x = self.layers(x)  # (batch_size, hidden_size//2)
        probs = self.classifier(x)  # (batch_size, naspect)
        return probs
