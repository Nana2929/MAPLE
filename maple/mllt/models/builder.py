import mmcv
from torch import nn

from mllt.utils import build_from_cfg
from .registry import BACKBONES, NECKS, HEADS, LOSSES, CLASSIFIERS


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_classifier(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, CLASSIFIERS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
