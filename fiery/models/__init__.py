from importlib import import_module

from torch.nn import CrossEntropyLoss
from self_attention_cv.bottleneck_transformer import BottleneckModule

from fiery.layers.bev_self_attention import BEVSelfAttention
from fiery.models.encoder import ImageAttention
from fiery.models.head_wrappers.Anchor3DHeadWrapper import Anchor3DHeadWrapper
from fiery.models.head_wrappers.CenterHeadWrapper import CenterHeadWrapper


def import_obj(cfg):
    if cfg is None or (cfg.get('type', None) is None and cfg.get('NAME') is None):
        return None
    classname = cfg.get('type', None) or cfg.get('NAME', None)
    if classname not in globals():
        globals()[classname] = getattr(import_module(classname[:classname.rfind('.')]), classname.split('.')[-1])
    classname = globals()[classname]
    return classname


def build_obj(cfg):
    classname = import_obj(cfg)
    if classname is None:
        return None
    cfg = cfg.convert_to_dict()
    cfg.pop('type', None)
    cfg.pop('NAME', None)
    obj = classname(**cfg)
    return obj


__all__ = [
    'build_obj',
    'Anchor3DHeadWrapper',
    'BottleneckModule',
    'BEVSelfAttention',
    'CenterHeadWrapper',
    'CrossEntropyLoss',
    'ImageAttention',
]
