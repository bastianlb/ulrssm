import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils.options import VALID_BASIS_TYPES

from .fmnet_base_loop import FMNetBase


@MODEL_REGISTRY.register()
class FMNetModel(FMNetBase):
    def __init__(self, opt):
        super(FMNetModel, self).__init__(opt)