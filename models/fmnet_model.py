import torch
import torch.nn.functional as F

from utils.registry import MODEL_REGISTRY
from .fmnet_base_loop import FMNetBase


@MODEL_REGISTRY.register()
class FMNetModel(FMNetBase):
    def __init__(self, opt):
        super(FMNetModel, self).__init__(opt)