import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils.options import VALID_BASIS_TYPES


@MODEL_REGISTRY.register()
class FMNetWithPointNetModel(BaseModel):
    def __init__(self, opt):
        super(FMNetWithPointNetModel, self).__init__(opt)

    def data_forward(self, data_x, data_y):
        verts_x = data_x['verts'].transpose(2, 1)
        verts_y = data_y['verts'].transpose(2, 1)

        feat_x, _, _ = self.networks['feature_extractor'](verts_x)  # [B, Nx, C]
        feat_y, _, _  = self.networks['feature_extractor'](verts_y)  # [B, Ny, C]
        return feat_x, feat_y

