import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils.options import VALID_BASIS_TYPES

from .fmnet_base_loop import FMNetBase

import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import KNNGraph

@MODEL_REGISTRY.register()
class FMNetWithTAGModel(FMNetBase):
    def __init__(self, opt):
        super(FMNetWithTAGModel, self).__init__(opt)
        self.graph_former = KNNGraph(k=10)

    def data_forward(self, data_x, data_y):
        bs = data_x['verts'].shape[0]
        feat_x, feat_y = self.networks['feature_extractor'](data_x['graph'], data_y['graph'], bs)  # [B, Nx, C]
        return feat_x, feat_y
