import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv

from utils.registry import NETWORK_REGISTRY

@NETWORK_REGISTRY.register()
class DGCNNNet(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        self.mlp = MLP(
            [3 * 64, 1024, 512, out_channels],
            dropout=0.5, norm=None
        )

    def forward(self, data, batch_size):
        x, pos, batch = data.x, data.pos, data.batch
        x0 = torch.cat([x, pos], dim=-1)
        
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return out.view(batch_size, -1, self.out_channels)