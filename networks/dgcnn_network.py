import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
import torch_geometric
from utils.registry import NETWORK_REGISTRY


class GNNFeatExtractor(torch.nn.Module):
    """
    Defines the GNN feature extractor including the three
    TAG Conv layers.
    """

    def __init__(self, feat_size: int, in_channels: int = 3):
        super().__init__()
        self.tag_conv_1 = TAGConv(in_channels, feat_size // 4, 1)
        self.tag_conv_2 = TAGConv(feat_size // 4, feat_size // 2, 2)
        self.tag_conv_3 = TAGConv(feat_size // 2, feat_size, 3)
        self.fc_layer = nn.Linear(feat_size, feat_size)
        self.activation = nn.ReLU()

    def forward(self, data: torch_geometric.data.Data):
        """
        Forward pass of GNN. Takes a graph and processes it through
        the TAG Conv Layers and outputs the extracted features.
        Args:
            data:   Input Graph
        Return:
            x:      Extracted features per node
        """
        feat, edge_index, edge_attr = (data.x, data.edge_index, data.edge_attr)
        feat = self.activation(self.tag_conv_1(feat, edge_index, edge_attr))
        feat = self.activation(self.tag_conv_2(feat, edge_index, edge_attr))
        feat = self.activation(self.tag_conv_3(feat, edge_index, edge_attr))

        feat = self.fc_layer(feat)
        feat = F.normalize(feat, dim=-1)

        return feat

@NETWORK_REGISTRY.register()
class DGCNNNet(nn.Module):
    """
    Network to compute the functional maps based on GNN shape descriptors
    """

    def __init__(self, feat_size: int):
        super().__init__()
        self.feat_extractor = GNNFeatExtractor(feat_size, in_channels=3)

    def forward(
        self,
        graph_x: torch_geometric.data.Data,
        graph_y: torch_geometric.data.Data,
    ):
        """
        Based on two shapes, the network extracts shape descriptors through a GNN and
        calculates the functional mapping.
        Args:
            graph_x:        Graph of shape X
            graph_y:        Graph of shape Y
            evecs_trans_x:  Transposed LBO eigenvectors of shape X
            evecs_trans_y:  Transposed LBO eigenvectors of shape Y
        Returns:
            C1:         Functional mapping from X to Y
            C2:         Functional mapping from Y to X
            feat_x:     Features per node of X
            feat_y:     Features per node of Y
        """
        feat_x = self.feat_extractor(graph_x)
        feat_x = feat_x.view(1, -1, feat_x.shape[1])
        feat_y = self.feat_extractor(graph_y)
        feat_y = feat_y.view(1, -1, feat_y.shape[1]) # Batch size must be 1 here
 
        return feat_x, feat_y