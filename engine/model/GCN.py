'''
GCN model
'''

from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 256)
        self.conv2 = GCNConv(256, 16)
        self.conv3 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, edge_weight = None, **kwargs):
        # GCN 레이어 적용
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)