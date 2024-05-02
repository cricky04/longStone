'''
GCN model
'''

from .basenet import BaseNet
from torch_geometric.nn import GCN
from torch.nn import Linear
import torch

class SimpleGCN(BaseNet):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SimpleGCN, self).__init__()
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels, num_layers=num_layers)
        self.mlp = Linear(in_features=out_channels, out_features=out_channels)
    
    def forward(self, data):
        device = next(self.parameters()).device
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        x = self.gcn(x, edge_index, batch)
        x = torch.sigmoid(x)
        x = self.mlp(x)

        return x