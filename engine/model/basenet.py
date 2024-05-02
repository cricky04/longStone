import torch
import torch.nn as nn
import numpy as np

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rep_dim = None

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        print('Trainable parameters: {}'.format(params))
