import torch_geometric
import torch
import numpy as np
import os
import sys
import torch.nn.functional as F

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from tqdm import tqdm

def train(model, dataloader, epochs):
    model.train()
    optimizer = torch.optim.AdamW(params = model.parameters(),lr = 0.001)
    for e in range(epochs):
        optimizer.zero_grad()
        for data in tqdm(dataloader):
            output = model(data)
            # node classification
            loss = F.nll_loss(output, data.y)
            loss.backward()
        optimizer.step()
        
    return model