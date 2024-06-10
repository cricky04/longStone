import torch
import torch_geometric
import numpy as np
import pandas as pd
import sys
import multiprocessing
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')


sys.path.append('../')

from torch_geometric.contrib.nn.models import GRBCDAttack
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from engine.model.GCN import SimpleGCN


def BudgetFinder(attack, dataset, model, pred, d):
    for budget in range(1, 125, 5):
        pertubed, fl = attack.attack(dataset.x, dataset.edge_index, dataset.y, budget, d)
        ppred = model(dataset.x, pertubed).argmax(dim = 1)
        if pred[d] != ppred[d]:
            return budget
    return -1

if __name__ == "__main__":
    dataName = 'Citeseer'
    model = torch.load(f'../data/checkpoint/simpleGCN_{dataName}.pt').to('cuda')
    attack = GRBCDAttack(model , 10).to('cuda')

    dataset = Planetoid(f'./{dataName}', dataName, pre_transform=NormalizeFeatures())[0].to('cuda')

    trainData = dataset
    testData = dataset

    pred = model(dataset.x, dataset.edge_index).argmax(dim = 1)
    minBudget = []

    nCpu = multiprocessing.cpu_count()

    inputs = []
    for d in tqdm(range(len(trainData.x))):
        if pred[d] != dataset.y[d]:
            minBudget.append(0)
            continue
        if len(inputs) < nCpu:
            inputs.append((attack, dataset, model, pred, d))
        else:
            pool = multiprocessing.Pool()
            outputs = pool.starmap(BudgetFinder, inputs)
            minBudget.extend(outputs)
            inputs = []
            
    with open(f'../result/pickles/{dataName}Budget.pkl', 'wb') as f:
        pickle.dump(minBudget, f)