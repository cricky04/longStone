'''
for both real data and artificial data
'''
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch
import os.path as osp

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
    
    
    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
            
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
def dataLoader(dataRoot, batch_size):
    dataset = CustomDataset(dataRoot)
    loader = DataLoader(dataset, batch_size= batch_size, shuffle = True)
    return loader