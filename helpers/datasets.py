import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import numpy as np


class ConditionalDataset(Dataset):
    def __init__(self, data, context):
        
        self.data = data
        self.context = context
        
        self.dim_data = data.shape[-1]
        self.dim_cont = context.shape[-1]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_p = self.data[idx]
        context_p = self.context[idx]
        return torch.tensor(data_p).float(), torch.tensor(context_p).float()
    
    
  
    
class ToyDataModuleLit(pl.LightningDataModule):
    def __init__(self, dim_data, dim_context, num_train, num_val):
        super().__init__()
        
        self.dim_data = dim_data
        self.dim_context = dim_context
        self.num_train = num_train
        self.num_val = num_val
          
        self.batch_size = 32
        
        self.train_data = np.random.triangular(-2, -1, 3, size = (self.num_train, self.dim_data))
        self.train_context = np.random.normal(4, 5, size = (self.num_train, self.dim_context))

        self.valid_data = np.random.triangular(-2, -1, 3, size = (self.num_val, self.dim_data))
        self.valid_context = np.random.normal(4, 5, size = (self.num_val, self.dim_context))
        
        self.train_dataset = ConditionalDataset(self.train_data, self.train_context)
        self.valid_dataset = ConditionalDataset(self.valid_data, self.valid_context)
          
        
    def prepare_data(self):

        pass
  
    def setup(self, stage=None):
        
        pass
  
    def train_dataloader(self):
        
        # Generating train_dataloader
        return DataLoader(self.train_dataset, 
                          batch_size = self.batch_size, shuffle = True,
                         num_workers = 8, pin_memory = True)
  
    def val_dataloader(self):
        
          # Generating val_dataloader
        return DataLoader(self.valid_dataset,
                          batch_size = self.batch_size, shuffle = False,
                          num_workers = 8, pin_memory = True)

  