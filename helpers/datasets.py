import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

import numpy as np
import numpy.ma as ma


class ConditionalDataset(Dataset):
    def __init__(self, features, context):
        
        self.features = features
        self.context = context
        
        self.dim_features = features.shape[-1]
        self.dim_context = context.shape[-1]
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_p = self.features[idx]
        context_p = self.context[idx]
        return torch.tensor(feature_p), torch.tensor(context_p)
    
    
# function to generate the toy data
def Y(k, theta, x1, x2, sigma=1):
    return np.random.normal(k*(np.cos(theta)*x1 + np.sin(theta)*x2), sigma).astype(dtype=np.float32)
    
"""
For a CATHODE-style analysis
"""

class ToySingleDataModuleLit(pl.LightningDataModule):
    def __init__(self, dim_context, num_samples, k, theta):
        super().__init__()
        
        self.dim_context = dim_context
        self.num_samples = num_samples
        self.k = k
        self.theta = theta
          
        self.batch_size = 128
        
        # generate the contexts
        self.context = np.random.normal(0, 1, size = (self.num_samples, self.dim_context)).astype(dtype=np.float32)
        
        # generate the features
        if self.dim_context == 1:
            self.features = Y(self.k, self.theta, self.context, self.context)

        elif self.dim_context == 2:
            self.features = Y(self.k, self.theta, self.context[:,0], self.context[:,1])
        else:
            raise Exception("Toy dataset only implemented for dim_context = 1, 2.")
        
        
    def prepare_data(self):

        pass
  
    def setup(self, stage=None):
        
        # generate the context masks
        if self.dim_context == 1:
            mask_CR = self.context < 1
            mask_SR = self.context > 1
        
        elif self.dim_context == 2:
            mask_CR = np.logical_not((self.context[:,0] > 1) & (self.context[:,1] > 1))
            mask_SR = (self.context[:,0] > 1) & (self.context[:,1] > 1)
            
        else:
            raise Exception("Toy dataset only implemented for dim_context = 1, 2.")
            
        
        # split into train-val
        features_train, features_valid, context_train, context_valid, mask_CR_train, mask_CR_valid, mask_SR_train, mask_SR_valid = train_test_split(self.features, self.context, mask_CR, mask_SR, test_size=0.2)

        # make datasets
        self.train_dataset_CR = ConditionalDataset(features_train[mask_CR_train].reshape(-1, 1), context_train[mask_CR_train].reshape(-1, self.dim_context))
        self.valid_dataset_CR = ConditionalDataset(features_valid[mask_CR_valid].reshape(-1, 1), context_valid[mask_CR_valid].reshape(-1, self.dim_context))
        self.train_dataset_SR = ConditionalDataset(features_train[mask_SR_train].reshape(-1, 1), context_train[mask_SR_train].reshape(-1, self.dim_context))
        self.valid_dataset_SR = ConditionalDataset(features_valid[mask_SR_valid].reshape(-1, 1), context_valid[mask_SR_valid].reshape(-1, self.dim_context))
          
  
    def train_dataloader(self):
    
        return DataLoader(self.train_dataset_CR, 
                      batch_size = self.batch_size, shuffle = True,
                     num_workers = 8, pin_memory = True)
       
  
    def val_dataloader(self):

        return DataLoader(self.valid_dataset_CR,
                          batch_size = self.batch_size, shuffle = False,
                          num_workers = 8, pin_memory = True)
    
"""
For a FETA-style analysis
"""

    
class ToyJointDataModuleLit(pl.LightningDataModule):
    def __init__(self, dim_context, num_samples, k, theta):
        super().__init__()
        
        self.dim_context = dim_context
        self.num_samples = num_samples
        self.k = k
        self.theta = theta
          
        self.batch_size = 128
        
        # generate the contexts
        self.context = np.random.normal(0, 1, size = (self.num_samples, self.dim_context)).astype(dtype=np.float32)
        
        # generate the features
        if self.dim_context == 1:
            self.features = Y(self.k, self.theta, self.context, self.context)

        elif self.dim_context == 2:
            self.features = Y(self.k, self.theta, self.context[:,0], self.context[:,1])
        else:
            raise Exception("Toy dataset only implemented for dim_context = 1, 2.")
        
        
    def prepare_data(self):

        pass
  
    def setup(self, stage=None):
        
        # generate the context masks
        if self.dim_context == 1:
            mask_CR = self.context < 1
            mask_SR = self.context > 1
        
        elif self.dim_context == 2:
            mask_CR = np.logical_not((self.context[:,0] > 1) & (self.context[:,1] > 1))
            mask_SR = (self.context[:,0] > 1) & (self.context[:,1] > 1)
            
        else:
            raise Exception("Toy dataset only implemented for dim_context = 1, 2.")
            
        
        # split into train-val
        features_train, features_valid, context_train, context_valid, mask_CR_train, mask_CR_valid, mask_SR_train, mask_SR_valid = train_test_split(self.features, self.context, mask_CR, mask_SR, test_size=0.2)

        # make datasets
        self.train_dataset_CR = ConditionalDataset(features_train[mask_CR_train].reshape(-1, 1), context_train[mask_CR_train].reshape(-1, self.dim_context))
        self.valid_dataset_CR = ConditionalDataset(features_valid[mask_CR_valid].reshape(-1, 1), context_valid[mask_CR_valid].reshape(-1, self.dim_context))
        self.train_dataset_SR = ConditionalDataset(features_train[mask_SR_train].reshape(-1, 1), context_train[mask_SR_train].reshape(-1, self.dim_context))
        self.valid_dataset_SR = ConditionalDataset(features_valid[mask_SR_valid].reshape(-1, 1), context_valid[mask_SR_valid].reshape(-1, self.dim_context))
          
  
    def train_dataloader(self):
        

        return DataLoader(self.train_dataset_CR, 
                      batch_size = self.batch_size, shuffle = True,
                     num_workers = 8, pin_memory = True)
       
  
    def val_dataloader(self):

        return DataLoader(self.valid_dataset_CR,
                          batch_size = self.batch_size, shuffle = False,
                          num_workers = 8, pin_memory = True)
   
   