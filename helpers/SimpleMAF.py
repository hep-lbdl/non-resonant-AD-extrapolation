import matplotlib.pyplot as plt
import numpy as np
import logging

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from helpers.utils import EarlyStopping

from tqdm import tqdm

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

log = logging.getLogger("run")

# Turn off matplotlib DEBUG messages
plt.set_loglevel(level="warning")

class SimpleMAF:
    def __init__(self,
                 num_features,
                 num_context=None,
                 num_hidden_features=4,
                 num_layers=5,
                 learning_rate=1e-3,
                 base_dist=None,
                 act='relu',
                 device='cpu'):
        
        activations = {'relu': F.relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
        activation = activations[act]
        
        self.nfeat = num_features
        self.ncond = num_context
 
        if base_dist is not None:
            self.base_dist = base_dist
            # set the base flow to be static
            for param in self.base_dist.parameters():
                param.requires_grad = False
        else:
            self.base_dist = StandardNormal(shape=[num_features])

        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=num_features))
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features, 
                                                                  hidden_features=num_hidden_features, 
                                                                  context_features=num_context, 
                                                                  activation = activation))
        self.transform = CompositeTransform(transforms)
        self.flow = Flow(self.transform, self.base_dist).to(device)
        self.optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate)
        self.device = device

    def get_device(self):
        return next(self.flow.parameters()).device

    def to(self, device):
        self.flow.to(device)
        
    def np_to_torch(self, array):
    
        return torch.tensor(array.astype(np.float32))
    
    def process_data(self, data, batch_size, cond=None):
        
        if self.nfeat != data.ndim:
            raise RuntimeError("input data dimention doesn't match with number of features!")
        if self.nfeat == 1:
            data = data.reshape(-1, 1)
        
        if cond is not None:
            if self.ncond != cond.ndim:
                raise RuntimeError("input cond dimention doesn't match with number of cond features!")
            if self.ncond == 1:
                cond = cond.reshape(-1, 1)
            data = np.concatenate((data, cond), axis=1)
        
        x_train, x_val = train_test_split(data, test_size=0.2, shuffle=True)
        
        train_data = torch.utils.data.DataLoader(self.np_to_torch(x_train), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
        val_data = torch.utils.data.DataLoader(self.np_to_torch(x_val), batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
        
        return train_data, val_data
        
    
    def train(self, data, cond=None, n_epochs=1000, batch_size=512, seed=1, outdir="./", early_stop=True, patience=5, min_delta=0.005, save_model=False):
        
        update_epochs = 1
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        epochs, epochs_val = [], []
        losses, losses_val = [], []
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        
        train_data, val_data = self.process_data(data=data, batch_size=batch_size, cond=cond)
        
        for epoch in tqdm(range(n_epochs), ascii=' >='):
            
            losses_batch_per_e = []
            
            for batch_ndx, data in enumerate(train_data):
                data = data.to(self.device)
                if cond is not None:
                    x_ = data[:, :-self.ncond] if self.nfeat != 1 else data[:,0].reshape(-1, 1)
                    c_ = data[:, -self.ncond:] if self.ncond != 1 else data[:,-1].reshape(-1, 1)
                    loss = -self.flow.log_prob(inputs=x_, context=c_).mean()
                else:
                    x_ = data
                    loss = -self.flow.log_prob(inputs=x_).mean()  
                losses_batch_per_e.append(loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  

            epochs.append(epoch)
            losses.append(np.mean(losses_batch_per_e))
            
            if epoch % update_epochs == 0: # validation loss
                with torch.no_grad():

                    val_losses_batch_per_e = []

                    for batch_ndx, data in enumerate(val_data):
                        data = data.to(self.device)
                        self.optimizer.zero_grad()
                        if cond is not None:
                            x_ = data[:, :-self.ncond] if self.nfeat != 1 else data[:,0].reshape(-1, 1)
                            c_ = data[:, -self.ncond:] if self.ncond != 1 else data[:,-1].reshape(-1, 1)
                        else:
                            x_ = data
                            c_ = None
                        val_loss = -self.flow.log_prob(inputs=x_, context=c_).mean() 
                        val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())
               
                    epochs_val.append(epoch)
                    mean_val_loss = np.mean(val_losses_batch_per_e)
                    losses_val.append(mean_val_loss)
                    
                    if early_stop:
                        early_stopping(mean_val_loss)
            
                    log.debug(f"Epoch: {epoch} - loss: {loss} - val loss: {val_loss}")
        
            if early_stop:
                if early_stopping.early_stop:
                    break
        
        if save_model:

            model_path = f"{outdir}/MAF_final_model.pt"
            torch.save(self, model_path)

            log.info(f"The trained MAF model is saved at {model_path}.")

        plt.figure(figsize=(6,4))
        plt.plot(epochs, losses, label="loss")
        plt.plot(epochs_val, losses_val, label="val loss")
        plt.xlabel("number of epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show
        plt.savefig(f"{outdir}/MAF_loss.png")
        plt.close
                    

    def sample(self, num_samples, cond=None):
        cond = self.np_to_torch(cond).to(self.device)
        samples = self.flow.sample(num_samples=num_samples, context=cond)
        return samples.detach().cpu().numpy()