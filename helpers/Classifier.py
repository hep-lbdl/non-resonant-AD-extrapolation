import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from helpers.utils import EarlyStopping
from tqdm import tqdm
import yaml


class Model(nn.Module):
    def __init__(self, 
                 layers,
                 n_inputs,
                 device="cpu"):
        
        super().__init__()
        
        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)
        self.device = device

    def forward(self, x):
        
        return self.model_stack(x)

    def predict(self, x):
        
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x, device=self.device)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction

    
class Classifier():
    def __init__(self, 
                 n_inputs,
                 layers=[64,64,64], 
                 learning_rate=1e-3, 
                 loss_type="binary_crossentropy", 
                 device="cpu"):

        self.n_inputs = n_inputs
        self.device = device
        self.model = Model(layers, n_inputs=n_inputs).to(self.device)

        if loss_type == 'binary_crossentropy':
            self.loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)        
        
    def np_to_torch(self, array):
    
        return torch.tensor(array.astype(np.float32))
    
    def process_data(self, input_x, batch_size):
        
        if self.n_inputs != input_x.ndim-1:
            raise RuntimeError("input data dimention doesn't match with number of features!")
        
        input_train, input_val = train_test_split(input_x, test_size=0.2, shuffle=True)
        
        x_train = torch.utils.data.TensorDataset(self.np_to_torch(input_train[:,:-1]),
            self.np_to_torch(input_train[:,-1]).reshape(-1,1))
        x_val = torch.utils.data.TensorDataset(self.np_to_torch(input_val[:,:-1]),
            self.np_to_torch(input_val[:,-1]).reshape(-1,1))
        
        train_data = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)
        val_data = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
        
        return train_data, val_data
    
    def train(self, input_x, n_epochs=200, batch_size=512, weights=None, seed=1, plot=False, outdir="./", save_model=None, early_stop=True, patience = 5):
        
        update_epochs = 1
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        epochs, epochs_val = [], []
        losses, losses_val = [], []
        
        if early_stop:
            early_stopping = EarlyStopping(patience=patience, min_delta=0.00001)
            
        train_data, val_data = self.process_data(input_x, batch_size=batch_size)
        
        for epoch in tqdm(range(n_epochs), ascii=' >='):
            
            losses_batch_per_e = []
            
            self.model.train()
            
            for batch_ndx, data in enumerate(train_data):
                
                batch_inputs, batch_labels = data
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                batch_outputs = self.model(batch_inputs)
                loss = self.loss_func(batch_outputs, batch_labels, weight=weights)
                losses_batch_per_e.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()

            epochs.append(epoch)
            losses.append(np.mean(losses_batch_per_e))
            
            with torch.no_grad():

                self.model.eval()
                val_losses_batch_per_e = []

                for batch_ndx, data in enumerate(val_data):

                    batch_inputs, batch_labels = data
                    batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)

                    batch_outputs = self.model(batch_inputs)
                    val_loss = self.loss_func(batch_outputs, batch_labels, weight=weights)
                    val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

                epochs_val.append(epoch)
                mean_val_loss = np.mean(val_losses_batch_per_e)
                losses_val.append(mean_val_loss)

                if early_stop:
                    early_stopping(mean_val_loss)

                # if plot:
                #     print(f"Epoch: {epoch} - loss: {loss} - val loss: {val_loss}")
                    
            if early_stop:
                if early_stopping.early_stop:
                    break
                    
        if plot:
            plt.plot(losses, epochs, label="loss")
            plt.plot(losses_val, epochs_val, label="val loss")
            plt.legend()
            plt.show
            plt.savefig(f"{outdir}/classfier_loss.png")
            plt.close
        
        if save_model is not None:
            torch.save(self.model, save_model+"_ep"+str(epoch))
    
    def evaluation(self, X_test, y_test=None, outdir='./', plot=True):
        
        self.model.eval()
        
        with torch.no_grad():
            x_test = self.np_to_torch(X_test).to(self.device)
            outputs = self.model(x_test).detach().cpu().numpy()

            # calculate auc 
            if y_test is not None:
                auc = roc_auc_score(y_test, outputs)
                fpr, tpr, _ = roc_curve(y_test, outputs)

        if plot and y_test is not None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            ax.plot(fpr, tpr)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC: " + str(auc))
            fname = f"{outdir}/roc.png"
            fig.savefig(fname)

            np.save(f"{outdir}/fpr.npy", fpr)
            np.save(f"{outdir}/tpr.npy", tpr)

            if auc < 0.5:
                auc = 1.0 - auc

            print(f"AUC: {auc}.")