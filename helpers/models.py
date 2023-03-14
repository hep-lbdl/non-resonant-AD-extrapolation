import pytorch_lightning as pl


from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, PiecewiseCubicCouplingTransform

from nflows.transforms.permutations import ReversePermutation

from torch import optim


# define the Base Density trainer
class BaseDensityLit(pl.LightningModule):
    def __init__(self, dim_data, dim_cont, flow_args, lr, wd):
        super().__init__()
        
        self.lr = lr
        self.wd = wd
        
        self.dim_data = dim_data
        self.dim_cont = dim_cont
        self.num_layers = flow_args["num_layers"]
        self.num_nodes = flow_args["num_nodes"]
        self.num_blocks = flow_args["num_blocks"]
        self.num_bins = flow_args["num_bins"]
        
        self.tails = "linear"
        self.tail_bound = 3.5
        self.base_dist = StandardNormal(shape=[self.dim_data])
        
        # make the transforms
        
        self.transforms_list = []
        for _ in range(self.num_layers):
            self.transforms_list.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features = self.dim_data, 
                    hidden_features = self.num_nodes, num_blocks = self.num_blocks, tail_bound = self.tail_bound, 
                    context_features = self.dim_cont, tails = self.tails, num_bins = self.num_bins))     
            self.transforms_list.append(ReversePermutation(features = self.dim_data)) 
            
        self.transform = CompositeTransform(self.transforms_list)     
        self.flow = Flow(self.transform, self.base_dist)    
        
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        features, context = batch
        loss = -self.flow.log_prob(inputs=features, context = context).mean()  
        self.log("train_loss", loss, logger = True, on_epoch = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        features, context = batch
        loss = -self.flow.log_prob(inputs=features, context = context).mean()  
        self.log("val_loss", loss, logger = True, on_epoch = True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.flow.parameters(), lr = self.lr, weight_decay = self.wd)
        return optimizer
