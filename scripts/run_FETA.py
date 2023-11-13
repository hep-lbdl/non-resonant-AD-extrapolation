import argparse
import numpy as np
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div_phys
from helpers.utils import load_nn_config
from semivisible_jet.utils import *
import torch
import os
import logging


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    action="store",
    default=None,
    help=".npz file for input training samples and conditional inputs",
)
parser.add_argument(
    '-w', 
    "--weights",
    action="store",
    default="reweighting_run/weights.npz",
    help='Load weights.'
)
parser.add_argument(
    "-c",
    "--config",
    action="store",
    default=None,
    help="Classifier config file",
)
parser.add_argument(
    '-s', 
    "--samples",
    action="store",
    default=None,
    help='Directly load generated samples.'
)
parser.add_argument(
    '-m', 
    "--model",
    action="store_true",
    default=False,
    help='Load best trained MAF model.'
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="FETA_run",
    help="output directory",
)
parser.add_argument(
    "--toy",
    action="store_true",
    default=False,
    help="Load toy samples.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Verbose enable DEBUG",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

log_level = logging.DEBUG if args.verbose else logging.INFO
    
log = logging.getLogger("run")
log.setLevel(log_level)


def load_samples():
    # load input files
    inputs = np.load(args.input)
    # data and MC
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_feature = inputs["MC_feature"]
    MC_context = inputs["MC_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    inputs.close()
    
    # Get feature and contexts from data
    data_feat_CR = data_feature[data_mask_CR]
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]

    # Get feature and contexts from MC
    MC_feat_SR = MC_feature[MC_mask_SR]
    MC_cond_SR = MC_context[MC_mask_SR]

    if args.toy:
        data_feat_CR = data_feat_CR.reshape(-1, 1)
        data_feat_SR = data_feat_SR.reshape(-1, 1)
        MC_feat_SR = MC_feat_SR.reshape(-1, 1)
        MC_feature = MC_feature.reshape(-1, 1)
    
    return data_feat_CR, data_feat_SR, data_cond_CR, MC_feat_SR, MC_cond_SR, MC_feature, MC_context


def plot_reweighting(data_feat, pred_bkg, w_MC=None, RTag="SR"):
    
    feat = ["m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
    for i in range(len(feat)):
        name = name_map()[feat[i]]
        if feat[i]=="m_jj":
            bins = np.linspace(0, 4000, 40)
        else:
            bins = np.linspace(0, 1, 20)

        # plot SR distribution
        plot_kwargs = {"bins":bins, "name": name, "tag":f"{feat[i]}_{RTag}_unweighted", "outdir":args.outdir}
        plot_kl_div_phys(data_feat[:,i], pred_bkg[:,i], f"true bkg {RTag}", f"predicted bkg {RTag}", **plot_kwargs)
        
        # plot reweigted SR distribution
        if w_MC is not None:
            plot_kwargs = {"w2":[w_MC], "bins":bins, "name": name, "tag":f"{feat[i]}_{RTag}_weighted", "outdir":args.outdir}
            plot_kl_div_phys(data_feat[:,i], pred_bkg[:,i], f"true bkg {RTag}", f"predicted bkg {RTag}", **plot_kwargs)


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    data_feat_CR, data_feat_SR, data_cond_CR, MC_feat_SR, MC_cond_SR, MC_feature, MC_context = load_samples()

    # define useful variables
    nfeat = data_feat_CR.shape[1]
    ncond = data_cond_CR.shape[1]
    input_dim = nfeat
    load_model = args.model
    
    if args.samples is None:
        
        if load_model:
            # MAF model path
            trasport_model_path = f"{args.outdir}/MAF_final_model.pt"
            # Check if a model exist
            if os.path.isfile(trasport_model_path):
                # Load the trained model
                logging.info("Loading a trained MAF transport flow ...")
                transport_flow = torch.load(trasport_model_path)
                transport_flow.to(device)
            else:
                load_model = False

        # Train base density flow.
        logging.info("Training a base density flow to learn P_MC(x|m)...")
        base_density_flow = SimpleMAF(num_features=nfeat, num_context=ncond, device=device)
        base_density_flow.train(data=MC_feature, cond=MC_context, outdir=args.outdir)
        
        if not load_model:
            # Train trasport flow.
            logging.info("Training a transport flow to map P_MC(x|m) to P_data(x|m)...")
            transport_flow = SimpleMAF(num_features = nfeat, num_context=ncond, base_dist=base_density_flow.flow, device=device)
            transport_flow.train(data=data_feat_CR, cond=data_cond_CR, outdir=args.outdir, save_model=True, min_delta=0.01)
        
        # Standard scaling
        MC_feat_SR_scaled = transport_flow.scaler_transform_x(MC_feat_SR)
        MC_cond_SR_scaled = transport_flow.scaler_transform_c(MC_cond_SR)
        
        # Transport MC to data using trasport flow.
        if MC_feat_SR.ndim == 1:
            MC_feat_SR_torch = torch.tensor(MC_feat_SR_scaled.reshape(-1, 1), dtype=torch.float32).to(device)
        else:
            MC_feat_SR_torch = torch.tensor(MC_feat_SR_scaled, dtype=torch.float32).to(device)
        
        MC_cond_SR_torch = torch.tensor(MC_cond_SR_scaled, dtype=torch.float32).to(device)

        # transport data
        transport_data_feat_SR, _ = transport_flow.flow._transform.inverse(MC_feat_SR_torch, MC_cond_SR_torch)
        pred_bkg_SR_scaled = transport_data_feat_SR.detach().cpu().numpy()
        
        #inverse scaling
        pred_bkg_SR = transport_flow.scaler_inverse_x(pred_bkg_SR_scaled)

        # Save generated samples.
        np.savez(f"{args.outdir}/samples_data_feat_SR.npz", samples = pred_bkg_SR)
        
    else:
        # Load samples.
        pred_bkg_SR = np.load(args.samples)["samples"]

    # Load weights    
    w_MC = np.load(args.weights)["weights"]
    
    if not args.toy:
        plot_reweighting(data_feat_SR, pred_bkg_SR, w_MC, RTag="SR")
    
    # Create training data set for the classifier.
    input_feat_x = np.concatenate([pred_bkg_SR, data_feat_SR], axis=0)
    input_x = input_feat_x
    
    # Create labels for the classifier.
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape[0])
    data_feat_SR_label = np.ones(data_feat_SR.shape[0])
    input_y = np.hstack([pred_bkg_SR_label, data_feat_SR_label]).reshape(-1, 1)

    w_data = np.array([1.]*data_feat_SR.shape[0])
    input_weights = np.hstack([w_MC, w_data]).reshape(-1, 1)

    # Train a classifier for x.
    log.info("Training a classifer for signal vs background...")
    layers, lr, bs = load_nn_config(args.config)
    NN = Classifier(n_inputs=input_dim, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/signal_significance")
    NN.train(input_x, input_y, weights=input_weights, batch_size=bs, save_model=True, model_name="0")

    
    log.info("FETA style extrapolation done!")
    
if __name__ == "__main__":
    main()