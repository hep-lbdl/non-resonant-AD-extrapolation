import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div, plot_multi_dist, plot_SIC
from helpers.utils import equalize_weights
from sklearn.model_selection import train_test_split
import torch
import os
import sys
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
    '-s', 
    "--samples",
    action="store",
    default=None,
    help='Directly load generated samples.'
)
parser.add_argument(
    '-m', 
    "--model",
    action="store",
    default=None,
    help='Load trained MAF model from path.'
)
parser.add_argument(
    "--oversample",
    action="store_true",
    default=False,
    help="Verbose enable DEBUG",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="CATHODE_run",
    help="output directory",
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


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
        
    # load input files
    inputs = np.load(args.input)
    
    # data and MC
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_context = inputs["MC_context"]
    
    # sig and bkg
    sig_feature = inputs["sig_feature"]
    sig_context = inputs["sig_context"]
    bkg_feature = inputs["bkg_feature"]
    bkg_context = inputs["bkg_context"]
    
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    
    #sig and bkg SR masks
    sig_mask_SR = inputs["sig_mask_SR"]
    bkg_mask_SR = inputs["bkg_mask_SR"]
    
    inputs.close()

    # Get feature and contexts from data
    data_feat_CR = data_feature[data_mask_CR]
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]

    # Get only contexts from MC
    MC_cond_SR = MC_context[MC_mask_SR]
    
    # Get feature and contexts from sig
    

    # define useful variables
    nfeat = data_feat_CR.ndim
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values

    if args.samples is None:
        
        if args.model is None:        
            # Train a MAF for density estimation
            logging.info("Training a MAF to learn P(x|m)...")
            MAF = SimpleMAF(num_features = nfeat, num_context=ncond, device=device)
            MAF.train(data=data_feat_CR, cond = data_cond_CR, outdir=args.outdir, save_model=True)

        else:
            # Load the trained model
            logging.info("Loading a trained MAF...")
            MAF = torch.load(f"{args.model}")
            MAF.to(device)
        
        # sample from MAF
        n_sample = 1 if args.oversample else 1
        pred_bkg_SR = MAF.sample(n_sample, MC_cond_SR).flatten()
        log.debug(f"MAF generated {len(pred_bkg_SR)} bkg events in the SR. Oversampling is not avaliable.")

        # save generated samples
        np.savez(f"{args.outdir}/samples_data_feat_SR.npz", samples = pred_bkg_SR)
        
    else:
        # Load samples
        pred_bkg_SR = np.load(args.samples)["samples"]

    # load weights    
    w_MC = np.load(args.weights)["weights"]
    
    # Make validation plots.
    plot_kwargs = {"tag":"2DSR_unweighted", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
    plot_kl_div([data_feat_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
        
    plot_kwargs = {"weights2":[w_MC], "tag":"2DSR", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
    plot_kl_div([data_feat_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
    
    
    log.info("Training a classifer for signal vs background...")
    
    # Create training data set for the classifier.
    input_feat_x = np.hstack([pred_bkg_SR, data_feat_SR]).reshape(-1, 1)
    input_cond_x = np.vstack([MC_cond_SR, data_cond_SR])
    input_x = np.concatenate([input_feat_x, input_cond_x], axis=1)
    
    # Create labels for the classifier.
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape)
    data_feat_SR_label = np.ones(data_feat_SR.shape)
    input_y = np.hstack([pred_bkg_SR_label, data_feat_SR_label]).reshape(-1, 1)

    w_data = np.array([1.]*len(data_feat_SR))
    input_weights = np.hstack([w_MC, w_data]).reshape(-1, 1)
    
    # Train a classifier for x, m1 and m2.
    NN = Classifier(n_inputs=nfeat+ncond, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/signal_significance")
    NN.train(input_x, input_y, weights=input_weights, save_model=True)

    log.info("CATHODE style extrapolation done!")
    
if __name__ == "__main__":
    main()