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
    '-t', 
    "--trains",
    action="store",
    type=int,
    default=1,
    help='Number of trainings.'
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
    
    # SR mask
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    
    #sig and bkg SR masks
    sig_mask_SR = inputs["sig_mask_SR"]
    bkg_mask_SR = inputs["bkg_mask_SR"]
    
    inputs.close()

    # Get feature and contexts from data
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_SR = data_context[data_mask_SR]

    # Get only contexts from MC
    MC_cond_SR = MC_context[MC_mask_SR]

    # define useful variables
    nfeat = data_feat_SR.ndim
    ncond = data_cond_SR.ndim

    # Load samples
    pred_bkg_SR = np.load(args.samples)["samples"]

    # load weights    
    w_MC = np.load(args.weights)["weights"]   
    
    if args.verbose:
        # Make validation plots.
        plot_kwargs = {"tag":"2DSR_unweighted", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
        plot_kl_div([data_feat_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)

        plot_kwargs = {"weights2":[w_MC], "tag":"2DSR", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
        plot_kl_div([data_feat_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)

    # Prepare traning inputs to the AD Classifier
    
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
    
    # Train the AD Classifier
    
    log.info(f"Ensamble size: {args.trains}")
    
    for i in range(args.trains):
        # Train a classifier for x, m1 and m2.
        log.info(f"Training a classifer for signal vs background...")
        
        NN = Classifier(n_inputs=nfeat+ncond, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/signal_significance")
        NN.train(input_x, input_y, weights=input_weights, save_model=True, model_name=f"{i+1}")

    log.info("AD training done!")

    
if __name__ == "__main__":
    main()