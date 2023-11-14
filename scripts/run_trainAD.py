import argparse
import numpy as np
from math import sin, cos, pi
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div
from helpers.utils import load_nn_config
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
    '-s', 
    "--samples",
    action="store",
    default=None,
    help='Directly load generated samples.'
)
parser.add_argument(
    "-c",
    "--config",
    action="store",
    default=None,
    help="Classifier config file",
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

    # data
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    # MC
    MC_context = inputs["MC_context"]
    # SR mask
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    
    inputs.close()

    # Get feature and contexts from data
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_SR = data_context[data_mask_SR]
    MC_cond_SR = MC_context[MC_mask_SR]

    if args.toy:
        data_feat_SR = data_feat_SR.reshape(-1,1)

    return data_feat_SR, data_cond_SR, MC_cond_SR


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
        
    data_feat_SR, data_cond_SR, MC_cond_SR = load_samples()

    # define useful variables
    nfeat = data_feat_SR.shape[1]
    ncond = data_cond_SR.shape[1]
    input_dim = nfeat+ncond if args.toy else nfeat

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
    input_feat_x = np.concatenate([pred_bkg_SR, data_feat_SR], axis=0)
    if args.toy:
        input_cond_x = np.concatenate([MC_cond_SR, data_cond_SR], axis=0)
        input_x = np.concatenate([input_cond_x, input_feat_x], axis=1)
    else:
        input_x = input_feat_x
    
    # Create labels for the classifier.
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape[0])
    data_feat_SR_label = np.ones(data_feat_SR.shape[0])
    input_y = np.hstack([pred_bkg_SR_label, data_feat_SR_label]).reshape(-1, 1)

    w_data = np.array([1.]*len(data_feat_SR))
    input_weights = np.hstack([w_MC, w_data]).reshape(-1, 1)
    
    # Train the AD Classifier
    log.info(f"Ensamble size: {args.trains}")
    
    layers, lr, bs = load_nn_config(args.config)

    for i in range(args.trains):
        # Train a classifier for x, m1 and m2.
        log.info(f"Training a classifer for signal vs background...")
        
        NN = Classifier(n_inputs=input_dim, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/signal_significance")
        NN.train(input_x, input_y, weights=input_weights, batch_size=bs, save_model=True, model_name=f"{i+1}")
        

    log.info("AD training done!")

    
if __name__ == "__main__":
    main()