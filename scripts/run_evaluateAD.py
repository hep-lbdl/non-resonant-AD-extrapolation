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
    '-m', 
    "--model",
    action="store",
    default=None,
    help='Load trained model from path.'
)
parser.add_argument(
    '-n', 
    "--name",
    action="store",
    default="Model",
    help='Name of the model'
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="outputs",
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
    # sig and bkg
    sig_feature = inputs["sig_feature"]
    sig_context = inputs["sig_context"]
    bkg_feature = inputs["bkg_feature"]
    bkg_context = inputs["bkg_context"]

    # SR and CR masks
    sig_mask_SR = inputs["sig_mask_SR"]
    bkg_mask_SR = inputs["bkg_mask_SR"]
    inputs.close()

    # Get feature and contexts from data
    sig_feat_SR = sig_feature[sig_mask_SR]
    sig_cond_SR = sig_context[sig_mask_SR]
    
    # Get feature and contexts from bkg-only data
    bkg_feat_SR = bkg_feature[bkg_mask_SR]
    bkg_cond_SR = bkg_context[bkg_mask_SR]
    
    # Create training data set for classifier
    input_feat_x = np.hstack([bkg_feat_SR, sig_feat_SR]).reshape(-1, 1)
    input_cond_x = np.vstack([bkg_cond_SR, sig_cond_SR])
    input_x = np.concatenate([input_feat_x, input_cond_x], axis=1)
    
    # Create labels for classifier
    bkg_feat_SR_label = np.zeros(bkg_feat_SR.shape)
    sig_feat_SR_label = np.ones(sig_feat_SR.shape)
    input_y = np.hstack([bkg_feat_SR_label, sig_feat_SR_label]).reshape(-1, 1)
    
    # Load the trained model
    logging.info("Loading a trained NN...")
    
    NN = torch.load(f"{args.model}")
    NN.to(device)
    NN.set_outdir(f"{args.outdir}/signal_significance")
    
    # Evaluate the classifier.
    output = NN.evaluation(input_x, input_y)
    
    tpr = np.load(f"{args.outdir}/signal_significance/tpr.npy")
    fpr = np.load(f"{args.outdir}/signal_significance/fpr.npy")
    plot_SIC(tpr, fpr, args.name, f"{args.outdir}/signal_significance/")

    log.info(f"Evaluation of {args.name} AD done!")
    
if __name__ == "__main__":
    main()