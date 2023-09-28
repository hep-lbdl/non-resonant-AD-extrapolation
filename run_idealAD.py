import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div, plot_multi_dist, plot_SIC
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
    '-s', 
    "--samples",
    action="store",
    default=None,
    help='Directly load generated samples.'
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
    # data and bkg
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    bkg_feature = inputs["bkg_feature"]
    bkg_context = inputs["bkg_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    bkg_mask_CR = inputs["bkg_mask_CR"]
    bkg_mask_SR = inputs["bkg_mask_SR"]
    inputs.close()

    # Get feature and contexts from data
    data_feat_CR = data_feature[data_mask_CR]
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]
    
    # Get feature and contexts from bkg-only data
    bkg_feat_CR = bkg_feature[bkg_mask_CR]
    bkg_feat_SR = bkg_feature[bkg_mask_SR]
    bkg_cond_CR = bkg_context[bkg_mask_CR]
    bkg_cond_SR = bkg_context[bkg_mask_SR]

    # define useful variables
    nfeat = data_feat_CR.ndim
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values

    pred_bkg_SR = bkg_feat_SR.flatten()
    
    log.info("Training a classifer for signal vs background...")
    
    # create training data set for classifier
    input_feat_x = np.hstack([pred_bkg_SR, data_feat_SR]).reshape(-1, 1)
    input_cond_x = np.vstack([bkg_cond_SR, data_cond_SR])
    input_x = np.concatenate([input_feat_x, input_cond_x], axis=1)
    
    # create labels for classifier
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape)
    data_feat_SR_label = np.ones(data_feat_SR.shape)
    input_y = np.hstack([pred_bkg_SR_label, data_feat_SR_label]).reshape(-1, 1)

    # train classifier for x, m1 and m2
    NN = Classifier(n_inputs=nfeat+ncond, layers=[128, 128], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/signal_significance")
    NN.train(input_x, input_y, save_model=True, n_epochs=200, batch_size=512)
    

    log.info("Ideal AD done!")
    
if __name__ == "__main__":
    main()