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
    data_events_SR= inputs["data_events_SR"]
    ideal_bkg_events_SR= inputs["ideal_bkg_events_SR"]
    # done loading
    inputs.close()
    log.info(f"Datset loaded: N data={len(data_events_SR)}, N ideal bkg={len(ideal_bkg_events_SR)}")

    # define number of features
    nfeat = data_events_SR.shape[1]
    
    # create training data set for classifier
    input_x = np.concatenate([ideal_bkg_events_SR, data_events_SR], axis=0)

    # create labels for classifier
    ideal_bkg_SR_label = np.zeros(ideal_bkg_events_SR.shape[0])
    data_SR_label = np.ones(data_events_SR.shape[0])
    input_y = np.hstack([ideal_bkg_SR_label, data_SR_label]).reshape(-1, 1)
    
    # Train the AD Classifier
    log.info("Training a classifer for signal vs background...")
    log.info("\n")
    log.info(f"Ensamble size: {args.trains}")

    for i in range(args.trains):

        NN = Classifier(n_inputs=nfeat, layers=[32, 32], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/signal_significance")
        NN.train(input_x, input_y, save_model=True, n_epochs=200, batch_size=256, model_name=f"{i}")


    log.info("Ideal AD done!")


if __name__ == "__main__":
    main()