import argparse
import numpy as np
from helpers.Classifier import Classifier
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
    '-t', 
    "--trains",
    action="store",
    type=int,
    default=1,
    help='Number of trainings.'
)
parser.add_argument(
    "-c",
    "--config",
    action="store",
    default=None,
    help="Classifier config file",
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
    sig_SR= inputs["sig_events_SR"]
    bkg_SR= inputs["bkg_events_SR"]
    inputs.close()

    # define number of features
    nfeat = sig_SR.shape[1]
    
    log.info("Training a classifer for signal vs background...")
    
    # create training data set for classifier
    input_x = np.concatenate([sig_SR, bkg_SR], axis=0)
    
    # Create labels for classifier
    sig_SR_label = np.ones(sig_SR.shape[0])
    bkg_SR_label = np.zeros(bkg_SR.shape[0])
    input_y = np.concatenate([sig_SR_label, bkg_SR_label], axis=0).reshape(-1,1)
    
    # Train the AD Classifier
    log.info(f"Training a classifer for signal vs background...")
    log.info(f"Ensamble size: {args.trains}")

    layers, lr, bs = load_nn_config(args.config)

    for i in range(args.trains):
        NN = Classifier(n_inputs=nfeat, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/signal_significance")
        NN.train(input_x, input_y, save_model=True, n_epochs=200, batch_size=bs, model_name=f"{i}")

    log.info("Fully supervised learning done!")
    
if __name__ == "__main__":
    main()