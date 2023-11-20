import argparse
import numpy as np
from helpers.Classifier import Classifier
from helpers.utils import load_nn_config
from helpers.process_data import *
from helpers.plotting import *
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
    data_events = inputs["data_events"] if args.toy else inputs["data_events"][:, 2:]
    ideal_bkg_events = inputs["ideal_bkg_events"] if args.toy else inputs["ideal_bkg_events"][:, 2:]

    data_events_SR = data_events[toy_SR_mask(data_events)] if args.toy else data_events[phys_SR_mask(data_events)]
    ideal_bkg_events_SR = ideal_bkg_events[toy_SR_mask(ideal_bkg_events)] if args.toy else ideal_bkg_events[phys_SR_mask(ideal_bkg_events)]

    inputs.close()
    
    return data_events_SR, ideal_bkg_events_SR


def main():

    # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
        
    data_events_SR, ideal_bkg_events_SR = load_samples()
    
    if args.toy:
        plot_kl_div_toy(data_events_SR[:,0], ideal_bkg_events_SR[:,0], "data SR", "ideal bkg SR", name="m1", title="idealAD inputs", bins=30, outdir=args.outdir)
        plot_kl_div_toy(data_events_SR[:,1], ideal_bkg_events_SR[:,1], "data SR", "ideal bkg SR", name="m2", title="idealAD inputs", bins=30, outdir=args.outdir)
        plot_kl_div_toy(data_events_SR[:,2], ideal_bkg_events_SR[:,2], "data SR", "ideal bkg SR", name="x", title="idealAD inputs", bins=30, outdir=args.outdir)
    
    # define number of features
    nfeat = data_events_SR.shape[1]
    log.info(f"Datset loaded: N data={len(data_events_SR)}, N ideal bkg={len(ideal_bkg_events_SR)}, {nfeat} features")
    
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
    
    layers, lr, bs = load_nn_config(args.config)

    for i in range(args.trains):

        NN = Classifier(n_inputs=nfeat, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/signal_significance")
        NN.train(input_x, input_y, save_model=True, n_epochs=200, batch_size=bs, model_name=f"{i}")


    log.info("Ideal AD done!")


if __name__ == "__main__":
    main()