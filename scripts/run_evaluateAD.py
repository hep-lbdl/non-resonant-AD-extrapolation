import argparse
import numpy as np
from math import sin, cos, pi
from helpers.Classifier import Classifier
from helpers.plotting import plot_SIC, plot_all_variables
from helpers.utils import get_roc_curve
import torch
import os
import sys
import glob
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

    # # selecting appropriate device
    CUDA = torch.cuda.is_available()
    print("cuda available:", CUDA)
    device = torch.device("cuda" if CUDA else "cpu")
    
    os.makedirs(args.outdir, exist_ok=True)
        
    # load input files
    inputs = np.load(args.input)
    # sig and bkg
    sig_SR= inputs["sig_events_SR"][:, 2:]
    bkg_SR= inputs["bkg_events_SR"][:, 2:]
    inputs.close()
    
    # Create testing data set for classifier
    input_x = np.concatenate([sig_SR, bkg_SR], axis=0)
    
    # Create labels for classifier
    sig_SR_label = np.ones(sig_SR.shape[0])
    bkg_SR_label = np.zeros(bkg_SR.shape[0])
    input_y = np.concatenate([sig_SR_label, bkg_SR_label], axis=0).reshape(-1,1)

    if args.verbose:
        # Plot varibles
        var_names = ["m_jj", "tau21_j1", "tau21_j2", "tau32_j1", "tau32_j2"]
        sig_list = sig_SR.T
        bkg_list = bkg_SR.T
        plot_kwargs = {"name":f"sig_vs_bkg_testset_for_evaluation", "title":f"N sig={len(sig_SR)}, N bkg={len(bkg_SR)}", "outdir":args.outdir}
        plot_all_variables(sig_list, bkg_list, var_names, **plot_kwargs)

    # Load the trained model
    logging.info("Loading a trained NN...")
    
    model_paths = glob.glob(f"{args.outdir}/signal_significance/trained_AD_classifier*.pt")

    output_list = []
    n_model = 0
    for model in model_paths:
        
        NN = torch.load(model)
        NN.to(device)
        NN.set_outdir(f"{args.outdir}/signal_significance")

        log.info(f"Model {model} loaded.")

        # Evaluate the classifier.
        output = NN.evaluation(input_x)
        n_model += 1
        output_list.append(output.flatten())
    
    output_arr = np.stack(output_list, axis=0)
    mean_output = np.mean(output_arr, axis=0)

    get_roc_curve(mean_output, input_y.flatten(), outdir=f"{args.outdir}/signal_significance", model_name="")
    
    tpr = np.load(f"{args.outdir}/signal_significance/tpr.npy")
    fpr = np.load(f"{args.outdir}/signal_significance/fpr.npy")
    plot_SIC(tpr, fpr, args.name, f"{args.outdir}/signal_significance/")

    log.info(f"Evaluation of {args.name} AD done!")
    
if __name__ == "__main__":
    main()