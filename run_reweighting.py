import argparse
import numpy as np
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
    '-m', 
    "--model",
    action="store",
    default=None,
    help='Directly load trained model.'
)
parser.add_argument(
    "-s",
    "--save",
    action="store_true",
    default=False,
    help="Save trained model.",
)
parser.add_argument(
    "-o",
    "--outdir",
    action="store",
    default="reweighting_run",
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
    data_context = inputs["data_context"]
    MC_context = inputs["MC_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_CR = inputs["MC_mask_CR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    inputs.close()

    # Get contexts from data
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]

    # Get only contexts from MC
    MC_cond_CR = MC_context[MC_mask_CR]
    MC_cond_SR = MC_context[MC_mask_SR]

    # define useful variables
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values

    if args.model is None:
        
        # train classifer for reweighting
        log.info("Training a classifer for reweighting...")
        
        # create labels for classifier
        MC_cond_CR_label = np.zeros(len(MC_cond_CR)).reshape(-1,1)
        data_cond_CR_label = np.ones(len(data_cond_CR)).reshape(-1,1)
        
        # create training data set for classifier
        input_cond_x = np.vstack([MC_cond_CR, data_cond_CR])
        input_cond_y = np.vstack([MC_cond_CR_label, data_cond_CR_label])
        
        # train reweighting classifier
        NN_reweight = Classifier(n_inputs=ncond, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/NN_reweight")
        NN_reweight.train(input_cond_x, input_cond_y, save_model=args.save, batch_size=512, min_delta=0.0002)
    
    else:
        
        # load trained model
        NN_reweight = args.model
      
    
    # evaluate classifier and calculate the weights
    w_MC = NN_reweight.evaluation(MC_cond_SR)
    w_MC = (w_MC/(1.-w_MC)).flatten()

    os.makedirs(f"{args.outdir}/reweighting_plots", exist_ok=True)
    
    # plot reweigted distribution
    plot_kwargs = {"title":"Reweighted MC vs data in SR for m1", "xlabel":"m1", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting_plots"}
    plot_multi_dist([MC_cond_SR[:,0], MC_cond_SR[:,0], data_cond_SR[:,0]], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)

    plot_kwargs = {"title":"Reweighted MC vs data in SR for m2", "xlabel":"m2", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting_plots"}
    plot_multi_dist([MC_cond_SR[:,1], MC_cond_SR[:,1], data_cond_SR[:,1]], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)


    # save generated samples
    np.savez(f"{args.outdir}/weights.npz", weights = w_MC)


    log.info("Reweighting done!")
    
if __name__ == "__main__":
    main()