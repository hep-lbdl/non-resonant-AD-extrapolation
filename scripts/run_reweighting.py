import argparse
import numpy as np
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div, plot_multi_dist, plot_SIC
from semivisible_jet.utils import *
from helpers.utils import load_nn_config
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
    "-e",
    "--evaluation",
    action="store_true",
    default=False,
    help="Only evaluate the best reweighting classifier.",
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
    # data, MC, and bkg
    data_context = inputs["data_context"]
    MC_context = inputs["MC_context"]
    bkg_context = inputs["bkg_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_CR = inputs["MC_mask_CR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    bkg_mask_CR = inputs["bkg_mask_CR"]
    bkg_mask_SR = inputs["bkg_mask_SR"]
    # Signal injected
    sig_percent = inputs["sig_percent"]
    inputs.close()

    # Get contexts from data
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]

    # Get contexts from MC
    MC_cond_CR = MC_context[MC_mask_CR]
    MC_cond_SR = MC_context[MC_mask_SR]

    # Get contexts from bkg
    bkg_cond_CR = bkg_context[bkg_mask_CR]
    bkg_cond_SR = bkg_context[bkg_mask_SR]
    
    # define useful variables
    ncond = data_cond_CR.shape[1]
    evaluate_only = args.evaluation

    if evaluate_only:
        # load best trained model
        model_path = f"{args.outdir}/NN_reweight/trained_AD_classifier.pt"
        if os.path.isfile(model_path):
            NN_reweight = torch.load(model_path)
            NN_reweight.to(device)
        else:
            evaluate_only = False

    if not evaluate_only:
        
        # train classifer for reweighting
        log.info("Training a classifer for reweighting...")
        
        # create labels for classifier
        MC_cond_CR_label = np.zeros(MC_cond_CR.shape[0]).reshape(-1,1)
        data_cond_CR_label = np.ones(data_cond_CR.shape[0]).reshape(-1,1)
        
        # create training data set for classifier
        input_cond_x = np.vstack([MC_cond_CR, data_cond_CR])
        input_cond_y = np.vstack([MC_cond_CR_label, data_cond_CR_label])
        
        # train reweighting classifier
        layers, lr, bs = load_nn_config(args.config)
        NN_reweight_train = Classifier(n_inputs=ncond, layers=layers, learning_rate=lr, device=device, outdir=f"{args.outdir}/NN_reweight")
        NN_reweight_train.train(input_cond_x, input_cond_y, save_model=True, batch_size=bs)
        
        # load best trained model
        model_path = f"{args.outdir}/NN_reweight/trained_AD_classifier.pt"
        NN_reweight = torch.load(model_path)
        NN_reweight.to(device)
      
    
    # evaluate classifier and calculate the weights
    w_SR = NN_reweight.evaluation(MC_cond_SR)
    w_SR = (w_SR/(1.-w_SR)).flatten()
    
    if True:

        os.makedirs(f"{args.outdir}/reweighting_plots", exist_ok=True)
    
        # make validation plots in CR
        w_CR = NN_reweight.evaluation(MC_cond_CR)
        w_CR = (w_CR/(1.-w_CR)).flatten()
        
        # only apply to the physics samples
        # plot reweigted distribution
        names = [name_map()["ht"], name_map()["met"], ]
        units = [unit_map()["ht"], unit_map()["met"]]
        ymaxs = [2000, 400]

        for i in [0,1]:

            hlist = [MC_cond_SR[:,i], MC_cond_SR[:,i], bkg_cond_SR[:,i], MC_cond_CR[:,i], MC_cond_CR[:,i], data_cond_CR[:,i]]
            weights = [None, w_SR, None, None, w_CR, None]
            labels = [f"MC SR (num. events: {len(MC_cond_SR[:,i]):.0e})", "Reweighted MC SR", "True bkg SR", f"MC CR (num. events: {len(MC_cond_CR[:,i]):.1e})", "Reweighted MC CR", "Data CR"]
            htype = ["step", "stepfilled", "step"]*2
            lstyle = ["-"]*3 + ["--"]*3
            plot_kwargs = {"title":f"Reweighted MC vs data for {names[i]}, S/B={sig_percent*1e2:.1f}%", "name":f"MC vs data reweighting {names[i]}", "xlabel":f"{names[i]} {units[i]}", "ymin":0, "ymax":ymaxs[i], "outdir":f"{args.outdir}/reweighting_plots"}
            plot_multi_dist(hlist, labels, weights=weights, htype=htype, lstyle=lstyle, **plot_kwargs)


    # save generated samples
    np.savez(f"{args.outdir}/weights.npz", weights = w_SR)
    
    log.info("Reweighting done!")
    
if __name__ == "__main__":
    main()