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
    "-e",
    "--evaluation",
    action="store_true",
    default=False,
    help="Only evaluate the best reweighting classifier.",
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
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values

    if not args.evaluation:
        
        # train classifer for reweighting
        log.info("Training a classifer for reweighting...")
        
        # create labels for classifier
        MC_cond_CR_label = np.zeros(len(MC_cond_CR)).reshape(-1,1)
        data_cond_CR_label = np.ones(len(data_cond_CR)).reshape(-1,1)
        
        # create training data set for classifier
        input_cond_x = np.vstack([MC_cond_CR, data_cond_CR])
        input_cond_y = np.vstack([MC_cond_CR_label, data_cond_CR_label])
        
        # train reweighting classifier
        NN_reweight_train = Classifier(n_inputs=ncond, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/NN_reweight")
        NN_reweight_train.train(input_cond_x, input_cond_y, save_model=True, batch_size=512)
    
        
    # load best trained model
    model_path = f"{args.outdir}/NN_reweight/trained_AD_classifier.pt"
    NN_reweight = torch.load(model_path)
    NN_reweight.to(device)
      
    
    # evaluate classifier and calculate the weights
    w_SR = NN_reweight.evaluation(MC_cond_SR)
    w_SR = (w_SR/(1.-w_SR)).flatten()

    os.makedirs(f"{args.outdir}/reweighting_plots", exist_ok=True)
    
    # make validation plots in CR
    w_CR = NN_reweight.evaluation(MC_cond_CR)
    w_CR = (w_CR/(1.-w_CR)).flatten()
    
    # plot reweigted distribution
    for i in [0,1]:
    
        hlist = [MC_cond_SR[:,i], MC_cond_SR[:,i], bkg_cond_SR[:,i], MC_cond_CR[:,i], MC_cond_CR[:,i], data_cond_CR[:,i]]
        weights = [None, w_SR, None, None, w_CR, None]
        labels = [f"MC SR (Num Events: {len(MC_cond_SR[:,i]):.1e})", "reweighted MC SR", "true bkg SR", f"MC CR (Num Events: {len(MC_cond_CR[:,i]):.1e})", "reweighted MC CR", "data CR"]
        htype = ["step", "stepfilled", "step"]*2
        lstyle = ["-"]*3 + ["--"]*3
        plot_kwargs = {"title":f"Reweighted MC vs data for m{i+1}, S/B={sig_percent*1e2:.3f}%", "name":f"MC vs data reweighting m{i+1}", "xlabel":f"m{i+1}", "ymin":-10, "ymax":10, "outdir":f"{args.outdir}/reweighting_plots"}
        plot_multi_dist(hlist, labels, weights=weights, htype=htype, lstyle=lstyle, **plot_kwargs)


    # save generated samples
    np.savez(f"{args.outdir}/weights.npz", weights = w_SR)
    
    log.info("Reweighting done!")
    
if __name__ == "__main__":
    main()