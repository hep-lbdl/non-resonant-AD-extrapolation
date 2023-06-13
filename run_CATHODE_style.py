import argparse
import numpy as np
from math import sin, cos, pi
from helpers.SimpleMAF import SimpleMAF
from helpers.Classifier import Classifier
from helpers.plotting import plot_kl_div, plot_multi_dist, plot_SIC
import torch
import os
import sys
import logging


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    action="store",
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
    
    # load input files
    inputs = np.load(args.input)
    # data and MC
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_context = inputs["MC_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_CR = inputs["MC_mask_CR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    inputs.close()
    
    # Get feature and contexts from data
    data_CR = data_feature[data_mask_CR]
    data_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]
    
    # Get only contexts from MC
    MC_cond_CR = MC_context[MC_mask_CR]
    MC_cond_SR = MC_context[MC_mask_SR]
    
    # define useful variables
    nfeat = data_CR.ndim
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values
    
    if args.samples is None:
        
        # train classifer for reweighting
        log.info("Training a classifer for reweighting...")
        # create labels for classifier
        MC_cond_CR_label = np.zeros(len(MC_cond_CR)).reshape(-1,1)
        data_cond_CR_label = np.ones(len(data_cond_CR)).reshape(-1,1)
        
        # create training data set for classifier
        input_m_x = np.vstack([MC_cond_CR, data_cond_CR])
        input_m_y = np.vstack([MC_cond_CR_label, data_cond_CR_label])
        
        # train reweighting classifier
        NN_reweight = Classifier(n_inputs=2, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/reweighting")
        NN_reweight.train(input_m_x, input_m_y)

        # TODO: properly generate test dataset
        
        # evaluate classifier and calculate the weights
        w_MC = NN_reweight.evaluation(MC_cond_SR)
        w_MC = (w_MC/(1.-w_MC)).flatten()
        
        # plot reweigted distribution
        plot_kwargs = {"title":"Reweighted MC vs data in SR for m1", "xlabel":"m1", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting"}
        plot_multi_dist([MC_cond_SR[:,0], MC_cond_SR[:,0], data_cond_SR[:,0]], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)
        
        plot_kwargs = {"title":"Reweighted MC vs data in SR for m2", "xlabel":"m2", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting"}
        plot_multi_dist([MC_cond_SR[:,1], MC_cond_SR[:,1], data_cond_SR[:,1]], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)
        
        # train a NF for density estimation
        logging.info("Training a MAF to learn P(x|m)...")
        # train MAF with data in CR
        MAF = SimpleMAF(num_features = nfeat, num_context=ncond, device=device)
        MAF.train(data=data_CR, cond = data_cond_CR, outdir=args.outdir)
        
        # sample from MAF
        pred_bkg_SR = MAF.sample(1, MC_cond_SR).flatten()

        # save generated samples
        np.savez(f"{args.outdir}/samples_data_SR.npz", samples = pred_bkg_SR, weights = w_MC)
        
    else:
        # load samples
        pred_bkg_SR = np.load(args.samples)["samples"]
        w_MC = np.load(args.samples)["weights"]

    plot_kwargs = {"tag":"2DSR_unweighted", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
    plot_kl_div([data_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
        
    plot_kwargs = {"weights2":[w_MC], "tag":"2DSR", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
    plot_kl_div([data_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
    
    log.info("Training a classifer for signal vs background...")
    
    # create labels for classifier
    pred_bkg_SR_label = np.zeros(pred_bkg_SR.shape)
    data_SR_label = np.ones(data_SR.shape)
    
    # create training data set for classifier
    input_x = np.hstack([pred_bkg_SR, data_SR]).reshape(-1, 1)
    input_y = np.hstack([pred_bkg_SR_label, data_SR_label]).reshape(-1, 1)

    w_data = np.array([1.]*len(data_SR))
    input_weights = np.hstack([w_MC, w_data]).reshape(-1, 1)
    
    # train classifier
    NN = Classifier(n_inputs=1, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/signal_significance")
    NN.train(input_x, input_y, weights=input_weights)
    
    # evaluate classifier
    # TODO: properly generate test dataset
    output = NN.evaluation(input_x, input_y, weights=input_weights)
    
    tpr = np.load(f"{args.outdir}/signal_significance/tpr.npy")
    fpr = np.load(f"{args.outdir}/signal_significance/fpr.npy")
    plot_SIC(tpr, fpr, "CATHODE style", f"{args.outdir}/signal_significance/")

    log.info("CATHODE style extrapolation done!")
    
if __name__ == "__main__":
    main()