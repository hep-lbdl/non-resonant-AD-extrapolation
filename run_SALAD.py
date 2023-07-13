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
    # data and MC
    data_feature = inputs["data_feature"]
    data_context = inputs["data_context"]
    MC_feature = inputs["MC_feature"]
    MC_context = inputs["MC_context"]
    # SR and CR masks
    data_mask_CR = inputs["data_mask_CR"]
    data_mask_SR = inputs["data_mask_SR"]
    MC_mask_CR = inputs["MC_mask_CR"]
    MC_mask_SR = inputs["MC_mask_SR"]
    inputs.close()
    
    # Get feature and contexts from data
    data_feat_CR = data_feature[data_mask_CR]
    data_feat_SR = data_feature[data_mask_SR]
    data_cond_CR = data_context[data_mask_CR]
    data_cond_SR = data_context[data_mask_SR]

    # Get feature and contexts from MC
    MC_feat_CR = MC_feature[MC_mask_CR]
    MC_feat_SR = MC_feature[MC_mask_SR]
    MC_cond_CR = MC_context[MC_mask_CR]
    MC_cond_SR = MC_context[MC_mask_SR]

    # Get only contexts from MC
    MC_cond_CR = MC_context[MC_mask_CR]
    MC_cond_SR = MC_context[MC_mask_SR]

    # define useful variables
    nfeat = data_feat_CR.ndim
    ncond = data_cond_CR.ndim
    num_samples = 1 # can set to higher values

    if args.samples is None:
        
        # train classifer for reweighting
        log.info("Training a classifer for reweighting...")
        
        # create training data set for classifier
        input_cond_CR = np.concatenate([MC_cond_CR, data_cond_CR], axis=0)
        input_feat_CR = np.concatenate([MC_feat_CR, data_feat_CR], axis=0).reshape(-1,1)
        input_x_train_CR = np.concatenate([input_cond_CR, input_feat_CR], axis=1)
        
        # create labels for classifier
        MC_CR_label = np.zeros(len(MC_cond_CR)).reshape(-1,1)
        data_CR_label = np.ones(len(data_cond_CR)).reshape(-1,1)
        
        input_y_train_CR = np.concatenate([MC_CR_label, data_CR_label], axis=0)
        
        # train reweighting classifier
        NN_reweight = Classifier(n_inputs=nfeat + ncond, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/reweighting")
        NN_reweight.train(input_x_train_CR, input_y_train_CR)

        # TODO: properly generate test dataset
        
        # evaluate classifier and calculate the weights
        input_x_test = np.concatenate([MC_cond_SR, MC_feat_SR.reshape(-1,1)], axis=1)
        
        w_MC = NN_reweight.evaluation(input_x_test)
        w_MC = (w_MC/(1.-w_MC)).flatten()
        
        # plot reweigted distribution
        plot_kwargs = {"title":"Reweighted MC vs data in SR for m1", "xlabel":"m1", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting"}
        plot_multi_dist([MC_cond_SR[:,0], MC_cond_SR[:,0], data_cond_SR[:,0]], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)
        
        plot_kwargs = {"title":"Reweighted MC vs data in SR for m2", "xlabel":"m2", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting"}
        plot_multi_dist([MC_cond_SR[:,1], MC_cond_SR[:,1], data_cond_SR[:,1]], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)
        
        plot_kwargs = {"title":"Reweighted MC vs data in SR for x", "xlabel":"x", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}/reweighting"}
        plot_multi_dist([MC_feat_SR, MC_feat_SR, data_feat_SR], ["MC", "reweighted MC", "data"], [None, w_MC, None], **plot_kwargs)

        # save weights
        np.savez(f"{args.outdir}/SALAD_weights.npz", weights = w_MC)
        
    else:
        # load weights
        w_MC = np.load(args.samples)["weights"]

#     plot_kwargs = {"tag":"2DSR_unweighted", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
#     plot_kl_div([data_feat_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
        
#     plot_kwargs = {"weights2":[w_MC], "tag":"2DSR", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
#     plot_kl_div([data_feat_SR], [pred_bkg_SR], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
    
#     plot_kwargs = {"tag":"2DSR_from_truth", "ymin":-15, "ymax":15, "outdir":f"{args.outdir}"}
#     plot_kl_div([data_feat_SR], [pred_bkg_SR_from_truth], "true SR", "gen SR", [0.5], [pi/4], **plot_kwargs)
    
    log.info("Training a classifer for signal vs background...")
    
    # create training data set for classifier
    input_cond_SR = np.concatenate([MC_cond_SR, data_cond_SR], axis=0)
    input_feat_SR = np.concatenate([MC_feat_SR, data_feat_SR], axis=0).reshape(-1,1)
    input_x_train_SR = np.concatenate([input_cond_SR, input_feat_SR], axis=1)
    
    # create labels for classifier
    MC_SR_label = np.zeros(len(MC_cond_SR)).reshape(-1,1)
    data_SR_label = np.ones(len(data_cond_SR)).reshape(-1,1)
    input_y_train_SR = np.concatenate([MC_SR_label, data_SR_label], axis=0)

    w_data = np.array([1.]*len(data_feat_SR))
    input_weights = np.hstack([w_MC, w_data]).reshape(-1, 1)
    
    # train classifier for x, m1 and m2
    NN = Classifier(n_inputs=nfeat + ncond, layers=[64,128,64], learning_rate=1e-4, device=device, outdir=f"{args.outdir}/signal_significance")
    NN.train(input_x_train_SR, input_y_train_SR, weights=input_weights)
    
    # evaluate classifier
    # TODO: properly generate test dataset
    output = NN.evaluation(input_x_train_SR, input_y_train_SR, weights=input_weights)
    
    tpr = np.load(f"{args.outdir}/signal_significance/tpr.npy")
    fpr = np.load(f"{args.outdir}/signal_significance/fpr.npy")
    plot_SIC(tpr, fpr, "SALAD", f"{args.outdir}/signal_significance/")

    log.info("SALAD extrapolation done!")
    
if __name__ == "__main__":
    main()